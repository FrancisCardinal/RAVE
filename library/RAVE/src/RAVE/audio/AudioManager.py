import torch
import torchaudio
import yaml
import os
import numpy as np
import time

from matplotlib import pyplot as plt

from pyodas.core import Stft, IStft, KissMask, SpatialCov, DelaySum
from pyodas.utils import CONST, generate_mic_array, get_delays_based_on_mic_array

from .IO.IO_manager import IOManager
from .Neural_Network.AudioModel import AudioModel
from .Beamformer.Beamformer import Beamformer

TIME = float("inf")
# TIME = 30
FILE_PARAMS_MONO = (
    1,
    2,
    CONST.SAMPLING_RATE,
    0,
    "NONE",
    "not compressed",
)
TARGET = np.array([0, 1, 0.5])

EPSILON = 1e-9
MIN_DB_VAL = 20 * np.log10(EPSILON)


THOMAS = False
SHOW_GRAPHS = False


class AudioManager:
    """
    Class used as manager for all audio processes, containing the main loop of execution for the application.

    Args:
            debug (bool): Run in debug mode or not.
            mask (bool): Run with KISS mask (True) or model (False)
            use_timers (bool): Get and print timers.
    """

    scm_weight = 0.1
    window = "hann"

    def __init__(self, debug=False, mask=False, use_timers=False):

        # TODO: ADD PRIVATE ATTRIBUTES (_)

        # Argument variables
        self.debug = debug
        self.mask = mask
        self.get_timers = use_timers

        # Class variables init empty
        self.file_params_output = FILE_PARAMS_MONO
        self.in_subfolder_path = None
        self.out_subfolder_path = None
        self.source = None
        self.original_sink = None
        self.target = TARGET
        self.source_dict = {}
        self.sink_dict = {}
        self.timers = {}
        self.transformation = None
        # Spectrogram variables
        self.noise_mask_np = None
        self.speech_mask_np = None
        self.speech_mask_np_gt = None
        self.noise_mask_np_gt = None
        self.speech_np_gt = None
        self.torch_speech_mask_np = None
        self.torch_noise_mask_np = None

        # Set variables to false until changed at init or runtime
        self.sim_input = False
        self.is_delays = False
        self.passthrough_mode = True

        # TODO: ADD OUT CHANNELS
        # General configs
        self.path = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(self.path, "audio_general_configs.yaml")
        with open(config_path, "r") as stream:
            try:
                self.general_configs = yaml.safe_load(stream)
                if self.debug:
                    print(self.general_configs)
            except yaml.YAMLError as exc:
                print(exc)
                exit()
        # Get general configs
        self.mic_dict = self.general_configs["mic_dict"]
        self.mic_array = generate_mic_array(self.mic_dict)
        self.channels = self.mic_dict["nb_of_channels"]
        self.out_channels = self.general_configs["out_channels"]
        self.chunk_size = self.general_configs["chunk_size"]
        self.frame_size = self.chunk_size * 4
        self.beamformer_name = self.general_configs["beamformer"]
        self.jetson_source = self.general_configs["jetson_source"]
        self.jetson_sink = self.general_configs["jetson_sink"]

        self.file_params_multi = (
            4,
            2,
            CONST.SAMPLING_RATE,
            0,
            "NONE",
            "not compressed",
        )

        # Individual configs
        config_path = os.path.join(self.path, "audio_indiv_configs.yaml")
        with open(config_path, "r") as stream:
            try:
                self.individual_configs = yaml.safe_load(stream)
                if self.debug:
                    print(self.individual_configs)
            except yaml.YAMLError as exc:
                print(exc)
                exit()
        # Get individual configs
        self.default_output_dir = self.individual_configs["default_output_dir"]
        self.model_path = self.individual_configs["model_path"]
        # self.mic_array_index = self.individual_configs['mic_array_index']
        self.use_beamformer = self.individual_configs["use_beamformer"]
        self.speech_and_noise = self.individual_configs["use_groundtruth"]
        if self.speech_and_noise:
            self.speech_file = None
            self.noise_file = None
        self.torch_gt = self.individual_configs["use_torch"]
        self.print_specs = self.individual_configs["print_specs"]
        self.source_list = self.individual_configs["source"]
        self.sink_list = self.individual_configs["sinks"]

        # Check if device has cuda
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"

        # IO
        self.io_manager = IOManager()

        # # Masks
        # # TODO: Add abstraction to model for info not needed in main script
        if self.mask:
            self.masks = None
        else:
            self.model = None
            self.last_h = None
            self.delay_and_sum = None

        # Beamformer
        # TODO: simplify beamformer abstraction kwargs
        self.beamformer = Beamformer(name=self.beamformer_name, channels=self.channels)

    def init_sim_input(self, name, file):
        """
        Initialise simulated input (from .wav file).

        Args:
            name (str): Name of source.
            file (str): Path of .wav file to use as input.

        Returns:
            source: WavSource object created.
        """
        source = self.io_manager.add_source(source_name=name, source_type="sim", file=file, chunk_size=self.chunk_size)
        self.file_params_output = (
            1,
            source.wav_params[1],
            source.wav_params[2],
            source.wav_params[3],
            source.wav_params[4],
            source.wav_params[5],
        )
        self.file_params_multi = source.wav_params
        self.sim_input = True

        # Get input config file info
        # Paths
        self.in_subfolder_path = os.path.split(file)[0]
        config_path = os.path.join(self.in_subfolder_path, "configs.yaml")
        with open(config_path, "r") as stream:
            try:
                input_configs = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                exit()
        # TODO: GET MICROPHONE MATRIX FROM CONGIS FILE?
        self.target = input_configs["source_dir"]

        return source

    def init_sim_output(self, name, wav_params, path=None):
        """
        Initialise simulated output (.wav file).

        Args:
            name (str): Name of sink.
            path (str): Path where to save .wav file.

        Returns:
            sink: WavSink object created.
        """
        # Get output subfolder
        if not self.out_subfolder_path:
            if path:
                self.out_subfolder_path = path
            else:
                if self.in_subfolder_path:
                    self.out_subfolder_path = self.in_subfolder_path
                else:
                    index = 0
                    subfolder_path = os.path.join(self.default_output_dir, "run")
                    while os.path.exists(f"{subfolder_path}_{index}"):
                        index += 1
                    self.out_subfolder_path = f"{subfolder_path}_{index}"
        if not os.path.exists(self.out_subfolder_path):
            os.makedirs(self.out_subfolder_path)

        # WavSink
        sink_path = os.path.join(self.out_subfolder_path, f"{name}.wav")
        # TODO: SETUP FILE PARAMS IF SIM SINK WITH NO SIM SOURCE
        sink = self.io_manager.add_sink(
            sink_name=name, sink_type="sim", file=sink_path, wav_params=wav_params, chunk_size=self.chunk_size
        )
        return sink

    def init_mic_input(self, name, src_index):
        """
        Initialise microphone input source.

        Args:
            name (str): Name of microphone source.
            src_index (str): Contains device index to use as source (default uses default system device).

        Returns:
            source: MicSource object created
        """
        if src_index == "default":
            src_index = None
        else:
            src_index = int(src_index)
        source = self.io_manager.add_source(
            source_name=name,
            source_type="mic",
            mic_index=src_index,
            channels=self.channels,
            mic_arr=self.mic_dict,
            chunk_size=self.chunk_size,
        )
        return source

    def init_play_output(self, name, sink_index):
        """
        Initialise playback (speaker) output sink.

        Args:
            name (str): Name of microphone source.
            sink_index (str): Contains device index to use as sink (default uses default system device).

        Returns:
            source: PlaybackSink object created
        """
        if sink_index == "default":
            sink_index = None
        else:
            sink_index = int(sink_index)
        sink = self.io_manager.add_sink(
            sink_name=name,
            sink_type="play",
            device_index=sink_index,
            channels=self.out_channels,
            chunk_size=self.chunk_size,
        )
        return sink

    def reset_manager(self):
        """
        Resets the manager and containing objects that have memory (STFT, SCM, ISTFT, Spectrograms).
        """
        self.out_subfolder_path = None
        self.target = None

        # Reset STFTs
        for source_obj in self.source_dict.values():
            if "stft" in source_obj:
                source_obj["stft"].reset()

        # Reset SCMs and ISTFTs
        for sink_obj in self.sink_dict.values():
            if "scm_target" in sink_obj:
                sink_obj["scm_target"].reset()
            if "scm_noise" in sink_obj:
                sink_obj["scm_noise"].reset()
            if "istft" in sink_obj:
                sink_obj["istft"].reset()

        # Reset accumulating spectrogram variables
        if self.print_specs:
            self.noise_mask_np = None
            self.speech_mask_np = None
            self.speech_mask_np_gt = None
            self.noise_mask_np_gt = None
            self.speech_np_gt = None
            self.torch_speech_mask_np = None
            self.torch_noise_mask_np = None

    def set_target(self, target):
        """
        Sets the target on which to perform the sound enhancement.

        Args:
            target (ndarray): Array containing the target direction, time delay between microphones or None.
        """
        self.target = target

    def output_sink(self, data, sink_name):
        """
        Checks if sink exists, if yes, output to it

        Args:
            data: Data to be written to sink
            sink_name: name of sink to which output data

        Returns:
            True if success (sink exists), else False
        """

        # TODO: ADD POSSIBILITY TO CONTROL OUTPUT CHANNELS
        # TODO: ADD STEREO OUTPUT

        if sink_name in self.sink_dict:
            chosen_sink = self.sink_dict[sink_name]["sink"]
            chosen_sink(data)
            return True
        else:
            return False

    def check_time(self, name, is_start, unit="ms"):
        """
        Run timers and save in dict.

        Args:
            name (str): Name of timer on which to do some action.
            is_start (bool): Consider time as start time (True) or end time (False).
            unit (str): String containing units to use for time management. Currently only accepts ms, the default.

        Returns:
            If start, returns current time, if end, returns elapsed time.
        """
        if not self.get_timers or self.loop_i < 5:
            return

        if unit == "ms":
            unit_factor = 1000

        if name not in self.timers:
            self.timers[name] = {"unit": unit, "total": 0, "call_cnt": 0}
            # self.timers[name]['unit'] = unit
            # self.timers[name]['total'] = 0
            # self.timers[name]['call_cnt'] = 0

        current_time = time.perf_counter() * unit_factor
        if is_start:
            self.timers[name]["start"] = current_time
            return current_time
        else:
            self.timers[name]["end"] = current_time
            elapsed_time = current_time - self.timers[name]["start"]
            self.timers[name]["total"] += elapsed_time
            self.timers[name]["call_cnt"] += 1
            return elapsed_time

    def get_time(self, name, is_mean):
        """
        Prints the time associated to the timer name.

        Args:
            name (str): Name of the timer of which to print time.
            is_mean (bool): Bool determining if the time wanted is mean (divided by call_count) or total.

        Returns:
            Returns a string containing the name, time and units. Returns 'Timer not found.' if name not found in dict.
        """
        if name in self.timers:
            time = self.timers[name]["total"]
            unit = self.timers[name]["unit"]

            mean_str = "Total time"
            if is_mean:
                time /= self.timers[name]["call_cnt"]
                mean_str = "Mean time per loop"

            time_string = f'Timer: {mean_str} "{name}" = {time:.3f} {unit}'
        else:
            time_string = f'Timer "{name}" not found.'

        return time_string

    def print_times(self, max_time):
        """
        Print timers.

        Args:
            max_time (float): Max time for one loop.
        """
        # TODO: AUTOMATE WITH LOOP
        print(self.get_time(name="init_audio", is_mean=False))
        print(self.get_time(name="loop", is_mean=False))
        print(self.get_time(name="loop", is_mean=True))
        print(f"Timer: Longest time per loop: {max_time} ms.")
        print(self.get_time(name="beamformer", is_mean=True))
        if self.mask:
            print(self.get_time(name="masks", is_mean=True))
        else:
            print(self.get_time(name="data", is_mean=True))
            print(self.get_time(name="network", is_mean=True))

    def print_spectrograms(self):
        """
        Plots and prints (and shows if SHOW_GRAPHS) various spectrograms.
        """
        subplot_cnt = 3
        speech_idx = 0
        speech_pred_idx = 1
        noise_pred_idx = 2
        if self.speech_and_noise:
            subplot_cnt += 2
            speech_target_idx = 1
            speech_pred_idx += 1
            noise_target_idx = 3
            noise_pred_idx += 2
            if self.torch_gt:
                subplot_cnt += 2
                speech_torch_idx = 1
                speech_target_idx += 1
                speech_pred_idx += 1
                noise_torch_idx = 4
                noise_target_idx += 2
                noise_pred_idx += 2

        # Set subplot figure
        fig, axs = plt.subplots(subplot_cnt)
        fig.set_figheight(9)
        fig.suptitle("Spectrogrammes")

        # Add basic plots (speech ground truth, speech mask and noise mask)
        axs[speech_idx].set_ylabel("Speech")
        axs[speech_idx].pcolormesh(self.speech_np_gt, shading="gouraud")
        axs[speech_pred_idx].set_ylabel("Pred S")
        axs[speech_pred_idx].pcolormesh(self.speech_mask_np, shading="gouraud", vmin=0, vmax=1)
        axs[noise_pred_idx].set_ylabel("Pred N")
        axs[noise_pred_idx].pcolormesh(self.noise_mask_np, shading="gouraud", vmin=0, vmax=1)
        axs[noise_pred_idx].set_xlabel("Temps(s)")

        if self.speech_and_noise:
            axs[speech_target_idx].set_ylabel("Target S")
            axs[speech_target_idx].pcolormesh(self.speech_mask_np_gt, shading="gouraud", vmin=0, vmax=1)
            axs[noise_target_idx].set_ylabel("Target N")
            axs[noise_target_idx].pcolormesh(self.noise_mask_np_gt, shading="gouraud", vmin=0, vmax=1)

            if self.torch_gt:
                axs[speech_torch_idx].set_ylabel("Torch S")
                axs[speech_torch_idx].pcolormesh(self.torch_speech_mask_np, shading="gouraud", vmin=0, vmax=1)
                axs[noise_torch_idx].set_ylabel("Torch N")
                axs[noise_torch_idx].pcolormesh(self.torch_noise_mask_np, shading="gouraud", vmin=0, vmax=1)

        # Save and display
        save_name = os.path.join(self.out_subfolder_path, "out_spec_plots.jpg")
        plt.savefig(fname=save_name)
        if SHOW_GRAPHS:
            plt.show(block=True)
        plt.close()

    def compute_masks(self, signal):
        """
        Computes masks (through KISS or neural network) for speech and noise on noisy speech signal.

        Args:
            signal (ndarray): Noisy speech frequential signal on which to obtain masks.
        Returns:
            Speech and noise masks.
        """
        if self.mask:
            self.check_time(name="mask", is_start=True)
            speech_mask, noise_mask = self.masks(signal, self.target)
            self.check_time(name="mask", is_start=False)

            return speech_mask, noise_mask
        else:
            self.check_time(name="data", is_start=True)
            # Delay and sum
            if self.is_delays:
                delay = self.target
            else:
                target_np = np.array([self.target])
                delay = get_delays_based_on_mic_array(target_np, self.mic_array, self.frame_size)
            sum = self.delay_and_sum(signal, delay[0])
            sum_tensor = torch.tensor(sum).to(self.device)
            sum_db = 20 * torch.log10(torch.abs(sum_tensor) + EPSILON)

            # Mono
            signal_tensor = torch.tensor(signal).to(self.device)
            signal_mono = torch.mean(signal_tensor, dim=0, keepdims=True)
            signal_mono_db = 20 * torch.log10(torch.abs(signal_mono) + EPSILON)

            concat_spec = torch.cat([signal_mono_db, sum_db], dim=1)
            concat_spec = torch.reshape(concat_spec, (1, 1, concat_spec.shape[1], 1))

            self.check_time(name="network", is_start=True)
            with torch.no_grad():
                noise_mask, self.last_h = self.model(concat_spec, self.last_h)
            self.check_time(name="network", is_start=False)
            noise_mask = torch.squeeze(noise_mask).cpu().numpy()
            speech_mask = 1 - noise_mask
            self.check_time(name="data", is_start=False)

            return speech_mask, noise_mask

    def calculate_groundtruth(self, audio_signal_f):
        """
        Uses speech and noise dataset samples to calculate ground truth.

        Args:
            audio_signal_f (ndarray): Array containing input audio signal in frequential domain.
        """
        # Get signal
        s = self.source_dict["speech"]["src"]()
        n = self.source_dict["noise"]["src"]()

        # Temporal to frequential
        S = self.source_dict["speech"]["stft"](s)
        N = self.source_dict["noise"]["stft"](n)

        # Output speech and noise directly (source, sink, STFT and ISTFT sanity check)
        s_out = self.sink_dict["speech_out"]["istft"](S)
        n_out = self.sink_dict["noise_out"]["istft"](N)
        self.output_sink(data=s_out, sink_name="speech_out")
        self.output_sink(data=n_out, sink_name="noise_out")

        # Save speech to spectrogram
        if self.print_specs:
            s_mean = np.mean(S, axis=0)
            s_log = 20 * np.log10(abs(s_mean) + EPSILON)
            s_mean = np.expand_dims(s_log, axis=1)
            if self.speech_np_gt is not None:
                self.speech_np_gt = np.append(self.speech_np_gt, s_mean, axis=1)
            else:
                self.speech_np_gt = s_mean

        # Prepare target as done for the model
        # Get energy
        S_sq = np.abs(S) ** 2
        N_sq = np.abs(N) ** 2
        # Mono
        S_mono = np.mean(S_sq, axis=0, keepdims=False)
        N_mono = np.mean(N_sq, axis=0, keepdims=False)
        # Calculate energy ratio
        noise_mask_gt = N_mono / (N_mono + S_mono + EPSILON)
        speech_mask_gt = S_mono / (N_mono + S_mono + EPSILON)

        # Save for spectrogram
        if self.print_specs:
            speech_mask_gt_exp = np.expand_dims(speech_mask_gt, axis=1)
            noise_mask_gt_exp = np.expand_dims(noise_mask_gt, axis=1)
            if self.speech_mask_np_gt is not None and self.noise_mask_np_gt is not None:
                self.speech_mask_np_gt = np.append(self.speech_mask_np_gt, speech_mask_gt_exp, axis=1)
                self.noise_mask_np_gt = np.append(self.noise_mask_np_gt, noise_mask_gt_exp, axis=1)
            else:
                self.speech_mask_np_gt = speech_mask_gt_exp
                self.noise_mask_np_gt = noise_mask_gt_exp

        # Spatial covariance matrices
        if self.use_beamformer:
            target_scm_gt = self.sink_dict["output_gt"]["scm_target"](audio_signal_f, speech_mask_gt)
            noise_scm_gt = self.sink_dict["output_gt"]["scm_noise"](audio_signal_f, noise_mask_gt)

        # Beamform
        if self.use_beamformer:
            Y_gt = self.beamformer(signal=audio_signal_f, target_scm=target_scm_gt, noise_scm=noise_scm_gt)
        else:
            Y_gt = audio_signal_f * speech_mask_gt
            Y_gt = np.mean(Y_gt, axis=0)

        # ISTFT and save
        y_gt = self.sink_dict["output_gt"]["istft"](Y_gt)
        self.output_sink(data=y_gt, sink_name="output_gt")

    def torch_init(self):
        """
        Executes torch load from simulated source to sanity check files and procedures.

        Returns:
            Returns the speech and noise spectrograms (if self.speech_and_noise).
        """

        # Audio
        audio_data_t, _ = torchaudio.load(self.source_dict["audio"]["file"])
        audio_data = self.transformation(audio_data_t)
        audio_data = torch.abs(audio_data) ** 2
        audio_data = torch.mean(audio_data, dim=0, keepdim=False)
        speech_data = None
        noise_data = None
        if self.speech_and_noise:
            # Speech
            speech_data_t, _ = torchaudio.load(self.speech_file)
            speech_data_f = self.transformation(speech_data_t)
            speech_data = torch.abs(speech_data_f) ** 2
            speech_data = torch.mean(speech_data, dim=0, keepdim=False)
            # Noise
            noise_data_t, _ = torchaudio.load(self.noise_file)
            noise_data_f = self.transformation(noise_data_t)
            noise_data = torch.abs(noise_data_f) ** 2
            noise_data = torch.mean(noise_data, dim=0, keepdim=False)
            # Show
            # TODO: FIX TORCH SPECTROGRAMS
            if self.print_specs:
                fig, axs = plt.subplots(3)
                fig.suptitle("Spectrogrammes")
                axs[0].pcolormesh(audio_data.cpu().numpy(), shading="gouraud", vmin=0, vmax=1)
                axs[1].pcolormesh(speech_data.cpu().numpy(), shading="gouraud", vmin=0, vmax=1)
                axs[2].pcolormesh(noise_data.cpu().numpy(), shading="gouraud", vmin=0, vmax=1)
                axs[2].set_xlabel("Temps(s)")
                axs[0].set_ylabel("Audio")
                axs[1].set_ylabel("Speech")
                axs[2].set_ylabel("Noise")
        else:
            if self.print_specs:
                fig, axs = plt.subplots(1)
                fig.suptitle("Spectrogrammes")
                axs[0].pcolormesh(audio_data, shading="gouraud", vmin=0, vmax=1)
                axs[0].set_xlabel("Temps(s)")
                axs[0].set_ylabel("Audio")
        save_name = os.path.join(self.out_subfolder_path, "torch_plots.jpg")
        plt.savefig(fname=save_name)
        if SHOW_GRAPHS:
            plt.show(block=True)
        plt.close()

        return speech_data, noise_data

    def torch_run_loop(self, audio_signal, loop_idx, speech_data, noise_data):
        """
        Runs torch loop for sanity check and debugging purposes (if sim input).

        Args:
            audio_signal (ndarray): Audio signal loaded from main source (PyODAS)
            loop_idx (int): Counter indicating at which chunk the program is.
            speech_data (ndarray): Obtained from torch_init, spectrogram of speech signal.
            noise_data (ndarray): Obtained from torch_init, spectrogram of noise signal.
        """
        # Get sample from torch loaded
        torch_S = speech_data[:, loop_idx].numpy()
        torch_N = noise_data[:, loop_idx].numpy()

        # Energy ratio
        torch_speech_mask_gt = torch_S / (torch_N + torch_S + EPSILON)
        torch_noise_mask_gt = torch_N / (torch_N + torch_S + EPSILON)

        # Save for spectrograms
        if self.print_specs:
            torch_speech_mask_gt_exp = np.expand_dims(torch_speech_mask_gt, axis=1)
            torch_noise_mask_gt_exp = np.expand_dims(torch_noise_mask_gt, axis=1)
            if self.torch_speech_mask_np is not None and self.torch_noise_mask_np is not None:
                self.torch_speech_mask_np = np.append(self.torch_speech_mask_np, torch_speech_mask_gt_exp, axis=1)
                self.torch_noise_mask_np = np.append(self.torch_noise_mask_np, torch_noise_mask_gt_exp, axis=1)
            else:
                self.torch_speech_mask_np = torch_speech_mask_gt_exp
                self.torch_noise_mask_np = torch_noise_mask_gt_exp

        # SCM
        if self.use_beamformer:
            torch_target_scm_gt = self.sink_dict["torch_gt"]["scm_target"](audio_signal, torch_S)
            torch_noise_scm_gt = self.sink_dict["torch_gt"]["scm_noise"](audio_signal, torch_N)

        # Beamform
        if self.use_beamformer:
            torch_Y = self.beamformer(
                signal=audio_signal, target_scm=torch_target_scm_gt, noise_scm=torch_noise_scm_gt
            )
        else:
            torch_Y = audio_signal * torch_speech_mask_gt
            torch_Y = np.mean(torch_Y, axis=0)

        # ISTFT and save
        torch_y = self.sink_dict["torch_gt"]["istft"](torch_Y)
        self.output_sink(data=torch_y, sink_name="torch_gt")

    def initialise_audio(self, source=None, sinks=None, overwrite_sinks=False, save_path=None):
        """
        Method to initialise audio after start. Uses configs written in 'audio_general_configs.yaml' and
        'audio_indiv_configs.yaml)

        Args:
            source (dict): Overwrite configs source.
            sinks (list[dict]): Overwrite or add more sinks to the config sinks.
            overwrite_sinks (bool): Whether to overwrite or append sinks.
            save_path (str): Path at which to save the simulated sinks (.wav files).
        """

        # self.check_time(name='init_audio', is_start=True)

        # Params
        if save_path:
            self.out_subfolder_path = save_path

        # Inputs
        if source:
            self.source_list = source
        if self.source_list["type"] == "sim":
            self.source_dict["audio"] = {
                "file": self.source_list["file"],
                "src": self.init_sim_input(name=self.source_list["name"], file=self.source_list["file"]),
                "stft": Stft(self.channels, self.frame_size, self.window),
            }
            if self.speech_and_noise:
                # Try loading speech and noise ground truths if present
                try:
                    # Load speech file
                    self.speech_file = os.path.join(os.path.split(self.source_list["file"])[0], "speech.wav")
                    self.source_dict["speech"] = {
                        "file": self.speech_file,
                        "src": self.init_sim_input(name="speech_gt_source", file=self.speech_file),
                        "stft": Stft(self.channels, self.frame_size, self.window),
                    }
                    # Load noise file
                    self.noise_file = os.path.join(os.path.split(self.source_list["file"])[0], "noise.wav")
                    self.source_dict["noise"] = {
                        "file": self.noise_file,
                        "src": self.init_sim_input(name="noise_gt_source", file=self.noise_file),
                        "stft": Stft(self.channels, self.frame_size, self.window),
                    }
                    # Set argument to True if successful
                    self.speech_and_noise = True
                except Exception as e:
                    if self.debug:
                        print(e)
                    self.speech_and_noise = False
        else:
            self.source_dict["audio"] = {
                "src": self.init_mic_input(name=self.source_list["name"], src_index=self.source_list["idx"]),
                "stft": Stft(self.channels, self.frame_size, self.window),
            }

        # Outputs
        # TODO: ADD OUTPUT CHANNELS CONTROL
        if sinks:
            if overwrite_sinks:
                self.sink_list = sinks
            else:
                self.sink_list.extend(sinks)
        for sink in self.sink_list:
            # Add sink
            self.sink_dict[sink["name"]] = {}
            if sink["type"] == "sim":
                wav_params = self.file_params_output if sink["beamform"] else self.file_params_multi
                self.sink_dict[sink["name"]]["sink"] = self.init_sim_output(name=sink["name"], wav_params=wav_params)
            else:
                self.sink_dict[sink["name"]]["sink"] = self.init_play_output(name=sink["name"], sink_index=sink["idx"])
            # Add ISTFT and SCM (if needed for beamforming)
            if sink["beamform"]:
                self.sink_dict[sink["name"]]["scm_target"] = SpatialCov(
                    self.channels, self.frame_size, weight=self.scm_weight
                )
                self.sink_dict[sink["name"]]["scm_noise"] = SpatialCov(
                    self.channels, self.frame_size, weight=self.scm_weight
                )
                self.sink_dict[sink["name"]]["istft"] = IStft(1, self.frame_size, self.chunk_size, self.window)
            else:
                self.sink_dict[sink["name"]]["istft"] = IStft(
                    self.channels, self.frame_size, self.chunk_size, self.window
                )

        # Model
        if self.mask:
            self.masks = KissMask(self.mic_array, buffer_size=30)
        else:
            self.model = AudioModel(input_size=1026, hidden_size=128, num_layers=2)
            self.model.to(self.device)
            if self.debug:
                print(self.model)
            self.model.load_best_model(self.model_path, self.device)
            self.delay_and_sum = DelaySum(self.frame_size)

        if self.torch_gt:
            self.transformation = torchaudio.transforms.Spectrogram(
                n_fft=self.frame_size,
                hop_length=self.chunk_size,
                power=None,
            )

        # self.check_time(name='init_audio', is_start=False)

    def init_app(self, save_input, save_output, passthrough_mode, output_path=""):
        """
        Function used to init jetson application.

        Args:
            save_input (bool): Whether to save the input (original) or not.
            save_output (bool): Whether to save the output or not.
            passthrough_mode (bool): Whether for "no target mode" to be passthrough or muted.
            output_path (str): Path at which to save the simulated sinks (.wav files).
        """
        self.is_delays = True
        self.passthrough_mode = passthrough_mode

        # Init source
        self.source_dict[self.jetson_source["name"]] = {
            "src": self.init_mic_input(name=self.jetson_source["name"], src_index=self.jetson_source["idx"]),
            "stft": Stft(self.channels, self.frame_size, self.window),
        }

        # Init playback sink
        self.sink_dict[self.jetson_sink["name"]] = {
            "sink": self.init_play_output(name=self.jetson_sink["name"], sink_index=self.jetson_sink["idx"]),
            "scm_target": SpatialCov(self.channels, self.frame_size, weight=self.scm_weight),
            "scm_noise": SpatialCov(self.channels, self.frame_size, weight=self.scm_weight),
            "istft": IStft(1, self.frame_size, self.chunk_size, self.window),
        }

        # Init simulated sources (.wav) if needed
        if save_input:
            self.sink_dict["original"] = {
                "sink": self.init_sim_output(name="original", path=output_path, wav_params=self.file_params_multi)
            }
        if save_output:
            self.sink_dict["output"] = {
                "sink": self.init_sim_output(name="output", path=output_path, wav_params=self.file_params_output)
            }

    def start_app(self):
        """
        Simplified main function used by app. Similar to main_loop, but with less debugging options.
        """
        samples = 0
        max_time = 0
        loop_i = 0
        while samples / CONST.SAMPLING_RATE < TIME:
            self.check_time(name="loop", is_start=True)

            # Record from microphone
            x = self.source_dict[self.jetson_source["name"]]["src"]()
            if x is None:
                if self.debug:
                    print("End of transmission. Closing.")
                break
            self.output_sink(data=x, sink_name="original")

            # Check if no target selected
            if self.target is None:
                if self.passthrough_mode:
                    out = x[0]
                else:
                    out = np.zeros((1, self.chunk_size))
                self.output_sink(data=out, sink_name=self.jetson_sink["name"])
                self.output_sink(data=out, sink_name="output")
                continue

            # Temporal to Frequential
            self.check_time(name="stft", is_start=True)
            X = self.source_dict[self.jetson_source["name"]]["stft"](x)
            self.check_time(name="stft", is_start=False)

            # Compute the masks
            speech_mask, noise_mask = self.compute_masks(signal=X)
            # Save masks for spectrogram if in debug
            if self.print_specs:
                speech_mask_exp = np.expand_dims(np.real(speech_mask), axis=1)
                noise_mask_exp = np.expand_dims(np.real(noise_mask), axis=1)
                if self.noise_mask_np is not None and self.speech_mask_np is not None:
                    self.speech_mask_np = np.append(self.speech_mask_np, speech_mask_exp, axis=1)
                    self.noise_mask_np = np.append(self.noise_mask_np, noise_mask_exp, axis=1)
                else:
                    self.speech_mask_np = speech_mask_exp
                    self.noise_mask_np = noise_mask_exp

            # Spatial covariance matrices
            self.check_time(name="scm", is_start=True)
            if self.use_beamformer:
                target_scm = self.sink_dict[self.jetson_sink["name"]]["scm_target"](X, speech_mask)
                noise_scm = self.sink_dict[self.jetson_sink["name"]]["scm_noise"](X, noise_mask)
            self.check_time(name="scm", is_start=False)

            # MVDR
            self.check_time(name="beamformer", is_start=True)
            if self.use_beamformer:
                Y = self.beamformer(signal=X, target_scm=target_scm, noise_scm=noise_scm)
            else:
                Y = X * speech_mask
                Y = np.mean(Y, axis=0)
            self.check_time(name="beamformer", is_start=False)

            # ISTFT
            self.check_time(name="istft", is_start=True)
            y = self.sink_dict[self.jetson_sink["name"]]["istft"](Y)
            self.check_time(name="istft", is_start=False)

            # Output enhanced speech
            self.output_sink(data=y, sink_name=self.jetson_sink["name"])
            self.output_sink(data=y, sink_name="output")

            loop_time = self.check_time(name="loop", is_start=False)
            if self.get_timers:
                max_time = loop_time if loop_time > max_time else max_time

            if self.debug and samples % (self.chunk_size * 50) == 0:
                print(f"Samples processed: {samples}")
                if self.get_timers:
                    print(f"Time for loop: {loop_time} ms.")

            samples += self.chunk_size
            loop_i += 1

        # Plot target + prediction spectrograms
        if self.print_specs:
            self.print_spectrograms()

        print(f"Audio_Manager: Finished running main_audio ({samples} samples)")
        if self.get_timers:
            self.print_times(max_time=max_time)

    def main_loop(self):
        """
        Main audio loop.
        """

        # Check torchaudio values for sanity-check when whole input is known
        if self.torch_gt and self.speech_and_noise:
            speech_data, noise_data = self.torch_init()

        samples = 0
        max_time = 0
        self.loop_i = 0
        while samples / CONST.SAMPLING_RATE < TIME:
            self.check_time(name="loop", is_start=True)

            # Record from microphone
            x = self.source_dict["audio"]["src"]()
            if x is None:
                if self.debug:
                    print("End of transmission. Closing.")
                break
            self.output_sink(data=x, sink_name="original")

            # Check if no target selected
            if self.target is None:
                if self.passthrough_mode:
                    out = x[0]
                else:
                    out = np.zeros((1, self.chunk_size))
                self.output_sink(data=out, sink_name="output")
                continue

            # Temporal to Frequential
            self.check_time(name="stft", is_start=True)
            X = self.source_dict["audio"]["stft"](x)
            self.check_time(name="stft", is_start=False)

            # Compute the masks
            speech_mask, noise_mask = self.compute_masks(signal=X)

            # Save masks for spectrogram if in debug
            if self.print_specs:
                speech_mask_exp = np.expand_dims(np.real(speech_mask), axis=1)
                noise_mask_exp = np.expand_dims(np.real(noise_mask), axis=1)
                if self.noise_mask_np is not None and self.speech_mask_np is not None:
                    self.speech_mask_np = np.append(self.speech_mask_np, speech_mask_exp, axis=1)
                    self.noise_mask_np = np.append(self.noise_mask_np, noise_mask_exp, axis=1)
                else:
                    self.speech_mask_np = speech_mask_exp
                    self.noise_mask_np = noise_mask_exp

            # Spatial covariance matrices
            self.check_time(name="scm", is_start=True)
            if self.use_beamformer:
                target_scm = self.sink_dict["output"]["scm_target"](X, speech_mask)
                noise_scm = self.sink_dict["output"]["scm_noise"](X, noise_mask)
            self.check_time(name="scm", is_start=False)

            # MVDR
            self.check_time(name="beamformer", is_start=True)
            if self.use_beamformer:
                Y = self.beamformer(signal=X, target_scm=target_scm, noise_scm=noise_scm)
            else:
                Y = X * speech_mask
                Y = np.mean(Y, axis=0)
            self.check_time(name="beamformer", is_start=False)

            # ISTFT
            self.check_time(name="istft", is_start=True)
            y = self.sink_dict["output"]["istft"](Y)
            self.check_time(name="istft", is_start=False)

            # Output fully processed data
            # out = y
            # if self.out_channels == 1:
            #     channel_to_use = out.shape[0] // 2
            #     out = out[channel_to_use]
            # elif self.out_channels == 2:
            #     out = np.array([out[0], out[-1]])
            self.output_sink(data=y, sink_name="output")

            # Check additional sources and output in additional sinks
            # Speech and noise ground truth beamforming
            if self.speech_and_noise:
                self.calculate_groundtruth(X)

                # Offline torch ground truth
                if self.torch_gt:
                    self.torch_run_loop(X, self.loop_i, speech_data, noise_data)

            loop_time = self.check_time(name="loop", is_start=False)
            if self.get_timers and self.loop_i >= 5:
                max_time = loop_time if loop_time > max_time else max_time

            if self.debug and samples % (self.chunk_size * 50) == 0:
                print(f"Samples processed: {samples}")
                if self.get_timers and self.loop_i > 5:
                    print(f"Time for loop: {loop_time:.4f} ms.")

            samples += self.chunk_size
            self.loop_i += 1

        # Plot target + prediction spectrograms
        if self.print_specs:
            self.print_spectrograms()

        print(f"Audio_Manager: Finished running main_audio ({samples} samples)")
        if self.get_timers:
            self.print_times(max_time=max_time)
