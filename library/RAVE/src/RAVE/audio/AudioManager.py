import yaml
import os
import time
import numpy as np
from collections import namedtuple
from matplotlib import pyplot as plt

# PyODAS dependencies
from pyodas.core import KissMask
from .GPU import Stft, IStft, SpatialCov, DelaySum
from pyodas.utils import CONST, generate_mic_array, get_delays_based_on_mic_array

# Internal dependancies
from .IO.IO_manager import IOManager
from .Neural_Network.AudioModel import AudioModel
from .Beamformer.Beamformer import Beamformer

# PyTorch dependencies
import torch
import torchaudio
from torchmetrics import SignalNoiseRatio, SignalDistortionRatio

# Named Tuples
SourceTuple = namedtuple("SourceTuple", "file source stft")
SinkTuple = namedtuple("SinkTuple", "sink scm_target scm_noise istft")

# DEFINES
TIME = float("inf")
# TIME = 5
TARGET = [0, 1, 0.5]
EPSILON = 1e-9

SHOW_GRAPHS = False


class AudioManager:
    """
    Class used as manager for all audio processes, containing the main loop of execution for the application.

    Args:
            debug (bool): Run in debug mode or not.
            mask (bool): Run with KISS mask (True) or model (False)
            use_timers (bool): Get and print timers.
    """

    chunk_size = 256
    frame_size = chunk_size * 2
    window = "sqrt_hann"
    scm_weight = 0.1

    # Class variables init empty
    in_subfolder_path = None
    out_subfolder_path = None
    file_configs_path = None
    file_configs = None
    source = None
    original_sink = None
    target = TARGET
    current_delay = None
    source_dict = {}
    sink_dict = {}
    source_list = []
    timers = {}
    transformation = None

    # Spectrogram variables
    noise_mask_np = None
    speech_mask_np = None
    speech_mask_np_gt = None
    noise_mask_np_gt = None
    speech_np_gt = None

    # SDR and SNR variables
    old_snr = None
    old_sdr = None
    new_snr = None
    new_sdr = None

    # Speech and noise variables
    speech_file = None
    noise_file = None

    # Set variables to false until changed at init or runtime
    sim_input = False
    is_delays = False
    passthrough_mode = True
    gain = 1

    # Masks and models
    masks = None
    model = None
    last_h = None
    delay_and_sum = None

    def __init__(self, debug=False, mask=False, use_timers=False, model_path=None):

        # Argument variables
        self.debug = debug
        self.mask = mask
        self.get_timers = use_timers
        self.model_path = model_path

        # General configs
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.config_path = os.path.join(self.path, "audio_general_configs.yaml")
        with open(self.config_path, "r") as stream:
            try:
                self.general_configs = yaml.safe_load(stream)
                if self.debug:
                    print(self.general_configs)
            except yaml.YAMLError as exc:
                print(exc)
                exit()

        # Transfer general configs
        self.mic_dict = self.general_configs["mic_dict"]
        self.mic_array = generate_mic_array(self.mic_dict)
        self.channels = self.mic_dict["nb_of_channels"]

        self.output_stereo = self.general_configs["output_stereo"]
        self.out_channels = 2 if self.output_stereo else 1
        self.default_output_dir = self.general_configs["default_output_dir"]

        self.use_beamformer = self.general_configs["use_beamformer"]
        self.speech_and_noise = self.general_configs["use_groundtruth"]
        self.print_specs = self.general_configs["print_specs"]
        self.print_sdr = self.general_configs["print_sdr"]

        self.jetson_source = self.general_configs["jetson_source"]
        self.jetson_sink = self.general_configs["jetson_sink"]
        self.sink_list = self.general_configs["sinks"]

        # wav parameters
        self.wav_params_multi = (
            self.channels,
            2,
            CONST.SAMPLING_RATE,
            0,
            "NONE",
            "not compressed",
        )
        self.wav_params_out = (
            self.out_channels,
            self.wav_params_multi[1],
            self.wav_params_multi[2],
            self.wav_params_multi[3],
            self.wav_params_multi[4],
            self.wav_params_multi[5],
        )

        # Check if device has cuda
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"

        # IO
        self.io_manager = IOManager()

        # Beamformer
        self.beamformer = Beamformer(name='mvdr', channels=self.channels, device=self.device)

    def init_wav_input(self, name, file):
        """
        Initialise wav file input.

        Args:
            name (str): Name of source.
            file (str): Path of .wav file to use as input.

        Returns:
            source: WavSource object created.
        """
        source = self.io_manager.add_source(source_name=name, source_type="sim", file=file, chunk_size=self.chunk_size)
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
        # TODO: get microphone matrix from configs file
        self.target = input_configs["source_dir"]

        return source

    def init_wav_output(self, name, wav_params):
        """
        Initialise wav file output.

        Args:
            name (str): Name of sink.
            wav_params (Tuple): wav file parameters.

        Returns:
            sink: WavSink object created.
        """

        # Get output subfolder
        if self.out_subfolder_path is None:
            if self.in_subfolder_path:
                self.out_subfolder_path = self.in_subfolder_path
            else:
                index = 0
                subfolder_path = os.path.join(self.default_output_dir, "", "run")
                while os.path.exists(f"{subfolder_path}_{index}"):
                    index += 1
                self.out_subfolder_path = f"{subfolder_path}_{index}"
            if not os.path.exists(self.out_subfolder_path):
                os.makedirs(self.out_subfolder_path)

        # WavSink
        sink_path = os.path.join(self.out_subfolder_path, f"{name}.wav")
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
        # Setup mic input index
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
            queue_size=50,
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

        # Setup playback output index
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
        for source_tuple in self.source_dict.values():
            if source_tuple.stft in source_tuple:
                source_tuple.stft.reset()

        # Reset SCMs and ISTFTs
        for sink_tuple in self.sink_dict.values():
            if sink_tuple.scm_target is not None:
                sink_tuple.scm_target.reset()
            if sink_tuple.scm_noise is not None:
                sink_tuple.scm_noise.reset()
            if sink_tuple.istft is not None:
                sink_tuple.istft.reset()

        # Reset accumulating spectrogram variables
        if self.print_specs:
            self.noise_mask_np = None
            self.speech_mask_np = None
            self.speech_mask_np_gt = None
            self.noise_mask_np_gt = None
            self.speech_np_gt = None

        # Reset timers
        self.timers = {}

    def set_target(self, target):
        """
        Sets the target on which to perform the sound enhancement.

        Args:
            target (ndarray): Array containing the target direction, time delay between microphones or None.
        """
        self.target = target

    def set_gain(self, gain):
        """
        Sets the gain to control output volume.

        Args:
            gain (float): Array containing the target direction, time delay between microphones or None.
        """
        self.gain = gain

    def output_sink(self, data, sink_name):
        """
        Checks if sink exists, if yes, output to it

        Args:
            data: Data to be written to sink
            sink_name: name of sink to which output data

        Returns:
            True if success (sink exists), else False
        """

        # TODO: add stereo output

        # Output fully processed data
        # out = y
        # if self.out_channels == 1:
        #     channel_to_use = out.shape[0] // 2
        #     out = out[channel_to_use]
        # elif self.out_channels == 2:
        #     out = np.array([out[0], out[-1]])

        if sink_name in self.sink_dict:
            self.sink_dict[sink_name].sink(data)
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

        # Save and display
        save_name = os.path.join(self.out_subfolder_path, "out_spec_plots.png")
        plt.savefig(fname=save_name)
        if SHOW_GRAPHS:
            plt.show(block=True)
        plt.close()

    def calculate_sdr(self):

        # TODO: Runtime sdr instead of at the end
        offset = 370
        # offset = 0

        target_path = os.path.join(self.in_subfolder_path, 'target.wav')
        original_path = os.path.join(self.in_subfolder_path, 'audio.wav')
        output_path = os.path.join(self.out_subfolder_path, 'output.wav')

        # New
        prediction, _ = torchaudio.load(output_path)
        length = prediction.shape[1]
        prediction = prediction[:, offset:length]

        target, _ = torchaudio.load(target_path)
        target = torch.unsqueeze(torch.mean(target, dim=0), dim=0)
        target = target[:, 0:length - offset]

        # Old
        original, _ = torchaudio.load(original_path)
        original = original[:, :length]
        original = torch.mean(original, dim=0, keepdim=True)

        target_original, _ = torchaudio.load(target_path)
        target_original = torch.unsqueeze(torch.mean(target_original, dim=0), dim=0)
        target_original = target_original[:, :length]

        sdr = SignalDistortionRatio()
        self.old_sdr = sdr(original, target_original).item()
        self.new_sdr = sdr(prediction, target).item()
        if self.debug:
            print("SDR: Before: ", self.old_sdr, " After: ", self.new_sdr)

        snr = SignalNoiseRatio()
        self.old_snr = snr(original, target_original).item()
        self.new_snr = snr(prediction, target).item()
        if self.debug:
            print("SNR: Before: ", self.old_snr, " After: ", self.new_snr)

        # Add to config file
        self.file_configs['old_snr'] = self.old_snr
        self.file_configs['new_snr'] = self.new_snr
        self.file_configs['old_sdr'] = self.old_sdr
        self.file_configs['new_sdr'] = self.new_sdr
        with open(self.file_configs_path, "w") as outfile:
            yaml.dump(self.file_configs, outfile, default_flow_style=None)

    def compute_masks(self, signal):
        """
        Computes masks (through KISS or neural network) for speech and noise on noisy speech signal.

        Args:
            signal (ndarray): Noisy speech frequential signal on which to obtain masks.
        Returns:
            Speech and noise masks.
        """
        if self.mask:
            # If using KISS-Mask instead of NN
            self.check_time(name="mask", is_start=True)
            speech_mask, noise_mask = self.masks(signal, self.target)
            self.check_time(name="mask", is_start=False)
        else:
            self.check_time(name="data", is_start=True)
            # Delay and sum
            if self.is_delays:
                delay = self.current_delay
            else:
                target_np = np.array([self.target])
                delay = get_delays_based_on_mic_array(target_np, self.mic_array, self.frame_size)[0]
                delay = torch.from_numpy(delay).to(self.device).type(torch.float32)
            sum = self.delay_and_sum(signal, delay)
            sum_db = 10 * torch.log10(torch.abs(sum) ** 2 + EPSILON)

            # Mono
            signal_mono_db = 10 * torch.log10(torch.unsqueeze(torch.sum(torch.abs(signal) ** 2, dim=0), dim=0) + 1e-09)
            concat_spec = torch.cat([signal_mono_db, sum_db], dim=1)
            concat_spec = torch.reshape(concat_spec, (1, 1, concat_spec.shape[1], 1))

            self.check_time(name="network", is_start=True)
            with torch.no_grad():
                noise_mask, self.last_h = self.model(concat_spec, self.last_h)
            self.check_time(name="network", is_start=False)

            noise_mask = torch.squeeze(noise_mask)
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
        s = torch.from_numpy(self.source_dict["speech"].source()).to(self.device)
        n = torch.from_numpy(self.source_dict["noise"].source()).to(self.device)

        # Temporal to frequential
        S = self.source_dict["speech"].stft(s)
        N = self.source_dict["noise"].stft(n)

        # Output speech and noise directly (source, sink, STFT and ISTFT sanity check)
        s_out = self.sink_dict["speech_out"].istft(S).detach().cpu().numpy()
        n_out = self.sink_dict["noise_out"].istft(N).detach().cpu().numpy()
        self.output_sink(data=s_out, sink_name="speech_out")
        self.output_sink(data=n_out, sink_name="noise_out")

        # Save speech to spectrogram
        if self.print_specs:
            s_mean = torch.mean(S, dim=0)
            s_log = 20 * torch.log10(abs(s_mean) + EPSILON)
            s_mean = torch.unsqueeze(s_log, dim=1)
            if self.speech_np_gt is not None:
                self.speech_np_gt = torch.cat([self.speech_np_gt, s_mean], dim=1)
            else:
                self.speech_np_gt = s_mean

        # Prepare target as done for the model
        # Get energy
        S_sq = torch.abs(S) ** 2
        N_sq = torch.abs(N) ** 2
        # Mono
        S_mono = torch.mean(S_sq, dim=0, keepdim=False)
        N_mono = torch.mean(N_sq, dim=0, keepdim=False)
        # Calculate energy ratio
        noise_mask_gt = N_mono / (N_mono + S_mono + EPSILON)
        speech_mask_gt = S_mono / (N_mono + S_mono + EPSILON)

        # Save for spectrogram
        if self.print_specs:
            speech_mask_gt_exp = torch.unsqueeze(speech_mask_gt, dim=1)
            noise_mask_gt_exp = torch.unsqueeze(noise_mask_gt, dim=1)
            if self.speech_mask_np_gt is not None and self.noise_mask_np_gt is not None:
                self.speech_mask_np_gt = torch.cat([self.speech_mask_np_gt, speech_mask_gt_exp], dim=1)
                self.noise_mask_np_gt = torch.cat([self.noise_mask_np_gt, noise_mask_gt_exp], dim=1)
            else:
                self.speech_mask_np_gt = speech_mask_gt_exp
                self.noise_mask_np_gt = noise_mask_gt_exp

        # Spatial covariance matrices
        if self.use_beamformer:
            target_scm_gt = self.sink_dict["output_gt"].scm_target(audio_signal_f, speech_mask_gt)
            noise_scm_gt = self.sink_dict["output_gt"].scm_noise(audio_signal_f, noise_mask_gt)

        # Beamform
        if self.use_beamformer:
            Y_gt = self.beamformer(signal=audio_signal_f, target_scm=target_scm_gt, noise_scm=noise_scm_gt)
        else:
            Y_gt = audio_signal_f * speech_mask_gt
            Y_gt = torch.mean(Y_gt, dim=0)

        # ISTFT and save
        y_gt = self.sink_dict["output_gt"].istft(Y_gt)
        self.output_sink(data=y_gt.detach().cpu().numpy(), sink_name="output_gt")

    def init_model(self):
        """ Init model or masks, depending on which using """
        if self.mask:
            self.masks = KissMask(self.mic_array, buffer_size=30)
        else:
            self.model = AudioModel(input_size=514, hidden_size=512, num_layers=2)
            self.model.to(self.device)
            if self.debug:
                print(self.model)
            self.model.load_best_model(self.model_path, self.device)
            self.delay_and_sum = DelaySum(self.frame_size, self.device)

    def initialise_audio(self, source, sinks=None, overwrite_sinks=False, save_path=None):
        """
        Method to initialise audio after start. Uses configs written in 'audio_general_configs.yaml'.

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
        self.source_list = source
        if self.source_list["type"] == "sim":
            # Load sim (wav) input
            self.source_dict["audio"] = SourceTuple(
                self.source_list["file"],
                self.init_wav_input(name=self.source_list["name"], file=self.source_list["file"]),
                Stft(self.channels, self.frame_size, self.window, self.device)
            )

            # Try to load configs
            self.file_configs_path = os.path.join(os.path.dirname(self.source_list["file"]), 'configs.yaml')
            with open(self.file_configs_path, "r") as stream:
                try:
                    self.file_configs = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    print("Couldn't load audio file configs.")
                    print(exc)
                    exit()

            # If sim, try loading speech and noise ground truths
            if self.speech_and_noise:
                try:
                    # Load speech file
                    self.speech_file = os.path.join(os.path.split(self.source_list["file"])[0], "speech.wav")
                    self.source_dict["speech"] = SourceTuple(
                        self.speech_file,
                        self.init_wav_input(name="speech_gt_source", file=self.speech_file),
                        Stft(self.channels, self.frame_size, self.window, self.device)
                    )
                    # Load noise file
                    self.noise_file = os.path.join(os.path.split(self.source_list["file"])[0], "noise.wav")
                    self.source_dict["noise"] = SourceTuple(
                        self.noise_file,
                        self.init_wav_input(name="noise_gt_source", file=self.noise_file),
                        Stft(self.channels, self.frame_size, self.window, self.device)
                    )
                    # Set argument to True if successful
                    self.speech_and_noise = True
                except Exception as e:
                    if self.debug:
                        print(e)
                    self.speech_and_noise = False
        else:
            # Load mic input
            self.source_dict["audio"] = SourceTuple(
                None,
                self.init_mic_input(name=self.source_list["name"], src_index=self.source_list["idx"]),
                Stft(self.channels, self.frame_size, self.window, self.device)
            )

        # Outputs
        if sinks:
            if overwrite_sinks:
                self.sink_list = sinks
            else:
                self.sink_list.extend(sinks)
        for sink in self.sink_list:
            # Get sink object
            if sink["type"] == "sim":
                wav_params = self.wav_params_out if sink["beamform"] else self.wav_params_multi
                sink_obj = self.init_wav_output(name=sink["name"], wav_params=wav_params)
            else:
                sink_obj = self.init_play_output(name=sink["name"], sink_index=sink["idx"])
            # Add ISTFT and SCM (if needed for beamforming)
            if not sink["beamform"]:
                scm_target = scm_noise = None
                istft = IStft(self.channels, self.frame_size, self.chunk_size, self.window, self.device)
            else:
                scm_target = SpatialCov(self.channels, self.frame_size, weight=self.scm_weight, device=self.device)
                scm_noise = SpatialCov(self.channels, self.frame_size, weight=self.scm_weight, device=self.device)
                istft = IStft(self.out_channels, self.frame_size, self.chunk_size, self.window, self.device)
            # Add sink
            self.sink_dict[sink["name"]] = SinkTuple(sink_obj, scm_target, scm_noise, istft)

        # Model
        self.init_model()

    def init_app(self, save_input, save_output, passthrough_mode, output_path="", gain=1):
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
        self.target = None
        self.gain = gain

        # Model
        self.init_model()

        # Init source
        mic_source = self.init_mic_input(name=self.jetson_source["name"], src_index=self.jetson_source["idx"])
        self.source_dict[self.jetson_source["name"]] = SourceTuple(
            None,
            mic_source,
            Stft(self.channels, self.frame_size, self.window, self.device)
        )

        # Init playback sink
        self.sink_dict[self.jetson_sink["name"]] = SinkTuple(
            self.init_play_output(name=self.jetson_sink["name"], sink_index=self.jetson_sink["idx"]),
            SpatialCov(self.channels, self.frame_size, weight=self.scm_weight, device=self.device),
            SpatialCov(self.channels, self.frame_size, weight=self.scm_weight, device=self.device),
            IStft(self.out_channels, self.frame_size, self.chunk_size, self.window, self.device)
        )

        # Init simulated sources (.wav) if needed
        if save_input:
            self.sink_dict["original"] = SinkTuple(
                self.init_wav_output(name="original", path=output_path, wav_params=self.wav_params_multi),
                None,
                None,
                None
            )
        if save_output:
            self.sink_dict["original"] = SinkTuple(
                self.init_wav_output(name="output", path=output_path, wav_params=self.wav_params_out),
                None,
                None,
                None
            )

        return mic_source

    def start_app(self):
        """
        Simplified main function used by app. Similar to main_loop, but with less debugging options.
        """
        samples = 0
        max_time = 0
        self.loop_i = 0
        while samples / CONST.SAMPLING_RATE < TIME:
            self.check_time(name="loop", is_start=True)

            self.current_delay = self.target

            # Record from microphone
            x = self.source_dict[self.jetson_source["name"]].source()
            if x is None:
                if self.debug:
                    print("End of transmission. Closing.")
                break
            self.output_sink(data=x, sink_name="original")

            # Check if no target selected
            if self.current_delay is None:
                if self.passthrough_mode:
                    out = x[0][None, ...]
                else:
                    out = np.zeros((1, self.chunk_size))
                out *= self.gain
                self.output_sink(data=out, sink_name=self.jetson_sink["name"])
                self.output_sink(data=out, sink_name="output")
                continue

            # Switch to GPU (if possible)
            x = torch.from_numpy(x).to(self.device)

            # Temporal to Frequential
            self.check_time(name="stft", is_start=True)
            X = self.source_dict[self.jetson_source["name"]].stft(x)
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
                target_scm = self.sink_dict[self.jetson_sink["name"]].scm_target(X, speech_mask)
                noise_scm = self.sink_dict[self.jetson_sink["name"]].scm_noise(X, noise_mask)
            self.check_time(name="scm", is_start=False)

            # MVDR
            self.check_time(name="beamformer", is_start=True)
            if self.use_beamformer:
                Y = self.beamformer(signal=X, target_scm=target_scm, noise_scm=noise_scm)
            else:
                Y = X * speech_mask
                Y = torch.mean(Y, dim=0)
            self.check_time(name="beamformer", is_start=False)

            # ISTFT
            self.check_time(name="istft", is_start=True)
            y = self.sink_dict[self.jetson_sink["name"]].istft(Y)
            self.check_time(name="istft", is_start=False)

            # Output enhanced speech
            y *= self.gain
            y = y.detach().cpu().numpy()
            self.output_sink(data=y, sink_name=self.jetson_sink["name"])
            self.output_sink(data=y, sink_name="output")

            # Calculate loop time
            loop_time = self.check_time(name="loop", is_start=False)
            if self.get_timers and loop_time is not None:
                max_time = loop_time if loop_time > max_time else max_time

            if self.debug and samples % (self.chunk_size * 50) == 0:
                print(f"Samples processed: {samples}")
                if self.get_timers:
                    print(f"Time for loop: {loop_time} ms.")

            samples += self.chunk_size
            self.loop_i = samples / self.chunk_size

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

        samples = 0
        max_time = 0
        self.loop_i = 0
        while samples / CONST.SAMPLING_RATE < TIME:
            self.check_time(name="loop", is_start=True)

            # Record from microphone
            x = self.source_dict["audio"].source()
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

            # To GPU (if possible)
            x = torch.from_numpy(x).to(self.device)

            # Temporal to Frequential
            self.check_time(name="stft", is_start=True)
            X = self.source_dict["audio"].stft(x)
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
                target_scm = self.sink_dict["output"].scm_target(X, speech_mask)
                noise_scm = self.sink_dict["output"].scm_noise(X, noise_mask)
            self.check_time(name="scm", is_start=False)

            # MVDR
            self.check_time(name="beamformer", is_start=True)
            if self.use_beamformer:
                Y = self.beamformer(signal=X, target_scm=target_scm, noise_scm=noise_scm)
            else:
                Y = X * speech_mask
                if self.output_stereo:
                    Y = torch.split(Y, 4)
                    Y = [torch.mean(Y[0], dim=0), torch.mean(Y[1], dim=0)]
                    Y = torch.stack(Y, dim=0)
                else:
                    Y = torch.mean(Y, dim=0)
                # TODO: stereo output
            self.check_time(name="beamformer", is_start=False)

            # ISTFT
            self.check_time(name="istft", is_start=True)
            y = self.sink_dict["output"].istft(Y)
            self.check_time(name="istft", is_start=False)

            # To CPU and output
            y = y.detach().cpu().numpy()
            self.output_sink(data=y, sink_name="output")

            # Check additional sources and output in additional sinks
            # Speech and noise ground truth beamforming
            if self.speech_and_noise:
                self.calculate_groundtruth(X)

            # Calculate loop time
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

        if self.print_sdr:
            # TODO: Check input is sim
            if self.speech_and_noise:
                self.calculate_sdr()
            else:
                print(f"Cannot measure SDR if no ")

        print(f"Audio_Manager: Finished running main_audio ({samples} samples)")
        if self.get_timers:
            self.print_times(max_time=max_time)