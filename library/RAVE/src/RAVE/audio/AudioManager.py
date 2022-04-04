import pyodas.core
import torch
import torchaudio
import yaml
import os
import numpy as np
import time

from matplotlib import pyplot as plt

from pyodas.core import (
    Stft,
    IStft,
    KissMask,
    SpatialCov,
    DelaySum
)
from pyodas.utils import CONST, generate_mic_array, load_mic_array_from_ressources, get_delays_based_on_mic_array
from pyodas.visualize import Monitor, Spectrogram

from .IO.IO_manager import IOManager
from .Neural_Network.AudioModel import AudioModel
from .Beamformer.Beamformer import Beamformer

TIME = float('inf')
FILE_PARAMS = (
            4,
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

    def __init__(self, debug, mask, use_timers):

        # Argument variables
        self.debug = debug
        self.mask = mask
        self.get_timers = use_timers

        # Class variables init empty
        self.file_params = None
        self.in_subfolder_path = None
        self.out_subfolder_path = None
        self.source = None
        self.original_sink = None
        self.target = None
        self.source_dict = {}
        self.sink_dict = {}
        self.timers = {}
        self.transformation = None

        # Set variables to false until changed at init or runtime
        self.sim_input = False
        self.speech_and_noise = False
        self.torch_gt = True

        # TODO: ADD OUT CHANNELS
        # General configs
        self.path = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(self.path, 'audio_general_configs.yaml')
        with open(config_path, "r") as stream:
            try:
                self.general_configs = yaml.safe_load(stream)
                if self.debug:
                    print(self.general_configs)
            except yaml.YAMLError as exc:
                print(exc)
                exit()
        # Get general configs
        self.mic_dict = self.general_configs['mic_dict']
        self.mic_array = generate_mic_array(self.mic_dict)
        self.channels = self.mic_dict['nb_of_channels']
        self.out_channels = self.general_configs['out_channels']
        self.chunk_size = self.general_configs['chunk_size']
        self.frame_size = self.chunk_size * 4
        self.beamformer_name = self.general_configs['beamformer']
        self.jetson_source = self.general_configs['jetson_source']
        self.jetson_sink = self.general_configs['jetson_sink']

        # Individual configs
        config_path = os.path.join(self.path, 'audio_indiv_configs.yaml')
        with open(config_path, "r") as stream:
            try:
                self.individual_configs = yaml.safe_load(stream)
                if self.debug:
                    print(self.individual_configs)
            except yaml.YAMLError as exc:
                print(exc)
                exit()
        # Get individual configs
        self.default_output_dir = self.individual_configs['default_output_dir']
        self.model_path = self.individual_configs['model_path']
        self.mic_array_index = self.individual_configs['mic_array_index']
        self.use_beamformer = self.individual_configs['use_beamformer']
        self.source_list = self.individual_configs['source']
        self.sink_list = self.individual_configs['sinks']
        self.debug_sink_list = self.individual_configs['debug_sinks']
        if self.debug:
            self.sink_list.extend(self.debug_sink_list)

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
        # TODO: FIXE FILE PARAMS
        source = self.io_manager.add_source(source_name=name, source_type='sim', file=file, chunk_size=self.chunk_size)
        self.file_params = (1, source.wav_params[1], source.wav_params[2],
                            source.wav_params[3], source.wav_params[4], source.wav_params[5])
        # self.file_params = source.wav_params
        self.sim_input = True

        # Get input config file info
        # Paths
        self.in_subfolder_path = os.path.split(file)[0]
        config_path = os.path.join(self.in_subfolder_path, 'configs.yaml')
        with open(config_path, "r") as stream:
            try:
                input_configs = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                exit()
        # TODO: GET MICROPHONE MATRIX FROM CONGIS FILE?
        self.target = input_configs['source_dir']

        return source

    def init_sim_output(self, name, path=None):
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
                    subfolder_path = os.path.join(self.default_output_dir, 'run')
                    while os.path.exists(f'{subfolder_path}_{index}'):
                        index += 1
                    self.out_subfolder_path = f'{subfolder_path}_{index}'
        if not os.path.exists(self.out_subfolder_path):
            os.makedirs(self.out_subfolder_path)

        # WavSink
        sink_path = os.path.join(self.out_subfolder_path, f'{name}.wav')
        # TODO: SETUP FILE PARAMS IF SIM SINK WITH NO SIM SOURCE
        # self.file_params[0] = 1
        sink = self.io_manager.add_sink(sink_name=name, sink_type='sim',
                                        file=sink_path, wav_params=self.file_params, chunk_size=self.chunk_size)
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
        if src_index == 'default':
            src_index = None
        else:
            src_index = int(src_index)
        source = self.io_manager.add_source(source_name=name, source_type='mic', mic_index=src_index,
                                       channels=self.channels, mic_arr=self.mic_dict, chunk_size=self.chunk_size)
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
        if sink_index == 'default':
            sink_index = None
        else:
            sink_index = int(sink_index)
        sink = self.io_manager.add_sink(sink_name=name, sink_type='play', device_index=sink_index,
                                        channels=self.out_channels, chunk_size=self.chunk_size)
        return sink

    def reset_manager(self):
        self.out_subfolder_path = None

        # Reset STFTs
        for source_obj in self.source_dict.values():
            if 'stft' in source_obj:
                source_obj['stft'].reset()

        # Reset SCMs and ISTFTs
        for sink_obj in self.sink_dict.values():
            if 'scm_target' in sink_obj:
                sink_obj['scm_target'].reset()
            if 'scm_noise' in sink_obj:
                sink_obj['scm_noise'].reset()
            if 'istft' in sink_obj:
                sink_obj['istft'].reset()

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
            chosen_sink = self.sink_dict[sink_name]['sink']
            chosen_sink(data)
            return True
        else:
            return False

    def check_time(self, name, is_start, unit='ms'):
        """
        Run timers and save in dict.

        Args:
            name (str): Name of timer on which to do some action.
            is_start (bool): Consider time as start time (True) or end time (False).
            unit (str): String containing units to use for time management. Currently only accepts ms, the default.

        Returns:
            If start, returns current time, if end, returns elapsed time.
        """
        if not self.get_timers:
            return

        if unit == 'ms':
            unit_factor = 1 / 10000000

        if name not in self.timers:
            self.timers[name]['unit'] = unit
            self.timers[name]['total'] = 0
            self.timers[name]['call_cnt'] = 0

        current_time = time.perf_counter_ns() * unit_factor
        if is_start:
            self.timers[name]['start'] = current_time
            return current_time
        else:
            self.timers[name]['end'] = current_time
            elapsed_time = self.timers[name]['start'] - current_time
            self.timers[name]['total'] += elapsed_time
            self.timers[name]['call_cnt'] += 1
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
            time = self.timers[name]['total']
            unit = self.timers[name]['unit']

            mean_str = 'Total time'
            if is_mean:
                time /= self.timers[name]['call_cnt']
                mean_str = 'Mean time per loop'

            time_string = f'Timer: {mean_str} "{name}" = {time} {unit}'
        else:
            time_string = 'Timer not found.'

        return time_string

    def print_times(self, max_time):
        """
        Print timers.

        Args:
            max_time (float): Max time for one loop.
        """
        print(self.get_time(name='total', is_mean=False))
        print(self.get_time(name='total', is_mean=True))
        print(f'Timer: Longest time per loop: {max_time} ms.')
        print(self.get_time(name='beamformer', is_mean=True))
        if self.mask:
            print(self.get_time(name='masks', is_mean=True))
        else:
            print(self.get_time(name='data', is_mean=True))
            print(self.get_time(name='network', is_mean=True))

    def compute_masks(self, signal):
        """
        Computes masks (through KISS or neural network) for speech and noise on noisy speech signal.

        Args:
            signal (ndarray): Noisy speech frequential signal on which to obtain masks.

        Returns:
            Speech and noise masks.
        """
        if self.mask:
            self.check_time(name='mask', is_start=True)
            speech_mask, noise_mask = self.masks(signal, self.target)
            self.check_time(name='mask', is_start=False)

            return speech_mask, noise_mask
        else:
            self.check_time(name='data', is_start=True)
            # Delay and sum
            target_np = np.array([self.target])
            delay = get_delays_based_on_mic_array(target_np, self.mic_array, self.frame_size)
            sum = self.delay_and_sum(signal, delay[0])
            sum_tensor = torch.tensor(sum)
            sum_db = 20 * torch.log10(torch.abs(sum_tensor) + EPSILON)

            # Mono
            signal_tensor = torch.tensor(signal)
            signal_mono = torch.mean(signal_tensor, dim=0, keepdims=True)
            signal_mono_db = 20 * torch.log10(torch.abs(signal_mono) + EPSILON)

            concat_spec = torch.cat([signal_mono_db, sum_db], dim=1)
            concat_spec = torch.reshape(concat_spec, (1, 1, concat_spec.shape[1], 1))

            self.check_time(name='network', is_start=True)
            with torch.no_grad():
                noise_mask, self.last_h = self.model(concat_spec, self.last_h)
            self.check_time(name='network', is_start=False)
            noise_mask = torch.squeeze(noise_mask).numpy()
            speech_mask = 1 - noise_mask
            self.check_time(name='data', is_start=False)

            return speech_mask, noise_mask

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
        # Params
        if save_path:
            self.out_subfolder_path = save_path

        # Inputs
        window = 'hann'
        if source:
            self.source_list = source

        self.source_dict['audio'] = {}
        if self.source_list['type'] == 'sim':
            self.source_dict['audio']['src'] = self.init_sim_input(name=self.source_list['name'],
                                                                   file=self.source_list['file'])
            self.source_dict['audio']['file'] = self.source_list['file']
            if self.debug:
                # Try loading speech and noise ground truths if present
                try:
                    # Load speech file
                    self.source_dict['speech'] = {}
                    self.speech_file = os.path.join(os.path.split(self.source_list['file'])[0], 'speech.wav')
                    self.source_dict['speech']['file'] = self.speech_file
                    self.source_dict['speech']['src'] = self.init_sim_input(name='speech_gt_source', file=self.speech_file)
                    self.source_dict['speech']['stft'] = Stft(self.channels, self.frame_size, window)
                    # Load noise file
                    self.source_dict['noise'] = {}
                    self.noise_file = os.path.join(os.path.split(self.source_list['file'])[0], 'noise.wav')
                    self.source_dict['noise']['file'] = self.noise_file
                    self.source_dict['noise']['src'] = self.init_sim_input(name='noise_gt_source', file=self.noise_file)
                    self.source_dict['noise']['stft'] = Stft(self.channels, self.frame_size, window)
                    # Set argument to True if successful
                    self.speech_and_noise = True
                except:
                    self.speech_and_noise = False
        else:
            self.source_dict['audio']['src'] = self.init_mic_input(name=self.source_dict['name'],
                                                                   src_index=self.source_dict['idx'])
        self.source_dict['audio']['stft'] = Stft(self.channels, self.frame_size, window)

        # Outputs
        # TODO: ADD OUTPUT CHANNELS CONTROL
        scm_weight = 0.1
        if sinks:
            if overwrite_sinks:
                self.sink_list = sinks
            else:
                self.sink_list.extend(sinks)
        for sink in self.sink_list:
            # Add sink
            self.sink_dict[sink['name']] = {}
            if sink['type'] == 'sim':
                self.sink_dict[sink['name']]['sink'] = self.init_sim_output(name=sink['name'])
            else:
                self.sink_dict[sink['name']]['sink'] = self.init_play_output(name='original', sink_index=sink['idx'])
            # Add ISTFT and SCM (if needed for beamforming)
            if sink['beamform']:
                self.sink_dict[sink['name']]['scm_target'] = SpatialCov(self.channels, self.frame_size, weight=scm_weight)
                self.sink_dict[sink['name']]['scm_noise'] = SpatialCov(self.channels, self.frame_size, weight=scm_weight)
                self.sink_dict[sink['name']]['istft'] = IStft(1, self.frame_size, self.chunk_size, window)
            else:
                self.sink_dict[sink['name']]['istft'] = IStft(self.channels, self.frame_size, self.chunk_size, window)

        # Model
        if self.mask:
            self.masks = KissMask(self.mic_array, buffer_size=30)
        else:
            self.model = AudioModel(input_size=1026, hidden_size=512, num_layers=2)
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

    def init_app(self, save_input, save_output, output_path=None):
        """
        Function used to init jetson application.

        Args:
            save_input (bool): Whether to save the input (original) or not.
            save_output (bool): Whether to save the output or not.
            output_path (str): Path at which to save the simulated sinks (.wav files).
        """
        # Init sources
        # TODO: FIX JETSON APP
        self.source = self.init_mic_input(name=self.jetson_source['name'], src_index=self.jetson_source['idx'])

        # Init sinks
        self.sink_dict[self.jetson_sink['name']] = self.init_play_output(name=self.jetson_sink['name'],
                                                                         sink_index=self.jetson_sink['idx'])
        if save_input:
            self.sink_dict['original'] = self.init_sim_output(name='original', path=output_path)
        if save_output:
            self.sink_dict['output'] = self.init_sim_output(name='output', path=output_path)

    def main_loop(self):
        """
        Main audio loop.
        """

        # TODO: SIMPLIFIED MAIN LOOP FOR "IN SYSTEM" USAGE (REMOVE ALL DEBUG OPTIONS AND SUCH)?

        # Check torchaudio values for sanity-check when whole input is known
        if self.debug and self.torch_gt:
            # Audio
            audio_data_t, _ = torchaudio.load(self.source_dict['audio']['file'])
            audio_data = self.transformation(audio_data_t)
            audio_data = torch.abs(audio_data) ** 2
            audio_data = torch.mean(audio_data, dim=0, keepdim=False)
            if self.speech_and_noise:
                # Speech
                speech_data_t, _ = torchaudio.load(self.speech_file)
                speech_data = self.transformation(speech_data_t)
                speech_data = torch.abs(speech_data) ** 2
                speech_data = torch.mean(speech_data, dim=0, keepdim=False)
                # Noise
                noise_data_t, _ = torchaudio.load(self.noise_file)
                noise_data = self.transformation(noise_data_t)
                noise_data = torch.abs(noise_data) ** 2
                noise_data = torch.mean(noise_data, dim=0, keepdim=False)
                # Show
                fig, axs = plt.subplots(3)
                fig.suptitle('Spectrogrammes')
                axs[0].pcolormesh(audio_data, shading='gouraud')
                axs[1].pcolormesh(speech_data, shading='gouraud')
                axs[2].pcolormesh(noise_data, shading='gouraud')
                axs[2].set_xlabel("Temps(s)")
                axs[0].set_ylabel("Audio")
                axs[1].set_ylabel("Speech")
                axs[2].set_ylabel("Noise")
            else:
                fig, axs = plt.subplots(1)
                fig.suptitle('Spectrogrammes')
                axs[0].pcolormesh(audio_data, shading='gouraud')
                axs[0].set_xlabel("Temps(s)")
                axs[0].set_ylabel("Audio")
            save_name = os.path.join(self.out_subfolder_path, 'torch_plots.jpg')
            plt.savefig(fname=save_name)
            if SHOW_GRAPHS:
                plt.show(block=True)
            plt.close()

        # Arrays used to store spectrogram information
        if self.debug:
            noise_mask_np = None
            speech_mask_np = None
            if self.speech_and_noise:
                speech_mask_np_gt = None
                noise_mask_np_gt = None
                speech_np_gt = None
            if self.torch_gt:
                torch_speech_mask_np = None
                torch_noise_mask_np = None

        samples = 0
        max_time = 0
        loop_i = 0
        while samples / CONST.SAMPLING_RATE < TIME:
            self.check_time(name='loop', is_start=True)

            # Check App-Manager state
            # TODO: ADD CHECK TO STATE MANAGER

            # Record from microphone
            x = self.source_dict['audio']['src']()
            if x is None:
                if self.debug:
                    print('End of transmission. Closing.')
                break
            self.output_sink(data=x, sink_name='original')

            # Temporal to Frequential
            self.check_time(name='stft', is_start=True)
            X = self.source_dict['audio']['stft'](x)
            self.check_time(name='stft', is_start=False)

            # Compute the masks
            speech_mask, noise_mask = self.compute_masks(signal=X)
            # Save masks for spectrogram if in debug
            if self.debug:
                speech_mask_exp = np.expand_dims(np.real(speech_mask), axis=1)
                noise_mask_exp = np.expand_dims(np.real(noise_mask), axis=1)
                if noise_mask_np is not None and speech_mask_np is not None:
                    speech_mask_np = np.append(speech_mask_np, speech_mask_exp, axis=1)
                    noise_mask_np = np.append(noise_mask_np, noise_mask_exp, axis=1)
                else:
                    speech_mask_np = speech_mask_exp
                    noise_mask_np = noise_mask_exp

            # Spatial covariance matrices
            self.check_time(name='scm', is_start=True)
            if self.use_beamformer:
                target_scm = self.sink_dict['output']['scm_target'](X, speech_mask)
                noise_scm = self.sink_dict['output']['scm_noise'](X, noise_mask)
            self.check_time(name='scm', is_start=False)

            # MVDR
            self.check_time(name='beamformer', is_start=True)
            if self.use_beamformer:
                Y = self.beamformer(signal=X, target_scm=target_scm, noise_scm=noise_scm)
            else:
                Y = X * speech_mask
                Y = np.mean(Y, axis=0)
            self.check_time(name='beamformer', is_start=False)

            # ISTFT
            self.check_time(name='istft', is_start=True)
            y = self.sink_dict['output']['istft'](Y)
            self.check_time(name='istft', is_start=False)

            # Output fully processed data
            # out = y
            # if self.out_channels == 1:
            #     channel_to_use = out.shape[0] // 2
            #     out = out[channel_to_use]
            # elif self.out_channels == 2:
            #     out = np.array([out[0], out[-1]])
            self.output_sink(data=y, sink_name='output')

            # If in debug, check additional sources and output in additional sinks
            if self.debug:
                # Speech and noise ground truth beamforming
                if self.speech_and_noise:
                    # Get signal
                    s = self.source_dict['speech']['src']()
                    n = self.source_dict['noise']['src']()

                    # Temporal to frequential
                    S = self.source_dict['speech']['stft'](s)
                    N = self.source_dict['noise']['stft'](n)

                    # Output speech and noise directly (source, sink, STFT and ISTFT sanity check)
                    n_out = self.sink_dict['speech_out']['istft'](N)
                    s_out = self.sink_dict['noise_out']['istft'](S)
                    self.output_sink(data=s_out, sink_name='speech_out')
                    self.output_sink(data=n_out, sink_name='noise_out')

                    # Save speech to spectrogram
                    if self.debug:
                        s_mean = np.mean(S, axis=0)
                        s_log = 20 * np.log10(abs(s_mean) + EPSILON)
                        s_mean = np.expand_dims(s_log, axis=1)
                        if speech_np_gt is not None:
                            speech_np_gt = np.append(speech_np_gt, s_mean, axis=1)
                        else:
                            speech_np_gt = s_mean

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
                    speech_mask_gt_exp = np.expand_dims(speech_mask_gt, axis=1)
                    noise_mask_gt_exp = np.expand_dims(noise_mask_gt, axis=1)
                    if speech_mask_np_gt is not None and noise_mask_np_gt is not None:
                        speech_mask_np_gt = np.append(speech_mask_np_gt, speech_mask_gt_exp, axis=1)
                        noise_mask_np_gt = np.append(noise_mask_np_gt, noise_mask_gt_exp, axis=1)
                    else:
                        speech_mask_np_gt = speech_mask_gt_exp
                        noise_mask_np_gt = noise_mask_gt_exp

                    # Spatial covariance matrices
                    if self.use_beamformer:
                        target_scm_gt = self.sink_dict['output_gt']['scm_target'](X, speech_mask)
                        noise_scm_gt = self.sink_dict['output_gt']['scm_noise'](X, noise_mask)

                    # Beamform
                    if self.use_beamformer:
                        Y_gt = self.beamformer(signal=X, target_scm=target_scm_gt, noise_scm=noise_scm_gt)
                    else:
                        Y_gt = X * speech_mask_gt
                        Y_gt = np.mean(Y_gt, axis=0)

                    # ISTFT and save
                    y_gt = self.sink_dict['output_gt']['istft'](Y_gt)
                    self.output_sink(data=y_gt, sink_name='output_gt')

                # Offline torch ground truth
                if self.torch_gt:
                    # Get sample from torch loaded
                    torch_S = speech_data[:, loop_i].numpy()
                    torch_N = noise_data[:, loop_i].numpy()

                    # Energy ratio
                    torch_speech_mask_gt = torch_S / (torch_N + torch_S + EPSILON)
                    torch_noise_mask_gt = torch_N / (torch_N + torch_S + EPSILON)

                    # Save for spectrograms
                    torch_speech_mask_gt_exp = np.expand_dims(torch_speech_mask_gt, axis=1)
                    torch_noise_mask_gt_exp = np.expand_dims(torch_noise_mask_gt, axis=1)
                    if torch_speech_mask_np is not None and torch_noise_mask_np is not None:
                        torch_speech_mask_np = np.append(torch_speech_mask_np, torch_speech_mask_gt_exp, axis=1)
                        torch_noise_mask_np = np.append(torch_noise_mask_np, torch_noise_mask_gt_exp, axis=1)
                    else:
                        torch_speech_mask_np = torch_speech_mask_gt_exp
                        torch_noise_mask_np = torch_noise_mask_gt_exp

                    # SCM
                    if self.use_beamformer:
                        torch_target_scm_gt = self.sink_dict['torch_gt']['scm_target'](X, torch_S)
                        torch_noise_scm_gt = self.sink_dict['torch_gt']['scm_noise'](X, torch_N)

                    # Beamform
                    if self.use_beamformer:
                        torch_Y = self.beamformer(signal=X, target_scm=torch_target_scm_gt, noise_scm=torch_noise_scm_gt)
                    else:
                        torch_Y = X * torch_speech_mask_gt
                        torch_Y = np.mean(torch_Y, axis=0)

                    # ISTFT and save
                    torch_y = self.sink_dict['torch_gt']['istft'](torch_Y)
                    self.output_sink(data=torch_y, sink_name='torch_gt')

            loop_time = self.check_time(name='loop', is_start=False)
            if self.get_timers:
                max_time = loop_time if loop_time > max_time else max_time

            if self.debug and samples % (self.chunk_size * 50) == 0:
                print(f'Samples processed: {samples}')
                if self.get_timers:
                    print(f'Time for loop: {loop_time} ms.')

            samples += self.chunk_size
            loop_i += 1

        # Plot target + prediction spectrograms
        if self.debug:
            if self.speech_and_noise:
                fig, axs = plt.subplots(7)
                fig.set_figheight(9)
                fig.suptitle('Spectrogrammes')
                axs[0].pcolormesh(speech_np_gt, shading='gouraud')
                axs[1].pcolormesh(torch_speech_mask_np, shading='gouraud', vmin=0, vmax=1)
                axs[2].pcolormesh(speech_mask_np_gt, shading='gouraud', vmin=0, vmax=1)
                axs[3].pcolormesh(speech_mask_np, shading='gouraud', vmin=0, vmax=1)
                axs[4].pcolormesh(torch_noise_mask_np, shading='gouraud', vmin=0, vmax=1)
                axs[5].pcolormesh(noise_mask_np_gt, shading='gouraud', vmin=0, vmax=1)
                axs[6].pcolormesh(noise_mask_np, shading='gouraud', vmin=0, vmax=1)
                axs[0].set_ylabel("Speech")
                axs[1].set_ylabel("Torch S")
                axs[2].set_ylabel("Target S")
                axs[3].set_ylabel("Pred S")
                axs[4].set_ylabel("Torch N")
                axs[5].set_ylabel("Target N")
                axs[6].set_ylabel("Pred N")
                axs[6].set_xlabel("Temps(s)")
                save_name = os.path.join(self.out_subfolder_path, 'out_spec_plots.jpg')
                plt.savefig(fname=save_name)
                if SHOW_GRAPHS:
                    plt.show(block=True)
                plt.close()
            else:
                fig, axs = plt.subplots(5)
                fig.set_figheight(9)
                fig.suptitle('Spectrogrammes')
                axs[0].pcolormesh(speech_np_gt, shading='gouraud')
                axs[1].pcolormesh(torch_speech_mask_np, shading='gouraud', vmin=0, vmax=1)
                axs[2].pcolormesh(speech_mask_np, shading='gouraud', vmin=0, vmax=1)
                axs[3].pcolormesh(torch_noise_mask_np, shading='gouraud', vmin=0, vmax=1)
                axs[4].pcolormesh(noise_mask_np, shading='gouraud', vmin=0, vmax=1)
                axs[0].set_ylabel("Speech")
                axs[1].set_ylabel("Torch S")
                axs[2].set_ylabel("Pred S")
                axs[3].set_ylabel("Torch N")
                axs[4].set_ylabel("Pred N")
                axs[4].set_xlabel("Temps(s)")
                save_name = os.path.join(self.out_subfolder_path, 'out_spec_plots.jpg')
                plt.savefig(fname=save_name)
                if SHOW_GRAPHS:
                    plt.show(block=True)
                plt.close()

        print(f'Audio_Manager: Finished running main_audio : {samples} samples')
        if self.get_timers:
            self.print_times(max_time=max_time)
