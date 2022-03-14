import torch
import yaml
import os
import numpy as np

from pyodas.core import (
    Stft,
    IStft,
    KissMask,
    SpatialCov,
    DelaySum
)
from pyodas.utils import CONST, generate_mic_array, load_mic_array_from_ressources, get_delays_based_on_mic_array

 # TODO: CHECK IF SIMPLER TO IN AND OUT INIT TO IO MANAGER
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


class AudioManager:
    """
    Class used as manager for all audio processes, containing the main loop of execution for the application.
    """

    def __init__(self, debug, mask):

        # Class variables init
        self.file_params = None
        self.in_subfolder_path = None
        self.out_subfolder_path = None
        self.source = None
        self.original_sink = None
        self.target = None

        # Argument variables
        self.debug = debug
        self.mask = mask

        # General configs
        self.path = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(self.path, 'audio/audio_general_configs.yaml')
        with open(config_path, "r") as stream:
            try:
                self.general_configs = yaml.safe_load(stream)
                if self.debug:
                    print(self.general_configs)
            except yaml.YAMLError as exc:
                print(exc)
                exit()
        # Get general configs
        self.mic_dict = self.general_configs.mic_dict
        self.mic_array = generate_mic_array(self.mic_dict)
        self.channels = self.mic_dict['nb_of_channels']
        self.out_channels = self.general_configs.out_channels
        self.chunk_size = self.general_configs.chunk_size
        self.frame_size = self.chunk_size * 2
        self.beamformer_name = self.general_configs.beamformer

        # Individual configs
        config_path = os.path.join(self.path, 'audio/audio_indiv_configs.yaml')
        with open(config_path, "r") as stream:
            try:
                self.individual_configs = yaml.safe_load(stream)
                if self.debug:
                    print(self.individual_configs)
            except yaml.YAMLError as exc:
                print(exc)
                exit()
        # Get general configs
        self.default_output_dir = self.individual_configs.default_output_dir
        self.mic_array_index = self.individual_configs.mic_array_index

        # Check if device has cuda
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"

        # IO
        self.io_manager = IOManager()

        # Masks
        # TODO: Add abstraction to model for info not needed in main script
        if self.mask:
            self.masks = KissMask(self.mic_array, buffer_size=30)
        else:
            self.model = AudioModel(input_size=self.frame_size, hidden_size=128, num_layers=2)
            self.model.to(self.device)
            if self.debug:
                print(self.model)
            self.delay_and_sum = DelaySum(self.frame_size)

        # Beamformer
        # TODO: simplify beamformer abstraction kwargs
        self.beamformer = Beamformer(name=self.beamformer_name, channels=self.channels)

        # Utils
        # TODO: Check if we want to handle stft and istft in IOManager class
        self.stft = Stft(self.channels, self.frame_size, "sqrt_hann")
        self.istft = IStft(self.channels, self.frame_size, self.chunk_size, "sqrt_hann")
        self.speech_spatial_cov = SpatialCov(self.channels, self.frame_size, weight=0.03)
        self.noise_spatial_cov = SpatialCov(self.channels, self.frame_size, weight=0.03)

    def init_app(self, save_input, output_path=None):

        # Init sources
        self.source = self.init_mic_input('mic_array', )

        # Init sinks
        if save_input:
            self.original_sink = self.init_sim_output(name='original', path=output_path)

    def init_sim_input(self, name, file):
        """
        Initialise simulated input (from .wav file).

        Args:
            name (str): Name of source.
            file (str): Path of .wav file to use as input.

        Returns:
            source: WavSource object created.
        """
        source = self.io_manager.add_source(source_name=name, source_type='sim', file=file, chunk_size=self.chunk_size)
        self.file_params = source.wav_params

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
        self.target = input_configs['source_dir']

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
        os.makedirs(self.out_subfolder_path)

        # WavSink
        sink_path = os.path.join(self.out_subfolder_path, f'{name}.wav')
        # TODO: SETUP FILE PARAMS IF SIM SINK WITH NO SIM SOURCE
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

    def initialise_audio(self):
        """
        Method to initialise audio after start.
        """
        pass

    def main_loop(self):
        """
        Main audio loop.
        """
        # Record for 6 seconds
        samples = 0
        while samples / CONST.SAMPLING_RATE < TIME:
            # Record from microphone
            x = self.source()
            if x is None:
                print('End of transmission. Closing.')
                exit()

            # Save the unprocessed recording
            if self.original_sink:
                self.original_sink(x)
            X = self.stft(x)

            # Compute the masks
            if self.mask:
                speech_mask, noise_mask = self.masks(X, target)
            else:
                delay = get_delays_based_on_mic_array(self.target, self.mic_array, self.frame_size)
                sum = self.delay_and_sum(X, delay)
                speech_mask = self.model(sum)
                noise_mask = 1 - speech_mask

            # Spatial covariance matrices
            target_scm = self.speech_spatial_cov(X, speech_mask)
            noise_scm = self.noise_spatial_cov(X, noise_mask)

            # MVDR
            Y = self.beamformer(signal=X, target_scm=target_scm, noise_scm=noise_scm)
            y = self.istft(Y)
            if self.out_channels == 1:
                out = y[y.shape[0] // 2]
            elif self.out_channels == 2:
                out = np.array([y[0], y[-1]])
            output_sink(out)

            samples += self.chunk_size

            if self.debug and samples % (self.chunk_size * 10) == 0:
                print(f'Samples processed: {samples}')

        print('Finished running main_audio')