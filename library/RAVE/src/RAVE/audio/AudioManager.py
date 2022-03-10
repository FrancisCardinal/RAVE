import torch
import yaml
import os

from pyodas.core import (
    Stft,
    IStft,
    KissMask,
    SpatialCov,
    DelaySum
)
from pyodas.utils import CONST, generate_mic_array, load_mic_array_from_ressources, get_delays_based_on_mic_array

from RAVE.audio.IO.IO_manager import IOManager
from RAVE.audio.Neural_Network.AudioModel import AudioModel
from RAVE.audio.Beamformer.Beamformer import Beamformer


class AudioManager:
    """
    Class used as manager for all audio processes, containing the main loop of execution for the application.
    """

    def __init__(self, debug, mic_type, mask, beamformer, ):

        self.file_params = None

        # Configs
        config_path = os.path.join(subfolder_path, 'configs.yaml')
        with open(config_path, "r") as stream:
            try:
                configs = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                exit()

        # Check if device has cuda
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"

        # IO
        self.io_manager = IOManager()

        # Masks
        # TODO: Add abstraction to model for info not needed in main script
        if mask:
            masks = KissMask(mic_array, buffer_size=30)
        else:
            model = AudioModel(input_size=FRAME_SIZE, hidden_size=128, num_layers=2)
            model.to(self.device)
            if debug:
                print(model)
            delay_and_sum = DelaySum(FRAME_SIZE)

        # Beamformer
        # TODO: simplify beamformer abstraction kwargs
        beamformer = Beamformer(name='mvdr', channels=channels)

        # Utils
        # TODO: Check if we want to handle stft and istft in IOManager class
        stft = Stft(channels, FRAME_SIZE, "sqrt_hann")
        istft = IStft(channels, FRAME_SIZE, CHUNK_SIZE, "sqrt_hann")
        speech_spatial_cov = SpatialCov(channels, FRAME_SIZE, weight=0.03)
        noise_spatial_cov = SpatialCov(channels, FRAME_SIZE, weight=0.03)

    def init_app(self):

        # Init sources

        # Init sinks

    def init_sim_input(self, name, file):

        # WavSource
        source = self.io_manager.add_source(source_name=name, source_type='sim',
                                       file=file, chunk_size=CHUNK_SIZE)
        self.file_params = source.wav_params

        # Get input config file info
        # Paths
        subfolder_path = os.path.split(INPUT)[0]
        config_path = os.path.join(subfolder_path, 'configs.yaml')
        with open(config_path, "r") as stream:
            try:
                input_configs = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                exit()
        target = input_configs['source_dir']

        # Microphone array
        mic_dict = {'mics': dict(), 'nb_of_channels': 0}
        for mic_idx, mic in enumerate(input_configs['mic_rel']):
            mic_dict['mics'][f'{mic_idx}'] = mic
            mic_dict['nb_of_channels'] += 1
        mic_array = generate_mic_array(mic_dict)
        channels = mic_dict['nb_of_channels']

    def init_mic_input(self):
        # Input source
        if INPUT == 'default' or INPUT.isdigit():
            # MicSource
            if INPUT == 'default':
                INPUT = None
            else:
                INPUT = int(INPUT)
            source = io_manager.add_source(source_name='MicSource1', source_type='mic', mic_index=INPUT,
                                           channels=CHANNELS, mic_arr=MIC_DICT, chunk_size=CHUNK_SIZE)
        else:

    def init_sim_output(self):
        # Output sink
        if OUTPUT == 'default' or OUTPUT.isdigit():
            # PlaybackSink
            if OUTPUT == 'default':
                OUTPUT = None
            else:
                OUTPUT = int(OUTPUT)
            output_sink = io_manager.add_sink(sink_name='original', sink_type='play', device_index=OUTPUT,
                                              channels=OUT_CHANNELS, chunk_size=CHUNK_SIZE)
        else:
            # If mic source (no subfolder), output subfolder
            if subfolder_path == None:
                index = 0
                subfolder_path = os.path.join(DEFAULT_OUTPUT_FOLDER, 'run')
                while os.path.exists(f'{subfolder_path}_{index}'):
                    index += 1
                os.makedirs(subfolder_path)

            # WavSink
            original = os.path.join(subfolder_path, 'original.wav')
            original_sink = io_manager.add_sink(sink_name='original', sink_type='sim',
                                                file=original, wav_params=FILE_PARAMS, chunk_size=CHUNK_SIZE)
            if OUTPUT == '':
                OUTPUT = os.path.join(subfolder_path, 'output.wav')
            output_sink = io_manager.add_sink(sink_name='output', sink_type='sim',
                                              file=OUTPUT, wav_params=FILE_PARAMS, chunk_size=CHUNK_SIZE)

    def init_play_output(self):

    def initialise_audio(self):
        """
        Method to initialise audio after start.
        """

        FILE_PARAMS = (
            4,
            2,
            CONST.SAMPLING_RATE,
            CONST.SAMPLING_RATE * TIME,
            "NONE",
            "not compressed",
        )
        mic_array = MIC_ARRAY
        channels = FILE_PARAMS[0]
        target = np.array([0, 1, 0.5])
        subfolder_path = None
        original_sink = None



    def main_loop(self):
        """
        Main audio loop.
        """
        # Record for 6 seconds
        samples = 0
        while samples / CONST.SAMPLING_RATE < TIME:
            # Record from microphone
            x = source()
            if x is None:
                print('End of transmission. Closing.')
                exit()

            # Save the unprocessed recording
            if original_sink:
                original_sink(x)
            X = stft(x)

            # Compute the masks
            if MASK:
                speech_mask, noise_mask = masks(X, target)
            else:
                delay = get_delays_based_on_mic_array(target, MIC_ARRAY, FRAME_SIZE)
                sum = delay_and_sum(X, delay)
                speech_mask = model(sum)
                noise_mask = 1 - speech_mask

            # Spatial covariance matrices
            target_scm = speech_spatial_cov(X, speech_mask)
            noise_scm = noise_spatial_cov(X, noise_mask)

            # ---- GEV ----
            # gev_Y = gev(X, target_scm, noise_scm)
            # gev_Y *= CHANNELS ** 2
            # gev_y = gev_istft(gev_Y)
            # kiss_gev_wav_sink(gev_y)

            # MVDR
            Y = beamformer(signal=X, target_scm=target_scm, noise_scm=noise_scm)
            y = istft(Y)
            if OUT_CHANNELS == 1:
                out = y[y.shape[0] // 2]
            elif OUT_CHANNELS == 2:
                out = np.array([y[0], y[-1]])
            output_sink(out)

            samples += CHUNK_SIZE

            if DEBUG and samples % (CHUNK_SIZE * 10) == 0:
                print(f'Samples processed: {samples}')

        print('Finished running main_audio')