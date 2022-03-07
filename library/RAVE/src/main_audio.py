import numpy as np
import argparse
from tkinter import filedialog
import torch
import os
import yaml

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

DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"

# Constants
# TODO: Uniformise microphone array in config file or such
# MIC_ARRAY = generate_mic_array({
#     'mics': {
#         '0': [-0.1, -0.1, 0],
#         '1': [-0.1, 0.1, 0],
#         '2': [0.1, -0.1, 0],
#         '3': [0.1, 0.1, 0]
#     },
#     'nb_of_channels': 4
# })
MIC_ARRAY = load_mic_array_from_ressources('ReSpeaker_USB')
CHANNELS = 4

CHUNK_SIZE = 256
FRAME_SIZE = 2 * CHUNK_SIZE

DEFAULT_OUTPUT_FOLDER = 'C:\GitProjet\pipeline'


def main(DEBUG, INPUT, OUTPUT, TIME, MASK):
    # TODO: add multiple source/sink support?
    # TODO: simplify io abstraction kwargs

    # Params: .wav file channels, int16 byte size, sampling rate, nb of samples,
    #         compression type, compression name
    FILE_PARAMS = (
        4,
        2,
        CONST.SAMPLING_RATE,
        CONST.SAMPLING_RATE * TIME,
        "NONE",
        "not compressed",
    )
    channels = FILE_PARAMS[0]
    target = np.array([0, 1, 0.5])
    subfolder_path = None
    original_sink = None

    # IO
    io_manager = IOManager()
    # Input source
    if INPUT == 'default' or INPUT.isdigit():
        # MicSource
        if INPUT == 'default':
            INPUT = None
        else:
            INPUT = int(INPUT)
        source = io_manager.add_source(source_name='MicSource1', source_type='mic', mic_index=INPUT,
                                       channels=CHANNELS, mic_arr=MIC_ARRAY, chunk_size=CHUNK_SIZE)
    else:
        # WavSource
        source = io_manager.add_source(source_name='WavSource1', source_type='sim',
                                       file=INPUT, chunk_size=CHUNK_SIZE)
        FILE_PARAMS = source.wav_params

        # If .wav source, get more info from files
        # Paths
        # TODO: fix output to .wav and playback?
        # TODO: fix folder path when no .wav input?
        subfolder_path = os.path.split(INPUT)[0]
        config_path = os.path.join(subfolder_path, 'configs.yaml')
        with open(config_path, "r") as stream:
            try:
                configs = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                exit()
        target = configs['source_dir']

        # Microphone array
        if configs['mic_rel']:
            mic_dict = {'mics': dict(), 'nb_of_channels': 0}
            for mic_idx, mic in enumerate(configs['mic_rel']):
                mic_dict['mics'][f'{mic_idx}'] = mic
                mic_dict['nb_of_channels'] += 1
            mic_array = generate_mic_array(mic_dict)
        else:
            mic_dict = MIC_ARRAY
        channels = mic_dict['nb_of_channels']

    # Output sink
    if OUTPUT == 'default' or OUTPUT.isdigit():
        # PlaybackSink
        if OUTPUT == 'default':
            OUTPUT = None
        else:
            OUTPUT = int(OUTPUT)
        output_sink = io_manager.add_sink(sink_name='original', sink_type='play', device_index=OUTPUT,
                                          channels=1, chunk_size=CHUNK_SIZE)
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

    # Masks
    # TODO: Add abstraction to model for info not needed in main script
    if MASK:
        masks = KissMask(mic_array, buffer_size=30)
    else:
        model = AudioModel(input_size=FRAME_SIZE, hidden_size=128, num_layers=2)
        model.to(DEVICE)
        if DEBUG:
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
        output_sink(x)

        samples += CHUNK_SIZE

        if DEBUG:
            print(f'Samples processed: {samples}')

    print('Finished running main_audio')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d", "--debug", action="store_true", help="Run the script in debug mode. Is more verbose."
    )
    parser.add_argument(
        "-m", "--mask", action="store_true", help="Use predefined masks instead of network."
    )

    parser.add_argument(
        "-i",
        "--input",
        action="store",
        type=str,
        default='tkinter',
        help="Source to use as input signals (can be microphone matrix or simulated .wav file)."
    )
    parser.add_argument(
        "-o",
        "--output",
        action="store",
        type=str,
        default='',
        help="Sink to use to output signals (can be speakers or simulated .wav file)."
    )

    parser.add_argument(
        "-t",
        "--time",
        action="store",
        type=int,
        default=float('inf'),
        help="Time to run audio for."
    )
    args = parser.parse_args()

    # parse sources
    input = args.input
    if input == 'tkinter':
        input = filedialog.askopenfilename(title="Input .wave file.")

    # parse noises
    output = args.output
    if output == 'tkinter':
        output = filedialog.askopenfilename(title="Output .wav file.")

    main(
        args.debug,
        input,
        output,
        args.time,
        args.mask
    )
