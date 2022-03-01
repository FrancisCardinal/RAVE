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
    SpatialCov
)
from pyodas.utils import CONST, generate_mic_array

from RAVE.audio.IO.IO_manager import IOManager
from RAVE.audio.Neural_Network.AudioModel import AudioModel
from RAVE.audio.Beamformer.Beamformer import Beamformer

DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"

# Constants
# TODO: Uniformise microphone array in config file or such
MIC_ARRAY = generate_mic_array({
    'mics': {
        '0': [-0.1, -0.1, 0],
        '1': [-0.1, 0.1, 0],
        '2': [0.1, -0.1, 0],
        '3': [0.1, 0.1, 0]
    },
    'nb_of_channels': 4
})

CHUNK_SIZE = 256
FRAME_SIZE = 2 * CHUNK_SIZE
UNPROCESSED_MIX = "../../../audio_files/unprocessed_mix.wav"
KISS_MVDR_MIX = "../../../audio_files/kiss_mvdr_ref_mix.wav"
KISS_GEV_MIX = "../../../audio_files/kiss_gev_mix.wav"
# TARGET = np.array([0, 1, 0.25])

# Params: .wav file channels, int16 byte size, sampling rate, nb of samples,
#         compression type, compression name
# FILE_PARAMS = (
#     4,
#     2,
#     CONST.SAMPLING_RATE,
#     CONST.SAMPLING_RATE * TIME,
#     "NONE",
#     "NONE",
# )


def main(DEBUG, INPUT, OUTPUT, TIME, MASK):
    # TODO: add multiple source/sink support?
    # TODO: simplify io abstraction kwargs

    # Paths
    subfolder_path = os.path.split(INPUT)[0]
    original = os.path.join(subfolder_path, 'original.wav')
    if OUTPUT == '':
        OUTPUT = os.path.join(subfolder_path, 'output.wav')
    config_path = os.path.join(subfolder_path, 'configs.yaml')
    with open(config_path, "r") as stream:
        try:
            configs = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            exit()

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

    # IO
    io_manager = IOManager()
    source = io_manager.add_source(source_name='source1', source_type='sim',
                                   file=INPUT, chunk_size=CHUNK_SIZE)
    wav_params = source.wav_params
    original_sink = io_manager.add_sink(sink_name='original', sink_type='sim',
                               file=original, wav_params=wav_params, chunk_size=CHUNK_SIZE)
    output_sink = io_manager.add_sink(sink_name='output', sink_type='sim',
                               file=OUTPUT, wav_params=wav_params, chunk_size=CHUNK_SIZE)

    # Masks
    # TODO: Add abstraction to model for info not needed in main script
    target = configs['source_dir']
    if MASK:
        masks = KissMask(mic_array, buffer_size=30)
    else:
        model = AudioModel(input_size=FRAME_SIZE, hidden_size=128, num_layers=2)
        model.to(DEVICE)
        if DEBUG:
            print(model)

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
        if x.all() == None:
            print('End of transmission. Closing.')
            exit()

        # Save the unprocessed recording
        original_sink(x)
        X = stft(x)

        # Compute the masks
        if MASK:
            speech_mask, noise_mask = masks(X, target)
        else:
            speech_mask = model(X)
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
        output_sink(y)

        samples += CHUNK_SIZE


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
