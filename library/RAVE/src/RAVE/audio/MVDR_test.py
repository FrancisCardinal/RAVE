import numpy as np
import os
from pyodas.core import (
    Stft,
    IStft,
    KissMask,
    SpatialCov,
    Mvdr,
    Gev,
)
from pyodas.io import MicSource, WavSink, WavSource
from pyodas.utils import CONST, generate_mic_array

# Constants
CHUNK_SIZE = 256
FRAME_SIZE = 2 * CHUNK_SIZE
CHANNELS = 4
TARGET = np.array([-1.45, 10.053, 0.0])
# Params: .wav file channels, int16 byte size, sampling rate, nb of samples,
#         compression type, compression name
FILE_PARAMS = (
    1,
    2,
    CONST.SAMPLING_RATE,
    0,
    "NONE",
    "NONE",
)

# MicArray
mic_dict = {
    'mics': {
         '0': [-0.1, -0.1, 0],
         '1': [-0.1, 0.1, 0],
         '2': [0.1, -0.1, 0],
         '3': [0.1, 0.1, 0]
    },
    'nb_of_channels': 4
}
mic_array = generate_mic_array(mic_dict)

# io
# mic_source = MicSource(CHANNELS, "ReSpeaker_USB", chunk_size=CHUNK_SIZE)
FILE = 'C:\\GitProjet\\MS-SNSD\\output\\no_reverb\\1\\clnsp0_AirConditioner_2'
source = WavSource(file=os.path.join(FILE, 'audio.wav'), chunk_size=CHUNK_SIZE)
speech = WavSource(file=os.path.join(FILE, 'speech.wav'), chunk_size=CHUNK_SIZE)
noise = WavSource(file=os.path.join(FILE, 'noise.wav'), chunk_size=CHUNK_SIZE)

original_wav_sink = WavSink(os.path.join(FILE, 'original.wav'), FILE_PARAMS, CHUNK_SIZE)
output_wav_sink = WavSink(os.path.join(FILE, 'output.wav'), FILE_PARAMS, CHUNK_SIZE)
output_wav_sink_kiss = WavSink(os.path.join(FILE, 'output_kiss.wav'), FILE_PARAMS, CHUNK_SIZE)

FILE_PARAMS = source.wav_params
speech_wav_sink = WavSink(os.path.join(FILE, 'speech_out.wav'), FILE_PARAMS, CHUNK_SIZE)
noise_wav_sink = WavSink(os.path.join(FILE, 'noise_out.wav'), FILE_PARAMS, CHUNK_SIZE)

# core
stft = Stft(CHANNELS, FRAME_SIZE, "sqrt_hann")
s_stft = Stft(CHANNELS, FRAME_SIZE, "sqrt_hann")
n_stft = Stft(CHANNELS, FRAME_SIZE, "sqrt_hann")
masks = KissMask(mic_array, buffer_size=30)
speech_spatial_cov = SpatialCov(CHANNELS, FRAME_SIZE, weight=0.05)
noise_spatial_cov = SpatialCov(CHANNELS, FRAME_SIZE, weight=0.05)
speech_spatial_cov_kiss = SpatialCov(CHANNELS, FRAME_SIZE, weight=0.05)
noise_spatial_cov_kiss = SpatialCov(CHANNELS, FRAME_SIZE, weight=0.05)

mvdr = Mvdr(CHANNELS)
mvdr_istft = IStft(1, FRAME_SIZE, CHUNK_SIZE, "sqrt_hann")
mvdr_istft_kiss = IStft(1, FRAME_SIZE, CHUNK_SIZE, "sqrt_hann")
s_istft = IStft(CHANNELS, FRAME_SIZE, CHUNK_SIZE, "sqrt_hann")
n_istft = IStft(CHANNELS, FRAME_SIZE, CHUNK_SIZE, "sqrt_hann")

# Record for 6 seconds
samples = 0
while samples / CONST.SAMPLING_RATE < float('inf'):
    # Record from microphone
    x = source()
    if x is None:
        print('End of transmission. Closing.')
        break
    s = speech()
    n = noise()

    # Save the unprocessed recording
    original_wav_sink(x[0:1])

    # Get the signal in the frequency domain
    X = stft(x)
    S = s_stft(s)
    N = n_stft(n)

    # Compute the masks with KISS
    speech_mask_kiss, noise_mask_kiss = masks(X, TARGET)
    np_S = np.real(S)
    np_N = np.real(N)
    noise_mask = (np_N ** 2) / ((np_N ** 2) + (np_S ** 2) + 1e-20)
    speech_mask = 1 - noise_mask
    # speech_mask_np_gt[:, loop_i] = speech_mask_gt[0]


    # Compute the spatial covariance matrices
    target_scm = speech_spatial_cov(X, speech_mask)
    noise_scm = noise_spatial_cov(X, noise_mask)
    target_scm_kiss = noise_spatial_cov(X, speech_mask_kiss)
    noise_scm_kiss = noise_spatial_cov(X, noise_mask_kiss)

    # ---- MVDR -----
    mvdr_Y = mvdr(X, target_scm, noise_scm)
    mvdr_Y_kiss = mvdr(X, target_scm_kiss, noise_scm_kiss)

    y_out = mvdr_istft(mvdr_Y)
    y_out_kiss = mvdr_istft_kiss(mvdr_Y_kiss)
    s_out = s_istft(S)
    n_out = n_istft(N)

    output_wav_sink(y_out)
    output_wav_sink_kiss(y_out_kiss)
    speech_wav_sink(s_out)
    noise_wav_sink(n_out)

    samples += CHUNK_SIZE
