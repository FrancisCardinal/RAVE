import json
import numpy as np
import torch
from itertools import product
from tqdm import tqdm

import pyodas.core as core
import pyodas.visualize as vis
import pyodas.io as io
from pyodas.utils import CONST

CHUNK_SIZE = 256
FRAME_SIZE = 2 * CHUNK_SIZE
CHANNELS = 6
INT16_BYTE_SIZE = 2
# NB_OF_FRAMES is 0 because we want to record
# for an indefinite amount of time
NB_OF_FRAMES = 0
FILE_PARAMS = (
    1,
    INT16_BYTE_SIZE,
    CONST.SAMPLING_RATE,
    NB_OF_FRAMES,
    "NONE",
    "NONE",
)

WIDTH = 640
HEIGHT = 480
ORDER_OF_POLYNOMIAL = 3

DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"

# Get calibration matrix
with open(
    "./calibration_6.json",
    "r",
    encoding="utf-8",
) as f:
    calibration_matrix = np.array(json.load(f))

# Used to reconstruct the full matrix delay
upper_triangle_indices = np.triu_indices(CHANNELS, k=1)
lower_triangle_indices = np.tril_indices(CHANNELS, k=-1)
delay = np.zeros((CHANNELS, CHANNELS), dtype=np.float32)

# Used to be able to pass from a pixel value to the
exponents = np.array(
    [[x, y] for x, y in product(range(ORDER_OF_POLYNOMIAL + 1), repeat=2)]
)

# Visualize
video_source = vis.VideoSource(0, WIDTH, HEIGHT)
m = vis.Monitor("Camera", video_source.shape, refresh_rate=10)

# Core
stft = core.Stft(CHANNELS, FRAME_SIZE, "sqrt_hann")
istft = core.IStft(CHANNELS, FRAME_SIZE, CHUNK_SIZE, "sqrt_hann")
delay_and_sum = core.DelaySum(FRAME_SIZE)
mvdr = core.Mvdr(CHANNELS)
gev = core.Gev()
speech_spatial_cov = core.SpatialCov(CHANNELS, FRAME_SIZE, weight=0.03)
noise_spatial_cov = core.SpatialCov(CHANNELS, FRAME_SIZE, weight=0.03)

# io
mic_source = io.MicSource(CHANNELS, "ReSpeaker Core v2", chunk_size=CHUNK_SIZE)
ds_sink = io.WavSink(
    "./audio_files/delay_and_sum.wav", FILE_PARAMS, CHUNK_SIZE
)
unprocessed_sink = io.WavSink(
    "./audio_files/unprocessed.wav", FILE_PARAMS, CHUNK_SIZE
)
output_sink = io.PlaybackSink(CHANNELS)

masks = core.KissMask(mic_source.mic_array, buffer_size=30)
TARGET = np.array([0, 1, 0.25])
x = mic_source()
X = stft(x)
speech_mask, noise_mask = masks(X, TARGET)

with tqdm(total=25000) as pbar:
    while True:
        # Get the audio signal and image frame
        x = mic_source()
        # frame = video_source()
        # unprocessed_sink(x[0])

        # Get the signal in the frequency domain
        X = stft(x)

        mouth_x = 300
        mouth_y = 400
        normalized_mouth_x = (2 * mouth_x - WIDTH - 1) / (WIDTH - 1)
        normalized_mouth_y = (2 * mouth_y - HEIGHT - 1) / (HEIGHT - 1)

        A = (
            normalized_mouth_x ** exponents[:, 0][:, np.newaxis]
            * normalized_mouth_y ** exponents[:, 1][:, np.newaxis]
        ).T

        delay_pairs = A @ calibration_matrix
        delay[
            upper_triangle_indices[0], upper_triangle_indices[1]
        ] = delay_pairs
        delay[lower_triangle_indices[0], lower_triangle_indices[1]] = (
            -1 * delay_pairs
        )

        speech_SCM = speech_spatial_cov(X, speech_mask)
        noise_SCM = speech_spatial_cov(X, noise_mask)
        X = mvdr(X, speech_SCM, noise_SCM)

        y = istft(X)
        output_sink(y)
        # ds_sink(y[0])

        pbar.update()
