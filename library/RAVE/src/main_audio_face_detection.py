# import json
# import numpy as np
# import torch
#
# import pyodas.core as core
# import pyodas.visualize as vis
# from pyodas.io import MicSource
# from pyodas.utils import anechoic_steering_vector, CONST
# from itertools import product
# from RAVE.face_detection.FaceDetectionModel import FaceDetectionModel
# from RAVE.face_detection.fpsHelper import FPS
#
# CHUNK_SIZE = 512
# FRAME_SIZE = 2 * CHUNK_SIZE
# CHANNELS = 4
#
# WIDTH = 640
# HEIGHT = 480
# ORDER_OF_POLYNOMIAL = 3
#
# DEVICE = "cpu"
# if torch.cuda.is_available():
#     DEVICE = "cuda"
#
# # Get calibration matrix
# with open(
#     "./calibration.json",
#     "r",
#     encoding="utf-8",
# ) as f:
#     calibration_matrix = np.array(json.load(f))
#
# pixels = np.array([300, 250])
#
# normalized_width_px = (2 * pixels[0] - WIDTH - 1) / (
#         WIDTH - 1
# )
#
# normalized_height_px = (2 * pixels[1] - HEIGHT - 1) / (
#         HEIGHT - 1
# )
#
# exponents = np.array(
#     [[x, y] for x, y in product(range(ORDER_OF_POLYNOMIAL + 1), repeat=2)]
# )
#
# A = (
#         normalized_width_px ** exponents[:, 0][:, np.newaxis]
#         * normalized_height_px ** exponents[:, 1][:, np.newaxis]
# ).T
#
# delay_pairs = A @ calibration_matrix
# delay_in_relation_to_first_mic = np.zeros((4, 1), dtype=np.float32)
# delay_in_relation_to_first_mic[1:, 0] = delay_pairs[0, :3]
# bins = (FRAME_SIZE // 2) + 1
# sound_vel = 343
# bins_range = np.tile(np.arange(bins), (CHANNELS, 1))
# constant = -1j * np.pi * CONST.SAMPLING_RATE / sound_vel / (bins - 1)
# transfer_fct = np.exp(bins_range * delay_in_relation_to_first_mic * constant)
#
# # Visualize
# video_source = vis.VideoSource(2, WIDTH, HEIGHT)
# m = vis.Monitor("Camera", video_source.shape, refresh_rate=10)
#
# # Core
# stft = core.Stft(CHANNELS, FRAME_SIZE, "hann")
# spatial_cov = core.SpatialCov(CHANNELS, FRAME_SIZE, weight=0.3)
# mic_source = MicSource(
#     CHANNELS,
#     "ReSpeaker_USB",
#     chunk_size=CHUNK_SIZE
# )
#
# while m.window_is_alive():
#     # Get the audio signal and image frame
#     x = mic_source()
#     frame = video_source()
#
#     # Get the signal in the frequency domain
#     X = stft(x)
#
#     # Compute the spatial covariance matrix
#     SCM = spatial_cov(X)
#
#     m.update("Camera", frame)

import json
import numpy as np

import pyodas.core as core
import pyodas.visualize as vis
from pyodas.io import MicSource

CHUNK_SIZE = 512
FRAME_SIZE = 2 * CHUNK_SIZE
CHANNELS = 4

WIDTH = 640
HEIGHT = 480

# Get calibration matrix
with open(
    "./calibration.json",
    "r",
    encoding="utf-8",
) as f:
    calibration_matrix = np.array(json.load(f))

# Visualize
video_source = vis.VideoSource(2, WIDTH, HEIGHT)
acoustic_image = vis.AcousticImage(WIDTH, HEIGHT, calibration_matrix)
m = vis.Monitor("Camera", video_source.shape, refresh_rate=10)

# Core
stft = core.Stft(CHANNELS, FRAME_SIZE, "hann")
spatial_cov = core.SpatialCov(CHANNELS, FRAME_SIZE, weight=0.3)
mic_source = MicSource(CHANNELS, "ReSpeaker_USB", chunk_size=CHUNK_SIZE)
svd_phat = core.SvdPhat(acoustic_image.delays, FRAME_SIZE, "./svd_weights.bin")

while m.window_is_alive():
    # Get the audio signal and image frame
    x = mic_source()
    frame = video_source()

    # If you are using a webcam, this might feel more natural, but
    # make sure the calibration was also done on a frame that was
    # flipped
    # frame = cv2.flip(frame, 1)

    # Get the signal in the frequency domain
    X = stft(x)

    # Compute the spatial covariance matrix
    SCM = spatial_cov(X)

    # Get acoustic image
    audio_projection = svd_phat(SCM)
    frame = acoustic_image(audio_projection, frame)

    m.update("Camera", frame)
