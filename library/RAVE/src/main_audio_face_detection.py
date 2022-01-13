import json
import numpy as np
import torch
from itertools import product

import pyodas.core as core
import pyodas.visualize as vis
import pyodas.io as io
from pyodas.utils import CONST


from RAVE.face_detection.FaceDetectionModel import FaceDetectionModel
from RAVE.common.image_utils import (
    xyxy2xywh,
    opencv_image_to_tensor,
    scale_coords,
    scale_coords_landmarks,
)
from RAVE.common.fpsHelper import FPS
from main_face_detection import show_results

CHUNK_SIZE = 512
FRAME_SIZE = 2 * CHUNK_SIZE
CHANNELS = 4
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
    "./calibration.json",
    "r",
    encoding="utf-8",
) as f:
    calibration_matrix = np.array(json.load(f))

face_detection_model = FaceDetectionModel(DEVICE)
fps = FPS()

# Used to reconstruct the full matrix delay
upper_triangle_indices = np.triu_indices(CHANNELS, k=1)
lower_triangle_indices = np.tril_indices(CHANNELS, k=-1)
delay = np.zeros((CHANNELS, CHANNELS), dtype=np.float32)

# Used to be able to pass from a pixel value to the
exponents = np.array(
    [[x, y] for x, y in product(range(ORDER_OF_POLYNOMIAL + 1), repeat=2)]
)

# Visualize
video_source = vis.VideoSource(4, WIDTH, HEIGHT)
m = vis.Monitor("Camera", video_source.shape, refresh_rate=10)

# Core
stft = core.Stft(CHANNELS, FRAME_SIZE, "sqrt_hann")
istft = core.IStft(CHANNELS, FRAME_SIZE, CHUNK_SIZE, "sqrt_hann")
delay_and_sum = core.DelaySum(FRAME_SIZE)

# io
mic_source = io.MicSource(CHANNELS, "ReSpeaker_USB", chunk_size=CHUNK_SIZE)
ds_sink = io.WavSink(
    "./audio_files/delay_and_sum.wav", FILE_PARAMS, CHUNK_SIZE
)
unprocessed_sink = io.WavSink(
    "./audio_files/unprocessed.wav", FILE_PARAMS, CHUNK_SIZE
)

fps.start()
while m.window_is_alive():
    # Get the audio signal and image frame
    x = mic_source()
    frame = video_source()
    unprocessed_sink(x[0])

    # Get predictions
    tensor = opencv_image_to_tensor(frame, DEVICE)
    tensor = torch.unsqueeze(tensor, 0)
    predictions = face_detection_model(tensor)[0]

    # Scale coords
    predictions[:, 5:15] = scale_coords_landmarks(
        tensor.shape[2:], predictions[:, 5:15], frame.shape
    ).round()
    predictions[:, :4] = scale_coords(
        tensor.shape[2:], predictions[:, :4], frame.shape
    ).round()

    # Draw predictions
    for i in range(predictions.size()[0]):
        gn = torch.tensor(frame.shape)[[1, 0, 1, 0]].to(
            DEVICE
        )  # normalization gain whwh
        gn_lks = torch.tensor(frame.shape)[[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]].to(
            DEVICE
        )  # normalization gain landmarks
        xywh = (
            (xyxy2xywh(predictions[i, :4].view(1, 4)) / gn).view(-1).tolist()
        )
        confidence = predictions[i, 4].cpu().item()
        landmarks = (
            (predictions[i, 5:15].view(1, 10) / gn_lks).view(-1).tolist()
        )
        frame = show_results(frame, xywh, confidence, landmarks)

    # Get the signal in the frequency domain
    X = stft(x)

    # If we have a face
    if predictions.size()[0] > 0:
        # Target pixels
        mouth_x = int(((landmarks[6] + landmarks[8]) * WIDTH) / 2)
        mouth_y = int(((landmarks[7] + landmarks[9]) * HEIGHT) / 2)
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

        X = delay_and_sum(X, delay)

    y = istft(X)
    ds_sink(y[0])

    fps.incrementFps()
    final_frame = fps.writeFpsToFrame(frame)

    m.update("Camera", frame)
