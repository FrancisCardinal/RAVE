import numpy as np
import json
import time

from itertools import product
from pyodas.core import BeamScan
from pyodas.utils import (
    icosahedron,
    generate_mic_array,
)
from pyodas.visualize import Monitor, ElevationAndAzimuth

CHUNK_SIZE = 256
CHANNELS = 8
ORDER_OF_POLYNOMIAL = 3
WIDTH = 640
HEIGHT = 480

icosahedron = icosahedron(2)

mic_dict = {
    "mics": {
        "0": [-0.07055, 0, 0],
        "1": [-0.07055, 0.0381, 0],
        "2": [-0.05715, 0.0618, 0],
        "3": [-0.01905, 0.0618, 0],
        "4": [0.01905, 0.0618, 0],
        "5": [0.05715, 0.0618, 0],
        "6": [0.07055, 0.0381, 0],
        "7": [0.07055, 0, 0],
    },
    "nb_of_channels": 8,
}

mic_array = generate_mic_array(mic_dict)

beam_scan = BeamScan(icosahedron, mic_array, CHUNK_SIZE, tolerance=3)

elevation_azimuth = ElevationAndAzimuth(CHANNELS, shape=(1280, 960))
# Define monitor
m = Monitor("Elevation and Azimuth", elevation_azimuth.shape)

# Get calibration matrix
with open(
    "./calibration.json",
    "r",
    encoding="utf-8",
) as f:
    calibration_matrix = np.array(json.load(f))

# Used to reconstruct the full matrix delay
upper_triangle_indices = np.triu_indices(CHANNELS, k=1)
lower_triangle_indices = np.tril_indices(CHANNELS, k=-1)
delay = np.zeros((CHANNELS, CHANNELS), dtype=np.float32)
exponents = np.array([[x, y] for x, y in product(range(ORDER_OF_POLYNOMIAL + 1), repeat=2)])

mouth_x = 0
mouth_y = 240

while mouth_x < WIDTH:
    normalized_mouth_x = (2 * mouth_x - WIDTH - 1) / (WIDTH - 1)
    normalized_mouth_y = (2 * mouth_y - HEIGHT - 1) / (HEIGHT - 1)

    A = (normalized_mouth_x ** exponents[:, 0][:, np.newaxis] * normalized_mouth_y ** exponents[:, 1][:, np.newaxis]).T

    delay_pairs = A @ calibration_matrix
    delay[upper_triangle_indices[0], upper_triangle_indices[1]] = delay_pairs
    delay[lower_triangle_indices[0], lower_triangle_indices[1]] = -1 * delay_pairs

    new_delay = delay[np.newaxis, ...]

    amplitude = np.ones((1, 8, 8), dtype=np.float32)

    direction, score = beam_scan(new_delay, amplitude)

    m.update("Elevation and Azimuth", elevation_azimuth.plot(direction, score))
    mouth_x += 1

time.sleep(10)
