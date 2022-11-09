import pyodas.visualize as vis
import cv2
import numpy as np
from pyodas.io import MicSource

CHUNK_SIZE = 256
FRAME_SIZE = 4 * CHUNK_SIZE
CHANNELS = 8

WIDTH = 640
HEIGHT = 480

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

# Visualization
calibration = vis.AcousticImageCalibration(
    CHANNELS,
    FRAME_SIZE,
    WIDTH,
    HEIGHT,
    save_path="./calibration.json",
)
video_source = vis.VideoSource(
    """v4l2src device=/dev/video0 ! video/x-raw, format=UYVY, width=640, height=480, framerate=60/1 !
     nvvidconv ! video/x-raw(memory:NVMM) ! nvvidconv ! video/x-raw, format=BGRx !  videoconvert ! video/x-raw,
      format=BGR ! appsink",""",
    WIDTH,
    HEIGHT,
)
m = vis.Monitor("Camera", video_source.shape, refresh_rate=100)

K = np.array([[340.60994606, 0.0, 325.7756748], [0.0, 341.93970667, 242.46219777], [0.0, 0.0, 1.0]])
D = np.array([[-3.07926877e-01, 9.16280959e-02, 9.46074597e-04, 3.07906550e-04, -1.17169354e-02]])

corrected_shape = (video_source.shape[1], video_source.shape[0])
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, D, corrected_shape, 1, corrected_shape)
mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, newcameramtx, corrected_shape, 5)

# Core
mic_source = MicSource(CHANNELS, mic_arr=mic_dict, chunk_size=CHUNK_SIZE, queue_size=10, mic_index=4)

while True:
    # Get the audio signal and image frame
    audio = mic_source()
    frame = video_source()

    # Undistort
    frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
    x, y, w, h = roi
    frame = frame[y : y + h, x : x + w]
    corrected_shape = (video_source.shape[1], video_source.shape[0])
    frame = cv2.resize(frame, corrected_shape)

    # If you are using a webcam of a camera facing yourself, this
    # might feel more natural
    # frame = cv2.flip(frame, 1)

    # Draw the targets on the frame and process the audio signal x
    frame = calibration(frame, audio, m.key_pressed)

    m.update("Camera", frame)
    # cv2.imshow('frame', np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8))
    # key = cv2.waitKey(1)
    if m.key_pressed == ord("q"):
        print("breaking")
        break

m.stop()
video_source.stop()
