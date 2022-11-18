import cv2

from RAVE.face_detection.trackers.TrackerPyTracking import TrackerPyTracking
from pyodas.visualize import VideoSource, Monitor

WIDTH = 640
HEIGHT = 480
DEVICE_INDEX = 0

video_source = VideoSource(DEVICE_INDEX, WIDTH, HEIGHT)
video_source.set(cv2.CAP_PROP_FPS, 30)
m = Monitor("Camera", video_source.shape)

tracker = TrackerPyTracking("dimp", "dimp18")

bbox = [252, 120, 230, 270]  # x, y, w, h
tracker.start(video_source(), bbox)

while m.window_is_alive():
    frame = video_source()
    frame_disp = frame.copy()

    succes, bbox = tracker.update(frame)

    if bbox is not None:
        bbox = [int(v) for v in bbox]
        cv2.rectangle(frame_disp, (bbox[0], bbox[1]), (bbox[2] + bbox[0], bbox[3] + bbox[1]), (255, 0, 0), 5)

    m.update("Camera", frame_disp)
