import cv2
from sympy import im

from ..common.LatestVideoCapture import LatestVideoCapture
from ..eye_tracker import EyeTrackerDataset


class EyeTrackerVideoCapture(LatestVideoCapture):
    def __init__(self, index):
        super().__init__(index)

    def setup(self):
        codec = 0x47504A4D  # MJPG
        self._video_feed.set(cv2.CAP_PROP_FPS, 30.0)
        self._video_feed.set(cv2.CAP_PROP_FOURCC, codec)

        self._video_feed.set(
            cv2.CAP_PROP_FRAME_WIDTH, EyeTrackerDataset.ACQUISITION_WIDTH
        )
        self._video_feed.set(
            cv2.CAP_PROP_FRAME_HEIGHT, EyeTrackerDataset.ACQUISITION_HEIGHT
        )

        self._video_feed.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)

        self._video_feed.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self._video_feed.set(cv2.CAP_PROP_FOCUS, 1000)
