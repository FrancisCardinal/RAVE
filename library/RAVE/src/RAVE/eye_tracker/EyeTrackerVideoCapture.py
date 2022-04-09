import cv2

from ..common.LatestVideoCapture import LatestVideoCapture


class EyeTrackerVideoCapture(LatestVideoCapture):
    """Child class of the LatestVideoCapture class, acquires frames from the
    camera for the eye tracking module
    """

    ACQUISITION_WIDTH, ACQUISITION_HEIGHT = 640, 480

    def __init__(self, index):
        """Constructor of the EyeTrackerVideoCapture class

        Args:
            index (int): opencv index of the device
        """
        super().__init__(index)

    def setup(self):
        """This method is an abstract method of the LatestVideoCapture class
        that is overidden by this child class as we wish to perform some
        operations before the frame acquisition begins. Notably, it sets
        the width and height of the capture, its exposure time, and disables
        autofocus.
        """
        codec = 0x47504A4D  # MJPG
        self._video_feed.set(cv2.CAP_PROP_FPS, 30.0)
        self._video_feed.set(cv2.CAP_PROP_FOURCC, codec)

        self._video_feed.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            self.ACQUISITION_WIDTH,
        )
        self._video_feed.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            self.ACQUISITION_HEIGHT,
        )

        # Set the auto exposure flag to true (i.e, don't use a fixed exposure
        # time)
        self._video_feed.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)

        # Disable autofocus and fix it to a manual value.
        self._video_feed.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self._video_feed.set(cv2.CAP_PROP_FOCUS, 1000)
