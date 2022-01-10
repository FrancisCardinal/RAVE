import time
import cv2


class FPS:
    """
    Class to compute the FPS
    """

    def __init__(self, refresh_rate=1):
        self.startTime = time.time()
        self.frameCount = 0
        self.frame_counter = 0
        self.fps = 0
        self.refresh_rate = refresh_rate

    def start(self):
        """
        To reset the start time, can be used if the part we want to compute is
        not right after the init.
        """
        self.startTime = time.time()
        self.fps_counter = 0

    def incrementFps(self):
        """
        Computes the FPS
        """
        self.frame_counter += 1
        current_time = time.time() - self.startTime
        if current_time >= self.refresh_rate:
            self.fps = self.frame_counter/current_time
            self.frame_counter = 0
            self.startTime = time.time()

    def getFps(self):
        """
        Accessor for the FPS
        """
        return round(self.fps, 2)

    def writeFpsToFrame(self, frame):
        """
        Write the FPS to an OpenCV image.

        Args:
            frame (ndarray): Image with shape (height, width, 3)
        """
        return cv2.putText(
            frame,
            "fps: " + str(int(self.fps)),
            (0, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
        )
