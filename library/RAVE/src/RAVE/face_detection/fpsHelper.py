import time
import cv2


class FPS:
    """
    Class to compute the FPS
    """

    def __init__(self):
        self.startTime = time.time()
        self.frameCount = 0
        self.fps = 0
        self.epsilon = 1e-20

    def start(self):
        """
        To reset the start time, can be used if the part we want to compute is
        not right after the init.
        """
        self.startTime = time.time()

    def setFps(self):
        """
        Computes the FPS
        """
        self.fps = 1 / (time.time() - self.startTime + self.epsilon)
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
