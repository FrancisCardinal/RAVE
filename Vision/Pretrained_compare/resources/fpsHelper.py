import time
import cv2


class FPS:
    def __init__(self):
        self.startTime = time.time()
        self.frameCount = 0

    def start(self):
        self.startTime = time.time()
        self.frameCount = 0

    def incrementFrameCount(self):
        self.frameCount += 1

    def getFps(self):
        return round(self.frameCount/(time.time() - self.startTime), 2)

    def writeFpsToFrame(self, frame):
        return cv2.putText(frame,
                           'fps: ' + str(self.getFps()),
                           (0, 15),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.5,
                           (0, 0, 255))