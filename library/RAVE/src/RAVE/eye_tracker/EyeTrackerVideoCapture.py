import cv2, queue, threading

# This class is a wrapper of the cv2.VideoCapture class.
# It ensures that a call to read() returns only the latest (newest) frame.
# Taken from https://stackoverflow.com/a/54755738
class EyeTrackerVideoCapture:
    def __init__(self, index, ACQUISITION_WIDTH, ACQUISITION_HEIGHT):
        self.index = index
        self.ACQUISITION_WIDTH = ACQUISITION_WIDTH
        self.ACQUISITION_HEIGHT = ACQUISITION_HEIGHT

        self.q = queue.Queue()
        self._should_run = True
        self._thread = threading.Thread(target=self._reader)
        self._thread.daemon = True
        self._thread.start()

    def stop(self):
        self._should_run = False

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        self._cap = cv2.VideoCapture(self.index)
        if not self._cap.isOpened():
            raise IOError("Cannot open specified device ({})".format(self.index))

        codec = 0x47504A4D  # MJPG
        self._cap.set(cv2.CAP_PROP_FPS, 30.0)
        self._cap.set(cv2.CAP_PROP_FOURCC, codec)

        self._cap.set(
            cv2.CAP_PROP_FRAME_WIDTH, self.ACQUISITION_WIDTH
        )
        self._cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT, self.ACQUISITION_HEIGHT
        )

        self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)

        self._cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self._cap.set(cv2.CAP_PROP_FOCUS, 1000)

        while self._should_run:
            ret, frame = self._cap.read()
            if not ret:
                continue
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)
        
        self._cap.release()

    def read(self):
        return self.q.get()

if __name__ == "__main__":
    cap = EyeTrackerVideoCapture(1, 640, 480)

    out = cv2.VideoWriter(
        "tmp.avi",
        cv2.VideoWriter_fourcc("M", "J", "P", "G"),
        30,
        (640, 480),
    )

    for i in range(5):
        frame = cap.read()
        out.write(frame)
    out.release()

    cap.stop()
