import cv2, queue, threading, time

# This class is a wrapper of the cv2.VideoCapture class.
# It ensures that a call to read() returns only the latest (newest) frame.
# Taken from https://stackoverflow.com/a/54755738
class LatestVideoCapture:
    def __init__(self, index):
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise IOError("Cannot open specified device ({})".format(index))

        self.q = queue.Queue()
        self._should_run = True
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    def __del__(self):
        self._should_run = False
        self._cap.release()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        time.sleep(0.25)

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

    def read(self):
        return self.q.get()

    def set(self, property, value):
        self._cap.set(property, value)

    def get(self, property):
        return self._cap.get(property)

    def isOpened(self):
        return self._cap.isOpened()
