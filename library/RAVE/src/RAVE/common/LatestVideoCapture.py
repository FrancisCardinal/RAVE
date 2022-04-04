import cv2, queue, threading, time

# This class is a wrapper of the cv2.VideoCapture class.
# It ensures that a call to read() returns only the latest (newest) frame.
# Taken from https://stackoverflow.com/a/54755738
class LatestVideoCapture:
    def __init__(self, index):
        self._index = index
        self.q = queue.Queue()
        self._should_run = True
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    def end(self):
        self._should_run = False
        self._video_feed.release()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        self._video_feed = cv2.VideoCapture(self._index)
        if not self._video_feed.isOpened():
            raise IOError(
                "Cannot open specified device ({})".format(self._index)
            )

        self.setup()

        while self._should_run:
            ret, frame = self._video_feed.read()
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

    def setup(self):
        pass

