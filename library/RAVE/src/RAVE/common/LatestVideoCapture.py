import cv2
import queue
import threading


class LatestVideoCapture:
    """This class is a wrapper of the cv2.VideoCapture class.
    It ensures that a call to read() returns only the latest (newest) frame.
    This is done by using a thread that reads frames as soon as they are
    available, and that stores only the most recent one in a queue.
    Taken from https://stackoverflow.com/a/54755738
    """

    def __init__(self, index):
        """Constructor of the LatestVideoCapture class

        Args:
            index (int): opencv index of the device to use
        """
        self._index = index
        self.q = queue.Queue()
        self._should_run = True
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    def end(self):
        """Ends the _reader thread and releases the opencv capture object"""
        self._should_run = False
        self._video_feed.release()

    def _reader(self):
        """Read frames as soon as they are available, keeping only the most
        recent one in the queue

        Raises:
            IOError: Raised if the opencv device can not be opened
        """
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
        """Get the most recent frame from the queue

        Returns:
            np array: The most recent frame
        """
        return self.q.get()

    def setup(self):
        """This method should be overidden by children classes if they
        wish to perform some operations before the frame acquisition begins
        (ex : set the width and height of the capture, its exposure time,
        and so on)
        """
        pass
