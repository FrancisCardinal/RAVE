from abc import ABC, abstractmethod


class Tracker(ABC):
    """
    Abstract class for trackers
    """

    @abstractmethod
    def start(self, frame, initial_bbox):
        """
        Initialize the tracker to start tracking

        Args:
            frame (np.ndarray): image containing the object to track
            initial_bbox (tuple: (x0, y0, w, h)):
                bounding box indicating location of object in 'frame'
        """
        raise NotImplementedError()

    @abstractmethod
    def update(self, frame):
        """
        Update the tracker with a new frame

        Args:
            frame (np.ndarray): new frame used to update the tracker
        """
        raise NotImplementedError()
