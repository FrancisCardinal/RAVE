import cv2
import dlib
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


# List of all the trackers offered by openCV
OPENCV_TRACKERS = {
    # Best accuracy, slower fps
    "csrt": cv2.TrackerCSRT_create,
    # Better fps than csrt, but bit less accuracy
    "kcf": cv2.TrackerKCF_create,
    # Super quick tracker, but has been moved to legacy in OpenCV 4.5.1
    "mosse": cv2.legacy_TrackerMOSSE.create,
    # Older tracker, low accuracy
    "mil": cv2.TrackerMIL_create,
    # NOTE: Could not find a working version
    "boosting": cv2.legacy.TrackerBoosting_create,
    # NOTE: Could not find a working version
    "tld": cv2.legacy.TrackerTLD_create,
    # NOTE: Could not find a working version
    "medianflow": cv2.legacy.TrackerMedianFlow_create,
}


class TrackerFactory:
    """
    Static factory class used to instantiate new Tracker objects
    """

    @staticmethod
    def create(tracker_type="kcf"):
        """
        Create new Tracker object of a specified type

        Args:
            tracker_type (str): Identifier for tracker type

        Returns:
            (Tracker):
                Returns the created Tracker object or None if unknown
                tracker_type supplied
        """
        if OPENCV_TRACKERS.get(tracker_type, None) is not None:
            return TrackerOpenCV(tracker_name=tracker_type)
        elif tracker_type == "dlib":
            return CorrelationTracker()
        else:
            print("Unknown tracker type:", tracker_type)
            return None


class TrackerOpenCV(Tracker):
    """
    Tracker class that acts as a wrapper for all OpenCV trackers
    (csrt, kcf, mosse, mil, boosting, tld, medianflow)

    Attributes:
        tracker (Tracker): The OpenCV tracker
    """

    def __init__(self, tracker_name="kcf"):
        self.tracker = OPENCV_TRACKERS[tracker_name]()

    def start(self, frame, initial_bbox):
        """
        Init the OpenCV tracker to start tracking

        Args:
            frame (np.ndarray): image containing the object to track
            initial_bbox (tuple: (x0, y0, w, h)):
                bounding box indicating location of object in 'frame'
        """
        self.tracker.init(frame, initial_bbox)

    def update(self, frame):
        """
        Update the tracker with a new frame

        Args:
            frame (np.ndarray): new frame used to update the tracker
        """
        return self.tracker.update(frame)


class CorrelationTracker(Tracker):
    """
    Tracker class acting as a wrapper for dlib's correlation_tracker

    Attributes:
        tracker (correlation_tracker): dlib's tracker
    """

    def __init__(self):
        self.tracker = dlib.correlation_tracker()

    def start(self, frame, bbox):
        """
        Init the OpenCV tracker to start tracking

        Args:
            frame (np.ndarray): image containing the object to track
            initial_bbox (tuple: (x0, y0, w, h)):
                bounding box indicating location of object in 'frame'
        """

        # Convert to RGB for dlib (openCV uses BGR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert (x0, y0, w, h) format to
        # (x0, y0, x1, x2) format for dlib rectangle
        rect_dlib = dlib.rectangle(
            bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
        )
        self.tracker.start_track(frame_rgb, rect_dlib)

    def update(self, frame):
        """
        Update the tracker with a new frame

        Args:
            frame (np.ndarray): new frame used to update the tracker
        """

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.tracker.update(frame_rgb)

        pos = self.tracker.get_position()
        start_x = int(pos.left())
        start_y = int(pos.top())
        end_x = int(pos.right())
        end_y = int(pos.bottom())

        # Convert to (x0, y0, w, h) format
        rectangle_out = (start_x, start_y, end_x - start_x, end_y - start_y)

        return True, rectangle_out
