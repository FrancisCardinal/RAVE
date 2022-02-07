import cv2
import dlib
import numpy as np
from filterpy.kalman import KalmanFilter
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
        elif tracker_type == "sort":
            return KalmanBoxTracker()
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


class KalmanBoxTracker:
    """
    This class represents the internal state of individual tracked objects
    observed as bbox.
    """

    def __init__(self):
        """
        Initialises Kalman filter.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ]
        )
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ]
        )

        self.kf.R[2:, 2:] *= 10.0
        # give high uncertainty to the unobservable initial velocities
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.started = True

    def start(self, frame, bbox):
        """
        Function to mimic opencvs tracker behavior and initialise the bbox
        Args:
            frame: to match opencv signature
            bbox: bbox to track
        """
        if not self.started:
            self.kf.x *= 0
            self.kf.x[:4] = convert_bbox_to_z(bbox)
        else:
            self.kf.update(convert_bbox_to_z(bbox))

    def update(self, frame):
        """
        Args:
            frame: to match opencv signature

        Returns:
            success, The predicted bbox in shape (x1,y1,w,h)
        """
        return True, self.predict()[0]

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding
        box estimate.

        Returns:
            The predicted bbox in shape (x1,y1,w,h)
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()

        return convert_x_to_bbox(self.kf.x)


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and
      r is the aspect ratio
    """
    w = bbox[2]
    h = bbox[3]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the
     form [x1,y1,w,h] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array([x[0] - w / 2.0, x[1] - h / 2.0, w, h]).reshape((1, 4))
    else:
        return np.array(
            [
                x[0] - w / 2.0,
                x[1] - h / 2.0,
                w,
                h,
                score,
            ]
        ).reshape((1, 5))
