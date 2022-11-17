import cv2

from .Tracker import Tracker


OPENCV_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    # Better fps than csrt, but bit less accuracy
    "kcf": cv2.TrackerKCF_create,
    # Super quick tracker, but has been moved to legacy in OpenCV 4.5.1
    "mosse": cv2.legacy.TrackerMOSSE.create,
    # Older tracker, low accuracy
    "mil": cv2.legacy.TrackerMIL_create,
    # NOTE: Could not find a working version
    "boosting": cv2.legacy.TrackerBoosting_create,
    # NOTE: Could not find a working version
    "tld": cv2.legacy.TrackerTLD_create,
    # NOTE: Could not find a working version
    "medianflow": cv2.legacy.TrackerMedianFlow_create,
}


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
