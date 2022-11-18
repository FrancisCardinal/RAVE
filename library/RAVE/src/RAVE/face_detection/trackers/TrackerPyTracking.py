import os
import sys

from collections import OrderedDict
from .pytracking.evaluation import Tracker as PyTracker
from .Tracker import Tracker as Tracker

env_path = os.path.join(os.path.dirname(__file__))
if env_path not in sys.path:
    sys.path.append(env_path)


class TrackerPyTracking(Tracker):
    """
    Tracker class that acts as a wrapper for all OpenCV trackers
    (csrt, kcf, mosse, mil, boosting, tld, medianflow)

    Attributes:
        tracker (Tracker): The OpenCV tracker
    """

    def __init__(self, tracker_name="dimp", tracker_params="dimp18"):
        tracker_class = PyTracker(tracker_name, tracker_params)
        params = tracker_class.get_parameters()

        params.tracker_name = tracker_class.name
        params.param_name = tracker_class.parameter_name

        self.tracker = tracker_class.create_tracker(params)

        self.sequence_object_ids = []
        self.prev_output = OrderedDict()

    def start(self, frame, initial_bbox):
        """
        Init the OpenCV tracker to start tracking

        Args:
            frame (np.ndarray): image containing the object to track
            initial_bbox (tuple: (x0, y0, w, h)):
                bounding box indicating location of object in 'frame'
        """
        self.id = 1
        info = OrderedDict()
        info["previous_output"] = self.prev_output
        info["object_ids"] = [
            self.id,
        ]
        info["init_bbox"] = initial_bbox

        out = self.tracker.initialize(frame, info)
        self.prev_output = OrderedDict(out)

    def update(self, frame):
        """
        Update the tracker with a new frame

        Args:
            frame (np.ndarray): new frame used to update the tracker
        """

        out = self.tracker.track(frame)

        bbox = out["target_bbox"]
        return (True, bbox)

    def reset(self, frame, bbox):
        info = OrderedDict()
        info["object_ids"] = [
            self.id,
        ]
        info["init_bbox"] = bbox
        self.tracker.resetbbox(frame, info)
