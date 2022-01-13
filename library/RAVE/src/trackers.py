import cv2
import dlib


class Tracker:

    # Initialize tracker
    def start(self, frame, initial_bbox):
        raise NotImplementedError()

    # Update tracker with new frames
    def update(self, frame):
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
    # "boosting": cv2.TrackerBoosting_create,
    # NOTE: Could not find a working version
    # "tld": cv2.TrackerTLD_create,
    # NOTE: Could not find a working version
    # "medianflow": cv2.TrackerMedianFlow_create,
}


class TrackerFactory:
    @staticmethod
    def create(tracker_type="kcf"):
        if OPENCV_TRACKERS.get(tracker_type, None) is not None:
            return TrackerOpenCV(tracker_name=tracker_type)
        elif tracker_type == "dlib":
            return CorrelationTracker()
        else:
            print("Unknown tracker type:", tracker_type)
            return None


class TrackerOpenCV(Tracker):
    def __init__(self, tracker_name="kcf"):
        self.tracker = OPENCV_TRACKERS[tracker_name]()

    def start(self, frame, bbox):
        self.tracker.init(frame, bbox)

    def update(self, frame):
        return self.tracker.update(frame)


# Wrapper for dlib's correlation_tracker
class CorrelationTracker(Tracker):
    def __init__(self):
        self.tracker = (
            dlib.correlation_tracker()
        )  # OPENCV_TRACKERS[tracker_name]()  # Select and create tracker

    def start(self, frame, bbox):
        # Convert to RGB for dlib (openCV uses BGR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert (x0, y0, w, h) format to
        # (x0, y0, x1, x2) format for dlib rectangle
        rect_dlib = dlib.rectangle(
            bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
        )
        self.tracker.start_track(frame_rgb, rect_dlib)

    def update(self, frame):
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
