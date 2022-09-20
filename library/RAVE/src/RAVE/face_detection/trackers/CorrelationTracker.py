import cv2
import dlib

from Tracker import Tracker


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
