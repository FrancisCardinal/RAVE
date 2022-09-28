import cv2

from .face_trackers import TrackerFactory
from .verifiers.Encoding import Encoding

NB_FRAMES_TO_CONFIRMED = 8
CONFIRMATION_THRESHOLD = 5
NB_FRAMES_TO_REJECT = 10
REJECTION_THRESHOLD = 2


class TrackedObject:
    """
    Container class representing an object that is being tracked. The term
    'object' is meant to be general and would be a face if the application
    is face tracking.

    Attributes:
        tracker (Tracker): Reference to the tracker that is linked to this
            object
        _tracker_type (str): The type identifier of the tracker
        _id (int): Unique identifier for this object to differentiate from
            other tracked objects
        bbox (tuple: (x0, y0, w, h)): Most recent or initial bounding box
            containing the object in a given frame
        _encoding (list): Variable length feature vector used to identify the
            object

        _evaluation_frames (int): Counter for the number of frames that the
            object has been evaluated in pre-process or post-process

        _confirmation_threshold (int): Number of re-detections required to
            complete the pre-process of the object
        _nb_of_frames_to_confirmed (int): Total number of re-detection
            attempts to perform during pre-process
        _confirmed_frames (int): Counter for the number of successful
            re-detections during pre-process
        _confirmed (bool): Flag indicating if the object is confirmed
            (completed pre-process)

        _rejection_threshold (int): Number of non-detections required to
            complete the post-process of the object
        _nb_of_frames_to_reject (int): Total number of non-detection
            attempts to perform during post-process
        _rejected_frames (int): Counter for the number of non-detections during
            post-processing
        _rejected (bool): Flag indicating is the object has been rejected
            (post-process complete)

        _relative_landmark (tuple: (x (int), y (int))): Coordinates of the
            relative position of the landmark of interest with respect to
            self.bbox
        tracker_started (bool): Flag indicating if the tracker has started
            tracking yet

    """

    def __init__(self, tracker_type, frame_object, bbox, mouth, identifier):
        self.tracker = TrackerFactory.create(tracker_type)
        self._tracker_type = tracker_type
        self._id = identifier
        self.bbox = bbox
        self._encoding = Encoding()
        self.current_frame = frame_object

        # Validation
        self._evaluation_frames = 0

        # Pre
        self._confirmation_threshold = CONFIRMATION_THRESHOLD
        self._nb_of_frames_to_confirmed = NB_FRAMES_TO_CONFIRMED
        self._confirmed_frames = 0
        self._confirmed = False

        # Post
        self._rejection_threshold = REJECTION_THRESHOLD
        self._nb_of_frames_to_reject = NB_FRAMES_TO_REJECT
        self._rejected_frames = 0
        self._rejected = False

        self._relative_landmark = None
        self.update_landmark(mouth)

        self.tracker.start(frame_object.frame, bbox)
        self.tracker_started = True

        # Flags used by TrackedObjectManager
        self.updating_missed_frames = False
        self.missed_frames_update_pending = False

    @property
    def id(self):
        """
        Returns:
            (int): The unique identifier for the tracked object
        """
        return self._id

    @property
    def encoding(self):
        """
        Returns:
            (list): The reference feature vector of the tracked object
        """
        return self._encoding

    @property
    def landmark(self):
        """
        Computes absolute coordinates of landmark (in the image) from the
        relative landmark (in the bounding box)

        Returns:
            tuple (int, int): The absolute coordinates of the landmark in the
                image
        """
        if not self._relative_landmark or not self.bbox:
            return None
        x_b, y_b, w_b, h_b = self.bbox
        x_rel, y_rel = self._relative_landmark
        x_abs = int(x_b + (x_rel * w_b))
        y_abs = int(y_b + (y_rel * h_b))
        return x_abs, y_abs

    @property
    def pending(self):
        """
        Returns:
            (bool): Whether the object is in a pending state (in pre-process)
        """
        if self.confirmed:
            return False
        else:
            return self._evaluation_frames < self._nb_of_frames_to_confirmed

    @property
    def confirmed(self):
        """
        Returns:
            (bool): If object is confirmed
        """
        return self._confirmed

    @property
    def rejected(self):
        """
        Returns:
            (bool): If object is rejected
        """
        return self._rejected

    def update_id(self, new_id):
        """
        Setter for id property
        """
        self._id = new_id

    def update_bbox(self, bbox):
        """
        Update the bounding box

        Args:
            bbox (tuple (int, int, int, int)): (x, y, w, h) bounding box of
                the object
        """
        self.bbox = bbox

    def update_encoding(self, feature, frame=None, bbox=None):
        """
        Args:
            feature (list): New feature vector representing the object
        """
        face_image = None
        if frame is not None and bbox is not None:
            x0, y0 = bbox[0], bbox[1]
            x1, y1 = x0 + bbox[2], y0 + bbox[3]
            face_image = frame[y0:y1, x0:x1]

        self._encoding.update(feature, face_image)

    def confirm(self):
        """Used to confirm a bbox in pre-processing"""
        if self.pending:
            self._confirmed_frames += 1
            self._evaluation_frames += 1
            if self._confirmed_frames >= self._confirmation_threshold:
                self._confirmed = True
                self._evaluation_frames = 0

    def reject(self):
        """Used to reject a bbox in post-processing"""
        if self._confirmed:
            self._rejected_frames += 1
            self._evaluation_frames += 1
            print(
                "Reject call:",
                self._rejected_frames,
                " / ",
                self._rejection_threshold,
            )

            if self._evaluation_frames > self._nb_of_frames_to_reject:
                self._evaluation_frames = 0
                self._rejected_frames = 0
                # print(f"Resetting evaluation frames for {self.id}")
            elif self._rejected_frames >= self._rejection_threshold:
                self._rejected = True

    def restore(self, pre_tracked_object):
        """
        Called when object is being restored and counters need to be reset
        as if this is a new tracked object

        Args:
            pre_tracked_object (TrackedObject):
                Object created upon detection of this face, but will now be
                replaced by the restored object. This object if useful because
                it contains information on the new detection (ex.: bbox
                position)
        """
        self._evaluation_frames = 0
        self._rejected_frames = 0
        self._rejected = False
        self.bbox = pre_tracked_object.bbox
        self.current_frame = pre_tracked_object.current_frame
        self.encoding.restore(pre_tracked_object)
        self._relative_landmark = pre_tracked_object._relative_landmark

    def increment_evaluation_frames(self):
        """
        Increment evaluation frame counter
        """
        self._evaluation_frames += 1

    def update_landmark(self, coordinates):
        """
        Update the coordinates of the landmark of interest. Coordinates are
        converted to be relative to the bbox
        """
        if coordinates is None:
            return
        x, y = coordinates
        x_b, y_b, w_b, h_b = self.bbox
        x_rel = (x - x_b) / w_b
        y_rel = (y - y_b) / h_b
        self._relative_landmark = (x_rel, y_rel)

    def reset(self, frame_object, bbox, landmark):
        """
        Used to refresh (reset) the bounding box and landmarks with new
        information

        Args:
            frame (np.ndarray): New image containing the object
            bbox (tuple (int, int, int, int)): new xywh bounding box containing
                the object
            landmark (tuple (int, int)): x & y coordinates for the landmark
        """
        self.tracker_started = False  # Tracker is not ready to use
        self.tracker = TrackerFactory.create(self._tracker_type)
        self.tracker.start(frame_object.frame, bbox)
        self.bbox = bbox
        self.current_frame = frame_object
        self.update_landmark(landmark)

        # Request missing frames update
        self.missed_frames_update_pending = True
        self.tracker_started = True  # Tracker is ready to use
        self._confirmed = True  # Useful when recovering from pre-processing

    def draw_prediction_on_frame(self, frame):
        """
        Draw the tracker prediction on the frame

        Args:
            frame (ndarray): current frame with shape HxWx3
        """

        x, y, w, h = self.bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if self.id is not None:
            cv2.putText(
                frame,
                self.id,
                (x, y - 2),
                0,
                1,
                [0, 0, 255],
                thickness=2,
                lineType=cv2.LINE_AA,
            )

        return frame
