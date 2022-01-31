from trackers import TrackerFactory

NB_FRAMES_TO_CONFIRMED = 10
CONFIRMATION_THRESHOLD = 8
NB_FRAMES_TO_REJECT = 20
REJECTION_THRESHOLD = 10


class TrackedObject:
    def __init__(self, tracker_type, frame, bbox, mouth, identifier):
        self.tracker = TrackerFactory.create(tracker_type)
        self._tracker_type = tracker_type
        self._id = identifier
        self.bbox = bbox

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

        self.tracker.start(frame, bbox)
        self.tracker_started = True

    @property
    def id(self):
        return self._id

    @property
    def landmark(self):
        if not self._relative_landmark or not self.bbox:
            return None
        x_b, y_b, w_b, h_b = self.bbox
        x_rel, y_rel = self._relative_landmark
        x_abs = int(x_b + (x_rel * w_b))
        y_abs = int(y_b + (y_rel * h_b))
        return x_abs, y_abs

    @property
    def pending(self):
        if self.confirmed:
            return False
        else:
            return self._evaluation_frames < self._nb_of_frames_to_confirmed

    @property
    def confirmed(self):
        return self._confirmed

    @property
    def rejected(self):
        return self._rejected

    def update_bbox(self, bbox):
        self.bbox = bbox

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
            elif self._rejected_frames >= self._rejection_threshold:
                self._rejected = True

    def increment_evaluation_frames(self):
        self._evaluation_frames += 1

    def update_landmark(self, coordinates):
        if coordinates is None:
            return
        x, y = coordinates
        x_b, y_b, w_b, h_b = self.bbox
        x_rel = (x - x_b) / w_b
        y_rel = (y - y_b) / h_b
        self._relative_landmark = (x_rel, y_rel)

    def reset(self, frame, bbox, mouth):
        self.tracker_started = False  # Tracker is not ready to use
        if self._tracker_type == "sort":
            self.tracker.update_tracker(bbox)
        else:
            self.tracker = TrackerFactory.create(self._tracker_type)
            self.tracker.start(frame, bbox)
        self.bbox = bbox
        self.update_landmark(mouth)
        self.tracker_started = True  # Tracker is ready to use
