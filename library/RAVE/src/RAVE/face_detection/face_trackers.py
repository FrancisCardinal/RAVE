# List of all the trackers offered by openCV
OPENCV_TRACKERS = [
    "csrt",
    "kcf",
    "mosse",
    "mil",
    "boosting",
    "tld",
    "medianflow",
]


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
        if tracker_type in OPENCV_TRACKERS:
            from .trackers.TrackerOpenCV import TrackerOpenCV

            return TrackerOpenCV(tracker_name=tracker_type)

        elif tracker_type == "pytracking":
            from .trackers.TrackerPyTracking import TrackerPyTracking

            return TrackerPyTracking()

        elif tracker_type == "dlib":
            from .trackers.CorrelationTracker import CorrelationTracker

            return CorrelationTracker()
        else:
            print("Unknown tracker type:", tracker_type)
            return None
