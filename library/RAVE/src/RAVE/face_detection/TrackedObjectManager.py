import time
import threading

from .TrackedObject import TrackedObject


class TrackedObjectManager:
    """
    Class used as a container to handle all tracked objects and their tracker
    objects
    Handles the creation of the threads running the tracking loops (for each
    object)

    Args:
        tracker_type (str): The type of tracker to use

    Attributes:
        tracker_type (str): type of tracker
        tracked_objects (dict of TrackedObject):
            dictionary of all faces currently being tracked having been
            confirmed
        pre_tracked_objects (dict of TrackedObject):
            dictionary of all faces currently being in the process of being
            confirmed
        rejected_objects (dict of TrackedObject):
            dictionary of all faces rejected by post-process
        count (int): for assigning ids.
        last_frame (ndarray):
            Last capture frame from stream. Is updated by TrackingManager
    """

    def __init__(self, tracker_type):
        self.tracker_type = tracker_type

        self.tracked_objects = {}
        self.pre_tracked_objects = {}
        self.rejected_objects = {}

        self.count_id = 0
        self.count_id_pre = 0
        self.last_frame = None

    def tracking_count(self):
        """
        Returns:
            int: The number of tracked objects
        """
        return len(self.tracked_objects)

    # Returns a dictionary combining pre-tracked and tracked objects
    def get_all_objects(self):
        """
        Returns:
            dict of TrackedObject:
                Dictionary of all faces being tracked confirmed and unconfirmed
        """
        return {**self.tracked_objects, **self.pre_tracked_objects}

    # Assumed to be called from main thread only
    def new_identifier(self):
        """
        Used to assign a new id to a new tracking object
        Returns:
            int:
                A new id
        """
        new_id = str(self.count_id_pre)
        self.count_id_pre += 1
        return new_id

        # Assumed to be called from main thread only

    def new_pre_identifier(self):
        """
        Used to assign a new id to a new pre-tracking object
        Returns:
            int:
                A new id for pre-tracked object
        """
        new_id = str(self.count_id)
        self.count_id += 1
        return new_id

    # Register a new object to tracker. Assumed to be called from main thread
    def add_pre_tracked_object(self, frame, bbox, mouth):
        """
        Creates a new TrackedObject for the new bbox and adds it to the
        pre-tracked list. It also starts the tracking thread.

        Args:
            frame (ndarray):
                current frame with shape HxWx3
            bbox (list):
                in format x,y,w,h (superior left corner)
            mouth (list or None):
                The position of the mouth in x,y
        """
        new_id = self.new_pre_identifier()
        new_tracked_object = TrackedObject(
            self.tracker_type, frame, bbox, mouth, new_id
        )
        self.pre_tracked_objects[new_tracked_object.id] = new_tracked_object
        self.start_tracking_thread(new_tracked_object)

    def start_tracking_thread(self, tracked_object):
        """
        Start the tracker with the tracked_object

        Args:
            tracked_object (TrackedObject): The object to track
        """
        new_thread = threading.Thread(
            target=self.track_loop, args=(tracked_object,), daemon=True
        )
        new_thread.start()

    def remove_tracked_object(self, identifier):
        """
        Remove tracked object from the tracked_object dictionary
        and add it to the _rejected_objects

        Args:
            identifier (int):
                id of the tracked object to be removed
        """
        rejected_object = self.tracked_objects.pop(identifier)
        self.rejected_objects[identifier] = rejected_object

    def restore_rejected_object(self, identifier, pre_tracked_object):
        """
        Restore old tracked object to resume tracking.
        Remove tracked object from the _rejected_objects dictionary
        and add it to the tracked_object. Also start the tracking thread

        Args:
            identifier (int):
                id of the rejected object to be restored
            pre_tracked_object (TrackedObject):
                Object created upon detection of this face, but will now be
                replaced by the restored object. This object if useful because
                it contains information on the new detection (ex.: bbox
                position)
        """
        restored_object = self.rejected_objects.pop(identifier)
        restored_object.restore(pre_tracked_object)
        self.tracked_objects[identifier] = restored_object
        self.start_tracking_thread(restored_object)

    def remove_pre_tracked_object(self, identifier):
        """
        Remove tracked object from the _pre_tracked_object dictionary

        Args:
            identifier (int):
                id of the tracked object to be removed
        """
        self.pre_tracked_objects.pop(identifier)

    def stop_tracking(self):
        """
        Remove all items from the tracked_objects dictionary
        """
        self.tracked_objects = {}
        self.pre_tracked_objects = {}
        self.rejected_objects = {}

    def track_loop(self, tracked_object):
        """
        Thread worker for calling the tracker on the TrackedObject

        Args:
            tracked_object (TrackedObject): The object to track
        """
        while (
            tracked_object in self.tracked_objects.values()
            or tracked_object in self.pre_tracked_objects.values()
        ):
            frame = self.last_frame

            if frame is None:
                print("No frame received")
                continue

            # Make sure tracker is ready to use
            if not tracked_object.tracker_started:
                continue

            success, box = tracked_object.tracker.update(frame)

            if success:
                xywh_rect = [int(v) for v in box]
                tracked_object.update_bbox(xywh_rect)

            time.sleep(0.05)

        print(f"Stopped tracking object {tracked_object.id}")
