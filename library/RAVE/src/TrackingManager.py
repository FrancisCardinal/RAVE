import time
import cv2
import threading
import uuid

from trackers import TrackerFactory
from face_detectors import YoloFaceDetector
from RAVE.common.image_utils import intersection
from RAVE.face_detection.fpsHelper import FPS
from pyodas.visualize import VideoSource, Monitor


class TrackedObject:
    def __init__(self, tracker, bbox, identifier):
        self.tracker = tracker
        self.bbox = bbox
        self._id = identifier

    @property
    def id(self):
        return self._id

    def update_bbox(self, bbox):
        self.bbox = bbox


class TrackingManager:
    def __init__(self, tracker_type, frequency):
        self._tracker_type = tracker_type

        self._tracked_objects = {}

        self._frequency = frequency
        self.count = 0

        self._detector = YoloFaceDetector()
        self._last_frame = None

    def tracking_count(self):
        return len(self._tracked_objects)

    def add_tracked_object(self, frame, bbox, identifier=None):
        new_tracker = TrackerFactory.create(self._tracker_type)
        new_id = str(identifier) if identifier else str(uuid.uuid4())
        new_tracked_object = TrackedObject(new_tracker, bbox, new_id)
        self._tracked_objects[new_tracked_object.id] = new_tracked_object

        new_thread = threading.Thread(
            target=self.track_loop, args=(new_tracked_object,), daemon=True
        )
        new_tracked_object.tracker.start(frame, bbox)
        new_thread.start()

    def remove_tracked_object(self, identifier):
        self._tracked_objects.pop(identifier)

    def stop_tracking(self):
        self._tracked_objects = {}

    def track_loop(self, tracked_object):

        while tracked_object in self._tracked_objects:
            frame = self._last_frame

            if frame is None:
                print("No frame received")
                continue

            success, box = tracked_object.tracker.update(frame)
            if success:
                xywh_rect = [int(v) for v in box]
                tracked_object.update_bbox(xywh_rect)

    def listen_keyboard_input(self, frame, key_pressed):
        key = key_pressed & 0xFF
        if key == ord("s"):
            # Select object to track manually
            selected_roi = cv2.selectROI(
                "Frame", frame, fromCenter=False, showCrosshair=True
            )
            self.add_tracked_object(frame, selected_roi)
        elif key == ord("x"):
            # Remove last tracked object
            if len(self._tracked_objects) > 0:
                self._tracked_objects.popitem()

    def start(self):
        cap = VideoSource(4)
        m = Monitor(
            "Detection", cap.shape, "Tracking", cap.shape, refresh_rate=30
        )
        cap.set(cv2.CAP_PROP_FPS, 60)
        fps = FPS()

        start_time = time.time()
        # Main display loop
        while m.window_is_alive():
            # TODO (JKealey): Assign directly to sel._last_frame
            #  and add mutex(?)
            frame = cap()
            self._last_frame = frame

            if frame is None:
                print("No frame received, exiting")
                break

            # Detect face for initial frame
            if self.tracking_count() == 0:
                face_frame, predicted_bboxes = self._detector.predict(
                    frame.copy(), draw_on_frame=True
                )
                if len(predicted_bboxes) > 0:
                    for predicted_bbox in predicted_bboxes:
                        self.add_tracked_object(frame, predicted_bbox)
                    m.update("Detection", face_frame)

            elif time.time() - start_time >= self._frequency:
                face_frame, predicted_bboxes = self._detector.predict(
                    frame.copy(), draw_on_frame=True
                )
                if len(predicted_bboxes) > 0:
                    # TODO (JKealey): Find a way to not declare new objects,
                    #  maybe with our own implementation

                    for predicted_bbox in predicted_bboxes:
                        # TODO (JKealey): Find a better way to link previous
                        #  ids to the new bboxes
                        intersection_scores = {}
                        for (
                            tracker_id,
                            tracked_object,
                        ) in self._tracked_objects.items():
                            intersection_scores[tracker_id] = intersection(
                                predicted_bbox, tracked_object.bbox
                            )

                        max_id = max(
                            intersection_scores, key=intersection_scores.get
                        )

                        if intersection_scores[max_id]:
                            new_id = max_id
                            self.remove_tracked_object(max_id)
                        else:
                            new_id = None

                        self.add_tracked_object(frame, predicted_bbox, new_id)
                    m.update("Detection", face_frame)

            # Draw bboxes from tracked objects
            for tracked_object in self._tracked_objects.values():
                if tracked_object.bbox is None:
                    continue
                x, y, w, h = tracked_object.bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    tracked_object.id,
                    (x, y - 2),
                    0,
                    1 / 3,
                    [225, 255, 255],
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )

            # Update display
            fps.setFps()
            frame = fps.writeFpsToFrame(frame)
            m.update("Tracking", frame)

            # Keyboard input controls
            self.listen_keyboard_input(frame, m.key_pressed)

        self.stop_tracking()


if __name__ == "__main__":
    tracking_manager = TrackingManager(tracker_type="kcf", frequency=1)
    tracking_manager.start()
