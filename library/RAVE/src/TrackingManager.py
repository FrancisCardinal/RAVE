import time
import cv2
import threading
import argparse

from trackers import TrackerFactory
from face_detectors import YoloFaceDetector
from RAVE.common.image_utils import intersection
from RAVE.face_detection.fpsHelper import FPS
from pyodas.visualize import VideoSource, Monitor


class TrackedObject:
    def __init__(self, tracker_type, frame, bbox, mouth, identifier):
        self.tracker = TrackerFactory.create(tracker_type)
        self.bbox = bbox
        self._tracker_type = tracker_type
        self._id = identifier

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

    def update_bbox(self, bbox):
        self.bbox = bbox

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
        self.tracker = TrackerFactory.create(self._tracker_type)
        self.tracker.start(frame, bbox)
        self.bbox = bbox
        self.update_landmark(mouth)
        self.tracker_started = True  # Tracker is ready to use


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

    # Assumed to be called from main thread only
    def new_identifier(self):
        new_id = str(self.count)
        self.count += 1
        return new_id

    # Register a new object to tracker. Assumed to be called from main thread
    def add_tracked_object(self, frame, bbox, mouth, identifier=None):
        new_id = str(identifier) if identifier else self.new_identifier()
        new_tracked_object = TrackedObject(
            self._tracker_type, frame, bbox, mouth, new_id
        )
        self._tracked_objects[new_tracked_object.id] = new_tracked_object

        new_thread = threading.Thread(
            target=self.track_loop, args=(new_tracked_object,), daemon=True
        )
        new_thread.start()

    def remove_tracked_object(self, identifier):
        self._tracked_objects.pop(identifier)

    def stop_tracking(self):
        self._tracked_objects = {}

    def track_loop(self, tracked_object):

        while tracked_object in self._tracked_objects.values():
            frame = self._last_frame

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

    def listen_keyboard_input(self, frame, key_pressed):
        key = key_pressed & 0xFF
        if key == ord("s"):
            # Select object to track manually
            selected_roi = cv2.selectROI(
                "Frame", frame, fromCenter=False, showCrosshair=True
            )
            self.add_tracked_object(frame, selected_roi, None)
        elif key == ord("x"):
            # Remove last tracked object
            if len(self._tracked_objects) > 0:
                self._tracked_objects.popitem()
        elif key == ord("q") or key == 27:
            return True

    def start(self, args):
        cap = VideoSource(args.video_source)
        shape = (
            (cap.shape[1], cap.shape[0])
            if args.flip_display_dim
            else cap.shape
        )
        m = Monitor("Detection", shape, "Tracking", shape, refresh_rate=30)
        cap.set(cv2.CAP_PROP_FPS, 60)
        fps = FPS()

        last_detect = 0
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
                face_frame, predicted_bboxes, mouth = self._detector.predict(
                    frame.copy(), draw_on_frame=True
                )
                last_detect = time.time()
                if len(predicted_bboxes) > 0:
                    for predicted_bbox in predicted_bboxes:
                        self.add_tracked_object(frame, predicted_bbox, mouth)
                    m.update("Detection", face_frame)

            elif time.time() - last_detect >= self._frequency:
                face_frame, predicted_bboxes, mouth = self._detector.predict(
                    frame.copy(), draw_on_frame=True
                )
                last_detect = time.time()
                if len(predicted_bboxes) > 0:
                    # TODO (JKealey): Find a way to not declare new objects,
                    #  maybe with our own implementation

                    current_ids = set(self._tracked_objects.keys())
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
                            self._tracked_objects[max_id].reset(
                                frame, predicted_bbox, mouth
                            )
                            current_ids.discard(max_id)
                        else:
                            self.add_tracked_object(
                                frame, predicted_bbox, mouth
                            )

                    # Remove tracked objects that are not re-detected
                    for tracker_id in current_ids:
                        print("Removing tracked object")
                        self._tracked_objects.pop(tracker_id, None)

                    m.update("Detection", face_frame)

            # Draw detections from tracked objects
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
                mouth = tracked_object.landmark
                if mouth is not None:
                    x_mouth, y_mouth = mouth
                    cv2.circle(frame, (x_mouth, y_mouth), 5, [0, 0, 255], -1)

            # Update display
            fps.setFps()
            # TODO: Fix fps? Delta too short?
            # frame = fps.writeFpsToFrame(frame)
            m.update("Tracking", frame)

            # Keyboard input controls
            terminate = self.listen_keyboard_input(frame, m.key_pressed)
            if terminate:
                break

        self.stop_tracking()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face tracking")
    parser.add_argument(
        "--video_source",
        dest="video_source",
        type=int,
        help="Video input source identifier",
        default=0,
    )
    parser.add_argument(
        "--flip_display_dim",
        dest="flip_display_dim",
        type=bool,
        help="If true, will flip window dimensions to (width, height)",
        default=False,
    )
    parser.add_argument(
        "--freq",
        dest="freq",
        type=float,
        help="Update frequency for the face detector (for adaptive scaling)",
        default=1,
    )
    args = parser.parse_args()

    frequency = args.freq
    tracking_manager = TrackingManager(tracker_type="kcf", frequency=frequency)
    tracking_manager.start(args)
