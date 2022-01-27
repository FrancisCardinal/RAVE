import time
import cv2
import threading
import argparse

from collections import defaultdict

# from tqdm import tqdm

from face_detectors import DetectorFactory
from RAVE.common.image_utils import intersection
from RAVE.face_detection.fpsHelper import FPS
from pyodas.visualize import VideoSource, Monitor
from TrackedObject import TrackedObject


class TrackingManager:
    def __init__(
        self,
        tracker_type,
        detector_type,
        frequency,
        intersection_threshold=0.5,
    ):
        self._tracker_type = tracker_type

        self._tracked_objects = {}
        self._pre_tracked_objects = {}

        self._frequency = frequency
        self._intersection_threshold = intersection_threshold
        self.count = 0

        self._detector = DetectorFactory.create(detector_type)
        self._last_frame = None
        self._last_detect = 0

    def tracking_count(self):
        return len(self._tracked_objects)

    # Returns a dictionary combining pre-tracked and tracked objects
    def get_all_objects(self):
        return {**self._tracked_objects, **self._pre_tracked_objects}

    # Assumed to be called from main thread only
    def new_identifier(self):
        new_id = str(self.count)
        self.count += 1
        return new_id

    # Register a new object to tracker. Assumed to be called from main thread
    def add_tracked_object(self, frame, bbox, mouth):
        new_id = self.new_identifier()
        new_tracked_object = TrackedObject(
            self._tracker_type, frame, bbox, mouth, new_id
        )
        self._pre_tracked_objects[new_tracked_object.id] = new_tracked_object

        new_thread = threading.Thread(
            target=self.track_loop, args=(new_tracked_object,), daemon=True
        )
        new_thread.start()

    def remove_tracked_object(self, identifier):
        self._tracked_objects.pop(identifier)

    def remove_pre_tracked_object(self, identifier):
        self._pre_tracked_objects.pop(identifier)

    def stop_tracking(self):
        self._tracked_objects = {}

    def track_loop(self, tracked_object):
        # with tqdm(desc=f"{tracked_object.id}", total=25000) as pbar:
        while (
            tracked_object in self._tracked_objects.values()
            or tracked_object in self._pre_tracked_objects.values()
        ):
            frame = self._last_frame

            if frame is None:
                print("No frame received")
                continue

            # Make sure tracker is ready to use
            if not tracked_object.tracker_started:
                continue

            success, box = tracked_object.tracker.update(frame)
            # pbar.update()
            if success:
                xywh_rect = [int(v) for v in box]
                tracked_object.update_bbox(xywh_rect)

        print(f"Stopped tracking object {tracked_object.id}")

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

    def draw_prediction_on_frame(self, frame, tracked_object):
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

        return frame

    def preprocess_faces(self, frame, monitor):

        (
            pre_face_frame,
            pre_face_bboxes,
            pre_face_mouths,
        ) = self._detector.predict(frame.copy(), draw_on_frame=True)

        finished_trackers = []
        pre_tracker_frame = frame.copy()
        for tracker_id, tracked_object in self._pre_tracked_objects.items():
            # TODO (JKealey): Find a better way to link previous
            #  ids to the new bboxes
            intersection_scores = list()
            for predicted_bbox in pre_face_bboxes:
                intersection_scores.append(
                    intersection(predicted_bbox, tracked_object.bbox)
                )

            max_score = (
                max(intersection_scores) if intersection_scores else None
            )
            if (
                intersection_scores
                and max_score > self._intersection_threshold
            ):
                tracked_object.confirm()
                max_index = intersection_scores.index(max_score)
                pre_face_bboxes.pop(max_index)
            else:
                tracked_object.increment_evaluation_frames()

            if tracked_object.confirmed:
                self._tracked_objects[tracker_id] = tracked_object

            if not tracked_object.pending:
                finished_trackers.append(tracker_id)

            if tracked_object.bbox is None:
                continue

            pre_tracker_frame = self.draw_prediction_on_frame(
                pre_tracker_frame, tracked_object
            )

        for tracker in finished_trackers:
            self.remove_pre_tracked_object(tracker)

        monitor.update("Pre-process", pre_tracker_frame)
        return pre_face_frame, pre_face_bboxes, pre_face_mouths

    def detector_update(self, frame, monitor, pre_detections):
        if any([elem is not None for elem in pre_detections]):
            face_frame, predicted_bboxes, mouths = pre_detections
        else:
            (face_frame, predicted_bboxes, mouths) = self._detector.predict(
                frame.copy(), draw_on_frame=True
            )
        self._last_detect = time.time()

        last_ids = set(self._tracked_objects.keys())
        for i, predicted_bbox in enumerate(predicted_bboxes):
            # TODO (JKealey): Find a better way to link previous
            #  ids to the new bboxes
            if mouths and len(mouths) > i:
                mouth = mouths[i]
            else:
                mouth = None
            intersection_scores = defaultdict(lambda: 0)

            all_tracked_objects = self.get_all_objects()
            for tracker_id, object in all_tracked_objects.items():
                intersection_scores[tracker_id] = intersection(
                    predicted_bbox, object.bbox
                )

            max_id = None
            if intersection_scores:
                max_id = max(intersection_scores, key=intersection_scores.get)

            if (
                intersection_scores
                and intersection_scores[max_id] > self._intersection_threshold
            ):
                if max_id in self._tracked_objects.keys():
                    self._tracked_objects[max_id].reset(
                        frame, predicted_bbox, mouth
                    )
                    self._tracked_objects[max_id].increment_evaluation_frames()
                    last_ids.discard(max_id)
            else:
                # A new object was discovered
                self.add_tracked_object(frame, predicted_bbox, mouth)

        # Remove tracked objects that are not re-detected
        for tracker_id in last_ids:
            self._tracked_objects[tracker_id].reject()
            if self._tracked_objects[tracker_id].rejected:
                # self._tracked_objects.pop(tracker_id, None)
                self.remove_tracked_object(tracker_id)
                print("Rejecting tracked object:", tracker_id)

        monitor.update("Detection", face_frame)

    def main_loop(self, monitor, cap, fps):
        while monitor.window_is_alive():
            # TODO (JKealey): Assign directly to sel._last_frame
            #  and add mutex(?)
            frame = cap()
            self._last_frame = frame

            if frame is None:
                print("No frame received, exiting")
                break

            # Do pre-processing of faces
            pre_frame, pre_bboxes, pre_mouths = None, None, None
            if self._pre_tracked_objects:
                pre_frame, pre_bboxes, pre_mouths = self.preprocess_faces(
                    frame, monitor
                )

            if time.time() - self._last_detect >= self._frequency:
                pre_detections = (pre_frame, pre_bboxes, pre_mouths)
                self.detector_update(frame, monitor, pre_detections)

            # Draw detections from tracked objects
            tracking_frame = frame.copy()
            for tracked_object in self._tracked_objects.values():
                if tracked_object.bbox is None:
                    continue
                self.draw_prediction_on_frame(tracking_frame, tracked_object)

                # Draw mouth point
                mouth = tracked_object.landmark
                if mouth is not None:
                    x_mouth, y_mouth = mouth
                    cv2.circle(
                        tracking_frame, (x_mouth, y_mouth), 5, [0, 0, 255], -1
                    )

            # Update display
            fps.setFps()
            # TODO: Fix fps? Delta too short?
            # frame = fps.writeFpsToFrame(frame)
            monitor.update("Tracking", tracking_frame)

            # Keyboard input controls
            terminate = self.listen_keyboard_input(frame, monitor.key_pressed)
            if terminate:
                break

    def start(self, args):
        cap = VideoSource(args.video_source)
        shape = (
            (cap.shape[1], cap.shape[0])
            if args.flip_display_dim
            else cap.shape
        )
        monitor = Monitor(
            "Detection",
            shape,
            "Tracking",
            shape,
            "Pre-process",
            shape,
            refresh_rate=30,
        )
        cap.set(cv2.CAP_PROP_FPS, 60)
        fps = FPS()

        # Main display loop
        self.main_loop(monitor, cap, fps)
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
    tracking_manager = TrackingManager(
        tracker_type="kcf", detector_type="yolo", frequency=frequency
    )
    tracking_manager.start(args)
