import cv2
import threading

from trackers import TrackerFactory
from face_detectors import YoloFaceDetector
from RAVE.face_detection.fpsHelper import FPS
from pyodas.visualize import VideoSource


class TrackedObject:
    def __init__(self, tracker, bbox):
        self.tracker = tracker
        self.bbox = bbox
        # self._tracked_id = tracked_id  # TODO: set unique id

    def update_bbox(self, bbox):
        self.bbox = bbox


class TrackingManager:
    def __init__(self, tracker_type):
        self._tracker_type = tracker_type

        self._tracked_objects = []
        self.count = 0

        self._detector = YoloFaceDetector()
        self._last_frame = None

    def tracking_count(self):
        return len(self._tracked_objects)

    def add_tracked_object(self, frame, bbox):
        new_tracker = TrackerFactory.create(self._tracker_type)
        new_tracked_object = TrackedObject(new_tracker, bbox)
        self._tracked_objects.append(new_tracked_object)

        new_thread = threading.Thread(
            target=self.track_loop, args=(new_tracked_object,)
        )
        new_tracker.start(frame, bbox)
        new_thread.start()

    def remove_tracked_object(self, index):
        self._tracked_objects.pop(index)

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

        print("Track loop ended")

    def listen_keyboard_input(self, frame):
        key = cv2.waitKey(30) & 0xFF
        if key == ord("s"):
            # Select object to track manually
            selected_roi = cv2.selectROI(
                "Frame", frame, fromCenter=False, showCrosshair=True
            )
            self.add_tracked_object(frame, selected_roi)
        elif key == ord("x"):
            # Remove last tracked object
            if len(self._tracked_objects) > 0:
                self._tracked_objects.pop()
        elif key == 27 or key == ord("q"):
            # Stop if escape key is pressed
            return True

    def start(self):
        cap = VideoSource(0)
        fps = FPS()

        # Main display loop
        while True:
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
                    # TODO (Anthony): choose a better criterion for selecting
                    # bbox (in case there are more than one)
                    self.add_tracked_object(frame, predicted_bboxes[0])
                    cv2.imshow("initial bbox", face_frame)

            # Draw bboxes from tracked objects
            for tracked_object in self._tracked_objects:
                if tracked_object.bbox is None:
                    continue
                x, y, w, h = tracked_object.bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Update display
            fps.setFps()
            frame = fps.writeFpsToFrame(frame)
            cv2.imshow("Tracking", frame)

            # Keyboard input controls
            terminate = self.listen_keyboard_input(frame)
            if terminate:
                break

        cap.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    tracking_manager = TrackingManager(tracker_type="kcf")
    tracking_manager.start()
