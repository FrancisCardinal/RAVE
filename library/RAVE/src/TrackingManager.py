import cv2

from trackers import TrackerOpenCV  # , CorrelationTracker
from face_detectors import YoloFaceDetector
from RAVE.face_detection.fpsHelper import FPS


class TrackingManager:
    def __init__(self, tracker):
        self.tracker = tracker
        self.detector = YoloFaceDetector()
        self.initial_bbox = None

    def set_initial_bbox(self, frame, bbox):
        self.initial_bbox = bbox
        self.tracker.start(frame, self.initial_bbox)

    def start(self):
        cap = cv2.VideoCapture(0)
        fps = FPS()

        while True:
            _, frame = cap.read()

            if frame is None:
                print("No frame received, exiting")
                break

            # Detect face for initial frame
            if self.initial_bbox is None:
                face_frame, predicted_bboxes = self.detector.predict(
                    frame.copy(), draw_on_frame=True
                )
                if (
                    len(predicted_bboxes) > 0
                ):  # TODO (Anthony): choose a better criterion for selecting
                    # bbox (in case there are more than one)
                    self.set_initial_bbox(frame, predicted_bboxes[0])
                    cv2.imshow("initial bbox", face_frame)
                    # cv2.waitKey(0)

            if self.initial_bbox is not None:
                # Update tracker with new frame
                (success, box) = self.tracker.update(frame)
                if success:
                    (x, y, w, h) = [int(v) for v in box]
                    cv2.rectangle(
                        frame, (x, y), (x + w, y + h), (0, 255, 0), 2
                    )

            fps.setFps()
            frame = fps.writeFpsToFrame(frame)
            cv2.imshow("Tracking", frame)

            k = cv2.waitKey(30) & 0xFF
            if k == ord("s"):
                # Select object to track manually
                selected_roi = cv2.selectROI(
                    "Frame", frame, fromCenter=False, showCrosshair=True
                )
                self.set_initial_bbox(frame, selected_roi)
            elif k == 27:
                # Stop if escape key is pressed
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    tracker = TrackerOpenCV()
    tracking_manager = TrackingManager(tracker)
    tracking_manager.start()
