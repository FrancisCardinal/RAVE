import time
import cv2
import threading
import numpy as np

from .TrackingUpdater import TrackingUpdater
from .TrackedObjectManager import TrackedObjectManager

from .fpsHelper import FPS
from pyodas.visualize import VideoSource, Monitor


class TrackingManager:
    """
    Used to coordinate the detector, the trackers and the verifier together.

    Args:
        tracker_type (str): The type of tracker to use
        detector_type (str): The type of detector to use
        verifier_type (str) The type of verifier to use
        frequency (float): Frequency at which the detector is called
        intersection_threshold (float):
            Threshold on the intersection criteria for reassigning an id to
            a bounding box
        verifier_threshold (float):
            Threshold for the verifier when comparing faces for a matches
        visualize (bool): If true, will show the different tracking frames

    Attributes:
        object_manager (TrackedObjectManager):
            Instance of class that handles the tracked objects
        updater (TrackingUpdater):
            Instance of the class that handles detector updates, face
            re-identification and association
        is_alive (bool): whether the loops should be running or not
        _visualize (bool): If true, will show the different tracking frames
    """

    def __init__(
        self,
        tracker_type,
        detector_type,
        verifier_type,
        frequency,
        intersection_threshold=0.2,
        verifier_threshold=0.25,
        visualize=True,
    ):
        self.object_manager = TrackedObjectManager(tracker_type)
        self.updater = TrackingUpdater(
            detector_type,
            verifier_type,
            self.object_manager,
            frequency,
            intersection_threshold,
            verifier_threshold,
        )

        self.is_alive = False
        self._visualize = visualize

    def kill_threads(self):
        """
        Call to kill loops in all threads
        """
        self.is_alive = False
        self.updater.is_alive = False

    def listen_keyboard_input(self, frame, key_pressed):
        """
        Opencv keyboard input handler for defining object to track manually
        or exiting the program.

        q or escape exits
        x removes the last tracked object
        f shows history of faces captured for each tracked object
        s lets you defined an object to track

        Args:
            frame (ndarray): current frame with shape HxWx3
            key_pressed (int): key pressed on the opencv window
        """
        key = key_pressed & 0xFF
        if key == ord("s"):
            # Select object to track manually
            selected_roi = cv2.selectROI(
                "Frame", frame, fromCenter=False, showCrosshair=True
            )
            self.object_manager.add_tracked_object(frame, selected_roi, None)
        elif key == ord("x"):
            # Remove last tracked object
            if len(self.object_manager.tracked_objects) > 0:
                self.object_manager.tracked_objects.popitem()
        elif key == ord("f"):
            # Show memory of faces for each tracked object
            tracked_objects = list(
                self.object_manager.tracked_objects.values()
            )
            if len(tracked_objects) == 0:
                return
            slot_count = max(
                [len(obj.encoding.all_faces) for obj in tracked_objects]
            )
            output = []
            for obj in tracked_objects:
                face_images = []
                for i in range(slot_count):
                    if len(obj.encoding.all_faces) >= i:
                        face_image = obj.encoding.all_faces[i]
                        face_image = cv2.resize(face_image, (150, 150))
                        face_images.append(face_image)
                    else:
                        face_images.append(np.zeros((150, 150)))
                output.append(np.concatenate(face_images, axis=1))

            if output:
                output = np.concatenate(output, axis=0)
                cv2.imshow("Detected faces", output)
                cv2.waitKey(10)

        elif key == ord("q") or key == 27:
            return True

    def draw_all_predictions_on_frame(self, tracking_frame):
        """
        Draw the prediction for each tracked object on a frame

        Args:
            frame (ndarray): current frame with shape HxWx3
        """

        all_tracked_objects = list(
            self.object_manager.tracked_objects.values()
        )
        for i in range(len(all_tracked_objects)):
            if len(all_tracked_objects) <= i:
                break

            tracked_object = all_tracked_objects[i]
            if tracked_object.bbox is None:
                continue
            tracked_object.draw_prediction_on_frame(tracking_frame)

            # Draw mouth point
            mouth = tracked_object.landmark
            if mouth is not None:
                x_mouth, y_mouth = mouth
                cv2.circle(
                    tracking_frame, (x_mouth, y_mouth), 5, [0, 0, 255], -1
                )

        return tracking_frame

    def main_loop(self, monitor, cap, fps):
        """
        Loop to be called on separate thread that handles retrieving new image
        frames from video input and displaying output in windows

        Args:
            monitor (Monitor):
                pyodas monitor to display the frame
            cap (VideoSource):
                pyodas video source object to obtain video feed from camera
        """

        while self.is_alive:

            # Capture image and pass to classes that need it
            frame = cap()
            frame = cv2.flip(frame, 0)
            # frame = cv2.imread("test_image_faces.png")
            self.updater.last_frame = frame
            self.object_manager.last_frame = frame

            if monitor is not None:
                # Draw detections from tracked objects
                tracking_frame = frame.copy()
                self.draw_all_predictions_on_frame(tracking_frame)

                # fps.setFps()
                # fps.writeFpsToFrame(tracking_frame)

                monitor.update("Tracking", tracking_frame)

                # Draw most recent pre-processing frame
                pre_process_frame = self.updater.pre_process_frame
                if pre_process_frame is not None:
                    monitor.update("Pre-process", pre_process_frame)
                    self.updater.pre_process_frame = None

                # Draw most recent detector frame
                detector_frame = self.updater.detector_frame
                if detector_frame is not None:
                    monitor.update("Detection", detector_frame)
                    self.updater.detector_frame = None

                # Keyboard input controls
                terminate = self.listen_keyboard_input(
                    frame, monitor.key_pressed
                )
                if terminate or not monitor.window_is_alive():
                    self.kill_threads()
                    break

            time.sleep(0.002)

    def start(self, args):
        """
        Start tracking faces in a real-time video feed
        Args:
            args:
                Arguments from argument parser, see main_tracking for more
                information
        """
        # cap = None
        # shape = (1422, 948)
        cap = VideoSource(args.video_source, args.width, args.height)
        shape = (
            (cap.shape[1], cap.shape[0])
            if args.flip_display_dim
            else cap.shape
        )
        self.is_alive = True
        if self._visualize:
            monitor = Monitor(
                "Detection",
                shape,
                "Tracking",
                shape,
                "Pre-process",
                shape,
                refresh_rate=30,
            )
        else:
            monitor = None
        cap.set(cv2.CAP_PROP_FPS, 60)
        fps = FPS()

        # Start update loop
        update_loop = threading.Thread(
            target=self.updater.update_loop, daemon=True
        )
        update_loop.start()

        # Start capture & display loop
        self.main_loop(monitor, cap, fps)
        self.object_manager.stop_tracking()
