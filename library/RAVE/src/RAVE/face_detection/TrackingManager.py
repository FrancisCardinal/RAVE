import cv2
import threading
import numpy as np
import torch
import time

from .TrackingUpdater import TrackingUpdater
from .TrackedObjectManager import TrackedObjectManager

from pyodas.visualize import Monitor
from RAVE.common.image_utils import opencv_image_to_tensor


class FrameObject:
    """
    Container for frames
    Associate a frame with an id
    """

    def __init__(self, frame, id, device):
        self.frame = frame
        self.frame_tensor = opencv_image_to_tensor(frame.copy(), device)
        self.id = id


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
        cap,
        tracker_type,
        detector_type,
        verifier_type,
        frequency,
        intersection_threshold=-0.25,
        verifier_threshold=0.25,
        visualize=True,
        debug_preprocess=False,
        debug_detector=False,
        tracking_or_calib=lambda: True,
    ):
        self.object_manager = TrackedObjectManager(tracker_type)
        self.updater = TrackingUpdater(
            detector_type,
            verifier_type,
            self.object_manager,
            frequency,
            intersection_threshold,
            verifier_threshold,
            visualize=visualize,
            debug_preprocess=debug_preprocess,
            debug_detector=debug_detector,
        )

        self._cap = cap
        self.is_alive = False
        self.frame_count = 0

        self._visualize = visualize
        self._debug_preprocess = debug_preprocess
        self._debug_detector = debug_detector
        self._tracking_or_calib = tracking_or_calib
        self._device = "cpu"
        if torch.cuda.is_available():
            self._device = "cuda"

        K = np.array([[347.33973783, 0.0, 324.87219961], [0.0, 345.76160498, 243.98890458], [0.0, 0.0, 1.0]])
        D = np.array([[-3.12182472e-01, 9.71248263e-02, -4.70897871e-05, 1.60933679e-04, -1.31438715e-02]])

        corrected_shape = (self._cap.shape[1], self._cap.shape[0])
        self.newcameramtx, self.roi = cv2.getOptimalNewCameraMatrix(K, D, corrected_shape, 1, corrected_shape)

        self.drawing_callbacks = list()

    def kill_threads(self):
        """
        Call to kill loops in all threads
        """
        self.is_alive = False
        self.updater.is_alive = False

    def listen_keyboard_input(self, frame_object, key_pressed):
        """
        Opencv keyboard input handler for defining object to track manually
        or exiting the program.

        q or escape exits
        x removes the last tracked object
        f shows history of faces captured for each tracked object

        Args:
            frame (ndarray): current frame with shape HxWx3
            key_pressed (int): key pressed on the opencv window
        """
        key = key_pressed & 0xFF
        if key == ord("x"):
            # Remove last tracked object
            if len(self.object_manager.tracked_objects) > 0:
                self.object_manager.tracked_objects.popitem()
        elif key == ord("f"):
            # Show memory of faces for each tracked object
            tracked_objects = list(self.object_manager.tracked_objects.values())
            if len(tracked_objects) == 0:
                return
            slot_count = max([len(obj.encoding.all_faces) for obj in tracked_objects])
            output = []
            for obj in tracked_objects:
                face_images = []
                for i in range(slot_count):
                    if len(obj.encoding.all_faces) > i:
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

        all_tracked_objects = list(self.object_manager.tracked_objects.values())
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
                cv2.circle(tracking_frame, (x_mouth, y_mouth), 5, [0, 0, 255], -1)

        return tracking_frame

    def process_frame(self, frame):
        if self.flip_frame:
            frame = cv2.flip(frame, 0)

        return frame

    def main_loop(self, monitor):
        """
        Loop to be called on separate thread that handles retrieving new image
        frames from video input and displaying output in windows

        Args:
            monitor (Monitor):
                pyodas monitor to display the frame
        """

        while self.is_alive:
            if self._tracking_or_calib():
                # Capture image and pass to classes that need it
                frame = self._cap()
                frame = self.process_frame(frame)

                frame_object = FrameObject(frame, self.frame_count, self._device)
                self.frame_count += 1

                self.updater.last_frame = frame_object
                self.object_manager.on_new_frame(frame_object)

                if self._visualize:
                    # Draw detections from tracked objects
                    tracking_frame = frame.copy()
                    self.draw_all_predictions_on_frame(tracking_frame)

                    for callback in self.drawing_callbacks:
                        callback(tracking_frame)

                    monitor.update("Tracking", tracking_frame)

                    if self._debug_preprocess:
                        # Draw most recent pre-processing frame
                        pre_process_frame = self.updater.pre_process_frame
                        if pre_process_frame is not None:
                            monitor.update("Pre-process", pre_process_frame)
                            self.updater.pre_process_frame = None

                    if self._debug_detector:
                        # Draw most recent detector frame
                        detector_frame = self.updater.detector_frame
                        if detector_frame is not None:
                            monitor.update("Detection", detector_frame)
                            self.updater.detector_frame = None

                    # Keyboard input controls
                    terminate = self.listen_keyboard_input(frame_object, monitor.key_pressed)
                    if terminate or not monitor.window_is_alive():
                        self.kill_threads()
                        break
            else:
                time.sleep(50)

    def start(self, args):
        """
        Start tracking faces in a real-time video feed
        Args:
            args:
                Arguments from argument parser, see main_tracking for more
                information
        """

        shape = self._cap.shape if args.flip_display_dim else (self._cap.shape[1], self._cap.shape[0])
        self.is_alive = True
        self.flip_frame = args.flip

        if self._visualize:
            debug_args = ["Tracking", shape]
            if self._debug_detector:
                debug_args.extend(["Detection", shape])
            if self._debug_preprocess:
                debug_args.extend(["Pre-process", shape])

            monitor = Monitor(*debug_args, refresh_rate=30)
        else:
            monitor = None

        # Start update loop
        update_loop = threading.Thread(target=self.updater.update_loop, name="Update loop", daemon=True)
        update_loop.start()

        # Start capture & display loop
        self.main_loop(monitor)
        self.stop()

    def stop(self):
        """
        Stop tracking loop and all related threads
        """
        self.kill_threads()
        self.object_manager.stop_tracking()
