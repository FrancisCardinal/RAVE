import time
import cv2
import threading
import torch
import numpy as np

from collections import defaultdict

# from tqdm import tqdm

from .face_detectors import DetectorFactory
from .face_verifiers import VerifierFactory, Encoding
from ..common.image_utils import intersection
from .fpsHelper import FPS
from pyodas.visualize import VideoSource, Monitor
from .TrackedObject import TrackedObject


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
        visualize (bool): If true, will show the different tracking frames

    Attributes:
        count (int): for assigning ids.
        last_frame (ndarray): Last frame acquired by the camera
        tracked_objects (dict of TrackedObject):
            dictionary of all faces currently being tracked having been
            confirmed
        _tracker_type (str): type of tracker
        _pre_tracked_objects (dict of TrackedObject):
            dictionary of all faces currently being in the process of being
            confirmed
        _rejected_objects (dict of TrackedObject):
            dictionary of all faces rejected by post-process
        _frequency (float):
            The frequency at which the detector/verifier are called
        _intersection_threshold (float):
            Threshold for reassigning an id to a new bbox
        _detector (Detector): Network doing the detection job
        _last_detect (float): The time of the last detection.
        _is_alive (bool): whether the loops should be running or not
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
        self._tracker_type = tracker_type

        self.tracked_objects = {}
        self._pre_tracked_objects = {}
        self._rejected_objects = {}

        self._frequency = frequency
        self._intersection_threshold = intersection_threshold
        self.count = 0

        self._detector = DetectorFactory.create(detector_type)
        self.last_frame = None
        self._last_detect = 0
        self._is_alive = False
        self._visualize = visualize

        self._pre_process_frame = None
        self._detector_frame = None

        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        self._verifier_threshold = verifier_threshold
        self._verifier = VerifierFactory.create(
            verifier_type, threshold=verifier_threshold, device=device
        )

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
        return {**self.tracked_objects, **self._pre_tracked_objects}

    # Assumed to be called from main thread only
    def new_identifier(self):
        """
        Used to assign a new id to a new tracking object
        Returns:
            int:
                A new id
        """
        new_id = str(self.count)
        self.count += 1
        return new_id

    # Register a new object to tracker. Assumed to be called from main thread
    def add_tracked_object(self, frame, bbox, mouth):
        """
        Creates a new TrackedObject for the new bbox and adds it to the
        pre-tracked list. It also starts the tracking thread.

        Args:
            frame (ndarray):
                current frame with shape HxWx3
            bbox (list):
                in format x,y,w,h (superior left corner)
            mouth (list):
                The position of the mouth in x,y
        """
        new_id = self.new_identifier()
        new_tracked_object = TrackedObject(
            self._tracker_type, frame, bbox, mouth, new_id
        )
        self._pre_tracked_objects[new_tracked_object.id] = new_tracked_object
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
        self._rejected_objects[identifier] = rejected_object

    def restore_rejected_object(self, identifier, pre_tracked_object):
        """
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
        restored_object = self._rejected_objects.pop(identifier)
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
        self._pre_tracked_objects.pop(identifier)

    def stop_tracking(self):
        """
        Remove all items from the tracked_objects dictionary
        """
        self.tracked_objects = {}
        self._pre_tracked_objects = {}
        self._rejected_objects = {}

    def track_loop(self, tracked_object):
        """
        Thread worker for calling the tracker on the TrackedObject

        Args:
            tracked_object (TrackedObject): The object to track
        """
        # with tqdm(desc=f"{tracked_object.id}", total=25000) as pbar:
        while (
            tracked_object in self.tracked_objects.values()
            or tracked_object in self._pre_tracked_objects.values()
        ):
            frame = self.last_frame

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

            time.sleep(0.05)

        print(f"Stopped tracking object {tracked_object.id}")

    def listen_keyboard_input(self, frame, key_pressed):
        """
        Opencv keyboard input handler for defining object to track manually
        or exiting the program.

        q or escape exits
        x removes the last tracked object
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
            self.add_tracked_object(frame, selected_roi, None)
        elif key == ord("x"):
            # Remove last tracked object
            if len(self.tracked_objects) > 0:
                self.tracked_objects.popitem()
        elif key == ord("f"):
            # Show memory of faces for each tracked object
            tracked_objects = list(self.tracked_objects.values())
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

    def draw_prediction_on_frame(self, frame, tracked_object):
        """
        Draw the tracker prediction on the frame

        Args:
            frame (ndarray): current frame with shape HxWx3
            tracked_object (TrackedObject): tracked object to be drawn
        """
        x, y, w, h = tracked_object.bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            tracked_object.id,
            (x, y - 2),
            0,
            1,
            [0, 0, 255],
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        return frame

    def draw_all_predictions_on_frame(self, tracking_frame):
        """
        Draw the prediction for each tracked object on a frame

        Args:
            frame (ndarray): current frame with shape HxWx3
        """

        all_tracked_objects = list(self.tracked_objects.values())
        for i in range(len(all_tracked_objects)):
            if len(all_tracked_objects) <= i:
                break

            tracked_object = all_tracked_objects[i]
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

        return tracking_frame

    def match_faces_by_iou(self, objects, detections):
        """
        Performs association between trackable objects and bouding boxes from
        detection by analyzing the intersection overlap and applying a
        threshold

        Args:
            objects (dict of id(str):objects): dict of objects to match
            detections (list of Detection): the detections containing the xywh
                bboxes to match

        Returns:
            matched_pairs (list of tuples: (object, detection)): contains the
                matched pairs between objects and detections
            unmatched_objects (dict of id(str):objects): dict of all objects
                that were not matched with predictions
            unmatched_detections (list of detections): list of detections that
                were not matched with an object
        """

        matched_pairs = []
        unmatched_objects = objects.copy()
        unmatched_detections = []
        for i, detection in enumerate(detections):
            intersection_scores = defaultdict(lambda: 0)

            for tracker_id, trackable_object in unmatched_objects.items():
                intersection_scores[tracker_id] = intersection(
                    detection.bbox, trackable_object.bbox
                )

            max_id = None
            if intersection_scores:
                max_id = max(intersection_scores, key=intersection_scores.get)

            if (
                intersection_scores
                and intersection_scores[max_id] > self._intersection_threshold
            ):
                # A face was matched
                matched_pairs.append((objects[max_id], detection))
                unmatched_objects.pop(max_id)
            else:
                # A new object was discovered
                unmatched_detections.append(detection)

        return matched_pairs, unmatched_objects, unmatched_detections

    def associate_faces(self, frame, detections):
        """
        Associate predictions to objects currently being tracked

        Args:
            frame (ndarray): current frame with shape HxWx3
            detections (list of Detection): each Detection contains a bounding
                box is in the x,y,w,h format and a mouth landmark point in
                the x,y format
        """

        # TODO: could include feature vector in face matching... (?)
        all_tracked_objects = self.get_all_objects()
        (
            matched_pairs,
            unmatched_objects,
            unmatched_detections,
        ) = self.match_faces_by_iou(all_tracked_objects, detections)

        # Handle matches
        for pair in matched_pairs:
            # A face was matched
            obj, detection = pair
            if obj.id in self.tracked_objects.keys():
                # Do not reset pre-tracked objects
                matched_object = self.tracked_objects[obj.id]
                matched_object.reset(frame, detection.bbox, detection.mouth)
                matched_object.increment_evaluation_frames()

                # Also extract new feature vector
                feature = self._verifier.get_features(frame, [detection.bbox])[
                    0
                ]

                # TODO: Verify that the new face actually matches (verifier)
                matched_object.update_encoding(feature, frame, detection.bbox)

        # Handle unmatched detections
        for new_detection in unmatched_detections:
            # A new object was discovered
            self.add_tracked_object(
                frame, new_detection.bbox, new_detection.mouth
            )

        # Reject unmatched tracked objects
        for obj_id, obj in unmatched_objects.items():
            if obj.id in self.tracked_objects.keys():
                tracked_object = self.tracked_objects[obj.id]
                tracked_object.reject()
                if tracked_object.rejected:
                    self.remove_tracked_object(obj.id)
                    print("Rejecting tracked object:", obj.id)

    def compare_encoding_to_objects(self, objects, encoding_to_compare):
        """
        WIP: will be modified
        Args:
            ...
        Returns:
            ...
        """
        reference_encodings = [obj.encoding for obj in objects]
        match_index, match_score = self._verifier.get_closest_face(
            reference_encodings, encoding_to_compare
        )

        if match_index is not None:
            print(f"Matched old face with score: {match_score}")
            return objects[match_index]

        return None

    def preprocess_faces(self, frame):
        """
        Associate predictions to objects currently being pre-tracked to confirm
        that they are faces.

        Args:
            frame (ndarray): current frame with shape HxWx3
        """

        if len(self._pre_tracked_objects) == 0:
            return

        annotated_frame, detections = self._detector.predict(
            frame.copy(), draw_on_frame=True
        )

        pre_tracked_objects = self._pre_tracked_objects.copy()
        matched_pairs, unmatched_objects, _ = self.match_faces_by_iou(
            pre_tracked_objects, detections
        )

        # Handle successful re-detections
        for pair in matched_pairs:
            pre_tracked_object, detection = pair

            # TODO: Maybe throttle/control when to call verifier
            # Compute encoding for detection
            feature = self._verifier.get_features(frame, [detection.bbox])[0]

            # Verify detection with past detections
            if not pre_tracked_object.encoding.is_empty:
                similarity_score = self._verifier.get_scores(
                    [pre_tracked_object.encoding],
                    Encoding(feature),  # TODO: param has to be encoding?
                )[0]

                if similarity_score >= self._verifier_threshold:
                    # Appearance matched last detection
                    # print("Encoding matched last encoding")
                    pre_tracked_object.update_encoding(
                        feature, frame, detection.bbox
                    )
                    pre_tracked_object.confirm()
                else:
                    pre_tracked_object.increment_evaluation_frames()
                    # print("Encoding did not match last")
            else:
                # Skip verifier compare on first detection
                pre_tracked_object.update_encoding(
                    feature, frame, detection.bbox
                )
                pre_tracked_object.confirm()

        # Handle unmatched objects
        for obj_id, obj in unmatched_objects.items():
            obj.increment_evaluation_frames()

        # Perform operations on all pre-tracked objects
        finished_trackers_id = set()
        pre_tracker_frame = frame.copy()
        for tracker_id, tracked_object in self._pre_tracked_objects.items():
            if tracked_object.confirmed:
                # Check if this object matches an old face
                restored_object = False
                rejected_objects = list(self._rejected_objects.values())
                if any(rejected_objects):
                    matched_object = self.compare_encoding_to_objects(
                        rejected_objects, tracked_object.encoding
                    )
                    if matched_object:
                        # Matched with old face: start tracking again
                        self.restore_rejected_object(
                            matched_object.id, tracked_object
                        )
                        restored_object = True

                if not restored_object:
                    self.tracked_objects[tracker_id] = tracked_object

            if not tracked_object.pending:
                finished_trackers_id.add(tracker_id)
                print("Adding finished tracker:", tracker_id)

            if tracked_object.bbox is None:
                continue

            pre_tracker_frame = self.draw_prediction_on_frame(
                pre_tracker_frame, tracked_object
            )

        for id in finished_trackers_id:
            self.remove_pre_tracked_object(id)

        self._pre_process_frame = pre_tracker_frame
        return annotated_frame, detections

    def detector_update(self, frame, pre_frame, pre_detections):
        """
        Obtain the detection predictions, unless the pre-process already
        called the detection then use those

        Args:
            frame (ndarray): current frame with shape HxWx3
            pre_frame (ndarray or None): annotated frame from pre-process step
            pre_detections (list(Detection)): Detections from the pre-process
        """
        if pre_detections is not None:
            face_frame = pre_frame
            detections = pre_detections
        else:
            face_frame, detections = self._detector.predict(
                frame.copy(), draw_on_frame=True
            )
        self._last_detect = time.time()
        self._detector_frame = face_frame

        self.associate_faces(frame, detections)

    def tracking_loop(self):
        """
        Tracking algorithm with pre and post process for confirming and
        rejecting bboxes.

        Args:
            monitor (Monitor):
                pyodas monitor to display the frame
            fps (FPS): To obtain the frames per second
        """

        # Wait for first frame to be available
        while self.last_frame is None:
            time.sleep(0.1)

        while self._is_alive:
            frame = self.last_frame

            if frame is None:
                print("No frame received, exiting")
                break

            # Do pre-processing of faces
            pre_frame, pre_detections = None, None
            if self._pre_tracked_objects:
                pre_frame, pre_detections = self.preprocess_faces(frame)
            if time.time() - self._last_detect >= self._frequency:
                self.detector_update(frame, pre_frame, pre_detections)

            time.sleep(0.05)

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

        while self._is_alive:
            self.last_frame = cap()

            if monitor is not None:
                # Draw detections from tracked objects
                tracking_frame = self.last_frame.copy()
                self.draw_all_predictions_on_frame(tracking_frame)

                # fps.setFps()
                # fps.writeFpsToFrame(tracking_frame)

                monitor.update("Tracking", tracking_frame)

                # Draw most recent pre-processing frame
                if self._pre_process_frame is not None:
                    monitor.update("Pre-process", self._pre_process_frame)
                    self._pre_process_frame = None

                # Draw most recent detector frame
                if self._detector_frame is not None:
                    monitor.update("Detection", self._detector_frame)
                    self._detector_frame = None

                # Keyboard input controls
                terminate = self.listen_keyboard_input(
                    self.last_frame, monitor.key_pressed
                )
                if terminate or not monitor.window_is_alive():
                    self._is_alive = False
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
        cap = VideoSource(args.video_source, args.width, args.height)
        shape = (
            (cap.shape[1], cap.shape[0])
            if args.flip_display_dim
            else cap.shape
        )
        self._is_alive = True
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

        # Image capture loop
        tracking_loop = threading.Thread(
            target=self.tracking_loop, daemon=True
        )
        tracking_loop.start()

        # Main display loop
        self.main_loop(monitor, cap, fps)
        self.stop_tracking()
