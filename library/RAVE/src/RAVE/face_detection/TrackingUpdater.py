import time
import torch
from collections import defaultdict

from .face_detectors import DetectorFactory
from .face_verifiers import VerifierFactory
from ..common.image_utils import intersection
from .verifiers.Encoding import Encoding


class TrackingUpdater:
    """
    Uses the verifier and detector to detect and re-assign faces during the
    tracking process
    Contains loop that performs periodic updates

    Args:
        detector_type (str): The type of detector to use
        verifier_type (str) The type of verifier to use
        object_manager (TrackedObjectManager):
            Reference to the object that handles the tracked objects
        frequency (float): Frequency at which the detector is called
        intersection_threshold (float):
            Threshold on the intersection criteria for reassigning an id to
            a bounding box
        verifier_threshold (float):
            Threshold for the verifier when comparing faces for a matches

    Attributes:
        detector (Detector): Instance used for face detection
        verifier (Verifier): Instance used for facial recognition
        object_manager (TrackedObjectManager):
            Reference to the object that handles the tracked objects
        last_frame (ndarray): Last frame acquired by the camera
        frequency (float):
            The frequency at which the detector/verifier are called
        intersection_threshold (float):
            Threshold on the intersection criteria for reassigning an id to
            a bounding box
        verifier_threshold (float):
            Threshold for the verifier when comparing faces for a matches
        pre_process_frame (ndarray):
            Most recent annotated frame from the pre-process phase
        detector_frame (ndarray):
            Most recent annotated frame from the detector
        last_detect (float): The time of the last detection.
        last_frame (ndarray):
            Last capture frame from stream. Is updated by TrackingManager
        is_alive (bool): whether the loops should be running or not
    """

    def __init__(
        self,
        detector_type,
        verifier_type,
        object_manager,
        frequency,
        intersection_threshold,
        verifier_threshold,
    ):

        self.detector = DetectorFactory.create(detector_type)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.verifier = VerifierFactory.create(
            verifier_type, threshold=verifier_threshold, device=device
        )

        self.object_manager = object_manager
        self.frequency = frequency
        self.intersection_threshold = intersection_threshold
        self.verifier_threshold = verifier_threshold

        self.pre_process_frame = None
        self.detector_frame = None

        self.last_detect = 0
        self.last_frame = None  # Updated by TrackingManager
        self.is_alive = True

    def match_faces_by_iou(self, objects, detections):
        """
        Performs association between trackable objects and bounding boxes from
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
                and intersection_scores[max_id] > self.intersection_threshold
            ):
                # A face was matched
                matched_pairs.append((objects[max_id], detection))
                unmatched_objects.pop(max_id)
            else:
                # A new object was discovered
                unmatched_detections.append(detection)

        return matched_pairs, unmatched_objects, unmatched_detections

    def associate_faces(self, frame_object, detections):
        """
        Associate predictions to objects currently being tracked

        Args:
            frame (ndarray): current frame with shape HxWx3
            detections (list of Detection): each Detection contains a bounding
                box is in the x,y,w,h format and a mouth landmark point in
                the x,y format
        """

        # TODO: could include feature vector in face matching... (?)
        all_tracked_objects = self.object_manager.get_all_objects()
        (
            matched_pairs,
            unmatched_objects,
            unmatched_detections,
        ) = self.match_faces_by_iou(all_tracked_objects, detections)

        tracked_objects = self.object_manager.tracked_objects

        # Handle matches
        for pair in matched_pairs:
            # A face was matched
            obj, detection = pair
            if obj.id in tracked_objects.keys():
                # Do not reset pre-tracked objects
                matched_object = tracked_objects[obj.id]
                matched_object.reset(
                    frame_object, detection.bbox, detection.mouth
                )
                matched_object.increment_evaluation_frames()

                # Also extract new feature vector
                # TODO: Could set verifier on its own update freq
                #  (larger than detector update)
                feature = self.verifier.get_features(
                    frame_object.frame, [detection.bbox]
                )[0]

                # TODO: Verify that the new face actually matches (verifier)
                matched_object.update_encoding(
                    feature, frame_object.frame, detection.bbox
                )

        # Handle unmatched detections
        for new_detection in unmatched_detections:
            # A new object was discovered
            self.object_manager.add_pre_tracked_object(
                frame_object, new_detection.bbox, new_detection.mouth
            )

        # Reject unmatched tracked objects
        for obj_id, obj in unmatched_objects.items():
            if obj.id in tracked_objects.keys():
                tracked_object = tracked_objects[obj.id]
                tracked_object.reject()
                if tracked_object.rejected:
                    self.object_manager.remove_tracked_object(obj.id)
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
        match_index, match_score = self.verifier.get_closest_face(
            reference_encodings, encoding_to_compare
        )

        if match_index is not None:
            print(f"Matched old face with score: {match_score}")
            return objects[match_index]

        return None

    def preprocess_faces(self, frame_object):
        """
        Associate predictions to objects currently being pre-tracked to confirm
        that they are faces.

        Args:
            frame (ndarray): current frame with shape HxWx3
        """
        pre_tracked_objects = self.object_manager.pre_tracked_objects
        if len(pre_tracked_objects) == 0:
            return

        annotated_frame, detections = self.detector.predict(
            frame_object.frame.copy(), draw_on_frame=True
        )

        pre_tracked_objects_copy = pre_tracked_objects.copy()
        matched_pairs, unmatched_objects, _ = self.match_faces_by_iou(
            pre_tracked_objects_copy, detections
        )

        # Handle successful re-detections
        for pair in matched_pairs:
            pre_tracked_object, detection = pair

            # TODO: Maybe throttle/control when to call verifier
            # Compute encoding for detection
            feature = self.verifier.get_features(
                frame_object.frame, [detection.bbox]
            )[0]

            # Verify detection with past detections
            if not pre_tracked_object.encoding.is_empty:
                similarity_score = self.verifier.get_scores(
                    [pre_tracked_object.encoding],
                    Encoding(feature),  # TODO: param has to be encoding?
                )[0]

                if similarity_score >= self.verifier_threshold:
                    # Appearance matched last detection
                    pre_tracked_object.update_encoding(
                        feature, frame_object.frame, detection.bbox
                    )
                    pre_tracked_object.confirm()
                else:
                    pre_tracked_object.increment_evaluation_frames()
            else:
                # Skip verifier compare on first detection
                pre_tracked_object.update_encoding(
                    feature, frame_object.frame, detection.bbox
                )
                pre_tracked_object.confirm()

        # Handle unmatched objects
        for obj_id, obj in unmatched_objects.items():
            obj.increment_evaluation_frames()

        # TODO: Could possibly split this function here ----------------

        # Perform operations on all pre-tracked objects
        finished_trackers_id = set()
        pre_tracker_frame = frame_object.frame.copy()
        rejected_objects = self.object_manager.rejected_objects
        tracked_objects = self.object_manager.tracked_objects
        for pre_tracker_id, pre_tracked_object in pre_tracked_objects.items():
            if pre_tracked_object.confirmed:
                matched_object = None

                # Check if this object doesn't match a face that is currently
                # being tracked, but that was not caught by IOU and is
                # mistaken for a new face
                tracked_objects_list = list(tracked_objects.values())
                if any(tracked_objects_list):
                    matched_object = self.compare_encoding_to_objects(
                        tracked_objects_list, pre_tracked_object.encoding
                    )
                    if matched_object:
                        # Recover object lost by tracker
                        # Get detection associated
                        detection = None

                        for pair in matched_pairs:
                            object_in_pair, detection_in_pair = pair
                            if object_in_pair == pre_tracked_object:
                                detection = detection_in_pair
                                break

                        if detection:
                            print(
                                "Recovering face that was lost by tracker:",
                                matched_object.id,
                            )
                            matched_object.reset(
                                frame_object, detection.bbox, detection.mouth
                            )
                        else:
                            # We wrongly assumed that the  confirmed object
                            # was in matched_pairs...
                            print("Couldn't find detection to recover face")

                # Check if this object matches an old face
                if matched_object is None:
                    rejected_objects_list = list(rejected_objects.values())
                    if any(rejected_objects_list):
                        matched_object = self.compare_encoding_to_objects(
                            rejected_objects_list, pre_tracked_object.encoding
                        )
                        if matched_object:
                            # Matched with old face: start tracking again
                            print("Matched with old face:", matched_object.id)
                            self.object_manager.restore_rejected_object(
                                matched_object.id, pre_tracked_object
                            )

                if matched_object is None:
                    # Passing object from pre-tracked to tracked
                    new_tracked_id = self.object_manager.new_identifier()
                    pre_tracked_object.update_id(new_tracked_id)
                    self.object_manager.tracked_objects[
                        new_tracked_id
                    ] = pre_tracked_object

            if not pre_tracked_object.pending:
                finished_trackers_id.add(pre_tracker_id)

            if pre_tracked_object.bbox is None:
                continue

            pre_tracker_frame = pre_tracked_object.draw_prediction_on_frame(
                pre_tracker_frame
            )

        for pre_id in finished_trackers_id:
            self.object_manager.remove_pre_tracked_object(pre_id)

        self.pre_process_frame = pre_tracker_frame
        return annotated_frame, detections

    def detector_update(self, frame_object, pre_frame, pre_detections):
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
            face_frame, detections = self.detector.predict(
                frame_object.frame.copy(), draw_on_frame=True
            )
        self.last_detect = time.time()
        self.detector_frame = face_frame

        self.associate_faces(frame_object, detections)

    def update_loop(self):
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

        while self.is_alive:
            frame_object = self.last_frame

            if frame_object is None:
                print("No frame received, exiting")
                break

            # Do pre-processing of faces
            pre_frame, pre_detections = None, None
            pre_tracked_objects = self.object_manager.pre_tracked_objects
            if pre_tracked_objects:
                pre_frame, pre_detections = self.preprocess_faces(frame_object)
            if time.time() - self.last_detect >= self.frequency:
                self.detector_update(frame_object, pre_frame, pre_detections)

            time.sleep(0.05)
