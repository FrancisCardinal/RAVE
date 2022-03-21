import numpy as np
import face_recognition

from .Verifier import Verifier
from .Encoding import Encoding


class DlibFaceRecognition(Verifier):
    """
    Wrapper for dlib's face_recognition module

    Attributes:
        score_threshold (float): Minimum similarity threshold to make a match
    """

    def __init__(self, score_threshold):
        self.score_threshold = score_threshold

    def get_encodings(self, frame, face_locations):
        """
        Get the encodings (feature vectors) for all request objects

        Args:
            frame (np.ndarray): The image containing the objects
            face_locations (list of tuples (int, int, int, int)): List of all
                the xywh bounding boxes for the objects in the image to compute
                the encodings for

        Returns:
            (list(float)): Variable length list containing the floats that
                represent the object
        """

        rgb_frame = frame[:, :, ::-1]

        # Convert xywh bboxes to (y0, x1, y1, x0)
        converted_face_locations = []
        for bbox in face_locations:
            x, y, w, h = bbox
            bbox_converted = (y, x + w, y + h, x)
            converted_face_locations.append(bbox_converted)

        face_feature_vectors = face_recognition.face_encodings(
            rgb_frame, converted_face_locations, num_jitters=1, model="small"
        )

        face_encodings = Encoding.create_encodings(face_feature_vectors)

        return face_encodings

    @staticmethod
    def get_scores(reference_encodings, face_encoding):
        """
        Get the similarity scores for each reference encoding compared to the
        supplied encoding

        Args:
            reference_encodings (list of Encodings): to compare with
            face_encoding (Encoding): encoding to compare against other
                encodings

        Returns:
            (list(float)): List of same length as reference_encodings
                indicating the similarity score between each reference encoding
                and supplied encoding
        """

        reference_features = [
            encoding.feature for encoding in reference_encodings
        ]
        target_feature = face_encoding.feature

        distances = face_recognition.face_distance(
            reference_features, target_feature
        )
        scores = [1 - distance for distance in distances]
        return scores

    def get_closest_face(self, reference_encodings, face_encoding):
        """
        Finds the most similar face in a set of faces compared with a supplied
        face encoding that also respects the set threshold

        Args:
            reference_encodings (list of Encodings): to compare with
            face_encoding (encoding): encoding to compare against other
                encodings

        Returns:
            (int, float): Returns the index of the encoding that best matched
                the supplied target encoding and the corresponding similarity
                score. Or (None, None) if no match was possible
        """

        face_scores = self.get_scores(reference_encodings, face_encoding)
        best_match_index = np.argmax(face_scores)
        best_score = face_scores[best_match_index]

        if best_score >= self.score_threshold:
            return best_match_index, best_score
        else:
            return None, None
