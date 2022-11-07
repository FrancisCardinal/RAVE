from abc import ABC, abstractmethod


class Verifier(ABC):
    """
    Abstract class for verifiers
    """

    @abstractmethod
    def get_features(self, frame, face_locations):
        """
        Get the feature vectors for all requested objects

        Args:
            frame (np.ndarray): The image containing the objects
            face_locations (list of tuples (int, int, int, int)): List of all
                the xywh bounding boxes for the objects in the image to compute
                the encodings for

        Returns:
            (list(float)): Variable length list containing the floats that
                represent the object
        """
        raise NotImplementedError()

    @abstractmethod
    def get_scores(self, reference_encodings, face_encoding):
        """
        Get the similarity scores for each reference encoding compared to the
        supplied encoding

        Args:
            reference_encodings (list of encodings): to compare with
            face_encoding (list): encoding to compare against other encodings

        Returns:
            (list(float)): List of same length as reference_encodings
                indicating the similarity score between each reference encoding
                and supplied encoding
        """
        raise NotImplementedError()

    @abstractmethod
    def get_closest_face(self, reference_encodings, face_encoding):
        """
        Finds the most similar face in a set of faces compared with a supplied
        face encoding that also respects the set threshold

        Args:
            reference_encodings (list of encodings): to compare with
            face_encoding (list): encoding to compare against other encodings

        Returns:
            (int, float): Returns the index of the encoding that best matched
                the supplied target encoding and the corresponding similarity
                score. Or (None, None) if no match was possible
        """
        raise NotImplementedError()
