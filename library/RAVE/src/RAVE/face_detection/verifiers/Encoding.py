import numpy as np


class Encoding:
    """
    Class used to contain feature vectors for a given face

    Attributes:
        feature (list of floats): the main feature vector of the encoding
    """

    MEMORY = 5

    def __init__(self, feature=None):
        self.feature = feature
        self.all_features = []
        self.all_faces = []

        if feature is not None:
            self.all_features.append(feature)

    @property
    def is_empty(self):
        """
        Check whether this encoding contains any feature vectors
        """
        return len(self.all_features) == 0

    @property
    def get_average(self):
        """
        Compute the average feature vector
        """
        if len(self.all_features) == 0:
            return None

        self.refresh()
        return list(np.mean(self.all_features, axis=0))

    def refresh(self):
        """
        Refresh memory to only hold recent data
        """
        count = Encoding.MEMORY
        self.all_features = self.all_features[-count:]
        self.all_faces = self.all_faces[-count:]

    def update(self, feature, face_image=None):
        """
        Update feature vector WIP
        """
        self.feature = feature
        self.all_features.append(feature)
        self.all_faces.append(face_image)
        self.refresh()

    def restore(self, pre_tracked_object):
        """
        Update features base on pre_tracked object that was associated with
        the object containing this encoding

        ...
        """
        pre_tracked_encoding = pre_tracked_object.encoding
        self.all_features.extend(pre_tracked_encoding.all_features)
        self.all_faces.extend(pre_tracked_encoding.all_faces)
        self.refresh()

    def get_last_feature(self):
        """
        Returns the last recorded feature vector
        """
        if not self.all_features:
            return None

        return self.all_features[-1]

    @staticmethod
    def create_encodings(feature_vectors):
        """
        Create encoding instances from list of feature vectors
        """
        all_encodings = []
        for feature in feature_vectors:
            new_encoding = Encoding(feature)
            all_encodings.append(new_encoding)

        return all_encodings
