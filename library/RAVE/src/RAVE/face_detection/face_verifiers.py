import face_recognition
import numpy as np
import os
import cv2
import torch
from abc import ABC, abstractmethod

from .verifiers.models.resnet import (
    resnet_face18,
    resnet_face34,
    resnet_face50,
)
from deepface.basemodels import ArcFace as arcface_model


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


class VerifierFactory:
    """
    Static factory class used to create verifier instances of different types
    """

    @staticmethod
    def create(verifier_type="dlib", threshold=0.0, device="cpu"):
        """
        Instantiate a verifier of a given type

        Args:
            verifier_type (str): The identifier for the desired verifier
            threshold (float): If applicable, the similarity score decision
                threshold to be used
            device (str): If applicable, the device to run the model on
                ("cpu" or "cuda")

        Returns:
            (Verifier): The newly created verifier or None
        """
        if verifier_type == "dlib":
            return DlibFaceRecognition(threshold)
        elif verifier_type == "resnet_face_18":
            return ResNetVerifier(threshold, device, 18)
        elif verifier_type == "resnet_face_34":
            return ResNetVerifier(threshold, device, 34)
        elif verifier_type == "resnet_face_50":
            return ResNetVerifier(threshold, device, 50)
        elif verifier_type == "arcface":
            return ArcFace(threshold)
        else:
            print("Unknown verifier type:", verifier_type)
            return None


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


class ResNetVerifier(Verifier):
    """
    Wrapper for a ResNet model pre-trained to recognize faces
    """

    PROJECT_PATH = os.getcwd()
    MODEL_PATH = os.path.join(
        PROJECT_PATH, "RAVE", "face_detection", "verifiers", "models"
    )

    def __init__(self, score_threshold, device="cpu", architecture=18):
        self.device = device
        self.score_threshold = score_threshold
        self.model = self._load_model(architecture)

    def preprocess_image(self, image):
        """
        Convert to image to format expected by the model

        Args:
            image (np.ndarray): image to convert
        """

        if image is None:
            return None

        img = cv2.resize(image, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float()
        if self.device == "cuda":
            device = torch.cuda.current_device()
            img = img.to(device)
        img.div_(255).sub_(0.5).div_(0.5)
        return img

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

        features = []
        for bbox in face_locations:
            roi = frame[
                int(bbox[1]) : int(bbox[1] + bbox[3]),
                int(bbox[0]) : int(bbox[0] + bbox[2]),
            ]
            image = self.preprocess_image(roi)
            # TODO JKealey: is context manager necessary if model is in eval?
            with torch.no_grad():
                # TODO JKealey: possibility to keep on gpu with numpy
                feature = self.model(image).cpu().numpy()
            features.append(feature)

        face_encodings = Encoding.create_encodings(features)

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

        dist = []
        for reference_encoding in reference_encodings:
            reference_feature = reference_encoding.feature
            target_feature = face_encoding.feature

            dist.append(
                np.dot(target_feature, reference_feature)
                / (
                    np.linalg.norm(target_feature)
                    * np.linalg.norm(reference_feature)
                )
            )
        return np.array(dist)

    def get_closest_face(self, reference_encodings, face_encoding):
        """
        Finds the most similar face in a set of faces compared with a supplied
        face encoding that also respects the set threshold

        Args:
            reference_encodings (list of Encodings): to compare with
            face_encoding (Encoding): encoding to compare against other
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

    def _load_model(self, architecture):
        """
        Helper method to load the model from .pth file

        Args:
            architecture (int): Identifier for the desired size of the model
                (18, 34 or 50)

        Returns:
            (Resnet model): The model loaded from .pth file
        """

        if self.device == "cuda":

            def map_fct(storage, loc):
                """
                ...
                """
                return storage.cuda(self.device)

        else:

            def map_fct(storage, loc):
                """
                ...
                """
                return storage

        if architecture == 18:
            # Load ResNet18
            model = resnet_face18(pretrained=False)
            pretrained_dict = torch.load(
                os.path.join(ResNetVerifier.MODEL_PATH, "resnet18.pth"),
                map_location=map_fct,
            )
        elif architecture == 34:
            # Load ResNet34
            model = resnet_face34(pretrained=False)
            pretrained_dict = torch.load(
                os.path.join(ResNetVerifier.MODEL_PATH, "resnet34.pth"),
                map_location=map_fct,
            )
        elif architecture == 50:
            # Load ResNet50
            model = resnet_face50(pretrained=False)
            pretrained_dict = torch.load(
                os.path.join(ResNetVerifier.MODEL_PATH, "resnet50.pth"),
                map_location=map_fct,
            )
        else:
            print(f"Unknown ResNet verifier architecture: {architecture}")
            return None

        model.load_state_dict(pretrained_dict)
        model = model.to(self.device)
        model.eval()

        return model


class ArcFace(Verifier):
    """
    Wrapper for ArcFace model used in the deepface package
    """

    def __init__(self, score_threshold):
        self.score_threshold = score_threshold
        self.model = arcface_model.loadModel()

    def preprocess_image(self, img):
        """
        Convert to image to format expected by the model

        Args:
            image (np.ndarray): image to convert
        """
        target_size = (112, 112)

        factor_0 = target_size[0] / img.shape[0]
        factor_1 = target_size[1] / img.shape[1]
        factor = min(factor_0, factor_1)

        dsize = (int(img.shape[1] * factor), int(img.shape[0] * factor))
        img = cv2.resize(img, dsize)

        # Then pad the other side to the target size by adding black pixels
        diff_0 = target_size[0] - img.shape[0]
        diff_1 = target_size[1] - img.shape[1]
        # Put the base image in the middle of the padded image
        img = np.pad(
            img,
            (
                (diff_0 // 2, diff_0 - diff_0 // 2),
                (diff_1 // 2, diff_1 - diff_1 // 2),
                (0, 0),
            ),
            "constant",
        )

        # Normalizing the image pixels
        # img_pixels = image.img_to_array(img)  # what this line doing? must?
        img_pixels = np.expand_dims(img, axis=0)
        img_pixels = img_pixels.astype(np.float64)
        img_pixels /= 255  # normalize input in [0, 1]

        return img_pixels

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
        features = []
        for bbox in face_locations:
            roi = frame[
                int(bbox[1]) : int(bbox[1] + bbox[3]),
                int(bbox[0]) : int(bbox[0] + bbox[2]),
            ]
            image = self.preprocess_image(roi)
            feature = self.model.predict(image)[0].tolist()
            features.append(feature)

        return features

    def get_scores(self, reference_encodings, face_encoding):
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

        dist = []
        for reference_encoding in reference_encodings:
            reference_feature = reference_encoding.get_average
            target_feature = face_encoding.get_average

            cosine_dist = ArcFace.find_cosine_distance(
                reference_feature, target_feature
            )
            dist.append(cosine_dist)

        scores = [1 - d for d in dist]
        return np.array(scores)

    def get_closest_face(self, reference_encodings, face_encoding):
        """
        Finds the most similar face in a set of faces compared with a supplied
        face encoding that also respects the set threshold

        Args:
            reference_encodings (list of Encodings): to compare with
            face_encoding (Encoding): encoding to compare against other
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

    @staticmethod
    def find_cosine_distance(source_representation, test_representation):
        """
        Find the cosine distance between two feature vectors

        Par
        """
        # TODO: This could be optimized to handle a batch of encodings in one
        #  computation, instead of one by one
        a = np.matmul(np.transpose(source_representation), test_representation)
        b = np.sum(np.multiply(source_representation, source_representation))
        c = np.sum(np.multiply(test_representation, test_representation))
        return 1 - (a / (np.sqrt(b) * np.sqrt(c)))
