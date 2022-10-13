import numpy as np
import cv2
import platform
import torch

from .Verifier import Verifier

if platform.release().split("-")[-1] == "tegra":
    from .models.arcface import ArcFace_trt as arcface_model
else:
    from .models.arcface import ArcFace_tf as arcface_model


class ArcFace(Verifier):
    """
    Wrapper for ArcFace model used in the deepface package
    """

    def __init__(self, score_threshold):
        self.score_threshold = score_threshold
        self.model = arcface_model.load_model()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

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
            image = ArcFace.preprocess_image(roi)

            # TODO: Change inference call here.. could make predict() func in
            #  both implementations
            if platform.release().split("-")[-1] == "tegra":
                # image = image.squeeze(0)
                # # TODO: No need to convert to tensor on cuda
                #  before bringing it back to numpy
                # tensor = ArcFace.opencv_image_to_tensor(
                #     image.copy(), self.device
                # )
                # tensor = torch.unsqueeze(tensor, 0)
                image = np.transpose(image, (0, 3, 1, 2))
                feature = self.model(image)
            else:
                feature = self.model(image)[0].numpy().tolist()

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
    def preprocess_image(img):
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

    # TODO: Move with copies of this function to some common location
    @staticmethod
    def opencv_image_to_tensor(image, DEVICE):
        """
        OpenCV BGR image to tensor RGB image

        Args:
            Image (ndarray): Image with shape (width, height, 3).
            DEVICE (string): Pytorch device.
        """
        tensor = torch.from_numpy(image).to(DEVICE)
        tensor = tensor.permute(2, 0, 1).float()
        tensor /= 255
        return tensor
