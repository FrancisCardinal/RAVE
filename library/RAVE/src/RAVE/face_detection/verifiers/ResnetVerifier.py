import os
import cv2
import numpy as np
import torch

from .models.arcface.trt_model import TrtModel
import platform
import torchvision

from .Verifier import Verifier

from .models.resnet import (
    resnet_face18,
    resnet_face34,
    resnet_face50,
)


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
        self.transform = torchvision.transforms.Resize((112, 112))

    def preprocess_image(self, image):
        """
        Convert to image to format expected by the model

        Args:
            image (tensor): image to convert
        """

        if image is None:
            return None

        img = self.transform(image).type(torch.float).unsqueeze(0)
        img = (img - 0.5)/0.5 # TODO: Determine if these values are necessary and correct
        return img

    def get_features(self, frame, face_locations):
        """
        Get the encodings (feature vectors) for all request objects

        Args:
            frame (tensor): The image containing the objects
            face_locations (list of tuples (int, int, int, int)): List of all
                the xywh bounding boxes for the objects in the image to compute
                the encodings for

        Returns:
            (list(float)): Variable length list containing the floats that
                represent the object
        """

        batch = None
        for bbox in face_locations:
            roi = frame[:,
                int(bbox[1]) : int(bbox[1] + bbox[3]),
                int(bbox[0]) : int(bbox[0] + bbox[2]),
            ]
            image = self.preprocess_image(roi)
            if batch is None:
                batch = image
            else:
                batch = torch.cat((batch, image), dim=0)


        with torch.no_grad():
            features = self.model(batch).cpu().numpy()

        features = [v for v in features]
        return features

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

        # TODO: is this really a score or a dist?
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
            # if platform.release().split("-")[-1] == "tegra":
            #     # TODO: Note: to be modified for use on the Jetson
            #     model = TrtModel()
            #     model_path = os.path.join(
            #         ResNetVerifier.MODEL_PATH, "resnet18", "resnet18_trt.pth"
            #     )
            #     pretrained_dict = torch.load(model_path)
            # else:
            model = resnet_face18(pretrained=False)

            model_path = os.path.join(ResNetVerifier.MODEL_PATH, "resnet18", "resnet18.pth"),
            try:
                pretrained_dict = torch.load(
                    model_path,
                    map_location=map_fct,
                )
            except Exception as e:
                print(f"""Failed to load resnet18 model, be sure to import resnet18.pth 
                at {model_path}""")
                print(e)
                
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
