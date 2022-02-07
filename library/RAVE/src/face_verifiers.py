import face_recognition
import numpy as np
import os
import cv2
import torch

from RAVE.face_detection.verifiers.models.resnet import (
    resnet_face18,
    resnet_face34,
    resnet_face50,
)


class Verifier:
    def get_encodings(self, frame, face_locations):
        raise NotImplementedError()

    def get_scores(self, reference_encoding, face_encoding):
        raise NotImplementedError()

    def get_closest_face(self, reference_encodings, face_encoding):
        raise NotImplementedError()


class VerifierFactory:
    @staticmethod
    def create(verifier_type="dlib", threshold=10, device="cpu"):
        if verifier_type == "dlib":
            return DlibFaceRecognition(threshold)
        if verifier_type == "resnet_face_18":
            return ResNetVerifier(threshold, device, 18)
        if verifier_type == "resnet_face_34":
            return ResNetVerifier(threshold, device, 34)
        if verifier_type == "resnet_face_50":
            return ResNetVerifier(threshold, device, 50)
        else:
            print("Unknown verifier type:", verifier_type)
            return None


class DlibFaceRecognition(Verifier):
    def __init__(self, score_threshold):
        self.score_threshold = score_threshold

    def get_encodings(self, frame, face_locations):
        """
        frame: image containing the faces
        face_locations: list of tuples for face locations in format (x,y,w,h)
        returns: list of encodings matching each element in face_locations
        """
        rgb_frame = frame[:, :, ::-1]
        # t = time.time()

        # Convert xywh bboxes to (y0, x1, y1, x0)
        converted_face_locations = []
        for bbox in face_locations:
            x, y, w, h = bbox
            bbox_converted = (y, x + w, y + h, x)
            converted_face_locations.append(bbox_converted)

        face_encodings = face_recognition.face_encodings(
            rgb_frame, converted_face_locations, num_jitters=1, model="small"
        )

        # print("Verifier time:", time.time() - t)
        return face_encodings

    @staticmethod
    def get_scores(reference_encodings, face_encoding):
        distances = face_recognition.face_distance(
            reference_encodings, face_encoding
        )
        scores = [1 - distance for distance in distances]
        return scores

    def get_closest_face(self, reference_encodings, face_encoding):
        face_scores = self.get_scores(reference_encodings, face_encoding)
        best_match_index = np.argmax(face_scores)
        best_score = face_scores[best_match_index]

        if best_score >= self.score_threshold:
            return best_match_index, best_score
        else:
            return None, None


class ResNetVerifier(Verifier):

    PROJECT_PATH = os.getcwd()
    MODEL_PATH = os.path.join(
        PROJECT_PATH, "RAVE", "face_detection", "verifiers", "models"
    )

    def __init__(self, score_threshold, device="cpu", architecture=18):
        self.device = device
        self.score_threshold = score_threshold
        self.model = self._load_model(architecture)

    def preprocess_image(self, image):
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
            if len(features) == 0:
                features = feature
            else:
                features = np.concatenate((features, feature), axis=0)

        return features

    @staticmethod
    def get_scores(reference_encodings, face_encoding):
        dist = []
        for reference_encoding in reference_encodings:
            dist.append(
                np.dot(face_encoding, reference_encoding)
                / (
                    np.linalg.norm(face_encoding)
                    * np.linalg.norm(reference_encoding)
                )
            )
        return np.array(dist)

    def get_closest_face(self, reference_encodings, face_encoding):
        face_distances = self.get_scores(reference_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        best_distance = face_distances[best_match_index]

        if best_distance <= self.score_threshold:
            return best_match_index, best_distance
        else:
            return None, None

    def _load_model(self, architecture):
        if self.device == "cuda":

            def map_fct(storage, loc):
                return storage.cuda(self.device)

        else:

            def map_fct(storage, loc):
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
            print(f"Unkown ResNet verifier architecture: {architecture}")
            return None

        model.load_state_dict(pretrained_dict)
        model = model.to(self.device)
        model.eval()

        return model
