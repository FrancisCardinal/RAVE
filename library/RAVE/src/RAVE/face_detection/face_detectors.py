import torch
import cv2
import numpy as np
import os

from abc import ABC, abstractmethod
from ..common.image_utils import (
    xyxy2xywh,
    opencv_image_to_tensor,
    scale_coords,
    scale_coords_landmarks,
)
from .FaceDetectionModel import FaceDetectionModel


class Detector(ABC):
    """
    Abstract class for detectors.
    """

    @abstractmethod
    def predict(self, frame, draw_on_frame):
        """
        Method to obtain the prediction on our model

        Args:
            frame (np.ndarray):
                image form which we want to detect faces with shape (3xHxW)
            draw_on_frame (bool):
                Whether or not to draw the predictions on the return frame.

        Return:
            (np.ndarray, list, list):
                A tuple containing the frame with or without the predictions
                drawn on it, a list of bounding boxes and a list of mouth
                positions in x,y.
        """
        raise NotImplementedError()


class DetectorFactory:
    """
    Factory for instantiating a Detector object
    """

    @staticmethod
    def create(detector_type="yolo"):
        """
        Creates the specified Detector object

        Args:
            detector_type (string):
             The type of the Detector. As of now, there's yolo or dnn.
        """
        if detector_type == "yolo":
            return YoloFaceDetector()
        elif detector_type == "dnn":
            return DnnFaceDetector()
        else:
            print("Unknown detector type:", detector_type)
            return None


class DnnFaceDetector(Detector):
    """
    Detector using a dnn model from dlib

    Attributes:
        self.model (dnn_Net): Dnn model from dlib
    """

    def __init__(self):
        resources_dir = os.path.join(
            os.path.dirname(__file__), "detectors", "models", "dnn"
        )
        config_file = os.path.join(resources_dir, "deploy.prototxt.txt")
        model_file = os.path.join(
            resources_dir, "res10_300x300_ssd_iter_140000.caffemodel"
        )
        self.model = cv2.dnn.readNetFromCaffe(config_file, model_file)

    def predict(self, frame, draw_on_frame=False):
        """
        Method to obtain the prediction of dnn

        Args:
            frame (np.ndarray):
                image form which we want to detect faces with shape (3xHxW)
            draw_on_frame (bool):
                Whether or not to draw the predictions on the return frame.

        Return:
            (np.ndarray, list, None):
                A tuple containing the frame with or without the predictions
                drawn on it, a list of bounding boxes and a list of mouth
                positions in x,y. For dnn, return None instead of a list
                because it does not extract features.
        """
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 117.0, 123.0),
        )
        self.model.setInput(blob)
        faces = self.model.forward()

        predicted_bboxes = []
        for i in range(faces.shape[2]):
            confidence = faces[0, 0, i, 2]

            if confidence > 0.5:
                box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                xywh_rect = (x, y, int(x1 - x), int(y1 - y))
                predicted_bboxes.append(xywh_rect)

                if draw_on_frame:
                    line_thickness = frame.shape[0] // 240 or 1
                    cv2.rectangle(
                        frame, (x, y), (x1, y1), (0, 0, 255), line_thickness
                    )
                    cv2.putText(
                        frame,
                        f"{confidence:.3f}",
                        ((x + x1) // 2, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        frame.shape[0] / 960,
                        (0, 0, 255),
                        line_thickness,
                    )  # Display confidence

        return frame, predicted_bboxes, None


class YoloFaceDetector(Detector):
    """
    Detector using yolov5_face model

    @article{
        YOLO5Face,
        title = {YOLO5Face: Why Reinventing a Face Detector},
        author = {Delong Qi and Weijun Tan and Qi Yao and Jingfeng Liu},
        booktitle = {ArXiv preprint ArXiv:2105.12931},
        year = {2021}
    }

    Attributes:
        self.device (string): cpu or cuda
        self.model (FaceDetectionModel): yolov5_face model
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = FaceDetectionModel(self.device)
        self.model.to(self.device)

    # TODO (Anthony): Code copied from main_face_detection;
    #  combine or remove one or the other eventually
    def predict(self, frame, draw_on_frame=False):
        """
        Method to obtain the prediction of yolov5_face

        Args:
            frame (np.ndarray):
                image form which we want to detect faces with shape (3xHxW)
            draw_on_frame (bool):
                Whether or not to draw the predictions on the return frame.

        Return:
            (np.ndarray, list, None):
                A tuple containing the frame with or without the predictions
                drawn on it, a list of bounding boxes and a list of mouth
                positions in x,y. For dnn, return None instead of a list
                because it does not extract features.
        """
        tensor = opencv_image_to_tensor(frame, self.device)
        tensor = torch.unsqueeze(tensor, 0)
        predictions = self.model(tensor)[0]

        # Scale coords
        predictions[:, 5:15] = scale_coords_landmarks(
            tensor.shape[2:], predictions[:, 5:15], frame.shape
        ).round()
        predictions[:, :4] = scale_coords(
            tensor.shape[2:], predictions[:, :4], frame.shape
        ).round()

        predicted_bbox = []
        mouths = []
        for i in range(predictions.size()[0]):
            gn = torch.tensor(frame.shape)[[1, 0, 1, 0]].to(
                self.device
            )  # normalization gain whwh
            gn_lks = torch.tensor(frame.shape)[
                [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
            ].to(
                self.device
            )  # normalization gain landmarks
            xywh = (
                (xyxy2xywh(predictions[i, :4].view(1, 4)) / gn)
                .view(-1)
                .tolist()
            )
            confidence = predictions[i, 4].cpu().item()
            landmarks = (
                (predictions[i, 5:15].view(1, 10) / gn_lks).view(-1).tolist()
            )

            # Bounding box
            height, width, _ = frame.shape
            bbox_scaled = [
                int(xywh[0] * width - 0.5 * xywh[2] * width),
                int(xywh[1] * height - 0.5 * xywh[3] * height),
                int(xywh[2] * width),
                int(xywh[3] * height),
            ]
            predicted_bbox.append(bbox_scaled)

            mouth_x = int(((landmarks[6] + landmarks[8]) * width) / 2)
            mouth_y = int(((landmarks[7] + landmarks[9]) * height) / 2)
            mouth = (mouth_x, mouth_y)
            mouths.append(mouth)

            if draw_on_frame:
                frame = YoloFaceDetector.show_results(
                    frame, xywh, confidence, landmarks
                )

        return frame, predicted_bbox, mouths

    @staticmethod
    def show_results(img, xywh, confidence, landmarks):
        """
        Writes the bounding box, the confidence score and the landmarks to the
        frame. For now, it only shows the result of one prediction.

        Args:
            img (ndarray):
                OpenCV image with shape (height, width, 3)
            xywh (ndarray):
                Normalized values to define the bounding box with shape
                (x, y, width height).
            confidence (float):
                Confidence score.
            landmarks (list):
                Normalized landmarks of length 10. The x and y alternate in the
                list like so: [x1, y1, x2, y2, ..., x5, y5]
        """
        height, width, _ = img.shape
        img = np.ascontiguousarray(img, dtype=np.uint8)
        line_thickness = 1 or round(0.002 * (height + width) / 2) + 1

        # Bounding box
        x1 = int(xywh[0] * width - 0.5 * xywh[2] * width)
        y1 = int(xywh[1] * height - 0.5 * xywh[3] * height)
        x2 = int(xywh[0] * width + 0.5 * xywh[2] * width)
        y2 = int(xywh[1] * height + 0.5 * xywh[3] * height)
        img = cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            thickness=line_thickness,
            lineType=cv2.LINE_AA,
        )

        colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (0, 255, 255),
        ]

        # Landmarks
        for i in range(5):
            point_x = int(landmarks[2 * i] * width)
            point_y = int(landmarks[2 * i + 1] * height)
            img = cv2.circle(
                img, (point_x, point_y), line_thickness + 1, colors[i], -1
            )

        mouth_x = int(((landmarks[6] + landmarks[8]) * width) / 2)
        mouth_y = int(((landmarks[7] + landmarks[9]) * height) / 2)

        img = cv2.circle(
            img, (mouth_x, mouth_y), line_thickness + 1, (255, 0, 0), -1
        )

        # Confidence score
        font_thickness = max(line_thickness - 1, 1)  # font thickness
        label = f"{str(confidence)[:5]}"
        cv2.putText(
            img,
            label,
            (x1, y1 - 2),
            0,
            line_thickness / 3,
            [225, 255, 255],
            thickness=font_thickness,
            lineType=cv2.LINE_AA,
        )

        return img
