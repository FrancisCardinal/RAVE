import cv2
import numpy as np
import torch

from .Detector import Detector, Detection
from .FaceDetectionModel import FaceDetectionModel
from ...common.image_utils import (
    xyxy2xywh,
    opencv_image_to_tensor,
    scale_coords,
    scale_coords_landmarks,
)


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
        threshold (float):
                Threshold for detection confidence score (0-1)
    """

    def __init__(self, threshold=0.5):
        self.threshold = threshold
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
                image form which we want to detect faces with shape (HxWx3)
            draw_on_frame (bool):
                Whether or not to draw the predictions on the return frame.

        Return:
            (np.ndarray, list of Detections):
                A tuple containing the frame with or without the predictions
                drawn on it, a list of detections containing the bounding box
                and mouth coordinate. For dnn, the mought is always None
                instead because it does not extract features.
        """
        original_frame = frame.copy()
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

        detections = []
        for i in range(predictions.size()[0]):

            # Make sure confidence is over threshold for each detection
            confidence = predictions[i, 4].cpu().item()
            if confidence < self.threshold:
                print(
                    f"Rejecting face detection below threshold"
                    f": {confidence} < {self.threshold}"
                )
                continue

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

            # Landmarks
            for i in range(5):
                landmarks[2 * i] = int(landmarks[2 * i] * width)
                landmarks[2 * i + 1] = int(landmarks[2 * i + 1] * height)

            mouth_x = int((landmarks[6] + landmarks[8]) / 2)
            mouth_y = int((landmarks[7] + landmarks[9]) / 2)
            mouth = (mouth_x, mouth_y)

            new_detection = Detection(
                original_frame.copy(), bbox_scaled, mouth, landmarks
            )
            detections.append(new_detection)

            if draw_on_frame:
                frame = YoloFaceDetector.show_results(
                    frame, bbox_scaled, confidence, landmarks, mouth
                )

        return frame, detections

    @staticmethod
    def show_results(img, xywh, confidence, landmarks, mouth):
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

        Returns:
            (ndarray) The image with the predictions on it
        """
        height, width, _ = img.shape
        img = np.ascontiguousarray(img, dtype=np.uint8)
        line_thickness = 1 or round(0.002 * (height + width) / 2) + 1

        # Bounding box
        x1 = xywh[0]
        y1 = xywh[1]
        x2 = xywh[0] + xywh[2]
        y2 = xywh[1] + xywh[3]
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
            point_x = landmarks[2 * i]
            point_y = landmarks[2 * i + 1]
            img = cv2.circle(
                img, (point_x, point_y), line_thickness + 1, colors[i], -1
            )

        img = cv2.circle(
            img, (mouth[0], mouth[1]), line_thickness + 1, (255, 0, 0), -1
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
