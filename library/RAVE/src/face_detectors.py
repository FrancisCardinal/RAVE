import torch
import cv2
import numpy as np

from RAVE.common.image_utils import (
    # tensor_to_opencv_image,
    # inverse_normalize,
    xyxy2xywh,
    opencv_image_to_tensor,
    scale_coords,
    scale_coords_landmarks,
)
from RAVE.face_detection.FaceDetectionModel import FaceDetectionModel


class Detector:
    def predict(self, frame):
        raise NotImplementedError()


class YoloFaceDetector(Detector):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = FaceDetectionModel(self.device)
        self.model.to(self.device)

    # TODO (Anthony): Code copied from main_face_detection;
    #  combine or remove one or the other eventually
    def predict(self, frame, draw_on_frame=False):
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

            if draw_on_frame:
                frame = YoloFaceDetector.show_results(
                    frame, xywh, confidence, landmarks
                )

        return frame, predicted_bbox

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
