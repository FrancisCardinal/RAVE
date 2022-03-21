import os
import cv2
import numpy as np

from .Detector import Detector, Detection


class DnnFaceDetector(Detector):
    """
    Detector using a dnn model from dlib

    Attributes:
        self.model (dnn_Net): Dnn model from dlib
        self.threshold (float):
                Threshold for detection confidence score (0-1)
    """

    def __init__(self, threshold=0.5):
        self.threshold = threshold

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
            (np.ndarray, list of Detections):
                A tuple containing the frame with or without the predictions
                drawn on it, a list of detections containing the bounding box
                and mouth coordinate. For dnn, the mought is always None
                instead because it does not extract features.
        """
        original_frame = frame.copy()
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 117.0, 123.0),
        )
        self.model.setInput(blob)
        faces = self.model.forward()

        detections = []
        for i in range(faces.shape[2]):
            confidence = faces[0, 0, i, 2]

            if confidence >= self.threshold:
                box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                xywh_rect = (x, y, int(x1 - x), int(y1 - y))

                new_detection = Detection(original_frame.copy(), xywh_rect)
                detections.append(new_detection)

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

        return frame, detections
