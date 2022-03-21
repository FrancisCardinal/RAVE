from .detectors.DnnFaceDetector import DnnFaceDetector
from .detectors.YoloFaceDetector import YoloFaceDetector


class DetectorFactory:
    """
    Factory for instantiating a Detector object
    """

    @staticmethod
    def create(detector_type="yolo", threshold=0.5):
        """
        Creates the specified Detector object

        Args:
            detector_type (string):
                The type of the Detector. As of now, there's yolo or dnn.
            threshold (float):
                Threshold for detection confidence score (0-1)
        """
        if detector_type == "yolo":
            return YoloFaceDetector(threshold=threshold)
        elif detector_type == "dnn":
            return DnnFaceDetector(threshold=threshold)
        else:
            print("Unknown detector type:", detector_type)
            return None
