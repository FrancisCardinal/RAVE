from abc import ABCMeta, abstractmethod


class Detector:
    """
    Abstract class for detectors.
    """

    __metaclass__ = ABCMeta

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
            (np.ndarray, list of Detections):
                A tuple containing the frame with or without the predictions
                drawn on it, a list of detections containing the bounding box
                and mouth coordinate.
        """
        raise NotImplementedError()


class Detection:
    """
    Container class for detections.

    Attributes:
        frame (ndarray): image taken at the moment of detection
        bbox (list(int)) xywh bounding box of detection
        mouth (tuple (int)): x,y coordinates of mouth landmark
    """

    def __init__(self, frame, bbox, mouth=None, landmarks=None):
        self.frame = frame
        self.bbox = bbox
        self.mouth = mouth
        self.landmarks = landmarks
