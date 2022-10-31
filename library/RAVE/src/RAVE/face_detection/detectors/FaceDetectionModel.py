import torch
import os
import platform
import sys
from torch import nn

if platform.release().split("-")[-1] == "tegra":
    from .models.yolov5.models.trt_model import TrtModel
    from .models.yolov5.utils.general import non_max_suppression_face


class FaceDetectionModel(nn.Module):
    """
    Model of the neural network that detects faces for facial detection

    Args:
        DEVICE (string): Pytorch device
    """

    CONFIDENCE_THRESHOLD = 0.5
    INTERSECTION_OVER_UNION_THRESHOLD = 0.5

    def __init__(self, DEVICE):
        """
        Defines the architecture of the model.
        Uses a pretrained version of yolov5-face nano.
        """
        super(FaceDetectionModel, self).__init__()

        # Add model path to be able to load a saved model from a different
        # project
        PROJECT_PATH = os.getcwd()
        MODEL_PATH = os.path.join(
            PROJECT_PATH,
            "RAVE",
            "face_detection",
            "detectors",
            "models",
            "yolov5",
        )
        sys.path.append(MODEL_PATH)
        self.device = DEVICE

        # if platform.release().split("-")[-1] == "tegra":
        #     self.model = TrtModel(os.path.join(MODEL_PATH, "yolov5n-face.trt"))
        # else:
        # TODO-JKealey: load to cpu to avoid ram/gpu surge as
        #  suggested in doc
        self.model = (
            torch.load(
                os.path.join(MODEL_PATH, "yolov5n-face.pt"),
                map_location=DEVICE,
            )["model"]
            .float()
            .fuse()
            .eval()
            .nms(
                conf=self.CONFIDENCE_THRESHOLD,
                iou=self.INTERSECTION_OVER_UNION_THRESHOLD,
            )
        )
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        Forward pass of the model

        Args:
            x (pytorch tensor):
                The input of the network (images)

        Returns:
            pytorch tensor:
                The predictions of the network
        """
        # if platform.release().split("-")[-1] == "tegra":
        #     # TODO JKealey: don't pass to GPU before inference
        #     x = self.model(x.cpu().numpy()).reshape(1, 18900, 16)
        #     x = non_max_suppression_face(
        #         torch.from_numpy(x).to(self.device),
        #         conf_thres=self.CONFIDENCE_THRESHOLD,
        #         iou_thres=self.INTERSECTION_OVER_UNION_THRESHOLD,
        #     )
        # else:
        x = self.model(x)
        return x
