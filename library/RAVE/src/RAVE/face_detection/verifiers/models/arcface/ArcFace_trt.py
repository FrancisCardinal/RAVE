import os

from trt_model import TrtModel
from trt_model import ONNX_to_TRT


def load_model():
    """
    Load the model
    """
    project_path = os.getcwd()
    model_path = os.path.join(
        project_path,
        "RAVE",
        "face_detection",
        "verifiers",
        "models",
        "arcface",
        "arcface.trt",
    )

    model = TrtModel(model_path)
    return model


def convert_model():
    """
    Load tensorRT model
    """

    # Add model path to be able to load a saved model from a different
    # project
    project_path = os.getcwd()
    model_path = os.path.join(
        project_path,
        "RAVE",
        "face_detection",
        "verifiers",
        "models",
        "arcface",
        "arcface.onnx",
    )

    ONNX_to_TRT(model_path)


if __name__ == "__main__":
    convert_model()
