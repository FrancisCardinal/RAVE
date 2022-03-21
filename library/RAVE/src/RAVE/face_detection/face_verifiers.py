from .verifiers.DlibFaceVerifier import DlibFaceRecognition
from .verifiers.ResnetVerifier import ResNetVerifier
from .verifiers.ArcfaceVerifier import ArcFace


class VerifierFactory:
    """
    Static factory class used to create verifier instances of different types
    """

    @staticmethod
    def create(verifier_type="arcface", threshold=0.0, device="cpu"):
        """
        Instantiate a verifier of a given type

        Args:
            verifier_type (str): The identifier for the desired verifier
            threshold (float): If applicable, the similarity score decision
                threshold to be used
            device (str): If applicable, the device to run the model on
                ("cpu" or "cuda")

        Returns:
            (Verifier): The newly created verifier or None
        """
        if verifier_type == "dlib":
            return DlibFaceRecognition(threshold)
        elif verifier_type == "resnet_face_18":
            return ResNetVerifier(threshold, device, 18)
        elif verifier_type == "resnet_face_34":
            return ResNetVerifier(threshold, device, 34)
        elif verifier_type == "resnet_face_50":
            return ResNetVerifier(threshold, device, 50)
        elif verifier_type == "arcface":
            return ArcFace(threshold)
        else:
            print("Unknown verifier type:", verifier_type)
            return None
