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
            from .verifiers.DlibFaceVerifier import DlibFaceRecognition

            return DlibFaceRecognition(threshold)

        resnet_size = None
        if verifier_type == "resnet_face_18":
            resnet_size = 18
        elif verifier_type == "resnet_face_34":
            resnet_size = 34
        elif verifier_type == "resnet_face_50":
            resnet_size = 50

        if resnet_size is not None:
            from .verifiers.ResnetVerifier import ResNetVerifier

            return ResNetVerifier(threshold, device, resnet_size)

        if verifier_type == "arcface":
            from .verifiers.ArcfaceVerifier import ArcFace

            return ArcFace(threshold)

        print("Unknown verifier type:", verifier_type)
        return None
