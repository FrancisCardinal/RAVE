import torch
from torch import nn


class EyeTrackerModel(nn.Module):
    """
    Model of the neural network that detects pupils for the eye tracker module
    """

    def __init__(self):
        """
        Defines the architecture of the model.
        Uses a pretrained version of resnet to do some transfer
        learning. The model also has some fully connected layers at the end,
        as they help the network to make better predictions. It has two heads,
        one for the regression (main) task, and the other for the domain
        classification task. The domain classification task is needed in order
        to implement the DANN algorithm, which we use because we have a lot
        of synthetic data, and not so much real data. The domain classification
        head is only useful / used during the training process.
        """
        super(EyeTrackerModel, self).__init__()
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        self.model = torch.hub.load(
            "pytorch/vision:v0.9.0", "resnet34", pretrained=True
        )
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.layer3.parameters():
            param.requires_grad = True

        for param in self.model.layer4.parameters():
            param.requires_grad = True

        n_inputs = self.model.fc.in_features

        DROPOUT = 0.02
        self.model.fc = nn.Sequential(
            nn.Linear(n_inputs, 2048),
            nn.BatchNorm1d(num_features=2048),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
        )
        self.regression_head = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(num_features=1024),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(512, 256),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(256, 128),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, 5),
        )
        DROPOUT_VISIBILITY = 0.20
        self.visibility_classification_head = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(num_features=1024),
            nn.ReLU(),
            nn.Dropout(DROPOUT_VISIBILITY),
            nn.Linear(1024, 128),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            nn.Dropout(DROPOUT_VISIBILITY),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        """
        Method of the Module class that must be overwritten by this class.
        Specifies how the forward pass should be executed
        The sigmoid is used to limit the ouput domain, as to converge more
        easily. Each parameter is normalized between 0 and 1, where 1
        represents the max pixel value of the corresponding axis of the
        parameter (or 2*pi radians for theta). For example, if the h
        parameter is 0.5 and the image width is 480, then the h value in
        pixels is 240.

        Args:
            x (pytorch tensor):
                The input of the network (images)

        Returns:
            pytorch tensor:
                The predictions of the network (ellipses parameters)
        """
        bottleneck = self.model(x)

        ellipse = self.regression_head(bottleneck)
        ellipse = torch.sigmoid(ellipse)

        visibility_classification = self.visibility_classification_head(
            bottleneck
        )
        visibility_classification = torch.sigmoid(visibility_classification)

        return ellipse, visibility_classification
