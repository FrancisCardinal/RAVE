import torch
from torch import nn


class EyeTrackerModel(nn.Module):
    """Model of the neural network that detects pupils for the eye tracker module
    """

    def __init__(self):
        """Constructor of the EyeTrackerModel class. Defines the architecture
           of the model.Uses a pretrained version of resnet to do some transfer
           learning. The model also has some fully connected layers at the end,
           as they help the network to make better predictions.
        """
        super(EyeTrackerModel, self).__init__()
        self.model = torch.hub.load(
            "pytorch/vision:v0.9.0", "resnet18", pretrained=True
        )
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.layer3.parameters():
            param.requires_grad = True

        for param in self.model.layer4.parameters():
            param.requires_grad = True

        n_inputs = self.model.fc.in_features

        DROPOUT = 0.05
        self.model.fc = nn.Sequential(
            nn.Linear(n_inputs, 512),
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
            nn.Linear(128, 64),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, 5),
        )

    def forward(self, x):
        """Method of the Dataset class that must be overwritten by this class.
           Specifies how the forward pass should be executed
           The sigmoid is used to limit the ouput domain, as to converge more
           easily. Each parameter is normalized between 0 and 1, where 1
           represents the max pixel value of the corresponding axis of the
           parameter (or 2*pi radians for theta). For example, if the h
           parameter is 0.5 and the image width is 480, then the h value in
           pixels is 240
        Args:
            x (pytorch tensor):
                The input of the network (images)

        Returns:
            pytorch tensor:
                The predictions of the network (ellipses parameters)
        """
        x = self.model(x)
        x = torch.sigmoid(x)
        return x
