import torch 
from torch import nn 

class EyeTrackerModel(nn.Module):
    def __init__(self):
        super(EyeTrackerModel, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.layer4[1].parameters():
            param.requires_grad = True

        n_inputs = self.model.fc.in_features

        self.model.fc = nn.Sequential(
            nn.Linear(n_inputs, 128),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(128, 64),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(64, 32),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(32, 5)
        )
        

    def forward(self, x):
        x = self.model(x) 
        x = torch.sigmoid(x) # Pour s'assurer qu'on n'ait pas de valeurs abbérantes et que le domaine de sortie soit limité, pour favoriser la convergence. Pour les quatres valeurs en pixels (h, k, a, b), cela signifie qu'une valeur de 1 correspond à la dimension originale de l'image (ex : h=1 --> h = ORIGINAL_IMAGE_WIDTH). Pour l'angle, 0 = 0 deg et 1 = 360 deg.
        return x 