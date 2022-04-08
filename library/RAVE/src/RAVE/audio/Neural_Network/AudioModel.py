import torch
from torch import nn
import torchaudio
import os

class AudioModel(nn.Module):
    """
    Model of the neural network that generate a mask to combine with a beamformer method to cancel noise for the audio module
    """

    def __init__(self, input_size, hidden_size, num_layers):
        super(AudioModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.BN = nn.BatchNorm2d(num_features=1)
        self.RNN = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False, dropout=0)
        self.fc = nn.Linear(hidden_size,32319)
        self.fc2 = nn.Conv2d(in_channels=hidden_size, out_channels=int(input_size/2), kernel_size=1)
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden=None):
        """
        Method used to go through the recurent neural network to estimate a noise mask
        Args:
            x: frequence signal (N x 1 x F x T) used to estimate the mask
            hidden: If given, the context will be passed to the recurent neural network layer, else the context is set
             to 0

        Returns: Output of the model which is the estimated noise mask to apply in a beamforming method

        """
        # N x 1 x F x T > N x 1 x T x F
        x = x.permute(0, 1, 3, 2)

        # N x 1 x T x F > N x 1 x T x F
        x = x.float()

        # N x 1 x T x F > N x 1 x T x F
        x = self.BN(x)

        # N x 1 x F x T > N x T x F x 1
        x = x.permute(0, 2, 3, 1)

        # N x T x F x 1 > N x T x F
        x = torch.reshape(x, (x.shape[0], x.shape[1], x.shape[2]*x.shape[3]))

        if hidden:
            # N x T x F > N x T x H
            x, h = self.RNN(x, hidden)
        else:
            # N x T x F > N x T x H
            x, h = self.RNN(x)

        # N x T x H > N x H x T
        x = x.permute(0, 2, 1)

        # N x H x T > N x H x T x 1
        x = torch.unsqueeze(x, 3)

        # N x H x T x 1 > N x F x T x 1
        x = self.fc2(x)

        # N x F x T x 1 > N x 1 x T x F
        x = x.permute(0, 3, 1, 2)

        # N x 1 x T x F > N x T x F
        x = torch.squeeze(x, dim=1)

        # N x T x F > N x T x F
        x = self.sig(x)
        return x, h


    def load_best_model(self, MODEL_DIR_PATH, device):
        """
        Used to get the best version of a model from disk
        Args:
            MODEL_DIR_PATH (string): Path to the best model on local disk
            device (string): Device used to perform the computations

        """
        checkpoint = torch.load(MODEL_DIR_PATH, map_location=device)
        self.load_state_dict(checkpoint["model_state_dict"])

        self.eval()


