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
        self.blstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False, dropout=0.5)
        self.fc = nn.Linear(hidden_size,32319)
        self.fc2 = nn.Conv2d(in_channels=hidden_size, out_channels=int(input_size/2), kernel_size=1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
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

        # N x T x F > N x T x H
        x, _ = self.blstm(x)

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
        return x


    def load_best_model(self, MODEL_DIR_PATH):
        """
        Used to get the best version of a model from disk

        Args:
            model (Module): Model on which to update the weights
        """
        checkpoint = torch.load(MODEL_DIR_PATH)
        self.load_state_dict(checkpoint["model_state_dict"])

        self.eval()


class AudioModel_old(nn.Module):
    """
    Model of the neural network that generate a mask to combine with a beamformer method to cancel noise for the audio module
    """
    def __init__(self, input_size, hidden_size, num_layers):
        super(AudioModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.BN = nn.BatchNorm1d(63)
        self.blstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False, dropout=0.5)
        self.fc = nn.Linear(hidden_size,32319)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        #h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)
        #c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)
        x = torch.einsum("ijk->ikj", x)
        x = x.float()
        x = self.BN(x)
        x, _ = self.blstm(x)
        x = self.fc(x[:,-1,:])
        #x = torch.tanh(x)
        x = self.sig(x)
        x = torch.reshape(x,(-1,513, 63))
        return x


