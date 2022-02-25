import torch
from torch import nn
import torchaudio

class AudioModel(nn.Module):
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





