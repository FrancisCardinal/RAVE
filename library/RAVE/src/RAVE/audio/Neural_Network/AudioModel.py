import torch
from torch import nn
import torchaudio

class AudioModel(nn.Module):
    """
    Model of the neural network that generate a mask to combine with a beamformer method to cancel noise for the audio module
    """
    def __init__(self, input_size, hidden_size, num_layers):
        """
        Defines the architecture of the model.
        """
        super(AudioModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.blstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size,1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        #h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)
        #c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)
        """
        # todo: move to datasetloader
        # x needs to be converted to a concatened tensor of 2 spectograms: the log mean spectogram of the array of mics, the spectogram of delay and sum signal
        x1 = self.logMeanSpect(x)
        x2 = self.delayAndSumSpect(x)
        x = torch.cat(x1, x2, dim=0)
        """
        x, _ = self.blstm(x)
        x = self.fc(x[:,-1,:])
        x = self.sig(x)
        return x
