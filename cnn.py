from torch import nn
from torchsummary import summary

class CNNNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        # 4 convolutional blocks / flatten layer / linear layer / softmax
        self.conv1 = nn.Sequential(  # We give layers for pytorch to link them sequentially
            nn.Conv2d(
                in_channels=1,  # We have 1 channel, mono
                out_channels=16,  # 16 filters in our conv layers
                kernel_size=3,  # usual value apparently
                stride=1,  #
                padding=2,  #
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(  # We give layers for pytorch to link them sequentially
            nn.Conv2d(
                in_channels=16,  # We pipe the output of the 1st conv into this so 16
                out_channels=32,
                kernel_size=3,  # usual value apparently
                stride=1,  #
                padding=2,  #
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv3 = nn.Sequential(  # We give layers for pytorch to link them sequentially
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,  # usual value apparently
                stride=1,  #
                padding=2,  #
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv4 = nn.Sequential(  # We give layers for pytorch to link them sequentially
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,  # usual value apparently
                stride=1,  #
                padding=2,  #
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128 * 5 * 4, 10)  # Shape of the data outputted by conv4, number of classes in our dataset
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        logits = self.linear(x)
        predictions = self.softmax(logits)
        return predictions


if __name__ == "__main__":
    cnn = CNNNetwork()
    summary(cnn, (1, 64, 44))  # summary takes a model and the shape of the network, in our case its the mel spectrogram
