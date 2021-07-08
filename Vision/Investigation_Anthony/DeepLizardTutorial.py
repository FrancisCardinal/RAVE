import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn

# -------------------
# PREPARE DATA
# -------------------

# Get Fashion-MNIST data from web [EXTRACT]
train_set = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST',
    train=True,
    download=True,

    # Transform data to tensors
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

# Load data (to get query capabilities on the data) (load into batches)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)

# Visualize data
print("train_set length: ", len(train_set))
print("labels: ", train_set.targets)
print("Freq of each class: ", train_set.targets.bincount())  # Check to see that dataset is balanced

# Extract one sample -------------------------------
# sample = next(iter(train_set))  # Error with pytorch 1.9.0 (works on 1.8.1), (torchvision 0.10.0 -> 0.9.1)
# image, label = sample
# # sample.shape = [1, 28, 28] (grayscale 28x28 image)
#
# # Show sample
# plt.imshow(image.squeeze(), 'gray')  # squeeze() removes all dimensions of size 1 (in this case, the color channel)
# plt.title(f"Label: {label}")
# plt.show()


# Extract a batch ------------------------------------
batch = next(iter(train_loader))
images, labels = batch  # Get 10 images and labels

# Plot all images in batch
grid = torchvision.utils.make_grid(images, nrow=10)  # torchvision function for combining images in a grid
plt.figure()
plt.imshow(np.transpose(grid, (1, 2, 0)))  # Need to transpose to reorder axis to match what imshow() expects
plt.title(labels)
plt.show()


# -------------------
# BUILD MODEL
# -------------------

# CNN Network
class Network(nn.Module):  # Extend nn base class (Module)! This class will track the weights during training
    def __init__(self):
        super(Network, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        # Linear layers
        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)  # fc: "Fully Connected" = linear layer
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)  # Final layer (output)

    def forward(self, t):
        t = self.layer(t)  # Transform tensor
        return t  # Return transformed tensor


# Create network
network = Network()