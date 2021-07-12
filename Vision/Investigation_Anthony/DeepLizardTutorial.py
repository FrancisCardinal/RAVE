import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

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
# train_loader = torch.utils.data.DataLoader(train_set, batch_size=10)

# Visualize data
# print("train_set length: ", len(train_set))
# print("labels: ", train_set.targets)
# print("Freq of each class: ", train_set.targets.bincount())  # Check to see that dataset is balanced

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
# batch = next(iter(train_loader))
# images, labels = batch  # Get 10 images and labels
#
# # Plot all images in batch
# grid = torchvision.utils.make_grid(images, nrow=10)  # torchvision function for combining images in a grid
# plt.figure()
# plt.imshow(np.transpose(grid, (1, 2, 0)))  # Need to transpose to reorder axis to match what imshow() expects
# plt.title(labels)
# plt.show()


# -------------------
# BUILD MODEL
# -------------------

# CNN Network
class Network(nn.Module):  # Extend nn base class (Module)! This class will track the weights during training
    def __init__(self):
        super(Network, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)  # 6 out_channels: means we want to apply 6 different filters (of size: kernel_size)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)  # Can also specify the STRIDE (step of kernel)

        # Linear layers
        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)  # Need to flatten from array to linear:
        self.fc2 = nn.Linear(in_features=120, out_features=60)  # fc: "Fully Connected" = linear layer
        self.out = nn.Linear(in_features=60, out_features=10)  # Final layer (output)

    def activation(self, t):
        return F.relu(t)

    def forward(self, t):

        # 1 input layer
        t = t  # Directly the input data

        # 2 hidden conv layer (conv1)
        t = self.conv1(t)
        t = F.relu(t)  # Activation function!
        t = F.max_pool2d(t, kernel_size=2, stride=2)  # Sometimes this is refered to as a "Pooling layer", but it does not have weights so it isn't really a layer per se

        # 3 hidden conv layer (conv2)
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)  # Pooling!

        # 4 hidden linear layer (fc1)
        t = t.reshape(-1, 12 * (4 * 4))  # We need to reshape: passing from conv to linear! (the tensors are 4x4 at this stage, because of convolution and pooling above)
        t = self.fc1(t)
        t = F.relu(t)

        # 5 hidden linear layer (fc2)
        t = self.fc2(t)
        t = F.relu(t)

        # 6 output layer (out)
        t = self.out(t)
        # t = F.softmax(t, dim=1)  # Could use softmax (don't use relu on last layer)

        return t  # Return transformed tensor

    # Override toString
    # def __repr__(self):
    #     return ""


# --------------------
# CALL NETWORK
# --------------------

# torch.set_grad_enabled(False)  # Disable learning for the moment

# Create network
network = Network()
#print(network)  # Detail the architecture

# Observe weights (and biases) in layer
# for name, parameter in network.named_parameters():
#     print(name, "\t\t\t", parameter.shape)

# sample = next(iter(train_set))
# image, label = sample
# image_batch = image.unsqueeze(0)  # Convert single image to a batch of size 1 with unsqueeze


def get_num_correct(predictions, labels):
    return predictions.argmax(dim=1).eq(labels).sum().item()


# Load data into batches
train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)
batch_iter = iter(train_loader)

optimizer = optim.Adam(network.parameters(), lr=0.01)  # Adam, or SGD: optimization algos. lr: learning rate

for epoch in range(50):
    total_loss = 0
    total_correct = 0

    for batch in train_loader:
        images, labels = batch  # unpack
        predictions = network(images)

        loss = F.cross_entropy(predictions, labels)  # Calculating the Loss function!
        optimizer.zero_grad()  # We need to reset previous training gradients (because or else they will add up)
        loss.backward()  # Calculating the gradients
        # print("Gradients of conv1 layer weights: ", network.conv1.weight.grad)  # Show the gradient of the weights (each weight has a gradient)

        # Take calculated gradients to update the weights (optimization step)

        #print("Current loss: ", loss.item())
        optimizer.step()  # Updating the weights!

        # print("Prediction probability: ", F.softmax(predictions, dim=1))
        # print("Max pred: ", predictions.argmax(dim=1), "Actual: ", labels)
        # print("Comparison: ", predictions.argmax(dim=1).eq(labels))  # Compare each result with label
        #print("Correct count: ", get_num_correct(predictions, labels))  # Count the number of correct predictions

        total_loss += loss.item()
        total_correct += get_num_correct(predictions, labels)

    print("Epoch:", epoch, "total_correct:", total_correct, "loss:", total_loss, "Result:", total_correct/len(train_set)*100, "%")

# View batch images
# grid = torchvision.utils.make_grid(images, nrow=10)  # torchvision function for combining images in a grid
# plt.figure()
# plt.imshow(np.transpose(grid, (1, 2, 0)))  # Need to transpose to reorder axis to match what imshow() expects
# plt.title(np.array2string(labels.numpy()))
# plt.show()























