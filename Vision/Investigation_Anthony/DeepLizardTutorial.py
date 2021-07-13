import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time
import pandas as pd
import json

from sklearn.metrics import confusion_matrix
from resources.plotcm import plot_confusion_matrix

import numpy as np
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter  # For tensorboard visualization

from itertools import product
from collections import OrderedDict
from collections import namedtuple

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


class RunBuilder:

    @staticmethod
    def get_runs(params):

        Run = namedtuple("Run", params.keys())
        runs = []
        for v in product(*params.values()):
           runs.append(Run(*v))

        return runs


# Need to implement main run loop described in 'CNN Training Loop Refactoring - Simultaneous Hyperparameter Testing' to use
class RunManager:

    def __init__(self):
        # TODO: can refactor into separate classes
        self.epoch_count = 0
        self.epoch_loss = 0
        self.epoch_num_correct = 0
        self.epoch_start_time = 0

        self.run_params = None
        self.run_count = 0
        self.run_data = []
        self.run_start_time = None

        self.network = None
        self.loader = None
        self.tensor_board = None

    def begin_run(self, run, network, loader):
        self.run_start_time = time.time()

        self.run_params = run
        self.run_count += 1

        self.network = network
        self.loader = loader
        self.tensor_board = SummaryWriter(comment=f'-{run}')

        images, labels = next(iter(self.loader))
        grid = torchvision.utils.make_grid(images)

        self.tb.add_image('images', grid)
        self.tb.add_graph(self.network, images)

    def end_run(self):
        self.tensor_board.close()
        self.epoch_count = 0

    def begin_epoch(self):
        self.epoch_start_time = time.time()

        self.epoch_count += 1
        self.epoch_loss = 0
        self.epoch_num_correct = 0

    def end_epoch(self):
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        loss = self.epoch_loss / len(self.loader.dataset)
        accuracy = self.epoch_num_correct / len(self.loader.dataset)

        self.tb.add_scalar('Loss', loss, self.epoch_count)
        self.tb.add_scalar('Accuracy', accuracy, self.epoch_count)

        for name, param in self.network.named_parameters():
            self.tb.add_histogram(name, param, self.epoch_count)
            self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_count)

        # Dictionary to save data ourselves
        results = OrderedDict()
        results["run"] = self.run_count
        results["epoch"] = self.epoch_count
        results['loss'] = loss
        results["accuracy"] = accuracy
        results['epoch duration'] = epoch_duration
        results['run duration'] = run_duration
        for k, v in self.run_params._asdict().items(): results[k] = v
        self.run_data.append(results)

        df = pd.DataFrame.from_dict(self.run_data, orient='columns')

    def track_loss(self, loss):
        self.epoch_loss += loss.item() * self.loader.batch_size

    def track_num_correct(self, preds, labels):
        self.epoch_num_correct += self._get_num_correct(preds, labels)

    @torch.no_grad()
    def _get_num_correct(self, preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()

    def save(self, filename):
        pd.DataFrame.from_dict(self.run_data, orient="columns").to_csv(f"{filename}.csv")

        with open(f"{filename}.json", "w", encoding="utf-8") as f:
            json.dump(self.run_data, f, ensure_ascii=False, indent=4)

# --------------------
# CALL NETWORK
# --------------------

# torch.set_grad_enabled(False)  # Disable learning for the moment

def get_num_correct(predictions, labels):
    # print("Prediction probability: ", F.softmax(predictions, dim=1))
    # print("Max pred: ", predictions.argmax(dim=1), "Actual: ", labels)
    # print("Comparison: ", predictions.argmax(dim=1).eq(labels))  # Compare each result with label
    # print("Correct count: ", get_num_correct(predictions, labels))  # Count the number of correct predictions
    return predictions.argmax(dim=1).eq(labels).sum().item()


def get_all_preds(network, loader):
    all_preds = torch.tensor([])
    for batch in loader:
        images, labels = batch
        predictions = network(images)

        all_preds = torch.cat((all_preds, predictions), dim=0)

    return all_preds


def build_confusion_matrix(targets, preds):
    # Build confusion matrix
    label_and_pred = torch.stack((targets, preds), dim=1)  # True value / pred pair

    confusion_mat = torch.zeros(10, 10, dtype=torch.int32)  # Initialize empty confusion matrix

    for pair in label_and_pred:
        label, pred = pair.tolist()
        confusion_mat[label, pred] += 1

    return confusion_mat


def show_confusion_matrix(targets, preds):
    # Plot confusion matrix
    names = ("t-shirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot")
    cm = confusion_matrix(targets, preds)

    plt.figure(figsize=(10, 10))
    plot_confusion_matrix(cm, names)
    plt.show()


# Hyperparameters
num_epochs = 5
parameters = OrderedDict(
    lr=[0.01, 0.001],
    batch_size=[10, 100, 1000],
    shuffle=[True, False]
)

# Create all combinations of hyperparameters
# param_values = [v for v in parameters.values()]  # list of parameters values

if __name__ == "__main__":
    all_runs = RunBuilder.get_runs(parameters)
    run_count = 0

    for run in all_runs:
        run_count += 1

        # Create network
        network = Network()

        # Load data into batches
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=run.batch_size, shuffle=run.shuffle, num_workers=1)  # num_workers: add 1 additional worker to get data reday for next loop ahead of time

        optimizer = optim.Adam(network.parameters(), lr=run.lr)  # Adam, or SGD: optimization algos. lr: learning rate

        # Tensor board
        comment = f"-{run}"
        tensor_board = SummaryWriter(comment=comment)  # Comment is used to identify run
        images, labels = next(iter(train_loader))
        grid = torchvision.utils.make_grid(images)
        tensor_board.add_image("images", grid)
        tensor_board.add_graph(network, images)

        print(f"{run_count}/{len(all_runs)}: ", comment)

        for epoch in range(num_epochs):
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
                optimizer.step()  # Updating the weights!

                total_loss += loss.item() * run.batch_size  # Multiplying by batch size so runs with different batch sizes can be comparable
                total_correct += get_num_correct(predictions, labels)

            # Add data to tensorboard
            tensor_board.add_scalar("Loss", total_loss, epoch)
            tensor_board.add_scalar("Number correct", total_correct, epoch)
            tensor_board.add_scalar("Accuracy", total_correct/len(train_set), epoch)

            for name, weight in network.named_parameters():
                tensor_board.add_histogram(name, weight, epoch)
                tensor_board.add_histogram(f"{name}.grad", weight.grad, epoch)

            print("Epoch: {}  total_correct: {}  total_loss: {:.4f}  result: {:.4f}%".format(epoch, total_correct, total_loss, total_correct/len(train_set)*100))

        # View batch images
        # grid = torchvision.utils.make_grid(images, nrow=10)  # torchvision function for combining images in a grid
        # plt.figure()
        # plt.imshow(np.transpose(grid, (1, 2, 0)))  # Need to transpose to reorder axis to match what imshow() expects
        # plt.title(np.array2string(labels.numpy()))
        # plt.show()

        # View results (only works if not shuffled)
        with torch.no_grad():  # Deactivate gradient tracking locally: because we don't want to train here
            test_train_loader = torch.utils.data.DataLoader(train_set, batch_size=run.batch_size, shuffle=False)
            train_preds = get_all_preds(network, test_train_loader)

        preds_correct = get_num_correct(train_preds, train_set.targets)
        print(f"Final: {preds_correct}/{len(train_set)} ({preds_correct/len(train_set)*100:.4f}%)")

        print(build_confusion_matrix(train_set.targets, train_preds.argmax(dim=1)))  # Print confusion matrix in console
        # show_confusion_matrix(train_set.targets, train_preds.argmax(dim=1))  # plot a nice confusion matrix

    tensor_board.close()

















