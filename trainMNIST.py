import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001

class FeedForwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential( # Pack layers together
            nn.Linear(28*28,256), # Images are 28*28
            nn.ReLU(),
            nn.Linear(256,10)
        )
        self.softmax = nn.Softmax(dim=1)
    def forward(self,input_data):
        flattened_data = self.flatten(input_data)
        logits = self.dense_layers(flattened_data)
        predicitions = self.softmax(logits)
        return predicitions

def download_mnist_datasets():
    train_data = datasets.MNIST(
        root="data", # download the dataset in data folder
        download=True, # Download if not downloaded
        train=True, # I want to train the model
        transform=ToTensor() #Apply transformation to dataset
    )
    validation_data = datasets.MNIST(
        root="data",  # download the dataset in data folder
        download=True,  # Download if not downloaded
        train=False,  # I dont't want to train the model
        transform=ToTensor()  # Apply transformation to dataset
    )
    return train_data, validation_data

def train_one_epoch(model,data_loader,loss_fn,optimiser,device):
    for inputs,targets in data_loader:
        inputs,targets = inputs.to(device), targets.to(device)

        # calculate loss
        predictions = model(inputs)
        loss = loss_fn(predictions,targets)
        # backpropagate loss and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    print(f"Loss : {loss.item()}")

def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_one_epoch(model,data_loader,loss_fn,optimiser,device)
        print("--------------")
    print("Training is done.")

if __name__ == "__main__":
    #download dataset
    train_data, _ = download_mnist_datasets()
    print("Dataset downloaded")
    # Data loader
    train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE)
    #Build model
    device = "cpu"
    feed_forward_net = FeedForwardNet().to(device)
    # get loss function + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(feed_forward_net.parameters(), lr=LEARNING_RATE)
    #Train model
    train(feed_forward_net, train_data_loader, loss_fn, optimiser, device, EPOCHS)
    #Store model
    torch.save(feed_forward_net.state_dict(), "feedforwardnet.pth")
    print("Model trained and stored")