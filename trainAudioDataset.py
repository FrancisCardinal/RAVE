import torch
from torch import nn
from torch.utils.data import DataLoader
import torchaudio

from urbansoundDataset import UrbanSoundDataset
from cnn import CNNNetwork

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001
#Instantiating our dataset object
ANNOTATIONS_FILE = "C:/Users/Jay/Desktop/TorchAudio/UrbanSound8K/metadata/UrbanSound8K.csv"
AUDIO_DIR = "C:/Users/Jay/Desktop/TorchAudio/UrbanSound8K/audio"
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050  # If sample_rate = num_samples it means we want 1 sec of audio

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

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,  # Frame size
        hop_length=512,  # Frame size / 2
        n_mels=64
    )

    usd = UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device)
    # Data loader
    train_data_loader = DataLoader(usd, batch_size=BATCH_SIZE)
    #Build model
    device = "cpu"
    cnn = CNNNetwork().to(device)
    # get loss function + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)
    #Train model
    train(cnn, train_data_loader, loss_fn, optimiser, device, EPOCHS)
    #Store model
    torch.save(cnn.state_dict(), "cnn_audio_network.pth")
    print("Model trained and stored")