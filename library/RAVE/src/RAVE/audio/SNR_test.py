import torch
import torchaudio
from torchmetrics import SignalNoiseRatio, SignalDistortionRatio
filename = "clnsp2_Babble_6_clnsp3_clnsp7"
folderPath = "reel_test/10nov1636/small/"

original_path = "/Users/felixducharmeturcotte/Desktop/"+ folderPath + filename + "/audio.wav"
output_path = "/Users/felixducharmeturcotte/Desktop/" + folderPath+ filename + "/output.wav"
groundtruth_path = "/Users/felixducharmeturcotte/Desktop/" + folderPath + filename + "/target.wav"

if __name__ == "__main__":
    offset = 100

    device = torch.device("mps")
    print("DEVICE:", device)
    prediction, _ = torchaudio.load(output_path)
    length = prediction.shape[1]
    prediction = prediction[:,offset:length]

    target, _ = torchaudio.load(groundtruth_path)
    target = torch.unsqueeze(torch.mean(target, dim=0), dim=0)
    target = target[:, 0:length-offset]

    original, _ = torchaudio.load(original_path)
    original = original[:, :length]
    original = torch.mean(original, dim=0, keepdim=True)

    targetOriginal, _ = torchaudio.load(groundtruth_path)
    targetOriginal = torch.unsqueeze(torch.mean(targetOriginal, dim=0), dim=0)
    targetOriginal = targetOriginal[:, 0:length]


    sdr = SignalDistortionRatio()
    print("Before: ", sdr(original, targetOriginal).item()," After: ", sdr(prediction, target).item())
    print("Done.")
