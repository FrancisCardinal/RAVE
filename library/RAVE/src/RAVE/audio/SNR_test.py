import torch
import torchaudio
from torchmetrics import SignalNoiseRatio, SignalDistortionRatio
filename = "p303_132_p266_001"

original_path = "/Users/felixducharmeturcotte/Desktop/S8_results/gru_50/" + filename + "/audio.wav"
output_path = "/Users/felixducharmeturcotte/Desktop/S8_results/gru_50/" + filename + "/output.wav"
groundtruth_path = "/Users/felixducharmeturcotte/Desktop/S8_results/gru_50/" + filename + "/target.wav"

if __name__ == "__main__":
    offset = 425

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
