import os
import re
import csv
import torchaudio
import torch
import random

from torch.utils.data import Dataset
from collections import namedtuple

random.seed(42)
torch.manual_seed(42)

Data = namedtuple('Data', 'position file')

class IROSDataset(Dataset):
    def __init__(self, dir, frame_size, hop_size, sample_rate=16000, max_sources=2, forceCPU=True) -> None:
        super().__init__()
        self.dir = dir
        self.sample_rate = sample_rate
        self.max_sources = max_sources
        self.gain = 1
        self.type = type

        self.targets = {
            "B": (640,309),
            "C": (324,295),
            "D": (0,307),
            "J": (640,284),
            "K": (328,272),
            "L": (0,275),
        }

        self.angles = {
            "A": 0,
            "B": 45,
            "C": 90,
            "D": 135,
            "E": 180,
            "F": 225,
            "G": 270,
            "H": 315,
            "I": 0,
            "J": 45,
            "K": 90,
            "L": 135,
            "M": 180,
            "N": 225,
            "O": 270,
            "P": 315,
        }

        if forceCPU:
            self.device = 'cpu'
        else:
            if (torch.cuda.is_available()):
                self.device = 'cuda'
            else:
                self.device = 'cpu'
        self.noise_path = os.path.join(self.dir, "noise")
        self.speech_path = os.path.join(self.dir, "speech")

        self.paths_to_target = []
        self.paths_to_all = []

        # All labels (including test and train)
        self.labels = {}
        for subdir, dirs, files in os.walk(self.speech_path):
            for file in files:
                if file[-3:] == "wav":
                    path = os.path.join(subdir, file)
                    position = subdir[-1]
                    self.paths_to_all.append(Data(position, path))
                    if position in self.targets.keys():
                        self.paths_to_target.append(Data(position, path))

        # for subdir, dirs, files in os.walk(self.noise_path):
        #     for file in files:
        #         if file[-3:] == "wav":
        #             path = os.path.join(subdir, file)
        #             position = subdir[-1]
        #             self.paths_to_all.append(Data(position, path))


    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        target_idx = random.randint(0, len(self.paths_to_target)-1)
        target_path = self.paths_to_target[target_idx]

        interference_path = None
        while True:
            interference_idx = random.randint(0, len(self.paths_to_all)-1)
            interference_path = self.paths_to_all[interference_idx]
            if self.angles[interference_path.position] != self.angles[target_path.position]:
                break

        target_audio, _ = torchaudio.load(target_path.file)
        interference_audio, _ = torchaudio.load(interference_path.file)
        target_audio = target_audio.to(self.device)
        interference_audio =interference_audio.to(self.device)

        # TODO: change number of seconds?
        target_audio = self.get_right_number_of_samples(target_audio, 10, True)
        interference_audio = self.get_right_number_of_samples(interference_audio, 10, True)
        
        mix = target_audio + interference_audio

        isolated_sources = torch.stack((target_audio, interference_audio))

        return mix*self.gain, isolated_sources*self.gain, self.targets[target_path.position]

    def get_right_number_of_samples(self, x, seconds, shuffle=False):
        nb_of_samples = seconds*self.sample_rate
        if x.shape[1] < nb_of_samples:
            x = torch.nn.functional.pad(x, (0, nb_of_samples-x.shape[1]), mode="constant", value=0)
        elif x.shape[1] > seconds*self.sample_rate:
            if shuffle:
                random_number = torch.randint(low=0, high=x.shape[-1]-nb_of_samples-1, size=(1,))[0].item()
                x = x[..., random_number:nb_of_samples+random_number]
            else:    
                x = x[..., :nb_of_samples]
            
        return x
    
    @staticmethod
    def normalize(X, augmentation = False):
        # Equation: 10*torch.log10((torch.abs(X)**2).mean()) = 0

        if augmentation:
            aug = torch.rand(1).item()*10 - 5
            augmentation_gain = 10 ** (aug/20)
        else:
            augmentation_gain = 1
        
        normalize_gain  = torch.sqrt(1/(torch.abs(X)**2).mean()) 
       
        return augmentation_gain * normalize_gain * X
