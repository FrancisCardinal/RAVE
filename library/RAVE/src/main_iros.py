import torch
import torchaudio
import csv
import os
import random

from tqdm import tqdm
from torchmetrics.functional.audio import signal_noise_ratio, scale_invariant_signal_distortion_ratio
from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality
from torchmetrics.functional.audio.stoi import short_time_objective_intelligibility
from RAVE.audio.Dataset.IROSDataset import IROSDataset
from RAVE.audio.AudioManager import AudioManager
from RAVE.face_detection.Pixel2Delay import Pixel2Delay
from collections import namedtuple
from multiprocessing import Queue
from torch.multiprocessing import set_start_method
from threading import Thread

path = "/home/jacob/dev/RAVE/IROS/3it"
frame_size = 512
hop_size = 256
device = 'cuda' if torch.cuda.is_available() else 'cpu'
random.seed(42)

Results = namedtuple('Results', 'init_si_sdr after_si_sdr init_snr after_snr init_pesq after_pesq init_stoi after_stoi')

def cuda_sqrt_hann_window(
    window_length, periodic=True, dtype=None, layout=torch.strided, device="cuda", requires_grad=False
):
    return torch.sqrt(
        torch.hann_window(
            window_length, periodic=periodic, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad
        )
    )

def si_snr(estimate, reference, epsilon=1e-8):
    estimate = estimate - estimate.mean()
    reference = reference - reference.mean()
    reference_pow = reference.pow(2).mean(axis=1, keepdim=True)
    mix_pow = (estimate * reference).mean(axis=1, keepdim=True)
    scale = mix_pow / (reference_pow + epsilon)

    reference = scale * reference
    error = estimate - reference

    reference_pow = reference.pow(2)
    error_pow = error.pow(2)

    reference_pow = reference_pow.mean(axis=1)
    error_pow = error_pow.mean(axis=1)

    si_snr = 10 * torch.log10(reference_pow) - 10 * torch.log10(error_pow)
    return si_snr.item()


def run_loop(loader, save=False):

    stft = torchaudio.transforms.Spectrogram(
        n_fft=512,
        hop_length=256,
        power=None,
        window_fn=cuda_sqrt_hann_window
    )
    istft = torchaudio.transforms.InverseSpectrogram(
        n_fft=512, hop_length=256, window_fn=cuda_sqrt_hann_window
    )
    SCM = torchaudio.transforms.PSD(normalize=False)
    mvdr_transform = torchaudio.transforms.SoudenMVDR()

    # Run audio manager
    audio_man = AudioManager(debug=False, mask=False, use_timers=False, save_output=True)
    pixel_to_delay = Pixel2Delay((480, 640), "./calibration_8mics.json", device=device)
    audio_man.iros_init()
    nb_of_iterations = len(loader.dataset)
    init_si_sdr = 0
    after_si_sdr = 0
    init_snr = 0
    after_snr = 0
    init_pesq = 0
    after_pesq = 0
    init_stoi = 0
    after_stoi = 0

    for idx, batch in enumerate(tqdm(loader)):
        # Get file in queue
        mix, isolated_sources, target = batch
        delay = pixel_to_delay.get_delay((target[0].item(), target[1].item()))
        audio_man.set_target(delay)
        output, target = audio_man.iros_test(mix, isolated_sources[0,0])

        if save:
            torchaudio.save("input.wav", mix[0].cpu(), 16000)
            torchaudio.save("output.wav", output.cpu(), 16000)
            torchaudio.save("target.wav", target.cpu(), 16000)

        mix = mix[:,0]
        isolated_sources = isolated_sources[:,0,0]
        target = target[0][None, ...]
        init_si_sdr += scale_invariant_signal_distortion_ratio(mix, isolated_sources).item()
        after_si_sdr += scale_invariant_signal_distortion_ratio(output, target).item()
        init_snr += si_snr(mix, isolated_sources)
        after_snr += si_snr(output, target)
        init_pesq += perceptual_evaluation_speech_quality(mix, isolated_sources, 16000, 'wb').item()
        after_pesq += perceptual_evaluation_speech_quality(output, target, 16000, 'wb').item()
        init_stoi += short_time_objective_intelligibility(mix, isolated_sources, 16000).item()
        after_stoi += short_time_objective_intelligibility(output, target, 16000).item()

    print(f"Initial SI-SDR: {init_si_sdr/nb_of_iterations}")
    print(f"After SI-SDR: {after_si_sdr/nb_of_iterations}")
    print(f"Initial SNR: {init_snr/nb_of_iterations}")
    print(f"After SNR: {after_snr/nb_of_iterations}")
    print(f"Initial PESQ: {init_pesq/nb_of_iterations}")
    print(f"After PESQ: {after_pesq/nb_of_iterations}")
    print(f"Initial STOI: {init_stoi/nb_of_iterations}")
    print(f"After STOI: {after_stoi/nb_of_iterations}")

def run_loop_mvdr(loader, save=False):

    stft = torchaudio.transforms.Spectrogram(
        n_fft=512,
        hop_length=256,
        power=None,
        window_fn=cuda_sqrt_hann_window
    )
    istft = torchaudio.transforms.InverseSpectrogram(
        n_fft=512, hop_length=256, window_fn=cuda_sqrt_hann_window
    )
    SCM = torchaudio.transforms.PSD(normalize=False)
    mvdr_transform = torchaudio.transforms.SoudenMVDR()

    # Run audio manager
    audio_man = AudioManager(debug=False, mask=False, use_timers=False, save_output=True)
    pixel_to_delay = Pixel2Delay((480, 640), "./calibration_8mics.json", device=device)
    audio_man.iros_init()
    nb_of_iterations = len(loader.dataset)
    init_si_sdr = 0
    after_si_sdr = 0
    init_snr = 0
    after_snr = 0
    init_pesq = 0
    after_pesq = 0
    init_stoi = 0
    after_stoi = 0

    for idx, batch in enumerate(tqdm(loader)):
        # Get file in queue
        mix, isolated_sources, target = batch
        delay = pixel_to_delay.get_delay((target[0].item(), target[1].item()))
        audio_man.set_target(delay)
        speech_mask, noise_mask = audio_man.iros_test_mvdr(mix)

        MIX = stft(mix)[0]
        scm_speech = SCM(MIX, speech_mask)
        scm_noise = SCM(MIX, noise_mask)

        MVDR = mvdr_transform(MIX, scm_speech, scm_noise, reference_channel=0)

        output = istft(MVDR)[None, ...]
        target = isolated_sources[0,0]

        if save:
            torchaudio.save("input.wav", mix[0].cpu(), 16000)
            torchaudio.save("output.wav", output.cpu(), 16000)
            torchaudio.save("target.wav", target.cpu(), 16000)

        mix = mix[:,0]
        isolated_sources = isolated_sources[:,0,0]
        target = target[0][None, ...]
        init_si_sdr += scale_invariant_signal_distortion_ratio(mix, isolated_sources).item()
        after_si_sdr += scale_invariant_signal_distortion_ratio(output, target).item()
        init_snr += si_snr(mix, isolated_sources)
        after_snr += si_snr(output, target)
        init_pesq += perceptual_evaluation_speech_quality(mix, isolated_sources, 16000, 'wb').item()
        after_pesq += perceptual_evaluation_speech_quality(output, target, 16000, 'wb').item()
        init_stoi += short_time_objective_intelligibility(mix, isolated_sources, 16000).item()
        after_stoi += short_time_objective_intelligibility(output, target, 16000).item()

    print(f"Initial SI-SDR: {init_si_sdr/nb_of_iterations}")
    print(f"After SI-SDR: {after_si_sdr/nb_of_iterations}")
    print(f"Initial SNR: {init_snr/nb_of_iterations}")
    print(f"After SNR: {after_snr/nb_of_iterations}")
    print(f"Initial PESQ: {init_pesq/nb_of_iterations}")
    print(f"After PESQ: {after_pesq/nb_of_iterations}")
    print(f"Initial STOI: {init_stoi/nb_of_iterations}")
    print(f"After STOI: {after_stoi/nb_of_iterations}")
    


def main(loader, WORKERS, dir):
    print("Applying mask")
    run_loop_mvdr(loader)
     


if __name__ == "__main__":
    dataset = IROSDataset(path, frame_size, hop_size, forceCPU=False)
    loader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=1)

    main(loader, 8, "/home/jacob/dev/RAVE/library/RAVE/src/results")



