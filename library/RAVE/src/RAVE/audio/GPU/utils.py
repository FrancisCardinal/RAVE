import torch
import numpy as np
from itertools import combinations, repeat
from pyodas.utils import CONST


def get_delays_based_on_mic_array(directions, mic_array, frame_size, sound_vel=340, device="cpu"):
    """
    Compute the theoretical delays between each microphone pair
    for each point passed.

    Args:
        directions (ndarray):
            Directions (x,y,z) in 3D with origin being the same as the
            microphone array with shape (nb_of_directions, 3)
        mic_array (MicArray):
            The configuration of the microphone array.
        frame_size (int):
            The frame size of the Fourier transform.
        sound_vel (float):
            The sound velocity of the environment in which the audio will be
            captured.

    Returns:
        (ndarray):
            The theoretical delays between each microphone pairs for each
            point with shape (nb_of_directions, channels, channels)
    """
    mic_array = torch.tensor(np.array(list(mic_array.mics.values())), dtype=torch.float32).to(device)

    delays = torch.tensor(
        np.array(list(map(_get_delay_based_on_mic_array, directions, repeat(mic_array), repeat(device))))
    ).to(device)

    delays *= CONST.SAMPLING_RATE / sound_vel

    # Make sure the delays are in the right range
    signed_delay = delays.detach().clone()
    delays = torch.abs(signed_delay)
    delays = delays % (frame_size / 2)
    delays = torch.copysign(delays, signed_delay)

    return delays


def _get_delay_based_on_mic_array(direction, mic_array, device):
    """
    Compute theoretical delay for a single direction.

    Args:
        direction (ndarray):
            The potential direction of arrival in (x, y, z)
        mic_array (ndarray):
            The configuration of the microphone array with shape
            (nb_of_mics, position)

    Returns:
        (ndarray):
            The theoretical delays between each microphone pairs with shape
            (channels, channels)

    """
    print(f"DEVICE: {device}")
    channels = mic_array.shape[0]
    delay = torch.zeros((channels, channels), dtype=torch.float32).to(device)
    for i, j in combinations(range(channels), 2):
        delay[i, j] = torch.dot(mic_array[j] - mic_array[i], direction)

    # Copy upper triangle in lower triangle
    delay[torch.tril_indices(channels, channels, -1)] = -1 * delay[torch.triu_indices(channels, channels, 1)]

    return delay.cpu()
