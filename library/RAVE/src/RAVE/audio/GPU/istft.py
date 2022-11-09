import torch

from pyodas.core import Module
from .windows import windows


class IStft(Module):
    """
    This module does the inverse short time Fourier transform of the signal

    Args:
        channels (int):
            The number of channels or microphones of the input
            signal
        frame_size (int):
            The length of the input signal in the time domain.
        hop_size (int):
            The length of the shift applied to insert the next signal.
        window (str | ndarray):
            A window used to attenuate discontinuity. If a string is passed the
            class will look for one of the windows in the windows dictionary
            from pyodas.utils. If an ndarray is passed, it uses it as the
            window if it's the shape (channels, frame_size).

    Raises:
        ValueError:
            If the string is not found in the dictionary or if the window
            passed is not the right shape
        TypeError:
            If the window is not the right type

    Example:
        Doing the istft on a signal of ones of length 257 with two channels::

            >>> import numpy as np

            >>> from pyodas.core import IStft
            >>> from pyodas.utils import TYPES

            >>> frame_size = 512
            >>> hop_size = frame_size // 4
            >>> channels = 2
            >>> istft = IStft(channels, frame_size, hop_size, "hann")
            >>> freq_sig = np.array(np.ones((channels, 257), dtype=TYPES.FREQ))
            >>> time_sig = istft(freq_sig)
    """

    def __init__(self, channels, frame_size, hop_size, window, device):
        if frame_size % 2 != 0:
            frame_size += 1
        self._frame_size = frame_size
        self._channels = channels
        self._xs = torch.zeros((self._channels, frame_size), device=device, dtype=torch.float32)
        self._hop_size = hop_size
        self.device = device

        # Get the window
        if isinstance(window, str):
            try:
                window_func = windows[window]
                self._ws = torch.tile(window_func(frame_size, device), (self._channels, 1))
            except KeyError:
                raise ValueError(f"The window {window} is not in the list" f" of supported windows")
        elif isinstance(window, torch.Tensor):
            if window.shape == (self._channels, frame_size):
                self._ws = window
            else:
                raise ValueError(
                    f"Window of shape {window.shape} is not the expected shape" f" {(channels, frame_size)}"
                )
        else:
            raise TypeError("window argument is not a string or an ndarray")

    def __call__(self, new_Xs):
        """
        Applies the inverse short time Fourier transform.

        Args:
            new_Xs (complex ndarray):
                Input data of the istft in the frequency domain with shape
                (channels, bins).

        Returns:
            ndarray:
                Output data of the istft in the time domain with shape
                (channels, hop_size).
        """
        self._xs += torch.fft.irfft(new_Xs, n=self._frame_size) * self._ws

        output = self._xs[:, 0 : self._hop_size]

        self._xs = torch.roll(self._xs, shifts=-1 * self._hop_size, dims=1)
        self._xs[:, -1 * self._hop_size :] = 0

        return output

    def reset(self):
        """
        This resets the memory of the istft. This can be used in a
        context where you are using this recursively.
        """
        self._xs = torch.zeros((self._channels, self._frame_size), device=self.device, dtype=torch.float32)
