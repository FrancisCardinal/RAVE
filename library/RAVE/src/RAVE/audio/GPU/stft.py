import torch

from pyodas.core import Module
from .windows import windows


class Stft(Module):
    """
    This module does a short time Fourier transform of the signal

    Args:
        channels (int):
            The number of channels or microphones of the input
            signal
        frame_size (int):
            The length of the segment on which the operations will be performed
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
        Doing the stft on a signal of ones of length 128 with two channels::

            >>> import numpy as np

            >>> from pyodas.core import Stft
            >>> from pyodas.utils import TYPES

            >>> frame_size = 512
            >>> chunk_size = frame_size // 4
            >>> channels = 1
            >>> stft = Stft(channels, frame_size, "hann")
            >>> time_sig = np.ones((channels, chunk_size), dtype=TYPES.TIME)
            >>> freq_sig = stft(time_sig)
    """

    def __init__(self, channels, frame_size, window, device):
        self._channels = channels
        self._frame_size = frame_size
        self.device = device
        self._xs = torch.zeros((self._channels, self._frame_size), device=device, dtype=torch.float32)

        # Get the window
        if isinstance(window, str):
            try:
                window_func = windows[window]
                self._ws = torch.tile(window_func(frame_size, device), (channels, 1))
            except KeyError:
                raise ValueError(f"The window {window} is not in the list" f" of supported windows")
        elif isinstance(window, torch.Tensor):
            if window.shape == (channels, frame_size):
                self._ws = window
            else:
                raise ValueError(
                    f"Window of shape {window.shape} is not the expected shape" f" {(channels, frame_size)}"
                )
        else:
            raise TypeError("window argument is not a string or an ndarray")

    def __call__(self, new_xs):
        """
        Applies the short time Fourier transform

        Args:
            new_xs (ndarray):
                Input data of the stft in the time domain with shape
                (channels, chunk_size).

        Returns:
            complex ndarray:
                Output data of the stft in the frequency domain with shape
                (channels, bins).


        """
        self._xs = torch.roll(self._xs, shifts=-1 * new_xs.shape[1], dims=1)
        self._xs[
            :,
            self._xs.shape[1] - new_xs.shape[1] : self._xs.shape[1],
        ] = new_xs

        return torch.fft.rfft(self._xs * self._ws)

    def reset(self):
        """
        This resets the memory of the stft. This can be used in a
        context where you are using this recursively.
        """
        self._xs = torch.zeros((self._channels, self._frame_size), device=self.device, dtype=torch.float32)
