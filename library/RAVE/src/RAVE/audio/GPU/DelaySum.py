import torch

from pyodas.core import Module


class DelaySum(Module):
    """
    Delay signals to have them aligned, once aligned add them together.

    Args:
        frame_size (int): The frame size of the signal

    Example:
        ::

            >>> import numpy as np

            >>> from pyodas.core import DelaySum
            >>> from pyodas.utils import TYPES

            # Declare a signal
            >>> x = np.array(
            ...     [
            ...         [1, 2, 3, 4, 5, 6],
            ...         [6, 1, 2, 3, 4, 5],
            ...     ],
            ...     dtype=TYPES.TIME,
            ... )

            # Get the Fourier transform of the signal
            >>> X = np.fft.rfft(x).astype(TYPES.FREQ)

            # Declare the DelaySum object and the delay. The delay is usually
            # obtain from the GccPhat object.
            >>> delay_and_sum = DelaySum(x.shape[1])
            >>> delay = np.array([[0, -1], [1, 0]])

            # Apply the delay to the signal
            >>> summation = delay_and_sum(X, delay)
            >>> result = np.fft.irfft(summation)

            >>> print(np.round_(result, 3))
            [[1. 2. 3. 4. 5. 6.]]
    """

    def __init__(self, frame_size):
        nb_of_bins = int(frame_size / 2) + 1
        self._bins = torch.arange(nb_of_bins, dtype=int).reshape(nb_of_bins, 1)
        self._frame_size = frame_size

    def __call__(self, freq_signal, delay):
        """
        Apply the delay to align the signals and add them

        Args:
            freq_signal (complex ndarray):
                The signal in the frequency domain with shape (channels, bins)
            delay (ndarray):
                Delay array of shape (nb_of_channels, nb_of_channels)

        Returns:
            complex ndarray:
                The signal delayed and summed, the output will only have one
                channel. The shape will then be (1, bins)
        """
        # Get the delay in frequency and normalize by the number of channels
        freq_delay = torch.exp((-2j * torch.pi * self._bins * delay[0, :] / self._frame_size)).T / freq_signal.shape[0]

        # Delay the channels and sum them together
        return torch.einsum("ij,ij->j", freq_signal, freq_delay)[None, :]
