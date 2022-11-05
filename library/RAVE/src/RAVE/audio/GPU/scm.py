import torch

from pyodas.core import Module


class SpatialCov(Module):
    """
    Computation of the covariance between channels

    Args:
        channels (int): Number of channels
        frame_size (int): Size used for computing the Fourier transform.
        weight (float):
            The weight (Between 0 and 1) associated to how much the instant
            spatial covariance is taken into account. Only used when in
            accumulator mode (buffer_size<0).
        buffer_size (int):
            Size of the buffer for the moving average. If the buffer_size is
            zero or smaller an accumulator is used instead.

    Example:
        With an accumulator ::

            >>> import numpy as np

            >>> from pyodas.core import SpatialCov
            >>> from pyodas.utils import TYPES

            >>> X = np.array(
            ...     [[2 + 1j, 3 + 2j], [1 - 10j, 2 + 4j]],
            ...     dtype=TYPES.FREQ
            ... )
            >>> spatial_cov = SpatialCov(2, 2, weight=1)

            >>> SCM = spatial_cov(X)

            >>> print(f'Spatial covariance matrix: {SCM}')
            Spatial covariance matrix: [[[  5. +0.j  -8.+21.j]
              [ -8.-21.j 101. +0.j]]
            <BLANKLINE>
             [[ 13. +0.j  14. -8.j]
              [ 14. +8.j  20. +0.j]]]

        With a buffer ::

            >>> import numpy as np

            >>> from pyodas.core import SpatialCov
            >>> from pyodas.utils import TYPES

            >>> X = np.array(
            ...     [[2 + 1j, 3 + 2j], [1 - 10j, 2 + 4j]],
            ...     dtype=TYPES.FREQ
            ... )
            >>> spatial_cov = SpatialCov(2, 2, buffer_size=4)

            # fill the buffer
            >>> for _ in range(4):
            ...     SCM = spatial_cov(X)

            >>> print(f'Spatial covariance matrix: {SCM}')
            Spatial covariance matrix: [[[  5. +0.j  -8.+21.j]
              [ -8.-21.j 101. +0.j]]
            <BLANKLINE>
             [[ 13. +0.j  14. -8.j]
              [ 14. +8.j  20. +0.j]]]
    """

    def __init__(self, channels, frame_size, weight=0.3, buffer_size=-1):
        self._buffer_size = buffer_size
        self._weight = torch.tensor(weight)
        self._channels = channels
        self._frame_size = frame_size
        self._scm = torch.zeros(
            (self._frame_size // 2 + 1, self._channels, self._channels),
            dtype=torch.cfloat,
        )
        self._mask = torch.ones(self._frame_size // 2 + 1, dtype=torch.float32)

        if self._buffer_size > 0:
            self._buffer = torch.zeros(
                (
                    self._buffer_size,
                    self._frame_size // 2 + 1,
                    self._channels,
                    self._channels,
                ),
                dtype=torch.cfloat,
            )

    def __call__(self, freq_sig, mask=None):
        """
        Return the covariance of Xs.

        Args:
            freq_sig (complex ndarray):
                The signal to calculate the covariance from. Expected shape is
                (channels, bins).
            mask (ndarray):
                The mask to be applied on the covariance. Values have to be
                between 0 and 1. Expected shape is (bins) or (channels, bins).

        Returns:
            (ndarray):
                The covariance between channels of all frequency bins. The
                shape will be (bins, channels, channels).

        """
        if mask is not None:
            self._mask = mask

        if self._buffer_size > 0:
            self._moving_average(freq_sig)
        else:
            self._accumulator(freq_sig)

        return self._scm.detach().clone()

    def reset(self):
        """
        This resets the memory of the spatial covariance. This can be used in a
        context where you are using this recursively.
        """
        self._scm = torch.zeros(
            (self._frame_size // 2 + 1, self._channels, self._channels),
            dtype=torch.cfloat,
        )
        self._mask = torch.ones(self._frame_size // 2 + 1, dtype=torch.float32)

        if self._buffer_size > 0:
            self._buffer = torch.zeros(
                (
                    self._buffer_size,
                    self._frame_size // 2 + 1,
                    self._channels,
                    self._channels,
                ),
                dtype=torch.cfloat,
            )

    def _accumulator(self, Xs):
        """
        Computes the covariance with an accumulator.

        Args:
            Xs (complex ndarray):
                The signal to calculate the covariance from
        """
        self._scm *= 1.0 - self._weight

        # For each bin multiple Xs with the weight and the mask then
        # with the conjugate transpose of Xs
        self._scm += torch.einsum(
            "d...,e...->...de",
            torch.einsum(",...b,...b->...b", self._weight, self._mask, Xs),
            torch.conj(Xs),
        )

    def _moving_average(self, Xs):
        """
        Computes the covariance with a moving average.

        Args:
            Xs (complex ndarray):
                The signal to calculate the covariance from
        """
        self._buffer = torch.roll(self._buffer, 1, dims=0)

        # For each bin multiple Xs with the weight and the mask then
        # with the conjugate transpose of Xs
        instant_scm = (
            torch.einsum(
                "d...,e...->...de",
                torch.einsum("...b,...b->...b", self._mask, Xs),
                torch.conj(Xs),
            )
            / self._buffer_size
        )

        # Updating the scm of the buffer
        self._scm += instant_scm - self._buffer[0, :]

        # Storing the last scm
        self._buffer[0, :] = instant_scm
