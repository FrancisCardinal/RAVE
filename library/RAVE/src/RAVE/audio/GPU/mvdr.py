import torch

from pyodas.core import Module


class Mvdr(Module):
    """
    Minimum Variance Distortionless Response (MVDR). This can be computed
    with :math:`\\boldsymbol{u}` as the reference vector or with a transfert
    function :math:`\\boldsymbol{v}` and :math:`\\text{Z}` as the output signal

    With a reference vector:

    .. math::
        \\textbf{F}_{\\text{MVDR}}(f) = \\frac
            {
                {{\\bf{\\Phi}_{\\textbf{NN}}^{-1}}(f)
                {\\bf{\\Phi}_{\\textbf{XX}}}}(f)
            }
            {\\text{Trace}
                ({{{\\bf{\\Phi}_{\\textbf{NN}}^{-1}}(f)
                \\bf{\\Phi}_{\\textbf{XX}}}(f))}
            }
            \\boldsymbol{u}

    With a transfer function:

    .. math::
        \\textbf{F}_{\\text{MVDR}}(f) = \\frac
        {
            {{\\bf{\\Phi}_{\\textbf{NN}}^{-1}}(f)
            {\\boldsymbol{v}}(f)}
        }
        {
            {\\boldsymbol{v}^{\\mathsf{H}}}(f)
            {\\bf{\\Phi}_{\\textbf{NN}}^{-1}}(f)
            {\\boldsymbol{v}}(f)
        }

    Applying the beamforming vector:

    .. math::
        Z(f) = \\textbf{F}_{\\text{MVDR}}^{H}(f)\\textbf{Y}(f)

    Args:
        channels (int):
            Number of channels of the signal.
        reference (int):
            The channel selected to be the reference.
            (ex: 0 selects the first channel)
        type (str):
            Either "ref_vector" or "transfer_fct" to select the type of MVDR.
        epsilon (float):
            Small value to add to a division to make sure we never divide by
            zero.



    Examples:
        The results of the following examples are not something meaningful,
        it just shows how to use the Mvdr module.

        With a reference vector::

            >>> import numpy as np

            >>> from pyodas.core import Mvdr, SpatialCov
            >>> from pyodas.utils import TYPES

            >>> x = np.array(
            ...     [
            ...         [1, 2, 3, 4, 5, 6],
            ...         [6, 1, 2, 3, 4, 5],
            ...     ],
            ...     dtype=TYPES.TIME,
            ... )

            # Get the Fourier transform of the signal
            >>> X = np.fft.rfft(x).astype(TYPES.FREQ)
            >>> channels, bins = X.shape
            >>> spatial_cov = SpatialCov(channels, (bins - 1) * 2)
            >>> target_scm = spatial_cov(X)
            >>> noise_scm = np.tile(
            ...     np.array([[7, 2], [-2, 1]], dtype=TYPES.FREQ),
            ...     (bins, 1, 1),
            ... )

            >>> mvdr = Mvdr(channels)

            >>> res = mvdr(X, target_scm, noise_scm)

            >>> print(np.round_(res, 3))
            [[-21.   +0.j     -5.96 -0.688j  -3.382+0.751j  -3.   +0.j   ]]

        With a transfer function::

            >>> import numpy as np
            >>> from pyodas.core import Mvdr
            >>> from pyodas.utils import (
            ...     TYPES, anechoic_steering_vector, generate_mic_array
            ... )

            # Declare a random signal
            >>> x = np.array(
            ...      [
            ...          [1, 2, 3, 4, 5, 6],
            ...          [6, 1, 2, 3, 4, 5],
            ...      ],
            ...      dtype=TYPES.TIME,
            ... )

            # Get the Fourier transform of the signal
            >>> X = np.fft.rfft(x).astype(TYPES.FREQ)
            >>> channels, bins = X.shape

            # Declare a random spatial covariance matrix for the noise
            >>> noise_scm = np.tile(
            ...     np.array([[7, 2], [-2, 1]], dtype=TYPES.FREQ),
            ...     (bins, 1, 1),
            ... )

            # Compute the anechoic steering vector
            >>> target = np.array([0, 1, 0.25])
            >>> mic_array = generate_mic_array(
            ...     {
            ...         "mics": {
            ...             "0": [1, 2, 0],
            ...             "1": [-1, 2, 0],
            ...         },
            ...         "nb_of_channels": 2,
            ...     }
            ... )
            >>> transfer_function = anechoic_steering_vector(
            ...     target, mic_array, 2*(bins-1)
            ... )

            # Compute the MVDR
            >>> mvdr = Mvdr(channels, type="transfer_fct")
            >>> res = mvdr(X, transfer_function,noise_scm)

            >>> print(np.round_(res, 2))
            [[-21.  +0.j    -3.75-5.2j   -3.75-1.73j  -3.75+0.j  ]]

    Raises:
        ValueError:
            When the reference is equal or higher than the number of channels

    """

    def __init__(self, channels, reference=0, type="ref_vector", epsilon=1e-6):
        if reference >= channels:
            raise ValueError(
                f"The reference {reference} needs to be smaller " f"than the number of channels {channels}"
            )
        assert type in ["ref_vector", "transfer_fct"], (
            "Unknown type provided. Must be one of " "[``ref_vector``, ``transfer_fct``]."
        )
        self.type = type
        self._ref_vector = torch.zeros((channels, 1), dtype=torch.cfloat)
        self._ref_vector[reference] = 1
        self._epsilon = epsilon

    def __call__(self, *args):
        """
        Compute the Minimum Variance Distortionless Response (MVDR)
        beamforming vector and apply it to the signal.

        If used with a reference vector:

        Args:
            signal (ndarray):
                Signal in the frequency domain with shape (channels, bins) to
                apply the beamforming vector to.
            target_scm (ndarray):
                The target spatial covariance matrix with shape
                (bins, channels, channels)
            noise_scm (ndarray):
                The noise spatial covariance matrix with shape
                (bins, channels, channels)

        If used with a transfer function:

        Args:
            signal (ndarray):
                Signal in the frequency domain with shape (channels, bins) to
                apply the beamforming vector to.
            transfer_fct (ndarray):
                The acoustic transfer function with shape
                (channels, bins)
            noise_scm (ndarray):
                The noise spatial covariance matrix with shape
                (bins, channels, channels)
        """
        assert len(args) == 3, f"Wrong number of arguments passed {len(args)}, expected 3"

        if self.type == "ref_vector":
            signal, target_scm, noise_scm = args
            beamforming_vector = self._beamform_with_reference_vector(target_scm, noise_scm)
        elif self.type == "transfer_fct":
            signal, transfer_fct, noise_scm = args
            beamforming_vector = self._beamform_with_transfer_fct(transfer_fct, noise_scm)
        else:
            raise ValueError("Only got one argument expected 2 or 3")

        # Phase transform to keep 1st channel's phase constant
        conj_vector = torch.conj(beamforming_vector[:, 0] / (torch.abs(beamforming_vector[:, 0]) + self._epsilon))
        beamforming_vector = torch.einsum("ij,i->ij", beamforming_vector, conj_vector)

        # Apply beamforming vector to signal:
        # Z(f) = F(f)Y(f)
        return torch.einsum("fc,cf->f", beamforming_vector.conj(), signal)[None, :]

    def _beamform_with_reference_vector(self, target_scm, noise_scm):
        """
        Compute the Minimum Variance Distortionless Response (MVDR)
        beamforming vector.

        Args:
            target_scm (ndarray):
                The target spatial covariance matrix with shape
                (bins, channels, channels)
            noise_scm (ndarray):
                The noise spatial covariance matrix with shape
                (bins, channels, channels)

        Returns:
            (ndarray):
                The MVDR beamforming vector applied to the signal
                with shape (1, bins)
        """
        # To make sure the noise matrix is not singular
        noise_scm += self._epsilon * torch.eye(noise_scm.shape[1])

        # noise_scm.inv() @ target_scm
        numerator = torch.linalg.solve(noise_scm, target_scm)
        denominator = self.trace(numerator, axis1=1, axis2=2).reshape(-1, 1) + self._epsilon

        return torch.squeeze(numerator / denominator[:, None] @ self._ref_vector)

    def _beamform_with_transfer_fct(self, transfer_fct, noise_scm):
        """
        Compute the Minimum Variance Distortionless Response (MVDR)
        beamforming vector.

        Args:
            transfer_fct (ndarray):
                The acoustic transfer function with shape
                (channels, bins)
            noise_scm (ndarray):
                The noise spatial covariance matrix with shape
                (bins, channels, channels)

        Returns:
            (ndarray):
                The MVDR beamforming vector applied to the signal
                with shape (1, bins)
        """
        noise_scm += self._epsilon * torch.eye(noise_scm.shape[1])

        trans_fct_vector = torch.swapaxes(transfer_fct, 1, 0)
        numerator = torch.linalg.solve(noise_scm, trans_fct_vector)
        denominator = torch.einsum("...d,...d->...", torch.conj(trans_fct_vector), numerator) + self._epsilon

        return numerator / denominator[:, None]

    def trace(self, input, axis1=0, axis2=1):
        """
        >>> torch.__version__
        '1.9.0.dev20210222+cpu'
        >>> x = torch.arange(1., 10.).view(3, 3)
        >>> x
        tensor([[1., 2., 3.],
                [4., 5., 6.],
                [7., 8., 9.]])
        >>> torch.trace(x)
        tensor(15.)
        >>> torch.trace(x.view(1, 3, 3))
        Traceback (most recent call last):
        ...
        RuntimeError: trace: expected a matrix, but got tensor with dim 3
        >>> trace(x)
        tensor(15.)
        >>> trace(x.view(3, 3, 1), axis1=0, axis2=1)
        tensor([15.])
        >>> trace(x.view(1, 3, 3), axis1=2, axis2=1)
        tensor([15.])
        >>> trace(x.view(3, 1, 3), axis1=0, axis2=2)
        tensor([15.])
        """
        assert input.shape[axis1] == input.shape[axis2], input.shape

        shape = list(input.shape)
        strides = list(input.stride())
        strides[axis1] += strides[axis2]

        shape[axis2] = 1
        strides[axis2] = 0

        input = torch.as_strided(input, size=shape, stride=strides)
        return input.sum(dim=(axis1, axis2))
