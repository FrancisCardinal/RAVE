import torch

from pyodas.utils import StaticDict


def hann(size, device="cpu"):
    """
    Helper method for creating Hann windows

    Args:
        size (int): size of the Hann window

    Returns:
        np.ndarray:
            Hann window in an array of float 32 the length of the given size.

    """
    return torch.hann_window(size, device=device, dtype=torch.float32)


def sqrt_hann(size, device="cpu"):
    """
    Helper method for creating square root Hann windows

    Args:
        size (int): size of the square root Hann window

    Returns:
        np.ndarray:
            square root Hann window in an array of float 32 the length of
            the given size.

    """
    return torch.sqrt(torch.hann_window(size, device=device, dtype=torch.float32))


windows = StaticDict(
    {
        "hann": hann,
        "sqrt_hann": sqrt_hann,
    }
)
