import numpy as np


def box_smooth(y, nb_pts):
    """Applies a box (mean) filter to a signal
        Taken from https://stackoverflow.com/a/26337730

    Args:
        y (np array): The signal to filter
        box_pts (int): Size of the filter (number of element that should be
            considered for a given output point)

    Returns:
        np array: The filtered signal
    """
    box = np.ones(nb_pts) / nb_pts
    y_smooth = np.convolve(y, box, mode="same")
    return y_smooth
