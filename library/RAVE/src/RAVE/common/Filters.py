import numpy as np


def box_smooth(y, box_pts):
    # https://stackoverflow.com/a/26337730
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode="same")
    return y_smooth
