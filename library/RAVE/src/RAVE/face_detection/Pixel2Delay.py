import numpy as np
import json
from itertools import product


class Pixel2Delay:
    """
    To do obtain a delay of a microphone array from a pixel. Prior to using
    this class a calibration needs to be done between the camera and the
    microphone array. The resulting calibration matrix path should be sent
    to this class.

    Args:
         img_shape (Tuple):
            The shape of the images acquired by the camera, should be in format
            (height, width)
         calibration_file (string):
            Path of the calibration matrix ex: "./calibration.json"

    Attributes:
        _calibration_matrix (ndarray):
            Matrix obtain from the calibration
        _order_of_polynomial (int):
            Order of the polynomial used in the calibration
        _channels (int):
            Number of microphones of the microphone array
        _upper_triangle_indices (ndarray):
            Used to transform the delay from (Pairs,) to
            (Nb_of_mics, Nb_of_mics)
        _lower_triangle_indices (ndarray):
            Used to transform the delay from (Pairs,) to
            (Nb_of_mics, Nb_of_mics)
        _delay (ndarray):
            The target delay for the audio
        _width (int):
            Width of the image acquired by the camera
        _height (int):
            Height of the image acquired by the camera

    """

    def __init__(self, img_shape, calibration_file):
        self._calibration_matrix = None
        self._order_of_polynomial = None
        self._channels = None
        self._upper_triangle_indices = None
        self._lower_triangle_indices = None
        self._delay = None
        self._exponents = None
        self._height = img_shape[0]
        self._width = img_shape[1]
        self.import_calibration_matrix(calibration_file)

    def import_calibration_matrix(self, calibration_file):
        """
        Sets the calibration matrix to the one provided by the file and updates
        the rest of the parameters

        Args:
            calibration_file (string):
                Path of the calibration matrix ex: "./calibration.json"

        """
        with open(calibration_file, "r", encoding="utf-8") as f:
            self._calibration_matrix = np.array(json.load(f))

        # Get the order of the polynomial and the channels from the shape of
        # the calibration matrix
        self._order_of_polynomial = int(
            np.sqrt(self._calibration_matrix.shape[0]) - 1
        )
        self._channels = int(
            np.roots([1, -1, -2 * self._calibration_matrix.shape[1]])[0]
        )

        # init indices to assign directly in the delay
        self._upper_triangle_indices = np.triu_indices(self._channels, k=1)
        self._lower_triangle_indices = np.tril_indices(self._channels, k=-1)

        # init delay to zeros
        self._delay = np.zeros(
            (self._channels, self._channels), dtype=np.float32
        )

        # Used to reconstruct the delay from a pixel
        self._exponents = np.array(
            [
                [x, y]
                for x, y in product(
                    range(self._order_of_polynomial + 1), repeat=2
                )
            ]
        )

    def get_delay(self, pixel_coords):
        """
        Does the transformation between a pixel and a delay of the microphone
        array

        Args:
            pixel_coords (Tuple): Pixel coordinates in (x,y) of the target

        Returns:
            (ndarray) The delay between the microphones
            with shape (Nb_of_mics, Nb_of_mics)

        """
        normalized_mouth_x = (2 * pixel_coords[0] - self._width - 1) / (
            self._width - 1
        )
        normalized_mouth_y = (2 * pixel_coords[1] - self._height - 1) / (
            self._height - 1
        )

        A = (
            normalized_mouth_x ** self._exponents[:, 0][:, np.newaxis]
            * normalized_mouth_y ** self._exponents[:, 1][:, np.newaxis]
        ).T

        delay_pairs = A @ self._calibration_matrix
        self._delay[
            self._upper_triangle_indices[0], self._upper_triangle_indices[1]
        ] = delay_pairs
        self._delay[
            self._lower_triangle_indices[0], self._lower_triangle_indices[1]
        ] = (-1 * delay_pairs)

        return self._delay
