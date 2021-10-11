import numpy as np
from numpy import sin, cos


class NormalizedEllipse:
    """Represents an ellipse with five parameters. h and k are the center position, a and b 
       are the horizontal and vertical axis length and theta is the rotation angle of the 
       ellipse relative to the x axis. The ellipse is normalized. 

       A 'normalized' ellipse is one where each parameters is no longer represented in pixels values. Each parameter is rather normalized
       between 0 and 1, where 1 represents the max pixel value of the corresponding axis of the parameter (or 2*pi radians for theta). 
       For example, if the h parameter was 240 and the image width is 480, then the new h value is 0.5 
    """

    def __init__(self, h,
                 k,
                 a,
                 b,
                 theta):
        """Constructor of the NormalizedEllipse class

        Args:
            h (float): x coordinate of the center of the ellipse (normalized)
            k (float): y coordinate of the center of the ellipse (normalized)
            a (float): length of the horizontal axis (normalized)
            b (float): length of the vertical axis (normalized)
            theta (float): rotation angle of the ellipse relative to the x axis (normalized)
        """
        self.h = h
        self.k = k
        self.a = a
        self.b = b
        self.theta = theta

    def rotate_around_image_center(self, phi):
        """Rotates a normalized ellipse around the image center point

        Args:
            phi (float): Angle of rotation (in rads)
        """
        h_relative_to_center = self.h - 0.5
        k_relative_to_center = self.k - 0.5

        self.h = h_relative_to_center * \
            cos(phi) - k_relative_to_center*sin(phi) + 0.5
        self.k = h_relative_to_center * \
            sin(phi) + k_relative_to_center*cos(phi) + 0.5

        self.theta += phi/(2*np.pi)

    def to_list(self):
        """Serializes the ellipse to a list

        Returns:
            List: The serialized ellipse
        """
        return [self.h, self.k, self.a, self.b, self.theta]

    @staticmethod
    def get_normalized_ellipse_from_list(list):
        """Creates a normalized ellipse object from a list 

        Args:
            list (List): List of the ellipse's parameters

        Returns:
            NormalizedEllipse: The normalized ellipse object
        """
        return NormalizedEllipse(list[0], list[1], list[2], list[3], list[4])

    @staticmethod
    def get_normalized_ellipse_from_opencv_ellipse(center_x,
                                                   ellipse_width,
                                                   center_y,
                                                   ellipse_height,
                                                   angle,
                                                   INPUT_IMAGE_WIDTH,
                                                   INPUT_IMAGE_HEIGHT):
        """Computes a normalized ellipse from an ellipse that is in the opencv format. 

        Args:
            center_x (int): x coordinate of the center of the ellipse (in pixels)
            ellipse_width (int): length of the horizontal axis (in pixels)
            center_y (int): y coordinate of the center of the ellipse (in pixels)
            ellipse_height (int): length of the vertical axis (in pixels)
            angle (int): rotation angle of the ellipse relative to the x axis (in pixels)
            INPUT_IMAGE_WIDTH (int): width of the input image (in pixels)
            INPUT_IMAGE_HEIGHT (int): height of the input image (in pixels)

        Returns:
            NormalizedEllipse: The normalized ellipse
        """
        h, k = center_x/INPUT_IMAGE_WIDTH, center_y/INPUT_IMAGE_HEIGHT
        a, b = ellipse_width / \
            (2*INPUT_IMAGE_WIDTH), ellipse_height/(2*INPUT_IMAGE_HEIGHT)
        theta = np.deg2rad(angle)/(2*np.pi)

        return NormalizedEllipse(h, k, a, b, theta)
