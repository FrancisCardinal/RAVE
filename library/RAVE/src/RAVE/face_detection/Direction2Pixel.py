import numpy as np
import math


class Direction2Pixel:
    """
    To convert an eye direction to a pixel on the facial camera

    Params:
        translation_eye_camera (nd.array):
            x,y,z translation in meters between to eye and the camera
        img_heigth (int):
            image heigth in pixels
        img_width (int):
            image width in pixels
    """

    def __init__(
        self,
        translation_eye_camera=np.array([0, 0, 0]),
        img_height=480,
        img_width=640,
    ):
        self._center_x = img_width / 2
        self._center_y = img_height / 2

        self._k_matrix = np.array(
            [
                [376.96798, 0.0, 314.09011],
                [0.0, 374.08737, 250.37452],
                [0.0, 0.0, 1.0],
            ]
        )
        self._distortion = np.array([-0.321459, 0.073634])
        self._translation_eye_camera = translation_eye_camera
        self._eps = 1e-20

    def get_pixel(self, angle_x, angle_y, dist):
        """
        Methode to get the pixel from the x and y angles
        Args:
            angle_x (float): horizontal angle in degrees
            angle_y (float): vertical angle in degrees
            dist (float): the distance in meters between the eye and the object

        Returns:
            (pixel_x, pixel_y): the pixel values for the position in the image
        """
        angle_x = math.radians(90 - angle_x)
        angle_y = math.radians(90 + angle_y)

        x_eye = dist * math.sin(angle_y) * math.cos(angle_x)
        y_eye = dist * math.cos(angle_y)
        z_eye = dist * math.sin(angle_y) * math.sin(angle_x)

        x_cam = x_eye + self._translation_eye_camera[0]
        y_cam = y_eye + self._translation_eye_camera[1]
        z_cam = z_eye + self._translation_eye_camera[2]

        a = x_cam / z_cam
        b = y_cam / z_cam

        r = math.sqrt(a**2 + b**2)
        theta = math.atan(r)

        theta_distorted = theta * (
            1
            + (self._distortion[0] * (theta**2))
            + (self._distortion[1] * (theta**4))
        )

        x_cam = theta_distorted * a / (r + self._eps)
        y_cam = theta_distorted * b / (r + self._eps)

        pixel_x = self._k_matrix[0, 0] * x_cam + self._k_matrix[0, 2]
        pixel_y = self._k_matrix[1, 1] * y_cam + self._k_matrix[1, 2]

        if pixel_x > self._center_x * 2:
            pixel_x = self._center_x * 2
        elif pixel_x < 0:
            pixel_x = 0

        if pixel_y > self._center_y * 2:
            pixel_y = self._center_y * 2
        elif pixel_y < 0:
            pixel_y = 0

        return (
            int(pixel_x),
            int(pixel_y),
        )

    def is_line_segment_in_rectangle(
        self, point1, point2, top_left_corner, bottom_right_corner
    ):
        """
        Args:
            point1 (Tuple of int): First point of the line in pixels (x, y)
            point2 (Tuple of int): Second point of the line in pixels (x, y)
            top_left_corner (Tuple of int):
                Top left corner of the rectangle in pixels (x, y)
            bottom_right_corner (Tuple of int):
                Bottom right corner of the rectangle in pixels (x, y)

        Returns:
            bool:
                Whether the line is segment is in the rectangle.
        """
        a, b = self.get_line_from_2_points(point1, point2)

        x_range = np.arange(point2[0], point1[0])

        y_range = (a * x_range + b).astype(int)

        points_on_line = np.vstack((x_range, y_range))

        return self.are_points_in_rectangle(
            points_on_line, top_left_corner, bottom_right_corner
        )

    @staticmethod
    def are_points_in_rectangle(
        points,
        top_left_corner,
        bottom_right_corner,
    ):
        """

        Args:
            points (nd.array): array of points (x, y)
                        top_left_corner (Tuple of int):
            top_left_corner (Tuple of int):
                Top left corner of the rectangle in pixels (x, y)
            bottom_right_corner (Tuple of int):
                Bottom right corner of the rectangle in pixels (x, y)

        Returns:
            bool:
                Whether the points are in the rectangle.

        """
        condition1 = points[0] > top_left_corner[0]
        condition2 = points[0] < bottom_right_corner[0]
        condition3 = points[1] > top_left_corner[1]
        condition4 = points[1] < bottom_right_corner[1]
        answer = sum(
            np.logical_and.reduce(
                (condition1, condition2, condition3, condition4)
            )
        )

        return answer >= 1

    @staticmethod
    def get_line_from_2_points(point1, point2):
        """
        Args:
            point1 (Tuple of int): First point of the line in pixels (x, y)
            point2 (Tuple of int): Second point of the line in pixels (x, y)

        Returns:
            (a, b):
                The parameters for the line y = ax+b
        """
        # TODO - JKealey: handle when line is vertical or horizontal
        #  if necessary because it seems our lines will never be.
        a = (point1[1] - point2[1]) / (point1[0] - point2[0])
        b = (point1[0] * point2[1] - point2[0] * point1[1]) / (
            point1[0] - point2[0]
        )

        return a, b


if __name__ == "__main__":
    converter = Direction2Pixel()

    x, y = converter.get_pixel(-25, 5.64, 1.77)
    print(f"Point: {x}, {y}")
