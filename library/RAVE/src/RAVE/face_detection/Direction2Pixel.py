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
        k_matrix,
        roi,
        translation_eye_camera=np.array([0, 0, 0]),
        img_height=480,
        img_width=640,
    ):
        T_eye_camera = np.array(
            [
                [0, -1, 0, translation_eye_camera[0]],
                [0, 0, -1, translation_eye_camera[1]],
                [1, 0, 0, translation_eye_camera[2]],
                [0, 0, 0, 1],
            ]
        )
        K = np.r_[k_matrix, [np.array([0, 0, 0])]]
        K = np.c_[K, np.array([0, 0, 0, 1])]

        self._P = np.dot(K, T_eye_camera)

        self._roi = roi
        self._img_width = img_width
        self._img_height = img_height

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
        phi = math.radians(-angle_x)
        theta = math.radians(90 - angle_y)

        x_eye = dist * math.sin(theta) * math.cos(phi)
        y_eye = dist * math.sin(theta) * math.sin(phi)
        z_eye = dist * math.cos(theta)

        p_eye = np.array([x_eye, y_eye, z_eye, 1])
        p_eye = np.dot(self._P, p_eye.T)
        p_eye = p_eye / (p_eye[2])

        pixel_x = self._clamp_pixel(p_eye[0], self._img_width)
        pixel_y = self._clamp_pixel(p_eye[1], self._img_height)

        # Undistort adjustements
        if self._roi is not None:
            origin_x, origin_y, cropped_width, cropped_height = self._roi

            pixel_x -= origin_x
            pixel_y -= origin_y

            pixel_x *= self._img_width / cropped_width
            pixel_y *= self._img_height / cropped_height

            pixel_x = self._clamp_pixel(pixel_x, self._img_width)
            pixel_y = self._clamp_pixel(pixel_y, self._img_height)

        return (
            int(pixel_x),
            int(pixel_y),
        )

    def _clamp_pixel(self, pixel_value, max_value):
        pixel_out = pixel_value
        if pixel_value > max_value:
            pixel_out = max_value
        elif pixel_value < 0:
            pixel_out = 0

        return pixel_out

    def is_line_segment_in_rectangle(self, point1, point2, top_left_corner, bottom_right_corner):
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

        return self.are_points_in_rectangle(points_on_line, top_left_corner, bottom_right_corner)

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
        answer = sum(np.logical_and.reduce((condition1, condition2, condition3, condition4)))

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
        b = (point1[0] * point2[1] - point2[0] * point1[1]) / (point1[0] - point2[0])

        return a, b


if __name__ == "__main__":
    converter = Direction2Pixel()

    x, y = converter.get_pixel(-25, 5.64, 1.77)
    print(f"Point: {x}, {y}")
