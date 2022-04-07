import numpy as np
from math import copysign


class Direction2Pixel:
    """
    To convert an eye direction to a pixel on the facial camera

    Params:
        x_offset (float):
            Translation in x in pixels between the eye axis and the camera axis
        y_offset (float):
            Translation in y in pixels between the eye axis and the camera axis
        max_x_angle (float):
            Max angle from right to left in degrees
        max_y_angle (float):
            Max angle from up to bottom in degrees
        img_heigth (int):
            image heigth in pixels
        img_width (int):
            image width in pixels
    """

    def __init__(
        self,
        x_offset,
        y_offset,
        max_x_angle=110,
        max_y_angle=82,
        img_height=480,
        img_width=640,
    ):
        self._center_x = img_width / 2
        self._center_y = img_height / 2
        self._u_x = max_x_angle / 2
        self._u_y = max_y_angle / 2
        self._u_x_squared = self._u_x ** 2
        self._u_y_squared = (max_y_angle / 2) ** 2
        self._pixel_radius_y_squared = (img_height / 2) ** 2
        self._x_offset = x_offset
        self._y_offset = y_offset

        self._eps = 1e-20

    def get_pixel(self, angle_x, angle_y):
        sign_x = copysign(1, angle_x)
        opposite_sign_y = -1 * copysign(1, angle_y)

        # Convert to abs
        angle_x = sign_x * angle_x
        angle_y = -1 * opposite_sign_y * angle_y

        if angle_x > self._u_x:
            angle_x = self._u_x
            print(
                "Warning: clipping in x when converting direction to a pixel"
            )

        if angle_y > self._u_y:
            angle_y = self._u_y
            print(
                "Warning: clipping in y when converting direction to a pixel"
            )

        y = np.sqrt(
            (
                self._pixel_radius_y_squared
                * (self._u_x_squared - (angle_x ** 2))
            )
            / (
                (
                    (self._u_y_squared * self._u_x_squared)
                    / (angle_y ** 2 + self._eps)
                )
                + (angle_x ** 2)
            )
        )

        x = (
            angle_x
            * self._center_x
            * np.sqrt(1 - (y ** 2 / self._pixel_radius_y_squared))
        ) / self._u_x

        pixel_x = self._center_x + (sign_x * (x + sign_x * self._x_offset))

        pixel_y = self._center_y + (
            opposite_sign_y * (y + opposite_sign_y * self._y_offset)
        )

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


class Direction2PixelFrom3D:
    def __init__(
        self,
        img_height=480,
        img_width=640,
    ):
        self._center_x = img_width / 2
        self._center_y = img_height / 2
        # self._x_offset = x_offset
        # self._y_offset = y_offset

        self._k_matrix = np.array(
            [
                [376.96798, 0.0, 314.09011],
                [0.0, 374.08737, 250.37452],
                [0.0, 0.0, 1.0],
            ]
        )
        self._translation_eye_camera = np.array([0.13, 0.092, 0])
        # self._translation_eye_camera = np.array([0.1246, 0.07485, 0])
        self._distortion = np.array([-0.321459, 0.073634])

        self._eps = 1e-20

    def get_pixel(self, angle_x, angle_y, dist):
        angle_x = np.deg2rad(90 + angle_x)
        angle_y = np.deg2rad(90 + angle_y)

        x_eye = dist * np.sin(angle_y) * np.cos(angle_x)
        y_eye = dist * np.cos(angle_y)
        z_eye = dist * np.sin(angle_y) * np.sin(angle_x)

        x_cam = x_eye + self._translation_eye_camera[0]
        y_cam = y_eye + self._translation_eye_camera[1]
        z_cam = z_eye + self._translation_eye_camera[2]

        a = x_cam / z_cam
        b = y_cam / z_cam

        r = np.sqrt(a ** 2 + b ** 2)
        theta = np.arctan(r)

        theta_distorted = theta * (
            1
            + (self._distortion[0] * (theta ** 2))
            + (self._distortion[1] * (theta ** 4))
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
        a, b = self.get_line_from_2_points(point1, point2)

        if point1[0] >= point2[0]:
            x_range = np.arange(point2[0], point1[0])
        else:
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
        # TODO - JKealey handle when line is vertical or horizontal
        a = (point1[1] - point2[1]) / (point1[0] - point2[0])
        b = (point1[0] * point2[1] - point2[0] * point1[1]) / (
            point1[0] - point2[0]
        )

        return a, b


if __name__ == "__main__":
    converter = Direction2PixelFrom3D()

    # start = time.time()
    # it = 100000
    # for _ in range(it):
    #     x, y = converter.get_pixel(0, -20)
    # end_time = time.time()
    # print(f"it/s: {it/(end_time - start)}")
    # x_plot = np.zeros((480, 640))
    # x_plot += -10000
    #
    # for x_angle in range(-40, 40, 1):
    #     for y_angle in range(-25, 25, 1):
    #         x, y = converter.get_pixel(x_angle, y_angle)
    #         x = 639 if int(x) >= 640 else x
    #         y = 479 if int(y) >= 480 else y
    #
    #         x_plot[int(y), int(x)] = np.abs(x_angle)
    #
    #
    #
    # plt.imshow(x_plot, interpolation='nearest')
    # plt.show()
    x, y = converter.get_pixel(-25, 5.64, 1.77)
    print(f"Point: {x}, {y}")
