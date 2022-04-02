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

        return (
            self._center_x + (sign_x * (x + sign_x * self._x_offset)),
            self._center_y
            + (opposite_sign_y * (y + opposite_sign_y * self._y_offset)),
        )


if __name__ == "__main__":
    converter = Direction2Pixel(-16, 21)

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
    x, y = converter.get_pixel(-15, 8.51)
    print(f"Point: {x}, {y}")
