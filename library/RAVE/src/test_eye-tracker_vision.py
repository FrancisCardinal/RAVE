import pickle
import cv2
import numpy as np

from RAVE.face_detection.Direction2Pixel import (
    Direction2Pixel,
    Direction2PixelFrom3D,
)


def main():
    angle_x = pickle.load(open("Xs.bin", "rb"))
    angle_y = pickle.load(open("Ys.bin", "rb"))

    video = cv2.VideoCapture("head_camera.avi")

    counter = 0

    algo_article = Direction2Pixel(31, 40)
    algo_3D = Direction2PixelFrom3D()

    while True:
        succes, frame = video.read()

        if not succes:
            break

        point1 = algo_3D.get_pixel(angle_x[counter], angle_y[counter], 1)
        point2 = algo_3D.get_pixel(angle_x[counter], angle_y[counter], 5)
        # point1 = algo_3D.get_pixel(-5, -20, 1)
        # point2 = algo_3D.get_pixel(-5, -20, 5)
        point3 = algo_3D.get_pixel(0, 0, 1)
        # point = algo_3D.get_pixel(angle_x[counter], angle_y[counter], 1.41)
        # point = algo_article.get_pixel(angle_x[counter], angle_y[counter])

        cv2.line(frame, point1, point2, color=(0, 0, 255), thickness=2)
        # cv2.drawMarker(frame, point1, color=(0, 0, 255), thickness=2)
        cv2.drawMarker(frame, point3, color=(255, 0, 0), thickness=2)

        rectangle_color = (0, 0, 255)
        rectangle_top_left_corner = (250, 180)
        rectangle_bottom_right_corner = (360, 280)

        if algo_3D.is_line_segment_in_rectangle(
            point1,
            point2,
            rectangle_top_left_corner,
            rectangle_bottom_right_corner,
        ):
            rectangle_color = (0, 255, 0)
        cv2.rectangle(
            frame,
            rectangle_top_left_corner,
            rectangle_bottom_right_corner,
            color=rectangle_color,
            thickness=2,
        )

        cv2.imshow("Facial camera", frame)
        cv2.waitKey(10)
        counter += 1

    print("done!")


if __name__ == "__main__":
    algo_3D = Direction2PixelFrom3D()
    unit_circle = np.linspace(0, 2 * np.pi, 360)
    x_coordinates = np.cos(unit_circle)
    y_coordinates = np.sin(unit_circle)

    x_angles = np.arctan2(x_coordinates, 3)
    y_angles = np.arctan2(y_coordinates, 3)

    x_angles, y_angles = np.rad2deg(x_angles), np.rad2deg(y_angles)

    for x_angle, y_angle in zip(x_angles, y_angles):
        point1 = algo_3D.get_pixel(x_angle, y_angle, 1)
        point2 = algo_3D.get_pixel(x_angle, y_angle, 5)

        frame = np.zeros((480, 640, 3))
        cv2.line(frame, point1, point2, color=(0, 0, 255), thickness=2)
        cv2.drawMarker(frame, point1, color=(255, 0, 0), thickness=2)
        cv2.drawMarker(frame, point2, color=(0, 255, 0), thickness=2)
        cv2.drawMarker(frame, (320, 240), color=(255, 255, 255), thickness=2)

        cv2.imshow("Test", frame)
        cv2.waitKey(10)
