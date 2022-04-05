import pickle
import cv2

from RAVE.face_detection.Direction2Pixel import (
    Direction2Pixel,
    Direction2PixelFrom3D,
)

if __name__ == "__main__":

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
