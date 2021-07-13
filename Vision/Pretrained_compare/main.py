import cv2

from models import haar

if __name__ == '__main__':
    frame = cv2.imread("resources/TonyFace1.jpg")
    final_frame = haar.detect_faces(frame)
    cv2.imshow("Final", final_frame)
    cv2.waitKey(0)

