import cv2

from models import haar
from resources.fpsHelper import FPS


def image_detect(detect_func, path="resources/TonyFace1/jpg"):
    frame = cv2.imread(path)
    final_frame = detect_func(frame)
    cv2.imshow("Final", final_frame)
    cv2.waitKey(0)


def stream_detect(detect_func):
    cap = cv2.VideoCapture(0)

    fps = FPS()
    while True:
        _, frame = cap.read()
        fps.incrementFrameCount()
        final_frame = detect_func(frame)
        final_frame = fps.writeFpsToFrame(final_frame)

        cv2.imshow("Final", final_frame)

        # Stop if escape key is pressed
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    # Test with image
    # image_detect("resources/TonyFace1.jpg")

    # Test with video stream haar
    stream_detect(haar.detect_faces)



