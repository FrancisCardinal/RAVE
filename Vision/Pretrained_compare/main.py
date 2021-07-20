import cv2
import glob
import os

from resources.fpsHelper import FPS
from models import haar, hog, dnn


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


def feed_images_detect(detect_func, in_folder="images_in/", out_folder="images_out/"):

    if not os.path.exists(out_folder):
        # Create new folder
        os.mkdir(out_folder)
    else:
        # Clear out folder
        for old_img in glob.glob(out_folder + "*"):
            os.remove(old_img)

    # Load, detect and save images in input folder
    for img in glob.glob(in_folder + "*"):
        final_img = detect_func(cv2.imread(img))
        cv2.imwrite(out_folder + os.path.basename(img), final_img)


if __name__ == '__main__':

    # Test with image
    # image_detect(haar.detect_faces, "images_in/16033.png")

    # Test with video stream
    # stream_detect(haar.detect_faces)

    # Feed multiple images to detect
    feed_images_detect(dnn.detect_faces, out_folder="images_out/images_out_dnn/")
    print("Done dnn")
    feed_images_detect(haar.detect_faces, out_folder="images_out/images_out_haar/")
    print("Done haar")
    feed_images_detect(hog.detect_faces, out_folder="images_out/images_out_hog/")
    print("Done hog")




