import cv2
import time

import numpy as np

from face_detectors import DnnFaceDetector

# Simulate the output to be expected from the face tracker
# Send this information to the server


# Called when we want to send updated data to the server
def send_data(frame, face_bboxes):

    # frame: image where the faces were detected

    # faces_bbox: list of bounding boxes for each face
    #   [list of tuples]: (x, y, width, height)

    face_images = []  # Individual images of the faces
    for bbox in face_bboxes:
        # Note: bboxes are in (x,y,w,h) format
        x0, y0 = bbox[0], bbox[1]
        x1, y1 = x0 + bbox[2], y0 + bbox[3]
        face_image = frame[y0:y1, x0:x1]

        # Resize
        face_image = cv2.resize(face_image, (150, 150))
        face_images.append(face_image)

    if len(face_images) > 0:
        cv2.imshow("faces", np.concatenate(face_images, axis=1))
        cv2.waitKey(1)

    # At the moment, the id for each face could be its index
    # in faces_bbox / face_images
    # -------------------------

    # TODO: Send data to server...
    print("Sending data to server...")


# Get faces from saved image (OPTION 1)
def image_detect(detect_func, image_path, freq):
    original_frame = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    last_detect = 0
    frame = None
    while True:

        now = time.time()
        if now - last_detect >= freq:
            last_detect = now
            frame = original_frame.copy()
            frame, faces, _ = detect_func(frame, draw_on_frame=True)
            send_data(original_frame, faces)

        if frame is not None:
            cv2.imshow("Detections", frame)
            cv2.waitKey(1)


# Get faces from video stream (OPTION 2)
def stream_detect(detect_func, freq):
    cap = cv2.VideoCapture(0)

    last_detect = 0
    while True:
        _, frame = cap.read()

        # Detect faces periodically
        now = time.time()
        if now - last_detect >= freq:
            last_detect = now
            original_frame = frame.copy()
            frame, faces, _ = detect_func(frame, draw_on_frame=True)
            send_data(original_frame, faces)

        cv2.imshow("Detections", frame)

        # Stop if escape key is pressed
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    SEND_FREQ = 5  # How often to send data (seconds)
    USE_STREAM = True  # Use webcam or not

    model = DnnFaceDetector()
    detect_func = model.predict

    if USE_STREAM:
        stream_detect(detect_func, SEND_FREQ)
    else:
        image_path = "test_image_faces.png"  # "test_image_faces2.png"
        image_detect(detect_func, image_path=image_path, freq=SEND_FREQ)
