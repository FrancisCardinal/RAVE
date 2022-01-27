import cv2
import time
import socketio
import base64

# import numpy as np

from face_detectors import DnnFaceDetector

# Simulate the output to be expected from the face tracker
# Send this information to the server

# socket io client
sio = socketio.Client()

# Called when we want to send updated data to the server


def send_data(frame, face_bboxes, sio):

    # frame: image where the faces were detected

    # faces_bbox: list of bounding boxes for each face
    #   [list of tuples]: (x, y, width, height)
    start = time.time()
    face_images = []  # Individual images of the faces
    for index, bbox in enumerate(face_bboxes):
        # Note: bboxes are in (x,y,w,h) format
        x0, y0 = bbox[0], bbox[1]
        x1, y1 = x0 + bbox[2], y0 + bbox[3]
        face_image = frame[y0:y1, x0:x1]

        # Resize
        face_image = cv2.resize(face_image, (150, 150))
        image_string = base64.b64encode(
            cv2.imencode('.jpg', face_image)[1]).decode()
        face_images.append({"img": image_string, "id": index, "dx": int(
            x0), "dy": int(y0), "width": int(x1), "height": int(y1)})

    if len(face_images) > 0:
        # cv2.imshow("faces", np.concatenate(face_images, axis=1))
        # cv2.waitKey(1)
        print("Sending data to server...")
        sio.emit("newFacesAvailable", face_images)

    end = time.time()
    print("Time elapsed:", end - start)


# Get faces from saved image (OPTION 1)
def image_detect(detect_func, image_path, freq, sio):
    original_frame = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    last_detect = 0
    frame = None
    while True:

        now = time.time()
        if now - last_detect >= freq:
            last_detect = now
            frame = original_frame.copy()
            frame, faces, _ = detect_func(frame, draw_on_frame=True)
            send_data(original_frame, faces, sio)

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


@sio.event
def connect():
    print('connection established to server')
    # Emit the socket id to the server to "authenticate yourself"
    sio.emit('pythonSocket', sio.get_sid())


@sio.on('forceRefresh')
def onForceRefresh(data):
    print("Client called force refresh, generating new faces")


@sio.on('setEyeTrackerMode')
def onSetEyeTrackerMode(data):
    print("Client togged eye tracker mode")


@sio.event
def disconnect():
    print('disconnected from server')


if __name__ == "__main__":
    SEND_FREQ = 10  # How often to send data (seconds)
    USE_STREAM = False  # Use webcam or not
    sio.connect("ws://localhost:9000")
    model = DnnFaceDetector()
    detect_func = model.predict

    if USE_STREAM:
        stream_detect(detect_func, SEND_FREQ)
    else:
        image_path = "test_image_faces.png"  # "test_image_faces2.png"
        image_detect(detect_func, image_path=image_path,
                     freq=SEND_FREQ, sio=sio)
