import cv2
import time
import socketio
import base64
from RAVE.face_detection.face_detectors import YoloFaceDetector

# Simulate the output to be expected from the face tracker
# Send this information to the server

# socket io client
sio = socketio.Client()


def emit(event_name, destination, payload):
    """
    Emits event to destination.
    Args:
        event_name (string): The name of the event to emit.
        destination (string):
            The destination to emit the event ("client" or "server").
        payload (dict): The information needed to be passed to the destination.
    """
    sio.emit(event_name, {"destination": destination, "payload": payload})


def send_data(frame, detections):

    # frame: image where the faces were detected

    # faces_bbox: list of bounding boxes for each face
    #   [list of tuples]: (x, y, width, height)
    start = time.time()
    boundingBoxes = []
    for index, detection in enumerate(detections):
        boundingBoxes.append(
            {
                "id": index,
                "dx": int(detection.bbox[0]),
                "dy": int(detection.bbox[1]),
                "width": int(detection.bbox[2]),
                "height": int(detection.bbox[3]),
            }
        )

    if len(detections) > 0:
        print("Sending data to server...")
        frame_string = base64.b64encode(
            cv2.imencode(".jpg", frame)[1]
        ).decode()
        emit(
            "newFrameAvailable",
            "client",
            {
                "base64Frame": frame_string,
                "dimensions": frame.shape,
                "boundingBoxes": boundingBoxes,
            },
        )

    end = time.time()
    print("Time elapsed:", end - start)


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

        # if frame is not None:
        #    cv2.imshow("Detections", frame)
        #    cv2.waitKey(1)


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
            frame, detections = detect_func(frame, draw_on_frame=True)
            send_data(original_frame, detections)

        # cv2.imshow("Detections", frame)

        # Stop if escape key is pressed
        # k = cv2.waitKey(30) & 0xFF
        # if k == 27:
        #    break

    cap.release()
    cv2.destroyAllWindows()


@sio.event
def connect():
    print("connection established to server")
    # Emit the socket id to the server to "authenticate yourself"
    emit("pythonSocketAuth", "server", {"socketId": sio.get_sid()})


@sio.on("forceRefresh")
def onForceRefresh():
    print("Client called force refresh, generating new faces")


@sio.on("targetSelect")
def onSelectTarget(payload):
    targetId = payload["targetId"]
    print(f"User selected id : {targetId}")
    emit("selectedTarget", "client", {"targetId": targetId})


@sio.on("setEyeTrackerMode")
def onSetEyeTrackerMode():
    print("Client togged eye tracker mode")


@sio.event
def disconnect():
    print("disconnected from server")


if __name__ == "__main__":
    SEND_FREQ = 0.05  # How often to send data (seconds)
    USE_STREAM = True  # Use webcam or not
    sio.connect("ws://localhost:9000")
    model = YoloFaceDetector()
    detect_func = model.predict

    if USE_STREAM:
        stream_detect(detect_func, SEND_FREQ)
    else:
        image_path = "test_image_faces.png"  # "test_image_faces2.png"
        image_detect(detect_func, image_path=image_path, freq=SEND_FREQ)
