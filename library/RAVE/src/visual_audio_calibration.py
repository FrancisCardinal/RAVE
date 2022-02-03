import pyodas.visualize as vis
import socketio
from pyodas.io import MicSource
from pynput.keyboard import Key, Controller

CHUNK_SIZE = 512
FRAME_SIZE = 2 * CHUNK_SIZE
CHANNELS = 1

WIDTH = 640
HEIGHT = 480

# socket io client
sio = socketio.Client()


@sio.event
def connect():
    print("connection established to server")
    # Emit the socket id to the server to "authenticate yourself"
    sio.emit("pythonSocket", sio.get_sid())


@sio.on("onNextCalibTarget")
def go_next_target():
    print("Client togged next target")
    # next_target = 32
    keyboard = Controller()
    keyboard.press(Key.space)
    keyboard.release(Key.space)


@sio.on("forceRefresh")
def onForceRefresh():
    print("Client called forc e refresh, generating new faces")


@sio.on("muteFunction")
def onMuteRequest(request):
    print("Client wants to mute?", request)


@sio.event
def disconnect():
    print("disconnected from server")


if __name__ == "__main__":
    sio.connect("ws://localhost:9000")
    # Visualization
    calibration = vis.AcousticImageCalibration(
        CHANNELS,
        FRAME_SIZE,
        WIDTH,
        HEIGHT,
        save_path="./visual_calibration.json",
    )
    video_source = vis.VideoSource(0, WIDTH, HEIGHT)
    m = vis.Monitor("Camera", (WIDTH, HEIGHT), refresh_rate=100)

    # Core
    mic_source = MicSource(CHANNELS, chunk_size=CHUNK_SIZE)

    # while m.window_is_alive():
    # Get the audio signal and image frame
    x = mic_source()
    frame = video_source()

    # If you are using a webcam of a camera facing yourself, this
    # might feel more natural
    # frame = cv2.flip(frame, 1)

    # Draw the targets on the frame and process the audio signal x
    frame = calibration(frame, x, m.key_pressed)

    m.update("Camera", frame)
