import cv2

from pyodas.visualize import VideoSource

DATA = """<?xml version='1.0'?><opencv_storage><cameraMatrix type_id='opencv-matrix'><rows>3</rows>
<cols>3</cols><dt>f</dt><data>340.60994606 0.0 325.7756748 0.0 341.93970667 242.46219777 0.0 0.0 1.0</data>
</cameraMatrix><distCoeffs type_id='opencv-matrix'><rows>5</rows><cols>1</cols><dt>f</dt>
<data>-3.07926877e-01 9.16280959e-02 9.46074597e-04 3.07906550e-04 -1.17169354e-02</data>
</distCoeffs></opencv_storage>"""

# Gstreamer_pipeline = '''v4l2src device=/dev/video0 ! video/x-raw, format=UYVY, width=640, heigth=480, framerate=60/1
#  ! nvvidconv ! video/x-raw(memory:NVMM)
#  ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR
#  ! appsink'''

Gstreamer_pipeline = f"""v4l2src device=/dev/video0 ! video/x-raw, format=UYVY, width=640, heigth=480,
 framerate=30/1 ! nvvidconv ! video/x-raw(memory:NVMM) ! nvvidconv ! video/x-raw,
  format=BGRx ! videoconvert ! video/x-raw, format=BGR ! videoconvert !
 cameraundistort  settings="{DATA}" ! videoconvert ! appsink"""


window_title = "USB Camera"

print(cv2.getBuildInformation())


def show_camera():

    # Full list of Video Capture APIs (video backends):
    # https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html
    # For webcams, we use V4L2
    video_capture = VideoSource(Gstreamer_pipeline, 640, 480)

    if True:

        cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
        # Window
        while True:
            frame = video_capture()
            # Check to see if the user closed the window
            # Under GTK+ (Jetson Default), WND_PROP_VISIBLE does not work correctly. Under Qt it does
            # GTK - Substitute WND_PROP_AUTOSIZE to detect if window has been closed by user
            if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                cv2.imshow(window_title, frame)
            else:
                break
            keyCode = cv2.waitKey(10) & 0xFF
            # Stop the program on the ESC key or 'q'
            if keyCode == 27 or keyCode == ord("q"):
                break

        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":

    show_camera()

# import cv2
# import torchvision.transforms as transforms

# img = cv2.imread("test_image.png")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# transform = transforms.ToTensor()
# img_tensor = transform(img)

# print(img_tensor)

# K = torch.tensor([[340.60994606, 0.0, 325.7756748], [0.0, 341.93970667, 242.46219777], [0.0, 0.0, 1.0]])
# D = torch.tensor([[-3.07926877e-01, 9.16280959e-02, 9.46074597e-04, 3.07906550e-04, -1.17169354e-02]])
# out_image = undistort_points(img_tensor, K, D)

# out_image.permute(1, 2, 0).numpy()
# cv2.imwrite("out_image.png", out_image)
