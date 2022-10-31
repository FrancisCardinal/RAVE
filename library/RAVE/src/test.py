# from pyodas.visualize import VideoSource, Monitor

Gstreamer_pipeline = 'v4l2src device=/dev/video0 ! video/x-raw, format=UYVY, width=640, heigth=480, framerate=60/1 ! nvvidconv ! video/x-raw(memory:NVMM) ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
# cap = VideoSource(Gstreamer_pipeline, 480, 640)

# monitor = Monitor("Image", cap.shape)

# while True:
#     img = cap()

#     monitor.update("Image", img)

#!/usr/bin/env python3
#
#  USB Camera - Simple
#
#  Copyright (C) 2021-22 JetsonHacks (info@jetsonhacks.com)
#
#  MIT License
#

import sys

import cv2

print(cv2.getBuildInformation())

window_title = "USB Camera"

# ASSIGN CAMERA ADRESS to DEVICE HERE!
pipeline = " ! ".join(["v4l2src device=/dev/video1",
                       "video/x-raw, width=640, height=480, framerate=30/1",
                       "videoconvert",
                       "video/x-raw, format=(string)BGR",
                       "appsink"
                       ])

# Sample pipeline for H.264 video, tested on Logitech C920
h264_pipeline = " ! ".join(["v4l2src device=/dev/video0",
                            "video/x-h264, width=1280, height=720, framerate=30/1, format=H264",
                            "avdec_h264",
                            "videoconvert",
                            "video/x-raw, format=(string)BGR",
                            "appsink sync=false"
                            ])

def show_camera():

    # Full list of Video Capture APIs (video backends): https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html
    # For webcams, we use V4L2
    video_capture = cv2.VideoCapture(Gstreamer_pipeline, cv2.CAP_GSTREAMER)

    if video_capture.isOpened():

        window_handle = cv2.namedWindow(
            window_title, cv2.WINDOW_AUTOSIZE)
        # Window
        while True:
            ret_val, frame = video_capture.read()
            # Check to see if the user closed the window
            # Under GTK+ (Jetson Default), WND_PROP_VISIBLE does not work correctly. Under Qt it does
            # GTK - Substitute WND_PROP_AUTOSIZE to detect if window has been closed by user
            if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                cv2.imshow(window_title, frame)
            else:
                break
            keyCode = cv2.waitKey(10) & 0xFF
            # Stop the program on the ESC key or 'q'
            if keyCode == 27 or keyCode == ord('q'):
                break
            
        video_capture.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":

    show_camera()