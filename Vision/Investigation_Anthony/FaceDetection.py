import numpy as np
import cv2
import time

# Load the cascade
face_cascade = cv2.CascadeClassifier('resources/haarcascade_frontalface_alt.xml')
profile_cascade = cv2.CascadeClassifier('resources/haarcascade_profileface.xml')

cap = cv2.VideoCapture(0)

# To use a video file as input
# cap = cv2.VideoCapture('filename.mp4')

# To use image as input
# frame = cv2.imread('ressources/TonyFace1.jpg')

while(True):
    # Capture frame-by-frame
    _, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect frontal faces
    frontal_faces = face_cascade.detectMultiScale(gray, 1.1, 4)  # NOTE: Can optimize by changing min and max expected sizes

    # Draw the rectangle around each frontal face
    for (x, y, w, h) in frontal_faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # cv2.circle(frame, (x + w // 2, y + h // 2), w // 2, (255, 0, 0), 2)
        # NOTE: we can use the face as a subregion to detect eyes or mouths (instead of running on entire image)

    # Detect profile faces
    profile_faces = profile_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw the rectangle around each profile face
    for (x, y, w, h) in profile_faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
        # cv2.circle(frame, (x + w//2, y + h//2), w//2, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
