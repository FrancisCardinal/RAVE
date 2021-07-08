import face_recognition
import numpy as np
import cv2
import time

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    _, image = cap.read()

    # Find all facial features in all the faces in the image
    face_landmarks_list = face_recognition.face_landmarks(image)

    for face_landmarks in face_landmarks_list:
        # Print the location of each facial feature in this image
        facial_features = [
            'chin',
            'left_eyebrow',
            'right_eyebrow',
            'nose_bridge',
            'nose_tip',
            'left_eye',
            'right_eye',
            'top_lip',
            'bottom_lip'
        ]

        for facial_feature in facial_features:
            print("The {} in this face has the following points: {}".format(facial_feature,
                                                                            face_landmarks[facial_feature]))

        # Let's trace out each facial feature in the image with a line!
        for facial_feature in facial_features:
            #cv2.line(face_landmarks[facial_feature], width=5)
            cv2.polylines(image,
                          [face_landmarks[facial_feature]],
                          isClosed=False,
                          color=(0, 255, 0),
                          thickness=3)
            # cv2.drawContours(image, face_landmarks[facial_feature] , 0, (255, 255, 255), 2)
            # cv2.line(image, face_landmarks[facial_feature][0], face_landmarks[facial_feature][1], (255, 255, 255))

    # Display the resulting frame
    cv2.imshow('frame', image)

    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
