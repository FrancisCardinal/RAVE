import numpy as np
import cv2
import time
import face_recognition

# Load the cascade
face_cascade = cv2.CascadeClassifier('resources/haarcascade_frontalface_alt.xml')
cap = cv2.VideoCapture(0)

# To use a video file as input
# cap = cv2.VideoCapture('filename.mp4')

# To use image as input
# frame = cv2.imread('ressources/TonyFace1.jpg')

# Load a sample picture and learn how to recognize it.
tony_image = face_recognition.load_image_file("resources/TonyFaceUpClose.jpg")
tony_face_encoding = face_recognition.face_encodings(tony_image)[0]

henri_image = face_recognition.load_image_file("resources/HenriUpClose.jpg")
henri_face_encoding = face_recognition.face_encodings(henri_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    tony_face_encoding,
    henri_face_encoding
]
known_face_names = [
    "Tony",
    "Henri"
]

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

        roi_frame = frame[y: y+h, x: x+w]

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_roi_frame = roi_frame[:, :, ::-1]

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_roi_frame)
        face_encodings = face_recognition.face_encodings(rgb_roi_frame, face_locations)

        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            # Display face name
            # cv2.rectangle(frame, (x, y+h - 35), (x+w, y+h), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (x + 2, y+h + 25), font, 1.0, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
