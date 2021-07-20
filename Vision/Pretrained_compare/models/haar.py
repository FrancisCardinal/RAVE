import os
import cv2  # pip install opencv-python

# Load pre-trained haar cascades
resources_dir = os.path.join(os.path.dirname(__file__), "../resources")
face_cascade = cv2.CascadeClassifier(os.path.join(resources_dir, "haarcascade_frontalface_default.xml"))
profile_cascade = cv2.CascadeClassifier(os.path.join(resources_dir, "haarcascade_profileface.xml"))


def detect_faces(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect frontal faces
    frontal_faces = face_cascade.detectMultiScale(gray, 1.1, 8)  # NOTE: Can optimize by changing min and max expected sizes

    line_thickness = frame.shape[0] // 240 or 1

    # Draw the rectangle around each frontal face
    for (x, y, w, h) in frontal_faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), line_thickness)

    # Uncomment to also detect profile faces:
    # # Detect profile faces
    # profile_faces = profile_cascade.detectMultiScale(gray, 1.1, 8)
    #
    # # Draw the rectangle around each profile face
    # for (x, y, w, h) in profile_faces:
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), line_thickness)

    return frame


if __name__ == "__main__":
    # For testing
    frame = cv2.imread("../resources/TonyFace1.jpg")
    frame = detect_faces(frame)
    cv2.imshow("Final", frame)
    cv2.waitKey(0)

