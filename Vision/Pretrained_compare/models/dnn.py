import cv2
import numpy as np
import os

# Load model
resources_dir = os.path.join(os.path.dirname(__file__), "../resources")
modelFile = os.path.join(resources_dir, "res10_300x300_ssd_iter_140000.caffemodel")
configFile = os.path.join(resources_dir, "deploy.prototxt.txt")
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)


def detect_faces(frame):

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0))
    net.setInput(blob)
    faces = net.forward()

    # Draw current box
    for i in range(faces.shape[2]):
        confidence = faces[0, 0, i, 2]

        # NOTE: Can modify confidence to alter sensitivity
        if confidence > 0.5:
            box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            cv2.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), 3)

    return frame


if __name__ == "__main__":
    # For testing
    frame = cv2.imread("../resources/TonyFace1.jpg")
    frame = detect_faces(frame)
    cv2.imshow("Final", frame)
    cv2.waitKey(0)
