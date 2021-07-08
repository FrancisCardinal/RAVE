import cv2
import numpy as np

modelFile = "resources/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "resources/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

cap = cv2.VideoCapture(0)


class Box:
    def __init__(self, x, y, x1, y1):
        self.x = x
        self.y = y
        self.x1 = x1
        self.y1 = y1

num_boxes = 5
previous_boxes = []

while(True):
    # Capture frame-by-frame
    _, frame = cap.read()

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0))
    net.setInput(blob)
    faces = net.forward()

    # Draw previous boxes
    if len(previous_boxes) > num_boxes:
        previous_boxes.pop(0)

    for box in previous_boxes:
        cv2.rectangle(frame, (box.x, box.y), (box.x1, box.y1), (0, 0, 150), 1)

    # Draw current box
    for i in range(faces.shape[2]):
        confidence = faces[0, 0, i, 2]
        if confidence > 0.5:
            box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            new_box = Box(x, y, x1, y1)
            previous_boxes.append(new_box)
            cv2.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), 3)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
