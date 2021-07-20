import dlib  # If trouble installing dlib follow: https://www.youtube.com/watch?v=jjRFCTmK2SY&ab_channel=Ritesh
import cv2

detector = dlib.get_frontal_face_detector()

def detect_faces(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)  # result

    line_thickness = frame.shape[0] // 240 or 1

    # Draw
    for result in faces:
        x = result.left()
        y = result.top()
        x1 = result.right()
        y1 = result.bottom()
        cv2.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), line_thickness)

    return frame
