import face_recognition
import numpy as np

# import time


class Verifier:
    def get_encodings(self, frame, face_locations):
        raise NotImplementedError()

    def get_closest_face(self, reference_encodings, face_encoding):
        raise NotImplementedError()


class VerifierFactory:
    @staticmethod
    def create(verifier_type="dlib", threshold=10):
        if verifier_type == "dlib":
            return DlibFaceRecognition(threshold)
        else:
            print("Unknown verifier type:", verifier_type)
            return None


class DlibFaceRecognition(Verifier):
    def __init__(self, distance_threshold):
        self.distance_threshold = distance_threshold

    def get_encodings(self, frame, face_locations):
        """
        frame: image containing the faces
        face_locations: list of tuples for face locations in format (x,y,w,h)
        returns: list of encodings matching each element in face_locations
        """
        rgb_frame = frame[:, :, ::-1]
        # t = time.time()

        # Convert xywh bboxes to (y0, x1, y1, x0)
        converted_face_locations = []
        for bbox in face_locations:
            x, y, w, h = bbox
            bbox_converted = (y, x + w, y + h, x)
            converted_face_locations.append(bbox_converted)

        face_encodings = face_recognition.face_encodings(
            rgb_frame, converted_face_locations, num_jitters=1, model="small"
        )

        # print("Verifier time:", time.time() - t)
        return face_encodings

    def get_distances(self, reference_encodings, face_encoding):
        return face_recognition.face_distance(
            reference_encodings, face_encoding
        )

    def get_closest_face(self, reference_encodings, face_encoding):
        face_distances = self.get_distances(reference_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        best_distance = face_distances[best_match_index]

        if best_distance <= self.distance_threshold:
            return best_match_index, best_distance
        else:
            return None, None
