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
    def __init__(self, score_threshold):
        self.score_threshold = score_threshold

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

    def get_scores(self, reference_encodings, face_encoding):
        distances = face_recognition.face_distance(
            reference_encodings, face_encoding
        )
        scores = [1 - distance for distance in distances]
        return scores

    def get_closest_face(self, reference_encodings, face_encoding):
        face_scores = self.get_scores(reference_encodings, face_encoding)
        best_match_index = np.argmax(face_scores)
        best_score = face_scores[best_match_index]

        if best_score >= self.score_threshold:
            return best_match_index, best_score
        else:
            return None, None
