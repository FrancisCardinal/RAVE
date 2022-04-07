import cv2
import time
import os
import tqdm
import pathlib

import numpy as np

from RAVE.face_detection.face_detectors import DetectorFactory
from RAVE.face_detection.face_verifiers import VerifierFactory
from RAVE.face_detection.verifiers.Encoding import Encoding


def saveImagesTest(freq):
    import os
    from deepface import DeepFace

    cap = cv2.VideoCapture(0)

    last_detect = 0
    last_image_id = -1
    while True:
        _, frame = cap.read()

        # Detect faces periodically
        now = time.time()
        if now - last_detect >= freq:
            last_detect = now

            last_image_id += 1
            image_path = os.path.join("imagesTest", f"image_capture{last_image_id}.jpg")
            cv2.imwrite(image_path, frame)
            # time.sleep(1)

            if last_image_id > 0:
                image_path1 = os.path.join("imagesTest", f"image_capture{last_image_id-1}.jpg")
                image_path2 = os.path.join("imagesTest", f"image_capture{last_image_id}.jpg")

                result = DeepFace.verify(img1_path=image_path1, img2_path=image_path2, model_name="Facenet", detector_backend="dlib", enforce_detection=False, distance_metric="euclidean")
                print(result)

                # Display face images
                image1 = cv2.imread(image_path1)
                image2 = cv2.imread(image_path2)
                cv2.imshow("faces", np.concatenate([image1, image2], axis=1))
                cv2.waitKey(0)


def photoshoot(start_ind=0):
    cap = cv2.VideoCapture(0)

    delay = 0.1
    last_image_id = start_ind - 1
    while True:
        _, frame = cap.read()
        cv2.imshow("Stream", frame)

        # Take photo
        if cv2.waitKey(10) == ord("s"):
            print(f"Taking picture in {delay}s...")
            time.sleep(delay)
            print("Taking picture!")
            _, img = cap.read()
            last_image_id += 1
            image_path = os.path.join("FaceTestRave", f"image_capture{last_image_id}.jpg")
            cv2.imwrite(image_path, img)
            print("Saved picture as:", image_path)

def deepFaceTests(model_name="Facenet", metric="default", threshold="default", verbose=False):
    model = None
    if model_name not in ["VGG-Face", "Facenet", "OpenFace", "DeepFace", "DeepID", "Dlib", "ArcFace"]:
        model = VerifierFactory.create(model_name)

    curr_path = pathlib.Path(__file__).parent.resolve()
    dir_name = os.path.join(
        curr_path,
        "FaceTestRaveCropped",
    )
    image_count = len(os.listdir(dir_name))

    identities = []
    for i in range(11):
        identities.append("Tony")
    for i in range(11, 18):
        identities.append("Oli")

    photo_pairs = []
    for i in range(image_count):
        for j in range(image_count):
            if i != j:
                photo_pairs.append((i, j))
    # random.shuffle(photo_pairs)

    correct = 0
    start_time = time.time()
    total_match_delta, total_match = 0, 0
    total_diff_delta, total_diff = 0, 0
    TP, TN, FP, FN = 0, 0, 0, 0
    for pair in tqdm.tqdm(photo_pairs, desc=f"{model_name} model with {metric} metric"):
        id1, id2 = pair
        image_path1 = os.path.join(dir_name, f"cropped_image_capture{id1}.jpg")
        image_path2 = os.path.join(dir_name, f"cropped_image_capture{id2}.jpg")

        score = None
        distance = None
        if model_name in ["VGG-Face", "Facenet", "OpenFace", "DeepFace", "DeepID", "Dlib", "ArcFace"]:
            result = DeepFace.verify(img1_path=image_path1, img2_path=image_path2, model_name=model_name,
                                     detector_backend="skip", enforce_detection=False, distance_metric=metric)

            distance = result["distance"]
            verified = result["verified"]
        else:
            image1 = cv2.imread(image_path1)
            image2 = cv2.imread(image_path2)

            full_bbox1 = [0, 0, image1.shape[1], image1.shape[0]]
            full_bbox2 = [0, 0, image2.shape[1], image2.shape[0]]
            encoding1 = Encoding(model.get_features(image1, [full_bbox1])[0])
            encoding2 = Encoding(model.get_features(image2, [full_bbox2])[0])

            # while True:
            #     image1 = cv2.imread(image_path1)
            #     full_bbox1 = [0, 0, image1.shape[1], image1.shape[0]]
            #     f = model.get_features(image1, [full_bbox1])
            #     t=2

            # image1 = cv2.imread(image_path1)
            # image2 = cv2.imread(image_path2)
            # image1 = cv2.resize(image1, (150, 150))
            # image2 = cv2.resize(image2, (150, 150))
            # cv2.imshow("faces", np.concatenate([image1, image2], axis=1))
            # cv2.waitKey(0)

            score = model.get_scores([encoding1], encoding2)[0]
            verified = score >= threshold
            distance = 1 - score

        if verbose:
            print(f"Test: {id1} & {id2}")
            if score is not None:
                print(verified, "Score:", score)
            else:
                print(verified, "Distance:", distance)

        is_match = (identities[id1] == identities[id2])
        if is_match:
            total_match_delta += distance
            total_match += 1
        else:
            total_diff_delta += distance
            total_diff += 1

        if verified == is_match:
            correct += 1

            if is_match is False:
                TN += 1
            else:
                TP += 1
            if verbose:
                print("Correct!")
        else:
            if is_match is False:
                FP += 1
            else:
                FN += 1
            if verbose:
                print("Failed")

        # Display face images
        if verbose:
            image1 = cv2.imread(image_path1)
            image2 = cv2.imread(image_path2)
            image1 = cv2.resize(image1, (150, 150))
            image2 = cv2.resize(image2, (150, 150))
            cv2.imshow("faces", np.concatenate([image1, image2], axis=1))
            if verified != is_match:
                cv2.waitKey(0)

    end_time = time.time()
    print("-----")
    print("Score: {}/{} ({:.2%})".format(correct, len(photo_pairs), correct/len(photo_pairs)))
    print(f"Time: {end_time-start_time}s")
    print(f"Distance with {metric} metric:")
    print("Match: {:.4}".format(total_match_delta/total_match))
    print("Different: {:.4}".format(total_diff_delta/total_diff))

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
    print("Accuracy: {:.3%}, Precision: {:.3%}, Recall: {:.3%}".format(accuracy, precision, recall))


def crop_images_to_face():
    model = DetectorFactory.create("yolo")
    detect_func = model.predict

    dir_name = "FaceTestRave"
    for filename in os.listdir(dir_name):
        image_path = os.path.join(dir_name, filename)
        image = cv2.imread(image_path)

        _, detections = detect_func(image, draw_on_frame=False)
        detection = detections[0]
        bbox = detection.bbox

        x, y, w, h = bbox
        cropped_image = image[y:y+h, x:x+w]
        out_file = os.path.join(dir_name+"Cropped", "cropped_"+filename)
        cv2.imwrite(out_file, cropped_image)

if __name__ == "__main__":
    SEND_FREQ = 0.1  # How often to send data (seconds)
    USE_STREAM = True  # Use webcam or not

    if USE_STREAM:
        # crop_images_to_face()
        # otherVerifierTests(model_name="dlib", show_images=True, verbose=True)
        # deepFaceTests(model="Facenet", verbose=True)
        # photoshoot(start_ind=11)
        # saveImagesTest(SEND_FREQ)
        # stream_detect(detect_func, SEND_FREQ)
        # dlib_loop(SEND_FREQ)

        verbose = False
        # deepFaceTests(model_name="dlib", threshold=0.32, verbose=verbose)
        # deepFaceTests(model_name="resnet_face_18", threshold=0.32, verbose=verbose)
        # deepFaceTests(model_name="resnet_face_34", threshold=0.32, verbose=verbose)
        # deepFaceTests(model_name="resnet_face_50", threshold=0.32, verbose=verbose)
        # deepFaceTests(model_name="dlib", threshold=0.4, verbose=verbose)
        # deepFaceTests(model_name="resnet_face_18", threshold=0.4, verbose=verbose)
        # deepFaceTests(model_name="resnet_face_34", threshold=0.4, verbose=verbose)
        # deepFaceTests(model_name="resnet_face_50", threshold=0.4, verbose=verbose)
        # deepFaceTests(model_name="Facenet", metric="euclidean", verbose=verbose)
        # deepFaceTests(model_name="VGG-Face", metric="cosine", verbose=verbose)
        # deepFaceTests(model_name="OpenFace", metric="euclidean", verbose=verbose)
        # deepFaceTests(model_name="ArcFace", metric="cosine", verbose=verbose)
        # deepFaceTests(model_name="DeepFace", metric="cosine", verbose=verbose)

        deepFaceTests(model_name="arcface", threshold=0.15, verbose=verbose)
    else:
        model = DetectorFactory.create("yolo")
        detect_func = model.predict
        image_path = "test_image_faces.png"  # "test_image_faces2.png"
        image_detect(detect_func, image_path=image_path, freq=SEND_FREQ)

"""
0-10: Anthony
11-17: Olivier
"""