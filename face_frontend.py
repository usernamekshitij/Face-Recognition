from PIL import Image
from keras.applications.vgg16 import preprocess_input
import base64
from io import BytesIO
import json
import random
import cv2
from keras.models import load_model
import numpy as np

from keras.preprocessing import image

model = load_model(
    "C:/Summer Training/Face Recognition/facefeatures_new_model_5classes.h5"
)

img = cv2.imread("C:/Summer Training/Face Recognition/daddario.jpg")
face_cascade = cv2.CascadeClassifier(
    "C:/Summer Training/Face Recognition/haarcascade_frontalface_default.xml"
)
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image

    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)

    if faces is ():
        return None

    # Crop all faces found
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cropped_face = img[y : y + h, x : x + w]

    return cropped_face


# Without bounding box
def face_extractor2(img):
    faces = face_cascade.detectMultiScale(img, 1.1, 5)

    if faces is ():
        return None
    for (x, y, w, h) in faces:
        x = x - 10
        y = y - 10
        cropped_face = img[y : y + h + 50, x : x + w + 50]

    return cropped_face


import os

c = 0
video_capture = cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read()
    # canvas = detect(gray, frame)
    # image, face =face_detector(frame)

    face = face_extractor(frame)
    if type(face) is np.ndarray:
        face = cv2.resize(face, (224, 224))
        im = Image.fromarray(face, "RGB")

        img_array = np.array(im)
        # Our keras model used a 4D tensor, (images x height x width x channel)

        img_array = np.expand_dims(img_array, axis=0)
        pred = model.predict(img_array)
        print(pred)
        name = "None matching"

        if name == "None matching":
            c += 1

        if pred[0][0] > 0.7:
            name = "Kshitij"
        elif pred[0][1] > 0.7:
            name = "Adriana"
        elif pred[0][2] > 0.7:
            name = "Alex Lawther"
        elif pred[0][3] > 0.7:
            name = "Alexandra Daddario"
        elif pred[0][4] > 0.7:
            name = "Alvaro Morte"
        cv2.putText(frame, name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(
            frame,
            "No face found",
            (50, 50),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (0, 255, 0),
            2,
        )
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
video_capture.release()
cv2.destroyAllWindows()


if c > 2:
    print("Do you want to add this to dataset??")
    print("y/n")
    x = input()
    if x == "y":
        print("enter a name")
        folder_name = input()
        print(type(folder_name))
        path_train = os.path.join(
            "C:/Summer Training/Face Recognition/Datasets/Train/", folder_name
        )
        path_test = os.path.join(
            "C:/Summer Training/Face Recognition//Datasets/Test/", folder_name
        )
        if not os.path.exists(path_train):
            os.mkdir(path_train)
        if not os.path.exists(path_test):
            os.mkdir(path_test)
        else:
            print("Folder already exists")

        vid = cv2.VideoCapture(0)
        count = 0
        while True:
            ret, frame = vid.read()
            if face_extractor2(frame) is not None:
                count += 1
                face = cv2.resize(face_extractor2(frame), (400, 400))
                file_name_path = path_train + "/" + folder_name + str(count) + ".jpg"
                if count > 70:
                    file_name_path = path_test + "/" + folder_name + str(count) + ".jpg"
                cv2.imwrite(file_name_path, face)
                cv2.putText(
                    face,
                    str(count),
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                    2,
                )
                cv2.imshow("Face Cropper", face)

            else:
                print("Face not found")
                pass
            if cv2.waitKey(1) == 13 or count == 100:  # 13 is the Enter Key
                break
        vid.release()
        cv2.destroyAllWindows()
        print("DONE but you still gotta train the model again!")

    else:
        print("Thank You")
else:
    print("Try running again")


"""
name = "None Matching"

face = face_extractor(img)
cv2.imshow("face", face)
cv2.waitKey(0)
cv2.destroyAllWindows()
if type(face) is np.ndarray:
    face = cv2.resize(face, (224, 224))
    im = Image.fromarray(face, "RGB")
    img_array = np.array(im)

    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    print("Pred", pred)

    if pred[0][2] > 0.5:
        name = "Daddario"
    print(name)




        vid = cv2.VideoCapture(0)
        count = 0

        while True:
            ret, frame = cap.read()
            
            if cv2.waitKey(1) == 13 or count == 100:  # 13 is the Enter Key
                break
        cap.release()
        cv2.destroyAllWindows()
        print("DONE")

    else:
        print("Thank You")
"""
