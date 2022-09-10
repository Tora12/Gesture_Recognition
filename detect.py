# Jenner Higgins
# CS404
import tensorflow as tf
import cv2 as cv
import numpy as np
import imutils
import os
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.layers import Activation, Dropout, Convolution2D, GlobalAveragePooling2D
from keras.models import Sequential, load_model
from squeezenet import SqueezeNet

GESTURE_MAP = {
    "fist": 0,
    "L": 1,
    "peace": 2,
    "ok": 3,
    "hand": 4,
    "nothing": 5 
}

CATEGORY_MAP = {
    0: "fist",
    1: "L",
    2: "peace",
    3: "ok",
    4: "hand",
    5: "nothing"
}

bg = None
model = load_model("gesture-model.h5")

def gestureMapper(value):
    return GESTURE_MAP[value]

def categoryMapper(value):
    return CATEGORY_MAP[value]

## Build Training Data ##
def makeTrainImg(image_label, num_images):
    train_img_dir = "training_images"
    img_label_dir = os.path.join(train_img_dir, image_label)
    
    capture = False
    font = cv.FONT_HERSHEY_PLAIN
    count = 0
    img_tag = 0

    try:
        os.mkdir(train_img_dir)
    except FileExistsError:
        pass
    
    try:
        os.mkdir(img_label_dir)
    except FileExistsError:
        img_tag = len(os.listdir(img_label_dir))

    webcam = cv.VideoCapture(0)
    while True:
        ret, img = webcam.read()
        img = imutils.resize(img, width = 750)
        img = cv.flip(img, 1)

        if count == num_images:
            break

        cv.rectangle(img, (700, 10), (400, 225), (255, 255, 255), 2) # top-right, white, 2px

        if capture:
            roi = img[10:225, 400:700] # region of interest
            save_path = os.path.join(img_label_dir, '{}.jpg'.format(img_tag + 1))
            cv.imwrite(save_path, roi)
            img_tag += 1
            count += 1

        cv.putText(img, "Press 's' to capture gesture in white box",
            (20, 30), font, 1, (255, 0, 0), 1, cv.LINE_AA)
        cv.putText(img, "Press 'q' to exit.",
            (20, 60), font, 1, (255, 0, 0), 1, cv.LINE_AA)
        cv.putText(img, "Image Count: {}".format(count),
            (20, 100), font, 1, (0, 255, 0), 2, cv.LINE_AA)
        cv.imshow("Get Training Images", img)

        key = cv.waitKey(10)
        if key == ord('q'):
            break
        if key == ord('s'):
            capture = not capture

    print("Process Complete")
    webcam.release()
    cv.destroyAllWindows()

## Load Training Data ##
def loadTrainImg():
    train_img_dir = "training_images"
    input_data = []
    for sub_dir in os.listdir(train_img_dir):
        if not sub_dir.startswith('.'):
            path = os.path.join(train_img_dir, sub_dir)
            for f in os.listdir(path):
                if f.endswith(".jpg"):
                    img = cv.imread(os.path.join(path, f))
                    img = cv.resize(img, (225, 225))
                    input_data.append([img, sub_dir])
    img_data, labels = zip(*input_data)
    labels = list(map(gestureMapper, labels))
    labels = np_utils.to_categorical(labels)

    return img_data, labels

## Build CNN using SqueezeNet ##
def createModel(image_data, labels):
    GESTURE_CATEGORY = len(GESTURE_MAP)
    model = Sequential()
    model.add(SqueezeNet(input_shape = (225, 225, 3), include_top = False))
    model.add(Dropout(0.5))
    model.add(Convolution2D(GESTURE_CATEGORY, (1, 1), padding = "valid"))
    model.add(Activation("relu"))
    model.add(GlobalAveragePooling2D())
    model.add(Activation("softmax"))

    model.compile(
        optimizer = Adam(lr = 0.0001),
        loss = "categorical_crossentropy",
        metrics = ["accuracy"]
    )

    model.fit(np.array(image_data), np.array(labels), epochs = 15)

    print("Training complete")
    model.save("gesture-model.h5")

def findRunAvg(image, weight):
    global bg
    if bg is None:
        bg = image.copy().astype("float")
        return
    
    cv.accumulateWeighted(image, bg, weight)

def segment(image, threshold = 25):
    global bg
    diff = cv.absdiff(bg.astype("uint8"), image)
    thresh = cv.threshold(diff, threshold, 255, cv.THRESH_BINARY)[1]
    cnts, heirarchy = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    for cnt in cnts:
        cnvxHull = cv.convexHull(cnt)
        cv.drawContours(image, [cnvxHull], -1, (255, 0, 0), 2)
    cv.imshow("Convex", image)

    if len(cnts) == 0:
        return
    else:
        segm = max(cnts, key = cv.contourArea)
        return thresh, segm

def predictGesture(image):
    predict = model.predict(image)
    gesture_numeric = np.argmax(predict)
    gesture_name = categoryMapper(gesture_numeric)
    
    return gesture_name


def main():
    
    makeTrainImg("fist", 250)
    makeTrainImg("L", 250)
    makeTrainImg("peace", 250)
    makeTrainImg("ok", 250)
    makeTrainImg("hand", 250)
    makeTrainImg("nothing", 250)
    img_data, labels = loadTrainImg()
    createModel(img_data, labels)

    prediction = 5
    weight = 0.5
    num_frames = 0
    webcam = cv.VideoCapture(0)
    while True:
        ret, frame = webcam.read()
        frame = imutils.resize(frame, width = 700)
        frame = cv.flip(frame, 1)
        clone = frame.copy()
        roi = frame[10:225, 400:700] # region of interest
        gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (7, 7), 0)
        
        if num_frames < 30:
            findRunAvg(gray, weight)
        else:
            hand = segment(gray, 25)
            if hand is not None:
                thresh, segm = hand
                cv.drawContours(clone, [segm + (400, 10)], -1, (0, 0, 255))
                cv.imshow("Theshold", thresh)
        cv.rectangle(clone, (700, 10), (400, 225), (0, 255, 0), 2)
        num_frames += 1

        cv.putText(clone, f"Prediction: {prediction}",
            (50, 30), cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
        cv.imshow("Video Feed", clone)
        img = cv.resize(frame, (225, 225))
        img = img.reshape(1, 225, 225, 3)
        prediction = predictGesture(img)       

        key = cv.waitKey(1)
        if key == ord('q'):
            break

    webcam.release()
    cv.destroyAllWindows() 

if __name__ == "__main__":
    main()