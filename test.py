import os
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

os.system("title \033[1;35m🖥️ Sign Language Interpreter 💬")

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Load models for alphabet mode
classifier_main_alphabets = Classifier("Model/f/keras_model100.h5", "Model/f/labels.txt")
classifier_amnst = Classifier("Model/i/Amnst/keras_model.h5", "Model/i/Amnst/labels.txt")
classifier_hgzpy = Classifier("Model/i/Hgzpy/keras_model.h5", "Model/i/Hgzpy/labels.txt")
classifier_uvkrw = Classifier("Model/i/Uvkrwb/keras_model.h5", "Model/i/Uvkrwb/labels.txt")
classifier_bcdefijloqx = Classifier("Model/i/Bcdefijloqx/keras_model.h5", "Model/i/Bcdefijloqx/labels.txt")

# Load model for digit mode
classifier_main_digits = Classifier("Model/i/numbers/keras_model.h5", "Model/i/numbers/labels.txt")

offset = 20
imgSize = 300

labels_main_alphabets = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V",
               "W", "X", "Y", "Z"]
labels_amnst = ["A", "M", "N", "S", "T"]
labels_hgzpy = ["H", "G", "Z", "P", "Y"]
labels_uvkrw = ["U", "V", "K", "R", "W","B"]
labels_bcdefijloqx = ["B", "C", "D", "E", "F", "I", "J", "L", "O", "Q", "X"]
labels_main_digits = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

min_confidence = 0.6

predicted_text = ""
mode = "alphabets"  # Default mode is set to alphabets

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        if x - offset >= 0 and y - offset >= 0 and x + w + offset <= img.shape[1] and y + h + offset <= img.shape[0]:

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            imgCropShape = imgCrop.shape
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                if not imgResize.any():
                    continue
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
                if mode == "alphabets":
                    prediction_main, index_main = classifier_main_alphabets.getPrediction(imgWhite, draw=True)
                else:
                    prediction_main, index_main = classifier_main_digits.getPrediction(imgWhite, draw=True)

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                if not imgResize.any():
                    continue
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                if mode == "alphabets":
                    prediction_main, index_main = classifier_main_alphabets.getPrediction(imgWhite, draw=True)
                else:
                    prediction_main, index_main = classifier_main_digits.getPrediction(imgWhite, draw=True)

            prob = prediction_main[index_main]
            if index_main > 0 and prediction_main[index_main - 1] < prediction_main[index_main]:
                print(prediction_main[index_main] * 100)

            if prob >= min_confidence:
                if mode == "alphabets":
                    if labels_main_alphabets[index_main] in labels_bcdefijloqx:
                        prediction, index = classifier_bcdefijloqx.getPrediction(imgWhite, draw=True)
                    elif labels_main_alphabets[index_main] in labels_amnst:
                        prediction, index = classifier_amnst.getPrediction(imgWhite, draw=True)
                    elif labels_main_alphabets[index_main] in labels_hgzpy:
                        prediction, index = classifier_hgzpy.getPrediction(imgWhite, draw=True)
                    elif labels_main_alphabets[index_main] in labels_uvkrw:
                        prediction, index = classifier_uvkrw.getPrediction(imgWhite, draw=True)
                    else:
                        prediction, index = prediction_main, index_main

                    ensemble_labels = labels_bcdefijloqx if labels_main_alphabets[index_main] in labels_bcdefijloqx else \
                        labels_amnst if labels_main_alphabets[index_main] in labels_amnst else \
                        labels_hgzpy if labels_main_alphabets[index_main] in labels_hgzpy else \
                        labels_uvkrw if labels_main_alphabets[index_main] in labels_uvkrw else \
                        labels_main_alphabets

                    predicted_label = ensemble_labels[index]
                else:
                    predicted_label = labels_main_digits[index_main]

                cv2.putText(imgOutput, predicted_label, (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (0, 0, 0), 4)
                cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

    cv2.rectangle(imgOutput, (10, 10), (550, 80), (255, 255, 255), cv2.FILLED)
    cv2.putText(imgOutput, predicted_text, (20, 60), cv2.FONT_HERSHEY_COMPLEX, 2, (50, 50, 50), 4)

    cv2.imshow("Sign Language Interpreter ", imgOutput)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break
    elif key == ord('a'):  # 'a' key is pressed to input the predicted label
        predicted_text += predicted_label
    elif key == ord('d'):  # 'd' key is pressed to delete the last predicted label
        predicted_text = predicted_text[:-1]
    elif key == ord('c'):  # 'c' key is pressed to clear the entire buffer
        predicted_text = ""
    elif key == ord('m'):  # 'm' key is pressed to switch modes between alphabets and digits
        if mode == "alphabets":
            mode = "digits"
        else:
            mode = "alphabets"

cap.release()
cv2.destroyAllWindows()
