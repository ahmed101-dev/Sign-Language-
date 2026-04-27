import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# SETUP

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

imgSize = 300
offset = 20
labels = ["A", "B", "C"]

# FUNCTION: PROCESS HAND

def process_hand(img, x, y, w, h):
    h_img, w_img, _ = img.shape

    # Safe crop
    x1 = max(0, x - offset)
    y1 = max(0, y - offset)
    x2 = min(w_img, x + w + offset)
    y2 = min(h_img, y + h + offset)

    imgCrop = img[y1:y2, x1:x2]

    if imgCrop.size == 0:
        return None, None

    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

    aspectRatio = h / w

    if aspectRatio > 1:
        k = imgSize / h
        wCal = math.ceil(k * w)
        imgResize = cv2.resize(imgCrop, (wCal, imgSize))
        wGap = (imgSize - wCal) // 2
        imgWhite[:, wGap:wGap + wCal] = imgResize
    else:
        k = imgSize / w
        hCal = math.ceil(k * h)
        imgResize = cv2.resize(imgCrop, (imgSize, hCal))
        hGap = (imgSize - hCal) // 2
        imgWhite[hGap:hGap + hCal, :] = imgResize

    return imgCrop, imgWhite


# =========================
# MAIN LOOP
# =========================
while True:
    success, img = cap.read()
    if not success:
        print("Camera not working")
        break

    # Flip for natural mirror view
    img = cv2.flip(img, 1)

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        handType = hand['type']  # 'Left' or 'Right'

        imgCrop, imgWhite = process_hand(img, x, y, w, h)

        if imgWhite is not None:
            # Prediction
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

            label = labels[index]
            confidence = round(prediction[index], 2)

            # Display label + confidence
            cv2.rectangle(img, (x, y - 60), (x + 200, y), (0, 255, 0), -1)
            cv2.putText(img, f"{label} ({confidence})",
                        (x + 10, y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 0), 2)

            # Show which hand
            cv2.putText(img, handType,
                        (x, y + h + 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 0, 0), 2)

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)

    key = cv2.waitKey(1)
    if key & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()