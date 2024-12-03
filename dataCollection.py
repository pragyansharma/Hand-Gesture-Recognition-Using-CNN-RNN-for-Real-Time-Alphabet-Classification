import cv2
from cvzone.HandTrackingModule import HandDetector
import os
import numpy as np
import time

# Parameters
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
sequence_length = 10
folder = "Data"  
os.makedirs(folder, exist_ok=True)
counter = 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if not success:
        print("Failed to capture image.")
        continue

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
        x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)
        imgCrop = img[y1:y2, x1:x2]

        aspectRatio = h / w
        if aspectRatio > 1:
            k = imgSize / h
            wCal = int(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = (imgSize - wCal) // 2
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = int(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = (imgSize - hCal) // 2
            imgWhite[hGap:hCal + hGap, :] = imgResize

        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        timestamp = time.time()
        cv2.imwrite(f'{folder}/Image_{counter}_{timestamp}.jpg', imgWhite)
        print(f"Saved image {counter}")
