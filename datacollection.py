import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np 
import math
import time

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
detector = HandDetector(maxHands=1)

offset = 20 
imgSize = 300

#this part collects data and clicks image

folder = "Data/C"
counter = 0 

while True:
    success, img = cap.read()

    if not success:
        print("Camera not working")
        break

    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']
        imgWhite = np.ones((imgSize , imgSize,3), np.uint8)*255


        imgCrop = img[y- offset : y + h + offset, x - offset:x + w+offset]
        h_img, w_img, _ = img.shape

        x1 = max(0, x - offset)
        y1 = max(0, y - offset)
        x2 = min(w_img, x + w + offset)
        y2 = min(h_img, y + h + offset)
        imgCrop = img[y1:y2, x1:x2]
        

       
       
        imgCropShape = imgCrop.shape
        
        

        aspectRatio = h/w 
        
        if aspectRatio >1 : 
            k = imgSize/h
            wCal = (math.ceil(k*w))
            imgResize = cv2.resize(imgCrop,(wCal,imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize-wCal)/2)
            #imgWhite[0:imgResizeShape[0],0:imgResizeShape[1]]= imgResize it means put imgcrop values in imgwhite value i.e to place one above of other
            imgWhite[:, wGap :wCal+wGap] = imgResize # brings the image at center in that window

        else : 
            k = imgSize/w
            hCal = (math.ceil(k*h))
            imgResize = cv2.resize(imgCrop,(imgSize , hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize-hCal)/2)
            imgWhite[hGap :hCal+hGap , :] = imgResize 

 
        cv2.imshow("ImageCrop" , imgCrop)
        cv2.imshow("ImageWhite" , imgWhite)  

    cv2.imshow("Image", img)

    key = cv2.waitKey(1)

    if key == ord("s"):
     counter += 1
     cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
     print(counter)

    elif key & 0xFF == 27:
     break

cap.release()
cv2.destroyAllWindows()