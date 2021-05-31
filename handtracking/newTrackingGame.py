import cv2
import mediapipe as mp
import time
import handtrackingModule as htm

pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)
detector = htm.handDetector()
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img)
    if len(lmList) != 0:
        print(lmList[8])
    #count fps
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    #display fps
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3,(225,0,225),3)
    
    cv2.imshow("Image", img)
    cv2.waitKey(1)