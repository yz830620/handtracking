import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
# Hands __init__(static_image_model要不要開啟常時偵測功能，
# 開啟後如果已經detect到，就用tracking就好，而如果信心不夠而掉追蹤，就會用下面參數重找
# max_num_hands =2, min_detection_confidence = 0.5,
# min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0
def distance(a,b):
     ax, ay = a
     bx, by = b
     return round(((ax-bx)**2+(ay-by)**2)**0.5, 2)
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            trigger_dict = {}
            for id, lm in enumerate(handLms.landmark):
                #print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)
                trigger_dict[id] = (cx,cy)
                if id == 8:
                    if distance(trigger_dict[4],trigger_dict[8]) <= 15:
                        cv2.circle(img, (cx,cy), 15, (255,0,255), cv2.FILLED)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    #count fps
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    #display fps
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3,(225,0,225),3)


    cv2.imshow("Image", img)
    cv2.waitKey(1)