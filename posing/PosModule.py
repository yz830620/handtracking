import cv2
import mediapipe as mp
import time

class poseDetector():
    def __init__(self,mode=False, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(mode ,upBody, smooth, detectionCon, trackCon)

    def findPose(self, img, draw=True):   
        img = cv2.resize(img, (720,480))
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if  self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)

        return img       
                # 
    
    def findPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                #print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx,cy])
                if draw:
                    if id in [12, 14, 16]:
                        cv2.circle(img, (cx,cy), 8, (255,0,0), cv2.FILLED)
        return lmList

def main():
    cap = cv2.VideoCapture('PosVideos/1.mp4')
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    detector = poseDetector()
    pTime = 0
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList)
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow("Image", img)

        cv2.waitKey(1)

if __name__ == "__main__":
    main()