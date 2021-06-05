import cv2
import mediapipe as mp

import time




mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()

class FaceDetector():
    """this is the class for face detection"""
    def __init__(self, minDetectionCon=0.5, cam=False):
        self.cam = cam
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw =  mp.solutions.drawing_utils
        self.faceDetection = mpFaceDetection.FaceDetection(minDetectionCon)

    def findFaces(self, img, draw=True):
        if self.cam:
            img = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img.flags.writeable = False
        self.results = self.faceDetection.process(img)
        bboxes = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                    int(bboxC.width * iw), int(bboxC.height * ih)
                bboxes.append([id, bbox, detection.score])
                if draw:
                    img = self.fancyDraw(img, bbox, l=int(bbox[3]/8))
                    cv2.putText(img, f"{int(detection.score[0]*100)}%", (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 255), 3)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img, bboxes

    def fancyDraw(self, img, bbox, l=30, t=8, rt=1):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h
        cv2.rectangle(img, bbox, (0, 225, 0), 4, rt)
        # top left
        cv2.line(img, (x,y),(x+l, y), (255, 0, 255), t)
        cv2.line(img, (x,y),(x, y+l), (255, 0, 255), t)
        # top right
        cv2.line(img, (x1-l,y),(x1, y), (255, 0, 255), t)
        cv2.line(img, (x1,y),(x1, y+l), (255, 0, 255), t)
        # bottom left
        cv2.line(img, (x,y1),(x+l, y1), (255, 0, 255), t)
        cv2.line(img, (x,y1-l),(x, y1), (255, 0, 255), t)
        #bottom right
        cv2.line(img, (x1-l,y1),(x1, y1), (255, 0, 255), t)
        cv2.line(img, (x1,y1-l),(x1, y1), (255, 0, 255), t)
        return img

def main():
    cap = cv2.VideoCapture(0)
    pTime = 0

    detector = FaceDetector(cam=True)
    while True:
        success, img = cap.read()
        img ,bboxes = detector.findFaces(img)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, f"{int(fps)}", (20, 100), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 3)

        print(fps)
        cv2.imshow("Image", img)
        cv2.waitKey(1)
        if cv2.waitKey(5) & 0xFF == 27:
            break


if __name__ == "__main__":
    main()