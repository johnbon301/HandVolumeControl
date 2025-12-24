import time

import mediapipe as mp
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = 'C:\\Development\\models\\hand_landmarker.task'

# BaseOptions = mp.tasks.BaseOptions
# HandLandmarker = mp.tasks.vision.HandLandmarker
# HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
# VisionRunningMode = mp.tasks.vision.RunningMode

class handDectector():
    def __init__(self, mode = False, maxH = 2, dectectionCount = 0.5, trackConf = 0.5):
        self.mode = mode
        self.maxH = maxH
        self.dectectionCount = dectectionCount
        self.trackConf = trackConf

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode = self.mode, max_num_hands = self.maxH,
                                         min_detection_confidence = self.dectectionCount, min_tracking_confidence = self.trackConf)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, frame, draw = True):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Blue-Green-Red 2 Red-Green-Blue
        self.results = self.hands.process(imgRGB) # analyzes img. and hands which returns an object
        # print(self.results.multi_hand_landmarks) # detects any hands

        if self.results.multi_hand_landmarks:
            for handsLM in self.results.multi_hand_landmarks: #loops through hands detected
                if draw:
                    self.mpDraw.draw_landmarks(frame, handsLM,
                                               self.mp_hands.HAND_CONNECTIONS) # does the 21 hand landmarks
        return frame

    def findPosition(self, frame, handNo = 0, draw = True):

        lmList = []
        if self.results.multi_hand_landmarks:
            handsLM = self.results.multi_hand_landmarks[handNo]
            for id, LM in enumerate(handsLM.landmark):
                # print(id, LM.x, LM.y)
                h, w, c = frame.shape
                cx, cy = int(LM.x * w), int(LM.y * h)
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
                # if draw:

        return lmList


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)  # creates a capture object that opens up the default camera (0)
    dectector = handDectector()

    while True:  # infinite loop
        success, frame = cap.read()  # returns to values that successfully captures the frame
        frame = dectector.findHands(frame)
        positions = dectector.findPosition(frame)
        if len(positions) != 0:
            print(positions)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(frame, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3,
            (255,0,255), 3)

        cv2.imshow("Image", frame) #displays frame in a window called "image"
        cv2.waitKey(1) #sleep timer that wait for key press

if __name__ == "__main__":
    main()
