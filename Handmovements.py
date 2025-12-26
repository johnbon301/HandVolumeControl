import time

import mediapipe as mp
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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
                h, w, c = frame.shape
                cx, cy = int(LM.x * w), int(LM.y * h)
                lmList.append([id, cx, cy])

        return lmList
