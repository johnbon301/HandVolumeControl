import time
import mediapipe as mp
import cv2
import Handmovements as hm


pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)  # creates a capture object that opens up the default camera (0)
dectector = hm.handDectector()

while True:  # infinite loop
    success, frame = cap.read()  # returns to values that successfully captures the frame
    frame = dectector.findHands(frame)
    positions = dectector.findPosition(frame)
    if len(positions) != 0:
        print(positions)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)

    cv2.imshow("Image", frame)  # displays frame in a window called "image"
    cv2.waitKey(1)  # sleep timer that wait for key press