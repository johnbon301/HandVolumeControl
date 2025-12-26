import time
import numpy as np
import cv2
import Handmovements as Hm
import math
from pycaw.pycaw import AudioUtilities


pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)  # creates a capture object that opens up the default camera (0)
cap.set(3, 640) # sets the height and width of the window size
cap.set(4, 480)

detector = Hm.handDectector() # creates an object called detector

device = AudioUtilities.GetSpeakers()
volume = device.EndpointVolume
# print(f"Audio output: {device.FriendlyName}")
# print(f"- Muted: {bool(volume.GetMute())}")
# print(f"- Volume level: {volume.GetMasterVolumeLevel()} dB")
# print(f"- Volume range: {volume.GetVolumeRange()[0]} dB - {volume.GetVolumeRange()[1]} dB")
volumerange = volume.GetVolumeRange()

minVolume = volumerange[0]
maxVolume = volumerange[1]
prevVol = minVolume

while True:  # infinite loop
    success, frame = cap.read()  # returns to values that successfully captures the frame (always have)
    frame = detector.findHands(frame) #outputs a frame from the camera which finds the hands
    LMpositions = detector.findPosition(frame) # lets us know where each landmark position is at

    if len(LMpositions) != 0:
        print(LMpositions[4], LMpositions[8])

        x1, y1 = LMpositions[4][1], LMpositions[4][2]
        x2, y2 = LMpositions[8][1], LMpositions[8][2]
        xm, ym = (x1 + x2) / 2, (y1 + y2) / 2

        cv2.circle(frame, (x1, y1), 2, (255, 0, 0), 3) # circle on thumb point tip
        cv2.circle(frame, (x2, y2), 2, (255, 0, 0), 3) # circle on the pointer finger tip
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 3) # a line between landmark 4 and 8
        cv2.circle(frame, (int(xm), int(ym)), 2, (255, 0, 0), 3) # the mid point of landmark 4 and 8

        length = math.hypot(x2 - x1, y2 - y1) # find the distance between two points

        # create a normal curve rather than a spontaneous jump or gamma curve
        norm = np.interp(length, [30,200], [0.0,1.0])  # normalize the distance to 0-1
        norm = norm ** 1.5  # perceptual curve
        vol = np.interp(norm, [0, 1], [minVolume, maxVolume]) # map to dB range
        vol = max(minVolume, min(vol, maxVolume)) # safe edge cases

        # smooth volume changes
        vol = prevVol + 0.2 * (vol - prevVol)
        prevVol = vol

        volume.SetMasterVolumeLevel(vol, None) # controls the volume


        if length < 30: # for when your fingers reach a min point
            cv2.circle(frame, (int(xm), int(ym)), 2, (128, 0, 128), 3)
        elif length > 200: # when your fingers reach a max point
            cv2.circle(frame, (int(xm), int(ym)), 2, (144, 238, 144), 3)

    # sets the fps
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(frame, f' fps: {str(int(fps))}', (10, 50), cv2.FONT_HERSHEY_PLAIN, 2,
                (255, 0, 255), 2) # adds and edits the window for "Image"
    cv2.imshow("Image", frame)  # displays frame in a window called "image"
    cv2.waitKey(1)  # sleep timer that wait for key press