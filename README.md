# Hand Volume Control

A learning project exploring computer vision with **OpenCV** and **Google MediaPipe**. It controls your system's audio volume by pinching your thumb and index finger together in front of a webcam.

## Why this project exists

I'm a computer science student learning computer vision, and this repo was my hands-on introduction to OpenCV and MediaPipe. I followed [this YouTube tutorial](https://youtu.be/01sAkU_NvOY?si=kWPnhl3t03WaDCc8) to get the base hand-tracking and volume-control logic working, so **most of the original code structure is not mine**, credit for the core approach goes to the video's author.

Once I had the base version running, I used it as a sandbox to practice extending someone else's code with my own ideas, including:

- **Two-hand lock/unlock gesture** — my own addition on top of the tutorial. Volume control stays **locked** while only one hand is visible, and **unlocks** only when a second hand enters the frame. This was a way to practice designing my own gesture-based interaction rather than just following along.
- Smoothing the volume changes with a perceptual curve (`norm ** 1.5`) and exponential smoothing instead of raw/jumpy values, to make the control feel more natural.

## How it works

1. `Handmovements.py` wraps MediaPipe's Hands solution in a `handDetector` class that detects hand landmarks in a webcam frame and can report how many hands are currently visible.
2. `VolumeControl.py` opens the webcam, tracks the distance between the thumb tip (landmark 4) and index fingertip (landmark 8), and maps that distance to the system's master volume via `pycaw`.
3. If two hands are detected, the volume is **unlocked** and updates live as you pinch. With only one hand (or none), the volume is **locked** and stays put.

## Tech stack

- [OpenCV](https://opencv.org/) — video capture and drawing
- [MediaPipe](https://developers.google.com/mediapipe) — hand landmark detection
- [pycaw](https://github.com/AndreMiras/pycaw) — Windows system audio control
- NumPy

## Running it

```bash
pip install opencv-python mediapipe numpy pycaw
python VolumeControl.py
```

Press `q` to quit.

> Note: `pycaw` controls the Windows audio endpoint, so this currently only works on Windows.

## Status

This is a personal learning project, not production software. It is meant to demonstrate my process of learning a new library (MediaPipe/OpenCV) and building on top of someone else's tutorial code with my own gesture-control idea.
