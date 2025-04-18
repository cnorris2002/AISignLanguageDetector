# test.py
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True)

print("MediaPipe and OpenCV are working!")
