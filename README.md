# AI Sign Language Detector
# CAP 4630 - Intro to AI Project 4

Team members: Gabriel Malabanan
              Lena Francisco
              Chris Norris

## Description

This is an application that recognizes letters A-Z of the American Sign Language (ASL) alphabet using your computer's webcam in real-time.

The approach uses a two-stage process:
1. Google's MediaPipe Hands library detects the hand and extracts 21 key landmarks
2. These landmarks are then fed into our Scikit-learn model (RandomForestClassifier) trained using a kaggle dataset of ASL letter images

The application then displays the live webcam feed with the detected landmarks and the predicted letter outputted.

## Installations Needed
- Python 3.12
- 'pip'
- A webcam

## How to Run
1. Clone repository
2. Create virtual environment: -m venv venv
3. Activate it: venv\Scripts\activate or source venv\bin\activate
4. Install dependencies: pip install -r requirements.txt
This will install needed libraries to run: (OpenCV, MediaPipe, NumPy, Scikit-learn, Matplotlib)
5. You must run train.py first to run data processing
6. Once asl_classifier.pkl is generated, we can then run predict.py

## How to Use
- A window should open up showing your webcam feed
- Position your hand clearly in the form and sign any ASL letter (A-Z)
- The program will then attempt to detect your hand, draw landmarks, and display the predicted letter
- Press q key while the OpenCV window is active to exit the application