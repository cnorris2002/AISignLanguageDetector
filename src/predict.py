import cv2
import mediapipe as mp
import numpy as np
import joblib
import os


try:
    # Using get_normalized_vector values from landmark_extractor.py
    # We will use these vectors for our model
    from landmark_extractor import get_normalized_vector
    print("Successfully imported get_normalized_vector function.")
except ImportError:
    print("Error. Could not import get_normalized_vector from landmark_extractor.py")
    get_normalized_vector = None


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# This will load our model and classes
MODEL_PATH = './models/asl_classifier.pkl'
CLASS_NAMES_PATH = './data/processed/class_names.npy'

print("Loading trained model...")
if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASS_NAMES_PATH):
    print(f"ERROR: Model ('{MODEL_PATH}') or Class Names ('{CLASS_NAMES_PATH}') not found.")
    print("Please run the training script (src/train.py) first.") # to make sure the training model is ran first
    exit()

try:
    model = joblib.load(MODEL_PATH)
    class_names = np.load(CLASS_NAMES_PATH)
    print("Model and class names loaded successfully.")
    print(f"Model type: {type(model)}")
    print(f"Available classes: {class_names}") # letters a-z labeled as "classes"
except Exception as e:
    print(f"Error loading model or class names: {e}")
    exit()


print("Initializing MediaPipe Hands for video...")
# We can adjust confidence levels if needed for real-time tracking
hands_detector_realtime = mp_hands.Hands(
    static_image_mode=False,     
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) 
print("MediaPipe Hands initialized.")

# Open the webcam
print("Initializing webcam...")
cap = cv2.VideoCapture(0) 
if not cap.isOpened():
    print("ERROR: Cannot open webcam.")
    exit()
print("Webcam initialized.")


print("Starting real-time prediction loop... Press 'q' to quit.")

while True:
    # Read whatever is on the screen
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    frame = cv2.flip(frame, 1) 

    # Landmark detection
    frame.flags.writeable = False
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector_realtime.process(image_rgb)
    frame.flags.writeable = True

    predicted_letter = "?" # Default to ? if nothing is recognized

    # Extracts landmarks and normalize them
    landmark_vector = None
    if results.multi_hand_landmarks:
        # Print to check if hand is detected
        # print("Hand Detected!", end='\r')
        hand_landmarks = results.multi_hand_landmarks[0] # Get landmarks for the first hand

        # function from landmark_extractor.py
        if get_normalized_vector:
            landmark_vector = get_normalized_vector(hand_landmarks)
            if landmark_vector is None:
                print("get_normalized_vector returned None", end='\r')
            else:
                print(f"Got landmark vector, shape: {landmark_vector.shape}", end='\r')
        else:
            landmark_vector = None

        # Visualition of the landmarks (dots)
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    
    else:
        print("No Hand Detected", end='\r')

    # This is where prediction happens, based on landmarks
    if landmark_vector is not None:
        try:
            prediction_index = model.predict([landmark_vector])[0] 
            predicted_letter = class_names[prediction_index]
        except Exception as e:
            print(f"Error during prediction: {e}")
            predicted_letter = "Error"

    # Display the predicted letter
    cv2.putText(frame, f"Prediction: {predicted_letter}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show frame
    cv2.imshow('ASL Letter Recognition', frame)

    # Exit
    if cv2.waitKey(5) & 0xFF == ord('q'):
        print("\nExiting...")
        break


# Release resources
cap.release()
cv2.destroyAllWindows()
print("Resources released.")