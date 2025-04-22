import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Hands solution once when the module is imported
mp_hands = mp.solutions.hands
hands_processor = mp_hands.Hands(
    static_image_mode=True,      # Process individual images
    max_num_hands=1,             # Expect only one hand
    min_detection_confidence=0.5 # Confidence threshold for detection
)

def extract_normalized_landmarks(image_path):
    """
    Loads an image, detects hand landmarks using MediaPipe, normalizes them 
    (relative to wrist, scaled), and returns them as a flat numpy array.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.ndarray or None: A flat numpy array of shape (63,) representing the 
                          normalized (x, y, z) landmarks if a hand is detected, 
                          otherwise None.
    """
    # --- Function logic will go here ---
        # 1. Check if image path exists
    if not os.path.exists(image_path):
        print(f"Error: Image path not found: {image_path}") # Optional warning
        return None

    # 2. Load image using OpenCV
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Error: Could not load image: {image_path}") # Optional warning
        return None

    # 3. Convert the BGR image to RGB (MediaPipe expects RGB)
    image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # --- Detection logic next ---
    # 4. Process the image and find hands
    results = hands_processor.process(image_rgb)


    landmark_array = None # Initialize landmark_array
    if results.multi_hand_landmarks:
        # Assuming only one hand (max_num_hands=1)
        hand_landmarks = results.multi_hand_landmarks[0]
        
        landmark_list = []
        for landmark in hand_landmarks.landmark:
            landmark_list.append([landmark.x, landmark.y, landmark.z])
        
        landmark_array = np.array(landmark_list) # Shape (21, 3)

        if landmark_array.shape != (21, 3):
             # print(f"Warning: Unexpected landmark array shape {landmark_array.shape} for {image_path}")
             return None 
             
    else: 
        # No hand detected
        return None
    
    # --- Normalization logic next ---
    scaled_landmarks = None # Initialize
    
    # a) Make relative to wrist (landmark 0)
    relative_landmarks = landmark_array - landmark_array[0] 

    # b) Scale based on max absolute value to roughly fit [-1, 1]
    max_abs_val = np.max(np.abs(relative_landmarks))
    if max_abs_val < 1e-6: # Avoid division by very small numbers (or zero)
        scaled_landmarks = relative_landmarks # Already all zeros or close
    else:
        scaled_landmarks = relative_landmarks / max_abs_val

        # --- Flattening and return next --- 
    if scaled_landmarks is not None:
        return scaled_landmarks.flatten() # Shape (63,)
    else:
        return None # Should not happen if normalization logic is correct, but safe fallback
    

# --- Example Usage (for testing when running this script directly) ---
if __name__ == "__main__":
    # Adjust this path based on your project structure
    # Assumes 'src' and 'data' are sibling directories
    test_image_path = 'data/raw/ASL_Alphabet_dataset/A/A1570.jpg' 

    print(f"Testing extraction on: {test_image_path}")
    
    landmarks_vector = extract_normalized_landmarks(test_image_path)

    if landmarks_vector is not None:
        print(f"Successfully extracted normalized landmarks!")
        print(f"Feature vector shape: {landmarks_vector.shape}") 
        print(f"First 5 features: {landmarks_vector[:5]}") 
    else:
        print(f"Could not extract landmarks from {test_image_path}.")

    # Optional: Test with a non-existent file
    print("\nTesting with a non-existent file:")
    landmarks_vector_none = extract_normalized_landmarks("path/to/non_existent_image.jpg")
    if landmarks_vector_none is None:
        print("Correctly returned None for non-existent file.")
    else:
        print("Error: Should have returned None for non-existent file.")