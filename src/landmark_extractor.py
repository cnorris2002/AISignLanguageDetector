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


    
def get_normalized_vector(hand_landmarks_object):
    """
    Takes MediaPipe hand landmarks object, extracts the coordinates
    and normalizes them (relatice to wrist), and returns them as a flat numpy
    array.

    Args:
        hand_landmarks_object: Specific hand_landmarks object from MediaPipe's
        results

    Returns:
        np.ndarray: A flat numpy array of shape (63,) representing the
        normalized (x,y,z) landmarks
    """
    if hand_landmarks_object is None:
        return None
    
    landmark_list = []

    # We then extract landmarks into a list
    for landmark in hand_landmarks_object.landmark:
        landmark_list.append([landmark.x, landmark.y, landmark.z])

    # Then convert to NumPy array
    landmark_array = np.array(landmark_list) # expected to be (21,3)

    # Same normalization as original function
    scaled_landmarks = None # used to initialize
    if landmark_array.shape == (21,3):
        # Should make it relative to the wrist
        relative_landmarks = landmark_array - landmark_array[0]

        # Scale based on max absolute value to roughly fit [-1, 1]
        max_abs_val = np.max(np.abs(relative_landmarks))
        if max_abs_val < 1e-6: 
            scaled_landmarks = relative_landmarks
        else:
            scaled_landmarks = relative_landmarks / max_abs_val

        # Flatten values and then return
        return scaled_landmarks.flatten()
    else:
        return None


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