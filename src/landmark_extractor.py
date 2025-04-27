import cv2
import mediapipe as mp
import numpy as np
import os

    
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
    if hasattr(hand_landmarks_object, 'landmark'): # Check if it has the landmark attribute
        for landmark in hand_landmarks_object.landmark:
            landmark_list.append([landmark.x, landmark.y, landmark.z])
    else:
        return None # Invalid input object

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

