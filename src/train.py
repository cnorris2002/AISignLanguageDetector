import numpy as np
import os
import joblib
import time
from sklearn.ensemble import RandomForestClassifier # our chosen classifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt # For plotting confusion matrix

# Constants and Parameters
PROCESSED_DATA_DIR = 'data/processed/'
X_SAVE_PATH = os.path.join(PROCESSED_DATA_DIR, 'X_landmarks.npy')
Y_SAVE_PATH = os.path.join(PROCESSED_DATA_DIR, 'y_labels.npy')
CLASS_NAMES_PATH = os.path.join(PROCESSED_DATA_DIR, 'class_names.npy')
MODEL_SAVE_PATH = 'models/asl_classifier.pkl'
CONFUSION_MATRIX_SAVE_PATH = 'confusion_matrix.png'

# Just making sure model directory exists
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

MODEL_CHOICE = RandomForestClassifier
MODEL_PARAMS = {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1} 

TEST_SPLIT_SIZE = 0.2
RANDOM_STATE = 42


# Load processed data, train and evaluate using the selected classifier
def train_model():
    
    # Load pre-processed data from npy files
    print(f"Loading pre-processed data from {PROCESSED_DATA_DIR}...")
    if not all(os.path.exists(p) for p in [X_SAVE_PATH, Y_SAVE_PATH, CLASS_NAMES_PATH]):
        print("Error: Pre-processed data files (.npy) not found.")
        print(f"Please run the data processing step (e.g., from the notebook) first to generate:")
        print(f"  - {X_SAVE_PATH}")
        print(f"  - {Y_SAVE_PATH}")
        print(f"  - {CLASS_NAMES_PATH}")
        return

    try:
        X = np.load(X_SAVE_PATH)
        y = np.load(Y_SAVE_PATH)
        class_names = np.load(CLASS_NAMES_PATH)
        print(f"Loaded X shape: {X.shape}, y shape: {y.shape}")
        print(f"Class names: {class_names}")
    except Exception as e:
        print(f"Error loading .npy files: {e}")
        return

    # Split the data
    print("\nSplitting data into train and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=TEST_SPLIT_SIZE, 
        random_state=RANDOM_STATE,
        stratify=y
    )
    print(f"Train set size: {X_train.shape[0]} samples")
    print(f"Validation set size: {X_val.shape[0]} samples")

    # Initialize and train our model
    print("\nInitializing and training model...")
    model = MODEL_CHOICE(**MODEL_PARAMS)
    print(f"Using model: {model}")
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    print(f"Model training finished in {end_time - start_time:.2f} seconds.")

    # Then evaluate model
    print("\nEvaluating model on validation data...")
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    try:
        # Using class_names loaded earlier
        report = classification_report(y_val, y_pred, target_names=class_names, zero_division=0)
        print(report)
    except Exception as e:
        print(f"Could not generate full classification report: {e}")

    # Code for confusion matrix generation
    print ("\nGenerating confusion matrix...")
    try:
        cm = confusion_matrix(y_val, y_pred)
        display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

        # for the plot
        fig, ax = plt.subplots(figsize=(12,12))
        display.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical') 
        plt.title("Confusion Matrix for ASL Alphabet Classification")

        # Save the plot to a file
        plt.savefig(CONFUSION_MATRIX_SAVE_PATH, bbox_inches='tight')
        print("Confusion matrix plot saved.")
        plt.close(fig)

    except Exception as e:
        print(f"Could not generate or save confusion matrix plot: {e}")

    # We then save the trained model
    print(f"\nSaving the trained model to: {MODEL_SAVE_PATH}")
    try:
        joblib.dump(model, MODEL_SAVE_PATH)
        print("Model saved successfully.")
    except Exception as e:
        print(f"Error saving model: {e}")

# Main execution block
if __name__ == "__main__":
    print("Starting model training process...")
    train_model()
    print("\nTraining process finished.")