"""
Live Webcam Candy Classification
Test the trained model in real-time using your webcam
"""

import tensorflow
import cv2 as cv
import numpy as np
from pathlib import Path

# Load the trained model
SCRIPT_DIR = Path(__file__).parent
MODEL_PATH = SCRIPT_DIR / "DNN_4_Candy_Model.keras"

print(f"Loading model from: {MODEL_PATH}")
model = tensorflow.keras.models.load_model(str(MODEL_PATH))

# Classes (alphabetical order as used by image_dataset_from_directory)
classes = ["Bad", "Good", "Ugly"]
class_colors = {
    "Good": (0, 255, 0),    # Green
    "Bad": (0, 0, 255),     # Red
    "Ugly": (0, 165, 255)   # Orange
}


def predict_frame(frame):
    """Predict the class of a frame"""
    # Resize to model input size
    img_resized = cv.resize(frame, (128, 128))
    # Convert BGR to RGB
    img_rgb = cv.cvtColor(img_resized, cv.COLOR_BGR2RGB)
    # Add batch dimension
    img_array = np.expand_dims(img_rgb, axis=0)
    
    # Predict
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][predicted_class_idx]
    
    return classes[predicted_class_idx], confidence, predictions[0]


def main():
    # Open webcam (try index 0, if not working try 1)
    cap = cv.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam. Trying camera index 1...")
        cap = cv.VideoCapture(1)
        if not cap.isOpened():
            print("Error: Could not open any webcam!")
            return
    
    print("\n" + "="*50)
    print("LIVE CANDY CLASSIFICATION")
    print("="*50)
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save current frame")
    print("  'p' - Pause/Resume")
    print("="*50 + "\n")
    
    paused = False
    frame_count = 0
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
        
        # Make prediction
        predicted_class, confidence, all_probs = predict_frame(frame)
        color = class_colors[predicted_class]
        
        # Create display frame
        display_frame = frame.copy()
        
        # Draw prediction box at top
        cv.rectangle(display_frame, (10, 10), (350, 130), (0, 0, 0), -1)
        cv.rectangle(display_frame, (10, 10), (350, 130), color, 2)
        
        # Main prediction text
        cv.putText(display_frame, f"Prediction: {predicted_class}", 
                   (20, 45), cv.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv.putText(display_frame, f"Confidence: {confidence:.1%}", 
                   (20, 75), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show all class probabilities
        prob_text = f"Bad:{all_probs[0]:.0%} Good:{all_probs[1]:.0%} Ugly:{all_probs[2]:.0%}"
        cv.putText(display_frame, prob_text, 
                   (20, 110), cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Draw center crosshair to help align candy
        h, w = display_frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        cv.line(display_frame, (center_x - 30, center_y), (center_x + 30, center_y), (255, 255, 255), 1)
        cv.line(display_frame, (center_x, center_y - 30), (center_x, center_y + 30), (255, 255, 255), 1)
        
        # Draw 128x128 preview box (what the model sees)
        preview_size = 128
        preview_x = w - preview_size - 20
        preview_y = 20
        preview = cv.resize(frame, (preview_size, preview_size))
        display_frame[preview_y:preview_y+preview_size, preview_x:preview_x+preview_size] = preview
        cv.rectangle(display_frame, (preview_x-2, preview_y-2), 
                     (preview_x+preview_size+2, preview_y+preview_size+2), (255, 255, 255), 2)
        cv.putText(display_frame, "Model Input", 
                   (preview_x, preview_y + preview_size + 20), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Pause indicator
        if paused:
            cv.putText(display_frame, "PAUSED", (w//2 - 50, h - 30), 
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Show frame
        cv.imshow("Live Candy Classification - Press 'q' to quit", display_frame)
        
        # Handle key presses
        key = cv.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save current frame
            filename = f"capture_{frame_count}.jpg"
            cv.imwrite(filename, frame)
            print(f"Saved: {filename} - Predicted: {predicted_class} ({confidence:.1%})")
            frame_count += 1
        elif key == ord('p'):
            paused = not paused
            print("Paused" if paused else "Resumed")
    
    cap.release()
    cv.destroyAllWindows()
    print("\nCamera closed.")


if __name__ == "__main__":
    main()
