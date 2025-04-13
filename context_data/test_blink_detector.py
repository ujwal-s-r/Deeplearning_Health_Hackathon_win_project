import cv2
import numpy as np
import time
import os
from blink_detector import BlinkDetector

def test_blink_detector():
    """
    Test the blink detector with a sample video or webcam feed.
    
    This script demonstrates how to use the BlinkDetector class
    to detect blinks in real-time using MediaPipe Face Mesh.
    """
    # Initialize blink detector
    blink_detector = BlinkDetector()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Variables for FPS calculation
    start_time = time.time()
    frame_count = 0
    fps = 0
    
    # Main loop
    while True:
        # Read frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to grab frame.")
            break
        
        # Calculate FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
        
        # Process frame for blink detection
        blink_data = blink_detector.detect_blink(frame)
        
        # Visualize results
        output_frame = blink_detector.visualize_blink_detection(frame, blink_data)
        
        # Add FPS to the frame
        cv2.putText(
            output_frame,
            f"FPS: {fps:.1f}",
            (10, output_frame.shape[0] - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        # Display the frame
        cv2.imshow("Blink Detector Test", output_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_blink_detector() 