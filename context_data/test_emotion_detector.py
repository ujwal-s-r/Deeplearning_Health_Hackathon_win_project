import cv2
import time
import numpy as np
import threading
import queue
import os
import tkinter as tk
from tkinter import filedialog
from emotion_detector import EmotionDetector

# Global variables for threaded capture
frame_queue = queue.Queue(maxsize=1)
stop_event = threading.Event()

def camera_capture_thread(camera_id=0, width=640, height=480, fps=30):
    """Thread function for camera capture to avoid blocking the main thread"""
    print(f"Starting camera capture thread with camera ID {camera_id}")
    
    # Try to open the camera
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        stop_event.set()  # Signal to stop
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    
    # Calculate the target frame time based on FPS
    frame_time = 1.0 / fps
    
    print(f"Camera initialized with resolution {width}x{height} at {fps} FPS")
    
    try:
        while not stop_event.is_set():
            # Read a frame with timeout
            start_time = time.time()
            ret, frame = cap.read()
            
            if not ret:
                print("Warning: Failed to capture frame")
                # Retry a few times before giving up
                retry_count = 0
                while not ret and retry_count < 5 and not stop_event.is_set():
                    time.sleep(0.1)  # Wait a bit before retrying
                    ret, frame = cap.read()
                    retry_count += 1
                
                if not ret:
                    print("Error: Could not capture frame after retries, releasing camera")
                    # Try to reopen the camera
                    cap.release()
                    time.sleep(0.5)
                    cap = cv2.VideoCapture(camera_id)
                    if not cap.isOpened():
                        print("Error: Failed to reopen camera, stopping thread")
                        stop_event.set()
                        break
                    continue
            
            # Put the frame in the queue, replacing any previous frame
            try:
                # Non-blocking put with clearing
                if frame_queue.full():
                    # Discard the old frame if we can't keep up
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                frame_queue.put_nowait(frame)
            except queue.Full:
                # Queue is full, just continue
                pass
            
            # Sleep to maintain the target frame rate
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_time - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    except Exception as e:
        print(f"Error in camera thread: {e}")
    finally:
        # Clean up
        cap.release()
        print("Camera thread stopped")

def select_photo():
    """Open a file dialog to select a photo"""
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(
        title="Select a photo to analyze",
        filetypes=[
            ("Image files", "*.jpg;*.jpeg;*.png;*.bmp"),
            ("All files", "*.*")
        ]
    )
    root.destroy()
    return file_path

def test_emotion_detector():
    """
    Test the EmotionDetector with webcam input.
    Displays live video with emotion detection visualization.
    Press 'q' to quit, 'p' to analyze a photo.
    """
    # Initialize EmotionDetector
    print("Initializing Advanced EmotionDetector...")
    emotion_detector = EmotionDetector()
    
    # Create output directory for photo analysis
    output_dir = "emotion_analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # For FPS calculation
    prev_time = time.time()
    frame_count = 0
    
    # For emotion history tracking
    emotion_history = []
    max_history = 30  # Keep track of the last 30 frames
    
    # Start camera capture thread
    stop_event.clear()  # Make sure it's cleared before starting
    camera_thread = threading.Thread(target=camera_capture_thread, args=(0, 640, 480, 30))
    camera_thread.daemon = True
    camera_thread.start()
    
    # Wait for camera to initialize
    time.sleep(1.0)
    
    # Global flag for showing photo
    showing_photo = False
    photo_frame = None
    photo_result = None
    
    print("Starting advanced emotion detection.")
    print("Controls:")
    print("  'q' - Quit the program")
    print("  'p' - Select and analyze a photo")
    print("  'r' - Return to live view from photo")
    
    try:
        while not stop_event.is_set():
            start_loop = time.time()
            
            # If not showing a photo, get frame from the queue
            if not showing_photo:
                try:
                    # Try to get a frame with a short timeout
                    frame = frame_queue.get(timeout=0.1)
                    frame_available = True
                except queue.Empty:
                    print("Warning: No frame available")
                    frame_available = False
                    # Create a blank frame to show an error message
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(
                        frame,
                        "Camera feed unavailable",
                        (50, 240),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2
                    )
            else:
                # If showing a photo, use the stored photo frame
                frame = photo_frame
                frame_available = True
            
            # Start timing for processing
            start_process = time.time()
            
            # Process the frame if available
            if frame_available:
                if not showing_photo:
                    # Real-time emotion detection
                    emotion_data = emotion_detector.detect_emotion(frame)
                    
                    # Update emotion history
                    if emotion_data['success']:
                        # Add to history with timestamp
                        emotion_history.append({
                            'time': time.time(),
                            'emotion': emotion_data['dominant_emotion']
                        })
                        
                        # Keep history limited to max_history
                        if len(emotion_history) > max_history:
                            emotion_history.pop(0)
                else:
                    # Use the stored photo result
                    emotion_data = photo_result
                
                # Visualize results
                output_frame = emotion_detector.visualize_emotion_detection(frame, emotion_data)
                
                # Add processing time
                process_time = time.time() - start_process
                cv2.putText(
                    output_frame,
                    f"Process time: {process_time*1000:.1f}ms",
                    (10, output_frame.shape[0] - 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1
                )
                
                if showing_photo:
                    # Add photo mode indicator
                    cv2.putText(
                        output_frame,
                        "PHOTO MODE - Press 'r' to return to live view",
                        (10, output_frame.shape[0] - 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1
                    )
                else:
                    # Calculate and display FPS (only in live view)
                    frame_count += 1
                    current_time = time.time()
                    elapsed = current_time - prev_time
                    
                    if elapsed >= 1.0:
                        fps = frame_count / elapsed
                        frame_count = 0
                        prev_time = current_time
                        
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
                
                # Only display enhanced image and emotion history in real-time mode
                if not showing_photo:
                    # Display preprocessed image
                    try:
                        # Create small preprocessed image to display in corner
                        enhanced_frame = emotion_detector._preprocess_image(frame)
                        h, w = output_frame.shape[:2]
                        small_enhanced = cv2.resize(enhanced_frame, (w//5, h//5))
                        
                        # Add text label for preprocessing view
                        cv2.putText(
                            small_enhanced,
                            "CLAHE Enhanced",
                            (5, 15),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (0, 0, 255),
                            1
                        )
                        
                        # Overlay in top-right corner
                        y_offset, x_offset = 10, output_frame.shape[1] - small_enhanced.shape[1] - 10
                        output_frame[y_offset:y_offset+small_enhanced.shape[0], 
                                    x_offset:x_offset+small_enhanced.shape[1]] = small_enhanced
                    except Exception as e:
                        print(f"Error showing preprocessed image: {e}")
                    
                    # Display emotion history as a simple bar at the bottom
                    if emotion_history:
                        # Define colors for emotions
                        colors = {
                            'angry': (0, 0, 255),    # Red
                            'disgust': (0, 128, 128), # Brown
                            'fear': (255, 0, 255),   # Purple
                            'happy': (0, 255, 255),  # Yellow
                            'sad': (255, 0, 0),      # Blue
                            'surprise': (0, 255, 0),  # Green
                            'neutral': (128, 128, 128) # Gray
                        }
                        
                        # Draw bar at bottom
                        bar_height = 20
                        bar_y = output_frame.shape[0] - bar_height - 5
                        bar_width = output_frame.shape[1] - 20
                        bar_x = 10
                        
                        # Draw background
                        cv2.rectangle(output_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                                    (50, 50, 50), -1)
                        
                        # Draw emotion segments
                        for i, entry in enumerate(emotion_history):
                            segment_width = bar_width // max_history
                            emotion_color = colors.get(entry['emotion'], (200, 200, 200))
                            
                            # Draw segment
                            segment_x = bar_x + i * segment_width
                            cv2.rectangle(output_frame, 
                                        (segment_x, bar_y), 
                                        (segment_x + segment_width, bar_y + bar_height), 
                                        emotion_color, -1)
                
                # Display the frame
                window_title = "Advanced Emotion Detection - Photo Mode" if showing_photo else "Advanced Emotion Detection - Live View"
                cv2.imshow(window_title, output_frame)
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            
            # Process key actions
            if key == ord('q'):
                print("Quitting...")
                stop_event.set()
                break
            elif key == ord('p') and not showing_photo:
                # Switch to photo mode
                print("Selecting photo...")
                photo_path = select_photo()
                if photo_path and os.path.exists(photo_path):
                    print(f"Analyzing photo: {photo_path}")
                    photo_result = emotion_detector.analyze_photo(photo_path)
                    if photo_result['success']:
                        # Load the photo for display
                        photo_frame = cv2.imread(photo_path)
                        showing_photo = True
                        print(f"Photo analysis complete: {photo_result['dominant_emotion']}")
                    else:
                        print(f"Failed to analyze photo: {photo_result.get('error', 'Unknown error')}")
                else:
                    print("No photo selected or file not found")
            elif key == ord('r') and showing_photo:
                # Return to live view
                showing_photo = False
                photo_frame = None
                photo_result = None
                print("Returning to live view")
            
            # Calculate loop time and add a small delay if needed
            loop_time = time.time() - start_loop
            if loop_time < 0.0333:  # Target ~30fps for display
                time.sleep(0.0333 - loop_time)
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
        # Clean up
        stop_event.set()
        if camera_thread.is_alive():
            camera_thread.join(timeout=1.0)
        cv2.destroyAllWindows()
        print("Test completed.")

if __name__ == "__main__":
    test_emotion_detector() 