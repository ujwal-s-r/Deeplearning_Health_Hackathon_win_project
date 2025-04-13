import cv2
import time
import argparse
from iris_tracker import IrisTracker

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test the iris tracking module.')
    parser.add_argument('--video', type=str, help='Path to video file (default: use webcam)')
    parser.add_argument('--output', type=str, help='Path to save output video (optional)')
    args = parser.parse_args()
    
    # Initialize iris tracker
    iris_tracker = IrisTracker()
    
    # Open video source (webcam or file)
    if args.video:
        cap = cv2.VideoCapture(args.video)
        print(f"Reading from video file: {args.video}")
    else:
        cap = cv2.VideoCapture(0)  # Use default webcam
        print("Reading from webcam")
    
    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Initialize video writer if output specified
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"Writing output to: {args.output}")
    else:
        out = None
    
    # Process frames
    frame_count = 0
    start_time = time.time()
    
    while True:
        # Read frame
        ret, frame = cap.read()
        
        if not ret:
            print("End of video stream")
            break
        
        # Process frame for iris tracking
        iris_data = iris_tracker.detect_iris(frame)
        
        # Visualize results
        output_frame = iris_tracker.visualize_iris_tracking(frame, iris_data)
        
        # Add processing information
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps_text = f"FPS: {frame_count / elapsed_time:.2f}"
        
        cv2.putText(
            output_frame,
            fps_text,
            (10, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        # Display the frame
        cv2.imshow('Iris Tracking Test', output_frame)
        
        # Write frame to output video if specified
        if out:
            out.write(output_frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("User interrupted")
            break
    
    # Clean up
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    
    # Print processing statistics
    if frame_count > 0:
        final_fps = frame_count / elapsed_time
        print(f"Processed {frame_count} frames in {elapsed_time:.2f} seconds ({final_fps:.2f} FPS)")

if __name__ == "__main__":
    main() 