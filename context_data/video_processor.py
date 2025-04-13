import cv2
import os
import numpy as np
from typing import Tuple, List, Dict, Any

def record_user_reaction(stimulus_video_path: str, output_path: str) -> bool:
    """
    Record user's reaction while playing the stimulus video.
    
    Args:
        stimulus_video_path: Path to the stimulus video file
        output_path: Path to save the recorded reaction video
        
    Returns:
        bool: True if recording was successful, False otherwise
    """
    try:
        # Open webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return False
        
        # Get webcam properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        # Open stimulus video
        stimulus = cv2.VideoCapture(stimulus_video_path)
        if not stimulus.isOpened():
            print("Error: Could not open stimulus video")
            cap.release()
            return False
        
        # Record while playing stimulus
        while stimulus.isOpened():
            ret_stimulus, frame_stimulus = stimulus.read()
            ret_webcam, frame_webcam = cap.read()
            
            if not ret_stimulus or not ret_webcam:
                break
            
            # Display stimulus video (can be shown in a window in development)
            cv2.imshow('Stimulus', frame_stimulus)
            
            # Write webcam frame to output video
            out.write(frame_webcam)
            
            # Break loop if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        stimulus.release()
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        return True
    
    except Exception as e:
        print(f"Error recording user reaction: {e}")
        return False

def extract_frames(video_path: str, output_dir: str, sample_rate: int = 1) -> List[str]:
    """
    Extract frames from a video file.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save the extracted frames
        sample_rate: Extract every nth frame (1 = all frames)
        
    Returns:
        List of saved frame file paths
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_path}")
            return []
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frames to extract
        frames_to_extract = range(0, total_frames, sample_rate)
        frame_paths = []
        
        # Extract frames
        for i in frames_to_extract:
            # Set position to the specific frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Calculate timestamp in seconds
            timestamp = i / fps
            
            # Save frame with timestamp in filename
            frame_filename = f"frame_{i:04d}_{timestamp:.2f}s.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            
            frame_paths.append(frame_path)
        
        # Release resources
        cap.release()
        
        return frame_paths
    
    except Exception as e:
        print(f"Error extracting frames: {e}")
        return []

def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Preprocess a frame for model input.
    
    Args:
        frame: Input frame (numpy array from OpenCV)
        
    Returns:
        Preprocessed frame
    """
    try:
        # Resize to a standard size (e.g., 640x480)
        frame = cv2.resize(frame, (640, 480))
        
        # Convert to RGB (models often expect RGB, while OpenCV uses BGR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        return frame_rgb
    
    except Exception as e:
        print(f"Error preprocessing frame: {e}")
        return frame  # Return original frame if preprocessing fails

def get_frame_timestamp_map(frames_dir: str) -> Dict[str, float]:
    """
    Create a mapping of frame filenames to their timestamps.
    
    Args:
        frames_dir: Directory containing extracted frames
        
    Returns:
        Dictionary mapping frame filenames to timestamps
    """
    frame_timestamp_map = {}
    
    try:
        # Get all frame files
        frame_files = [f for f in os.listdir(frames_dir) if f.startswith('frame_') and f.endswith('.jpg')]
        
        for frame_file in frame_files:
            # Extract timestamp from filename (assuming format frame_XXXX_XX.XXs.jpg)
            parts = frame_file.split('_')
            if len(parts) >= 3:
                timestamp_str = parts[2].replace('s.jpg', '')
                try:
                    timestamp = float(timestamp_str)
                    frame_timestamp_map[frame_file] = timestamp
                except ValueError:
                    pass
        
        return frame_timestamp_map
    
    except Exception as e:
        print(f"Error creating frame timestamp map: {e}")
        return {} 