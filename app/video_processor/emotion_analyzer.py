import cv2
import numpy as np
from deepface import DeepFace
from collections import Counter
import os
import time
from typing import Dict, List, Tuple, Union, Optional

class EmotionAnalyzer:
    def __init__(self):
        """Initialize the emotion analyzer with DeepFace."""
        # Supported emotions in DeepFace
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        # Emotion map for integer coding
        self.emotion_map = {
            'neutral': 0,
            'sad': 1,
            'happy': 2,
            'angry': 3,
            'fear': 4,
            'surprise': 5,
            'disgust': 6
        }
        
        # Create CLAHE for contrast enhancement
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Emotion biases - to make detection more sensitive to certain emotions
        self.emotion_biases = {
            'surprise': 1.4,  # Boost surprise detection
            'neutral': 0.8    # Reduce neutral bias
        }
        
        # Detector backends in preference order
        self.backends = ['opencv', 'retinaface', 'mtcnn']
        
        # Error handling
        self.last_successful_result = None
        self.consecutive_failures = 0
        self.max_failures = 5
        
        # Results storage
        self.emotion_counts = {emotion: 0 for emotion in self.emotions}
        self.all_frames_emotions = []
        self.dominant_emotion = None
        self.dominant_emotion_code = None
        
        # Cache for results to avoid redundant processing
        self.result_cache = {}
        self.cache_max_size = 30  # Limit cache size to prevent memory issues
        
    def _preprocess_image(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess image to improve emotion detection.
        
        Args:
            frame: Input frame from video
            
        Returns:
            Enhanced frame with improved contrast
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            enhanced = self.clahe.apply(gray)
            
            # Convert back to BGR for DeepFace
            enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            
            return enhanced_bgr
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return frame  # Return original frame if preprocessing fails
    
    def detect_emotion(self, frame: np.ndarray) -> Dict:
        """
        Detect emotions in a single frame.
        
        Args:
            frame: Input frame from video
            
        Returns:
            Dictionary containing emotion detection data:
            - emotions: Dictionary with emotion probabilities
            - dominant_emotion: String name of dominant emotion
            - dominant_emotion_code: Integer code of dominant emotion
            - success: Boolean indicating successful detection
            - face_region: Dictionary with face coordinates (if available)
        """
        # Initialize return data
        emotion_data = {
            'emotions': None,
            'dominant_emotion': None,
            'dominant_emotion_code': None,
            'success': False,
            'face_region': None
        }
        
        # Check if frame is valid
        if frame is None or frame.size == 0:
            print("Invalid frame provided to detect_emotion")
            if self.last_successful_result:
                result = self.last_successful_result.copy()
                result['fresh'] = False
                return result
            return emotion_data
        
        # Generate a cache key based on frame data
        try:
            small_frame = cv2.resize(frame, (100, 100))  # Downsize for faster hashing
            cache_key = hash(small_frame.tobytes())
            
            # Check if result is in cache
            if cache_key in self.result_cache:
                return self.result_cache[cache_key]
                
        except Exception as e:
            print(f"Error generating cache key: {e}")
            cache_key = None  # Disable caching for this frame
            
        try:
            # Pre-process the frame to improve face detection
            enhanced_frame = self._preprocess_image(frame)
            
            # Initialize lists to store results
            all_results = []
            all_emotions = {}
            face_region = None
            
            # Try different backends
            for backend in self.backends:
                try:
                    # Original frame analysis
                    result_original = DeepFace.analyze(
                        img_path=frame,
                        actions=['emotion'],
                        enforce_detection=False,
                        detector_backend=backend,
                        prog_bar=False
                    )
                    
                    # Enhanced frame analysis
                    result_enhanced = DeepFace.analyze(
                        img_path=enhanced_frame,
                        actions=['emotion'],
                        enforce_detection=False,
                        detector_backend=backend,
                        prog_bar=False
                    )
                    
                    # Standardize result format
                    results_list = []
                    
                    # Handle both single and multi-face results
                    if isinstance(result_original, dict):
                        results_list.append(result_original)
                        # Store face region for visualization
                        face_region = result_original.get('region', None)
                    else:
                        results_list.extend(result_original)
                        # Store first face region if available
                        if result_original and len(result_original) > 0:
                            face_region = result_original[0].get('region', None)
                        
                    if isinstance(result_enhanced, dict):
                        results_list.append(result_enhanced)
                    else:
                        results_list.extend(result_enhanced)
                    
                    # Add to all results
                    all_results.extend(results_list)
                    
                    # Break if we found at least one face
                    if results_list:
                        break
                        
                except Exception as e:
                    print(f"Backend {backend} failed: {str(e)}")
                    continue
            
            # If no results were obtained
            if not all_results:
                self.consecutive_failures += 1
                if self.last_successful_result:
                    result = self.last_successful_result.copy()
                    result['fresh'] = False
                    return result
                return emotion_data
            
            # Reset failures counter on success
            self.consecutive_failures = 0
            
            # Combine emotion scores from all results
            for result in all_results:
                for emotion, score in result['emotion'].items():
                    if emotion in all_emotions:
                        all_emotions[emotion] = all_emotions[emotion] + score
                    else:
                        all_emotions[emotion] = score
            
            # Average the scores
            for emotion in all_emotions:
                all_emotions[emotion] = all_emotions[emotion] / len(all_results)
            
            # Apply emotion biases
            for emotion, bias in self.emotion_biases.items():
                if emotion in all_emotions:
                    all_emotions[emotion] *= bias
            
            # Re-normalize to ensure sum is close to 100%
            total = sum(all_emotions.values())
            normalized_emotions = {
                emotion: (score / total) * 100 
                for emotion, score in all_emotions.items()
            }
            
            # Determine dominant emotion
            dominant_emotion = max(normalized_emotions, key=normalized_emotions.get)
            dominant_emotion_code = self.emotion_map.get(dominant_emotion, -1)
            
            # Populate return data
            emotion_data['emotions'] = normalized_emotions
            emotion_data['dominant_emotion'] = dominant_emotion
            emotion_data['dominant_emotion_code'] = dominant_emotion_code
            emotion_data['success'] = True
            emotion_data['face_region'] = face_region
            emotion_data['fresh'] = True
            
            # Store as last successful result
            self.last_successful_result = emotion_data.copy()
            
            # Cache the result if caching is enabled
            if cache_key is not None:
                self.result_cache[cache_key] = emotion_data.copy()
                
                # Limit cache size
                if len(self.result_cache) > self.cache_max_size:
                    # Remove an arbitrary key (first one)
                    self.result_cache.pop(next(iter(self.result_cache)))
            
        except Exception as e:
            # Handle errors gracefully
            print(f"Error in advanced emotion detection: {e}")
            self.consecutive_failures += 1
            if self.last_successful_result:
                # Return last successful result on error
                result = self.last_successful_result.copy()
                result['fresh'] = False
                return result
        
        return emotion_data

    def process_video(self, video_path: str, output_path: Optional[str] = None, fps: int = 10) -> int:
        """
        Process a video file to detect emotions at the specified FPS.
        
        Args:
            video_path: Path to the video file
            output_path: Optional path to save processed video
            fps: Frames per second to process (default 10)
            
        Returns:
            The dominant emotion code across the entire video
        """
        print(f"Starting video processing: {video_path}")
        
        # Reset counts
        self.emotion_counts = {emotion: 0 for emotion in self.emotions}
        self.all_frames_emotions = []
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return 0  # Return neutral as default
            
        # Get video properties
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, int(video_fps / fps))  # Process at specified FPS, minimum 1
        
        # Initialize writer if output is specified
        writer = None
        if output_path:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        processed_count = 0
        
        print(f"Video properties: {video_fps} FPS, {total_frames} frames")
        print(f"Processing at {fps} FPS (every {frame_interval} frames)")
        
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                
                # Process only every n-th frame to achieve target FPS
                if frame_count % frame_interval != 0:
                    continue
                    
                processed_count += 1
                
                # Detect emotion in the frame
                emotion_data = self.detect_emotion(frame)
                
                if emotion_data['success']:
                    # Track the dominant emotion
                    dominant_emotion = emotion_data['dominant_emotion']
                    self.emotion_counts[dominant_emotion] += 1
                    self.all_frames_emotions.append(dominant_emotion)
                    
                    # Update visualization if needed
                    if writer:
                        vis_frame = self.visualize_emotion_detection(frame, emotion_data)
                        writer.write(vis_frame)
                
                # Print progress periodically
                if processed_count % 10 == 0:
                    elapsed = time.time() - start_time
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% - Processed {processed_count} frames in {elapsed:.1f}s")
        
        except Exception as e:
            print(f"Error during video processing: {e}")
        
        finally:
            # Clean up
            cap.release()
            if writer:
                writer.release()
            
            # Calculate the dominant emotion across all frames
            if self.all_frames_emotions:
                most_common = Counter(self.all_frames_emotions).most_common(1)[0][0]
                self.dominant_emotion = most_common
                self.dominant_emotion_code = self.emotion_map[most_common]
                
                print(f"Analysis complete. Dominant emotion: {self.dominant_emotion} (code: {self.dominant_emotion_code})")
                print(f"Emotion distribution: {self.emotion_counts}")
            else:
                print("No emotions detected in the video")
                self.dominant_emotion = "neutral"
                self.dominant_emotion_code = 0
                
            return self.dominant_emotion_code
        
    def visualize_emotion_detection(self, frame: np.ndarray, emotion_data: Dict) -> np.ndarray:
        """
        Visualize emotion detection results on a frame.
        
        Args:
            frame: Input frame from video
            emotion_data: Dictionary from detect_emotion
            
        Returns:
            Frame with emotion visualization
        """
        output_frame = frame.copy()
        
        if emotion_data['success']:
            # Get emotion data
            emotions = emotion_data['emotions']
            dominant_emotion = emotion_data['dominant_emotion']
            
            # Draw emotion label with freshness indicator
            freshness_indicator = ""
            if 'fresh' in emotion_data and not emotion_data.get('fresh', True):
                freshness_indicator = " (cached)"
                
            cv2.putText(
                output_frame,
                f"Emotion: {dominant_emotion}{freshness_indicator}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # Draw emotion probabilities
            y_offset = 60
            for emotion, probability in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
                # Format probability as percentage
                text = f"{emotion}: {probability:.1f}%"
                
                cv2.putText(
                    output_frame,
                    text,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )
                y_offset += 25
                
            # Try to draw face region if available
            face_region = emotion_data.get('face_region', None)
            if face_region:
                try:
                    x, y, w, h = face_region['x'], face_region['y'], face_region['w'], face_region['h']
                    cv2.rectangle(output_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                except Exception as e:
                    print(f"Error drawing face region: {e}")
        else:
            # No emotion detected
            error_msg = "No face detected"
            if 'error' in emotion_data:
                error_msg = f"Error: {emotion_data['error']}"
                
            cv2.putText(
                output_frame,
                error_msg,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
        
        return output_frame 