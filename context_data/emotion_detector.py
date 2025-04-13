import cv2
import numpy as np
from deepface import DeepFace
from typing import Dict, List, Tuple, Union, Optional
import time
import os
from collections import Counter

class EmotionDetector:
    def __init__(self):
        """Initialize the emotion detector with DeepFace."""
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
        
        # Cache for results to avoid redundant processing
        self.result_cache = {}
        self.cache_max_size = 30  # Limit cache size to prevent memory issues
        
        # Detector backends in preference order
        self.backends = ['opencv', 'retinaface', 'mtcnn']
        
        # Create CLAHE for contrast enhancement
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Emotion biases - to make detection more sensitive to certain emotions
        self.emotion_biases = {
            'surprise': 1.4,
            'neutral':0.8    # Boost surprise detection    # Reduce neutral bias
        }
        
        # Error handling
        self.last_successful_result = None
        self.consecutive_failures = 0
        self.max_failures = 5  # Reset camera after this many failures
    
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
        Detect emotions in a frame using advanced DeepFace methods.
        
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
                # Return last successful result with a flag indicating it's not fresh
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
                    # Return last successful result with a flag indicating it's not fresh
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

    def analyze_photo(self, photo_path: str) -> Dict:
        """
        Analyze emotions in a photo file.
        
        Args:
            photo_path: Path to the photo file
            
        Returns:
            Dictionary containing emotion analysis results
        """
        try:
            # Read image from path
            frame = cv2.imread(photo_path)
            if frame is None:
                return {
                    'success': False,
                    'error': f"Failed to load image from {photo_path}"
                }
            
            # Call the regular detect_emotion function
            emotion_data = self.detect_emotion(frame)
            
            # Add the image path to the result
            emotion_data['photo_path'] = photo_path
            
            # Create a visualization
            if emotion_data['success']:
                visualization = self.visualize_emotion_detection(frame, emotion_data)
                
                # Save the visualization
                vis_filename = os.path.splitext(photo_path)[0] + "_analyzed.jpg"
                cv2.imwrite(vis_filename, visualization)
                emotion_data['visualization_path'] = vis_filename
            
            return emotion_data
            
        except Exception as e:
            print(f"Error analyzing photo {photo_path}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def analyze_emotion_sequence(self, emotions_over_time: List[Dict], timestamps: List[float]) -> Dict:
        """
        Analyze a sequence of emotions detected over time.
        
        Args:
            emotions_over_time: List of emotion data dictionaries
            timestamps: List of corresponding timestamps
            
        Returns:
            Dictionary with analysis results:
            - emotion_frequencies: Count of each emotion
            - dominant_emotion: Most common emotion over all frames
            - emotion_changes: Number of times dominant emotion changed
            - emotion_distribution: Percentage of time spent in each emotion
        """
        if not emotions_over_time or len(emotions_over_time) != len(timestamps):
            return {
                'emotion_frequencies': {},
                'dominant_emotion': None,
                'emotion_changes': 0,
                'emotion_distribution': {}
            }
        
        # Count emotion frequencies
        emotion_frequencies = {emotion: 0 for emotion in self.emotions}
        
        # Track emotion changes
        last_emotion = None
        emotion_changes = 0
        
        # Process all emotion data
        for emotion_data in emotions_over_time:
            if not emotion_data['success']:
                continue
                
            dominant_emotion = emotion_data['dominant_emotion']
            
            # Update frequency count
            if dominant_emotion in emotion_frequencies:
                emotion_frequencies[dominant_emotion] += 1
            
            # Check for emotion change
            if last_emotion is not None and dominant_emotion != last_emotion:
                emotion_changes += 1
            
            last_emotion = dominant_emotion
        
        # Calculate most frequent emotion
        total_valid_frames = sum(emotion_frequencies.values())
        
        if total_valid_frames == 0:
            return {
                'emotion_frequencies': emotion_frequencies,
                'dominant_emotion': None,
                'emotion_changes': 0,
                'emotion_distribution': {emotion: 0 for emotion in self.emotions}
            }
        
        dominant_emotion = max(emotion_frequencies, key=emotion_frequencies.get)
        
        # Calculate emotion distribution (percentage of time)
        emotion_distribution = {
            emotion: (count / total_valid_frames if total_valid_frames > 0 else 0) 
            for emotion, count in emotion_frequencies.items()
        }
        
        return {
            'emotion_frequencies': emotion_frequencies,
            'dominant_emotion': dominant_emotion,
            'emotion_changes': emotion_changes,
            'emotion_distribution': emotion_distribution
        }
    
    def get_emotional_reactivity(self, emotions_over_time: List[Dict], 
                                timestamps: List[float], 
                                event_timestamps: List[float],
                                window_size: float = 1.0) -> Dict:
        """
        Measure emotional reactivity to specific events.
        
        Args:
            emotions_over_time: List of emotion data dictionaries
            timestamps: List of corresponding timestamps
            event_timestamps: List of timestamps for significant events
            window_size: Size of the time window to consider before/after events (in seconds)
            
        Returns:
            Dictionary with reactivity metrics for each event:
            - event_index: {
                'before': average emotions before event,
                'after': average emotions after event,
                'delta': change in emotion probabilities
              }
        """
        if not emotions_over_time or len(emotions_over_time) != len(timestamps):
            return {}
        
        reactivity_data = {}
        
        # Process each event
        for i, event_time in enumerate(event_timestamps):
            # Find frames before and after the event
            before_frames = []
            after_frames = []
            
            for j, time in enumerate(timestamps):
                if event_time - window_size <= time < event_time:
                    # Frame is in the window before the event
                    if emotions_over_time[j]['success']:
                        before_frames.append(emotions_over_time[j])
                elif event_time <= time < event_time + window_size:
                    # Frame is in the window after the event
                    if emotions_over_time[j]['success']:
                        after_frames.append(emotions_over_time[j])
            
            # Skip if no frames in either window
            if not before_frames or not after_frames:
                continue
            
            # Calculate average emotion values before event
            before_emotions = {emotion: 0 for emotion in self.emotions}
            for frame in before_frames:
                for emotion in self.emotions:
                    before_emotions[emotion] += frame['emotions'].get(emotion, 0)
            
            # Average the values
            for emotion in before_emotions:
                before_emotions[emotion] /= len(before_frames)
            
            # Calculate average emotion values after event
            after_emotions = {emotion: 0 for emotion in self.emotions}
            for frame in after_frames:
                for emotion in self.emotions:
                    after_emotions[emotion] += frame['emotions'].get(emotion, 0)
            
            # Average the values
            for emotion in after_emotions:
                after_emotions[emotion] /= len(after_frames)
            
            # Calculate delta (change in emotion)
            delta_emotions = {
                emotion: after_emotions[emotion] - before_emotions[emotion]
                for emotion in self.emotions
            }
            
            # Store reactivity data for this event
            reactivity_data[i] = {
                'before': before_emotions,
                'after': after_emotions,
                'delta': delta_emotions
            }
        
        return reactivity_data
    
    def visualize_emotion_detection(self, frame: np.ndarray, emotion_data: Dict) -> np.ndarray:
        """
        Visualize emotion detection results on a frame.
        
        Args:
            frame: Input frame
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

class RealTimeEmotionDetector:
    def __init__(self):
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.emotion_map = {
            'neutral': 0,
            'sad': 1,
            'happy': 2,
            'angry': 3,
            'fear': 4,
            'surprise': 5,
            'disgust': 6
        }
        self.backends = ['opencv', 'retinaface']
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def _preprocess_image(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        enhanced = self.clahe.apply(gray)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    def detect_emotions(self, frame: np.ndarray) -> dict:
        emotion_data = {'emotions': None, 'dominant_emotion': None, 'success': False}
        try:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            if len(faces) == 0:
                return emotion_data
            for (x, y, w, h) in faces:
                face_frame = frame[y:y+h, x:x+w]
                enhanced_frame = self._preprocess_image(face_frame)
                all_results = []
                for backend in self.backends:
                    try:
                        result = DeepFace.analyze(face_frame, actions=['emotion'], enforce_detection=False, detector_backend=backend)
                        result_enhanced = DeepFace.analyze(enhanced_frame, actions=['emotion'], enforce_detection=False, detector_backend=backend)
                        all_results.extend(result if isinstance(result, list) else [result])
                        all_results.extend(result_enhanced if isinstance(result_enhanced, list) else [result_enhanced])
                    except Exception as e:
                        print(f"Backend {backend} failed: {str(e)}")
                if not all_results:
                    continue
                emotions = {}
                for result in all_results:
                    for emotion, score in result['emotion'].items():
                        emotions[emotion] = emotions.get(emotion, 0) + score
                for emotion in emotions:
                    emotions[emotion] /= len(all_results)
                dominant_emotion = max(emotions, key=emotions.get)
                emotion_data.update({'emotions': emotions, 'dominant_emotion': dominant_emotion, 'success': True})
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"Emotion: {dominant_emotion}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        except Exception as e:
            print(f"Error in emotion detection: {e}")
        return emotion_data

    def run_webcam(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame from camera. Exiting...")
                break
            self.detect_emotions(frame)
            cv2.imshow('Real-Time Emotion Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = RealTimeEmotionDetector()
    detector.run_webcam() 