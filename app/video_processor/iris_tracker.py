import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, List, Tuple, Union, Optional

class IrisTracker:
    def __init__(self):
        """Initialize the iris tracker with MediaPipe face mesh."""
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Use Face Mesh with iris refinement
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Define eye landmark indices
        # Left eye landmarks (right from the perspective of the viewer)
        self.LEFT_EYE_LANDMARKS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        # Right eye landmarks (left from the perspective of the viewer)
        self.RIGHT_EYE_LANDMARKS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        
        # Define iris landmarks
        # Left eye iris landmarks
        self.LEFT_IRIS_LANDMARKS = [468, 469, 470, 471, 472]
        # Right eye iris landmarks
        self.RIGHT_IRIS_LANDMARKS = [473, 474, 475, 476, 477]
        
        # Baseline pupil size (will be adjusted during detection)
        self.baseline_pupil_size = None
        
        # Store pupil sizes for analysis
        self.pupil_sizes = []
        self.avg_pupil_size = 0.0
        self.max_pupil_size = 0.0
        self.min_pupil_size = float('inf')
        self.pupil_dilation_delta = 0.0
        self.baseline_recorded = False
        self.baseline_pupil_sizes = []
        self.event_pupil_sizes = []
        self.baseline_frames = 30  # Number of frames to use for baseline
    
    def detect_iris(self, frame: np.ndarray) -> Dict:
        """
        Detect iris in a frame and compute related metrics.
        
        Args:
            frame: Input frame from video
            
        Returns:
            Dictionary containing iris tracking data:
            - left_iris_center: (x, y) coordinates of left iris center
            - right_iris_center: (x, y) coordinates of right iris center
            - left_pupil_size: diameter of left pupil
            - right_pupil_size: diameter of right pupil
            - landmarks: all detected face landmarks
            - success: boolean indicating successful detection
        """
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        
        # Process the frame and get results
        results = self.face_mesh.process(rgb_frame)
        
        # Initialize return data
        iris_data = {
            'left_iris_center': None,
            'right_iris_center': None,
            'left_pupil_size': None,
            'right_pupil_size': None,
            'landmarks': None,
            'success': False,
            'pupil_dilation_ratio': None
        }
        
        # Check if landmarks were detected
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            iris_data['landmarks'] = landmarks
            
            # Extract normalized landmark points
            points = np.array([[point.x, point.y, point.z] for point in landmarks.landmark])
            
            # Convert to pixel coordinates
            points[:, 0] = points[:, 0] * w
            points[:, 1] = points[:, 1] * h
            
            # Get iris landmarks
            left_iris = points[self.LEFT_IRIS_LANDMARKS]
            right_iris = points[self.RIGHT_IRIS_LANDMARKS]
            
            # Calculate iris centers (average of all iris landmarks)
            left_iris_center = np.mean(left_iris[:, :2], axis=0).astype(int)
            right_iris_center = np.mean(right_iris[:, :2], axis=0).astype(int)
            
            iris_data['left_iris_center'] = left_iris_center
            iris_data['right_iris_center'] = right_iris_center
            
            # Calculate pupil size (distance between opposing iris landmarks)
            left_pupil_size = np.linalg.norm(left_iris[1][:2] - left_iris[3][:2])
            right_pupil_size = np.linalg.norm(right_iris[1][:2] - right_iris[3][:2])
            
            iris_data['left_pupil_size'] = left_pupil_size
            iris_data['right_pupil_size'] = right_pupil_size
            
            # Calculate average pupil size
            avg_pupil_size = (left_pupil_size + right_pupil_size) / 2
            
            # Update min and max pupil sizes
            if avg_pupil_size > 0:
                self.avg_pupil_size = avg_pupil_size
                self.max_pupil_size = max(self.max_pupil_size, avg_pupil_size)
                self.min_pupil_size = min(self.min_pupil_size, avg_pupil_size) if self.min_pupil_size > 0 else avg_pupil_size
                
                # Add to pupil sizes list
                self.pupil_sizes.append(avg_pupil_size)
                
                # If baseline not set, collect baseline data
                if not self.baseline_recorded and len(self.baseline_pupil_sizes) < self.baseline_frames:
                    self.baseline_pupil_sizes.append(avg_pupil_size)
                    if len(self.baseline_pupil_sizes) == self.baseline_frames:
                        self.baseline_recorded = True
                        self.baseline_pupil_size = np.mean(self.baseline_pupil_sizes)
                elif self.baseline_recorded:
                    self.event_pupil_sizes.append(avg_pupil_size)
                    # Calculate pupil dilation delta once we have enough event data
                    if len(self.event_pupil_sizes) > 10:  # Wait for at least 10 frames of event data
                        self.pupil_dilation_delta = self.calculate_pupil_dilation_delta(
                            self.baseline_pupil_sizes, self.event_pupil_sizes)
                
                # Calculate pupil dilation ratio
                pupil_dilation_ratio = self.calculate_pupil_dilation(avg_pupil_size)
                iris_data['pupil_dilation_ratio'] = pupil_dilation_ratio
            
            iris_data['success'] = True
        
        return iris_data
    
    def calculate_pupil_dilation(self, current_size: float) -> Optional[float]:
        """
        Calculate pupil dilation as a ratio compared to baseline.
        
        Args:
            current_size: Current pupil size
            
        Returns:
            Pupil dilation ratio (>1 means dilated, <1 means constricted)
        """
        if self.baseline_pupil_size is None or current_size is None or self.baseline_pupil_size == 0:
            return None
        
        return current_size / self.baseline_pupil_size
    
    def get_eye_regions(self, frame: np.ndarray, iris_data: Dict) -> Dict:
        """
        Extract eye regions from a frame.
        
        Args:
            frame: Input frame
            iris_data: Dictionary from detect_iris
            
        Returns:
            Dictionary with 'left_eye' and 'right_eye' regions
        """
        if not iris_data['success']:
            return {'left_eye': None, 'right_eye': None}
        
        h, w, _ = frame.shape
        landmarks = iris_data['landmarks']
        
        # Get eye landmarks
        left_eye_landmarks = np.array([(landmarks.landmark[i].x * w, landmarks.landmark[i].y * h) 
                                       for i in self.LEFT_EYE_LANDMARKS], dtype=np.int32)
        right_eye_landmarks = np.array([(landmarks.landmark[i].x * w, landmarks.landmark[i].y * h) 
                                       for i in self.RIGHT_EYE_LANDMARKS], dtype=np.int32)
        
        # Get bounding boxes
        left_x, left_y, left_w, left_h = cv2.boundingRect(left_eye_landmarks)
        right_x, right_y, right_w, right_h = cv2.boundingRect(right_eye_landmarks)
        
        # Add padding to the bounding boxes
        padding = 10
        left_x = max(0, left_x - padding)
        left_y = max(0, left_y - padding)
        left_w = min(w - left_x, left_w + 2 * padding)
        left_h = min(h - left_y, left_h + 2 * padding)
        
        right_x = max(0, right_x - padding)
        right_y = max(0, right_y - padding)
        right_w = min(w - right_x, right_w + 2 * padding)
        right_h = min(h - right_y, right_h + 2 * padding)
        
        # Extract eye regions
        left_eye = frame[left_y:left_y+left_h, left_x:left_x+left_w]
        right_eye = frame[right_y:right_y+right_h, right_x:right_x+right_w]
        
        return {
            'left_eye': left_eye,
            'right_eye': right_eye,
            'left_box': (left_x, left_y, left_w, left_h),
            'right_box': (right_x, right_y, right_w, right_h)
        }
    
    def visualize_iris_tracking(self, frame: np.ndarray, iris_data: Dict) -> np.ndarray:
        """
        Draw iris tracking visualization on a frame.
        
        Args:
            frame: Input frame
            iris_data: Dictionary from detect_iris
            
        Returns:
            Frame with iris tracking visualization
        """
        output_frame = frame.copy()
        
        if iris_data['success']:
            landmarks = iris_data['landmarks']
            
            # Draw face mesh
            self.mp_drawing.draw_landmarks(
                image=output_frame,
                landmark_list=landmarks,
                connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
            )
            
            # Draw iris centers as circles
            if iris_data['left_iris_center'] is not None:
                cv2.circle(output_frame, 
                          tuple(iris_data['left_iris_center'].astype(int)), 
                          radius=2, 
                          color=(0, 255, 0),
                          thickness=-1)
            
            if iris_data['right_iris_center'] is not None:
                cv2.circle(output_frame, 
                          tuple(iris_data['right_iris_center'].astype(int)), 
                          radius=2,
                          color=(0, 255, 0),
                          thickness=-1)
            
            # Draw pupil size information
            h, w, _ = output_frame.shape
            
            left_pupil_size = iris_data['left_pupil_size']
            right_pupil_size = iris_data['right_pupil_size']
            
            if left_pupil_size is not None and right_pupil_size is not None:
                avg_pupil_size = (left_pupil_size + right_pupil_size) / 2
                dilation_ratio = self.calculate_pupil_dilation(avg_pupil_size)
                
                info_text = f"Pupil Size: {avg_pupil_size:.2f}px"
                if dilation_ratio:
                    info_text += f" (x{dilation_ratio:.2f})"
                
                cv2.putText(
                    output_frame,
                    info_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                # Display pupil dilation delta
                if self.pupil_dilation_delta != 0:
                    delta_text = f"Dilation Delta: {self.pupil_dilation_delta:.4f}"
                    cv2.putText(
                        output_frame,
                        delta_text,
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
            
            # Draw eye regions
            eye_regions = self.get_eye_regions(frame, iris_data)
            left_box = eye_regions['left_box']
            right_box = eye_regions['right_box']
            
            cv2.rectangle(
                output_frame,
                (left_box[0], left_box[1]),
                (left_box[0] + left_box[2], left_box[1] + left_box[3]),
                (255, 0, 0),
                2
            )
            
            cv2.rectangle(
                output_frame,
                (right_box[0], right_box[1]),
                (right_box[0] + right_box[2], right_box[1] + right_box[3]),
                (255, 0, 0),
                2
            )
        else:
            # No face detected
            cv2.putText(
                output_frame,
                "No face detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
        
        return output_frame
    
    def analyze_pupil_variability(self, pupil_sizes: List[float] = None) -> Dict:
        """
        Calculate statistical measures for pupil sizes over time.
        
        Args:
            pupil_sizes: List of pupil size measurements (defaults to stored pupil_sizes if None)
            
        Returns:
            Dictionary with pupil variability statistics:
            - mean: Average pupil size
            - std: Standard deviation of pupil size
            - min: Minimum pupil size
            - max: Maximum pupil size
            - range: Range of pupil sizes (max - min)
        """
        if pupil_sizes is None:
            pupil_sizes = self.pupil_sizes
            
        if not pupil_sizes or len(pupil_sizes) < 2:
            return {
                'mean': None,
                'std': None,
                'min': None,
                'max': None,
                'range': None
            }
        
        pupil_sizes = np.array(pupil_sizes)
        pupil_sizes = pupil_sizes[pupil_sizes > 0]  # Filter out any zero values
        
        if len(pupil_sizes) < 2:
            return {
                'mean': None,
                'std': None,
                'min': None,
                'max': None,
                'range': None
            }
        
        mean = np.mean(pupil_sizes)
        std = np.std(pupil_sizes)
        min_val = np.min(pupil_sizes)
        max_val = np.max(pupil_sizes)
        range_val = max_val - min_val
        
        return {
            'mean': mean,
            'std': std,
            'min': min_val,
            'max': max_val,
            'range': range_val
        }
    
    def calculate_pupil_dilation_delta(self, baseline_pupil_sizes: List[float] = None, 
                                       event_pupil_sizes: List[float] = None) -> float:
        """
        Calculate pupil size change in response to an event.
        
        Args:
            baseline_pupil_sizes: List of pupil sizes before event (defaults to stored baseline)
            event_pupil_sizes: List of pupil sizes after event (defaults to stored event pupil sizes)
            
        Returns:
            Normalized pupil dilation delta (positive = dilation, negative = constriction)
        """
        if baseline_pupil_sizes is None:
            baseline_pupil_sizes = self.baseline_pupil_sizes
            
        if event_pupil_sizes is None:
            event_pupil_sizes = self.event_pupil_sizes
            
        if not baseline_pupil_sizes or not event_pupil_sizes:
            return 0.0
        
        # Calculate average pupil size before and after event
        baseline_avg = np.mean(baseline_pupil_sizes)
        event_avg = np.mean(event_pupil_sizes)
        
        if baseline_avg == 0:
            return 0.0
        
        # Calculate normalized delta
        delta = (event_avg - baseline_avg) / baseline_avg
        
        return delta
    
    def get_metrics(self) -> Dict:
        """
        Get all iris tracking metrics.
        
        Returns:
            Dictionary containing all metrics
        """
        variability = self.analyze_pupil_variability()
        
        return {
            'avg_pupil_size': self.avg_pupil_size,
            'min_pupil_size': self.min_pupil_size,
            'max_pupil_size': self.max_pupil_size,
            'pupil_dilation_delta': self.pupil_dilation_delta,
            'pupil_variability': variability,
            'baseline_recorded': self.baseline_recorded,
            'baseline_pupil_size': self.baseline_pupil_size
        }
    
    def reset(self):
        """Reset all metrics and data."""
        self.baseline_pupil_size = None
        self.pupil_sizes = []
        self.avg_pupil_size = 0.0
        self.max_pupil_size = 0.0
        self.min_pupil_size = float('inf')
        self.pupil_dilation_delta = 0.0
        self.baseline_recorded = False
        self.baseline_pupil_sizes = []
        self.event_pupil_sizes = []
    
    def __del__(self):
        """Release MediaPipe resources."""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close() 