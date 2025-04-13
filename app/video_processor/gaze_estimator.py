import cv2
import numpy as np
import os
import mediapipe as mp

class GazeEstimator:
    """
    Estimates gaze direction based on detecting face and eyes using MediaPipe FaceMesh,
    and analyzing pupil position using a thresholding and percentage method.
    """
    def __init__(self, pupil_threshold=50):
        """
        Initializes the GazeEstimator.

        Args:
            pupil_threshold (int): The threshold value for isolating the pupil. Lower values
                                   make the thresholding stricter (requiring darker pixels).
        """
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        # Use static_image_mode=False for video streams, refine_landmarks=True for better iris/pupil landmarks
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1, 
            refine_landmarks=True, 
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        
        # Pupil threshold remains the same
        self.pupil_threshold = pupil_threshold
        
        # Store last known metrics
        self.last_gaze_direction = "N/A"
        self.last_avg_left_percentage = 0.0
        self.last_avg_right_percentage = 0.0
        self.last_left_pupil_coords = None
        self.last_right_pupil_coords = None
        self.face_detected = False
        self.eyes_detected = False
        
        # Define landmark indices for eye bounding boxes (approximate)
        # These can be adjusted based on visual inspection
        self.LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144] # Left eye outline
        self.RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380] # Right eye outline
        # Landmarks for pupils (if refine_landmarks=True)
        self.LEFT_PUPIL_LANDMARK = 473
        self.RIGHT_PUPIL_LANDMARK = 468
        
        # Counter for direction statistics
        self.looking_left_count = 0
        self.looking_right_count = 0
        self.looking_center_count = 0
        self.total_frames_processed = 0

    def _get_bounding_box(self, landmarks, frame_width, frame_height):
        """Calculate bounding box from a list of landmarks."""
        if not landmarks: return None
        
        xs = [lm.x * frame_width for lm in landmarks]
        ys = [lm.y * frame_height for lm in landmarks]
        
        if not xs or not ys: return None
        
        x_min, x_max = int(min(xs)), int(max(xs))
        y_min, y_max = int(min(ys)), int(max(ys))
        
        # Add some padding if needed, or ensure min size
        w = max(1, x_max - x_min)
        h = max(1, y_max - y_min)
        
        return x_min, y_min, w, h

    def _detect_face_and_eyes(self, frame):
        """Detect face and eyes in the frame using MediaPipe FaceMesh"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        frame_height, frame_width, _ = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Keep grayscale version

        self.face_detected = False
        self.eyes_detected = False
        face_landmarks = None
        face_coords = None
        eye_regions_relative = [] # Store eye boxes relative to face ROI
        roi_gray = None
        roi_color = None
        left_pupil_abs = None
        right_pupil_abs = None

        if results.multi_face_landmarks:
            self.face_detected = True
            face_landmarks = results.multi_face_landmarks[0] # Assuming one face
            
            # 1. Calculate Face Bounding Box (approximate from all landmarks)
            all_landmarks = [lm for lm in face_landmarks.landmark]
            face_bbox = self._get_bounding_box(all_landmarks, frame_width, frame_height)
            if face_bbox:
                fx, fy, fw, fh = face_bbox
                # Clamp coordinates to frame dimensions
                fx, fy = max(0, fx), max(0, fy)
                fw, fh = min(frame_width - fx, fw), min(frame_height - fy, fh)
                face_coords = (fx, fy, fw, fh)
                
                # Extract face ROI
                roi_gray = gray[fy:fy+fh, fx:fx+fw]
                roi_color = frame[fy:fy+fh, fx:fx+fw]
            else:
                # Fallback if face bbox calculation fails (shouldn't happen if landmarks exist)
                self.face_detected = False
                return None, None, None, None, None, None

            # 2. Calculate Eye Bounding Boxes (Absolute)
            left_eye_lm = [face_landmarks.landmark[i] for i in self.LEFT_EYE_LANDMARKS]
            right_eye_lm = [face_landmarks.landmark[i] for i in self.RIGHT_EYE_LANDMARKS]
            
            left_eye_bbox_abs = self._get_bounding_box(left_eye_lm, frame_width, frame_height)
            right_eye_bbox_abs = self._get_bounding_box(right_eye_lm, frame_width, frame_height)

            # Get pupil coordinates (absolute)
            try:
                left_pupil_lm = face_landmarks.landmark[self.LEFT_PUPIL_LANDMARK]
                right_pupil_lm = face_landmarks.landmark[self.RIGHT_PUPIL_LANDMARK]
                left_pupil_abs = (int(left_pupil_lm.x * frame_width), int(left_pupil_lm.y * frame_height))
                right_pupil_abs = (int(right_pupil_lm.x * frame_width), int(right_pupil_lm.y * frame_height))
            except IndexError:
                 # This might happen if refine_landmarks=False or landmarks aren't found
                 pass

            # 3. Convert Eye Coords to be relative to Face ROI
            if left_eye_bbox_abs and right_eye_bbox_abs and face_coords:
                lx_abs, ly_abs, lw, lh = left_eye_bbox_abs
                rx_abs, ry_abs, rw, rh = right_eye_bbox_abs
                fx, fy, _, _ = face_coords
                
                # Calculate relative coords
                lx_rel, ly_rel = lx_abs - fx, ly_abs - fy
                rx_rel, ry_rel = rx_abs - fx, ry_abs - fy
                
                # Clamp relative coords to ROI dimensions if needed (should ideally fit within)
                lx_rel, ly_rel = max(0, lx_rel), max(0, ly_rel)
                rx_rel, ry_rel = max(0, rx_rel), max(0, ry_rel)
                # Adjust width/height if clamping changed origin
                lw = min(fw - lx_rel, lw)
                lh = min(fh - ly_rel, lh)
                rw = min(fw - rx_rel, rw)
                rh = min(fh - ry_rel, rh)
                
                # Ensure width/height are positive
                lw, lh = max(1, lw), max(1, lh)
                rw, rh = max(1, rw), max(1, rh)

                # Store relative eye regions, ensuring left comes first
                eye_regions_relative = [
                    (lx_rel, ly_rel, lw, lh), 
                    (rx_rel, ry_rel, rw, rh)
                ]
                self.eyes_detected = True
            else:
                self.eyes_detected = False

        return face_coords, roi_color, roi_gray, eye_regions_relative, left_pupil_abs, right_pupil_abs

    def _process_eye(self, roi_gray, eye_region):
        """Process a single eye region. Expects eye_region coords relative to roi_gray."""
        ex, ey, ew, eh = eye_region
        
        # Ensure ROI coordinates are valid
        roi_h, roi_w = roi_gray.shape
        ex, ey = max(0, ex), max(0, ey)
        ew, eh = min(roi_w - ex, ew), min(roi_h - ey, eh)
        
        if ew <= 0 or eh <= 0:
            # Invalid eye region dimensions
            return (50.0, 50.0), (ew // 2, eh // 2) # Default

        eye_roi = roi_gray[ey:ey+eh, ex:ex+ew]
        
        if eye_roi.size == 0:
            # Eye ROI is empty
            return (50.0, 50.0), (ew // 2, eh // 2)

        # Apply histogram equalization to enhance contrast
        try:
            eye_roi_equalized = cv2.equalizeHist(eye_roi)
        except cv2.error as e:
             # Handle cases like single-color ROI which can cause errors
             eye_roi_equalized = eye_roi # Use original if equalization fails
        
        # Apply thresholding to isolate the pupil
        _, threshold_eye = cv2.threshold(eye_roi_equalized, self.pupil_threshold, 255, cv2.THRESH_BINARY_INV)
        
        # Divide the eye into left and right halves
        mid_x = ew // 2
        left_half = threshold_eye[:, :mid_x]
        right_half = threshold_eye[:, mid_x:]
        
        # Count non-zero pixels (pupil pixels) in each half
        left_pixels = cv2.countNonZero(left_half)
        right_pixels = cv2.countNonZero(right_half)
        
        total_pixels = left_pixels + right_pixels
        
        if total_pixels == 0:
            left_percentage = 50.0 # Default to center if no pupil detected
            right_percentage = 50.0
        else:
            left_percentage = (left_pixels / total_pixels) * 100
            right_percentage = (right_pixels / total_pixels) * 100
        
        # Find the centroid of the pupil for visualization
        moments = cv2.moments(threshold_eye)
        cx, cy = ew // 2, eh // 2 # Default centroid
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            
        # Return percentages and centroid coordinates relative to the eye ROI
        return (left_percentage, right_percentage), (cx, cy)

    def _determine_gaze_direction(self, left_eye_percentages, right_eye_percentages):
        """Determine overall gaze direction"""
        left_eye_left_perc, left_eye_right_perc = left_eye_percentages
        right_eye_left_perc, right_eye_right_perc = right_eye_percentages
        
        # Calculate average percentage for left and right across both eyes
        avg_left_percentage = (left_eye_left_perc + right_eye_left_perc) / 2
        avg_right_percentage = (left_eye_right_perc + right_eye_right_perc) / 2
        
        # Determine direction based on which side has the higher percentage
        threshold = 5 # Percentage difference threshold
        # If more pupil is on the left side of the eyes, user is looking LEFT (inverse from previous logic)
        if avg_left_percentage > avg_right_percentage + threshold:
            direction = "Looking Left"
            self.looking_left_count += 1
        # If more pupil is on the right side of the eyes, user is looking RIGHT (inverse from previous logic)
        elif avg_right_percentage > avg_left_percentage + threshold:
            direction = "Looking Right"
            self.looking_right_count += 1
        else:
            direction = "Looking Center"
            self.looking_center_count += 1
            
        self.last_gaze_direction = direction
        self.last_avg_left_percentage = avg_left_percentage
        self.last_avg_right_percentage = avg_right_percentage
        self.total_frames_processed += 1
            
        return direction, avg_left_percentage, avg_right_percentage

    def analyze_frame(self, frame):
        """
        Analyzes a single frame to detect face, eyes, and determine gaze direction.
        """
        annotated_frame = frame.copy()
        # Get face coords, ROIs, relative eye regions, and absolute pupil landmarks
        face_coords, face_roi_color, face_roi_gray, eye_regions_relative, left_pupil_abs, right_pupil_abs = self._detect_face_and_eyes(frame)

        # Update metrics based on detection status
        metrics = {
            "face_detected": self.face_detected,
            "eyes_detected": self.eyes_detected,
            "gaze_direction": "N/A",
            "avg_left_perc": 0.0,
            "avg_right_perc": 0.0,
            "left_pupil_coords": left_pupil_abs,
            "right_pupil_coords": right_pupil_abs,
        }
        self.last_left_pupil_coords = left_pupil_abs
        self.last_right_pupil_coords = right_pupil_abs

        if self.face_detected and face_coords and face_roi_gray is not None:
            fx, fy, fw, fh = face_coords
            cv2.rectangle(annotated_frame, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)

            if self.eyes_detected and eye_regions_relative and len(eye_regions_relative) == 2:
                left_eye_region_rel = eye_regions_relative[0]
                right_eye_region_rel = eye_regions_relative[1]

                # Process eyes using face ROI gray and relative coords
                left_percentages, left_centroid_rel = self._process_eye(face_roi_gray, left_eye_region_rel)
                right_percentages, right_centroid_rel = self._process_eye(face_roi_gray, right_eye_region_rel)

                # Determine gaze
                gaze_direction, avg_left_perc, avg_right_perc = self._determine_gaze_direction(left_percentages, right_percentages)
                
                metrics["gaze_direction"] = gaze_direction
                metrics["avg_left_perc"] = avg_left_perc
                metrics["avg_right_perc"] = avg_right_perc

                # --- Visualization --- 
                # Draw eye rectangles (absolute coordinates)
                lx_rel, ly_rel, lw, lh = left_eye_region_rel
                rx_rel, ry_rel, rw, rh = right_eye_region_rel
                lx_abs, ly_abs = fx + lx_rel, fy + ly_rel
                rx_abs, ry_abs = fx + rx_rel, fy + ry_rel
                cv2.rectangle(annotated_frame, (lx_abs, ly_abs), (lx_abs + lw, ly_abs + lh), (0, 255, 0), 1)
                cv2.rectangle(annotated_frame, (rx_abs, ry_abs), (rx_abs + rw, ry_abs + rh), (0, 255, 0), 1)

                # Draw dividing lines (absolute coordinates)
                cv2.line(annotated_frame, (lx_abs + lw // 2, ly_abs), (lx_abs + lw // 2, ly_abs + lh), (255, 255, 0), 1)
                cv2.line(annotated_frame, (rx_abs + rw // 2, ry_abs), (rx_abs + rw // 2, ry_abs + rh), (255, 255, 0), 1)

                # Draw pupil centroids (calculated from thresholding, absolute coordinates)
                pupil_left_abs_calc = (lx_abs + left_centroid_rel[0], ly_abs + left_centroid_rel[1])
                pupil_right_abs_calc = (rx_abs + right_centroid_rel[0], ry_abs + right_centroid_rel[1])
                cv2.circle(annotated_frame, pupil_left_abs_calc, 3, (0, 0, 255), -1)
                cv2.circle(annotated_frame, pupil_right_abs_calc, 3, (0, 0, 255), -1)

                # Put gaze direction text
                cv2.putText(annotated_frame, f"Gaze: {gaze_direction}", (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            elif self.face_detected: # Face detected, but not eyes
                 cv2.putText(annotated_frame, "Eyes not detected", (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                 self.last_gaze_direction = "N/A"
                 self.last_avg_left_percentage = 0.0
                 self.last_avg_right_percentage = 0.0
        else:
            # No face detected
            cv2.putText(annotated_frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            self.last_gaze_direction = "N/A"
            self.last_avg_left_percentage = 0.0
            self.last_avg_right_percentage = 0.0
        
        # Update final metrics with last known values if detection failed this frame
        metrics["gaze_direction"] = self.last_gaze_direction
        metrics["avg_left_perc"] = self.last_avg_left_percentage
        metrics["avg_right_perc"] = self.last_avg_right_percentage
        
        return annotated_frame, metrics

    def get_last_metrics(self):
        """Returns the last calculated metrics."""
        # Calculate ratio_gaze_on_roi (ratio of right/left gaze frames)
        ratio_gaze_on_roi = 0.0
        if self.looking_left_count > 0:
            ratio_gaze_on_roi = self.looking_right_count / self.looking_left_count
        
        return {
            "face_detected": self.face_detected,
            "eyes_detected": self.eyes_detected,
            "gaze_direction": self.last_gaze_direction,
            "avg_left_perc": round(self.last_avg_left_percentage, 1),
            "avg_right_perc": round(self.last_avg_right_percentage, 1),
            "left_pupil_coords": self.last_left_pupil_coords,
            "right_pupil_coords": self.last_right_pupil_coords,
            "looking_left_count": self.looking_left_count,
            "looking_right_count": self.looking_right_count,
            "looking_center_count": self.looking_center_count,
            "total_frames_processed": self.total_frames_processed,
            "ratio_gaze_on_roi": ratio_gaze_on_roi
        }

    def reset(self):
        """Reset the counters and metrics"""
        self.looking_left_count = 0
        self.looking_right_count = 0
        self.looking_center_count = 0
        self.total_frames_processed = 0
        self.last_gaze_direction = "N/A"
        self.last_avg_left_percentage = 0.0
        self.last_avg_right_percentage = 0.0

    def __del__(self): 
        # Release MediaPipe resources
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close() 