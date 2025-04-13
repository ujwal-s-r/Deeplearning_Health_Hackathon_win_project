from flask import render_template, redirect, url_for, request, session, flash, jsonify
from app.main import main
import os
import json
from datetime import datetime
from app.video_processor import EmotionAnalyzer, BlinkDetector, GazeEstimator, IrisTracker
from app.models import DepressionPredictor
import cv2

# Create game data directory if it doesn't exist
GAME_DATA_DIR = os.path.join('app', 'game_data')
if not os.path.exists(GAME_DATA_DIR):
    os.makedirs(GAME_DATA_DIR)

# Create webcam recordings directory if it doesn't exist
WEBCAM_RECORDINGS_DIR = os.path.join('app', 'webcam_recordings')
if not os.path.exists(WEBCAM_RECORDINGS_DIR):
    os.makedirs(WEBCAM_RECORDINGS_DIR)

# Create processed videos directory if it doesn't exist
PROCESSED_VIDEOS_DIR = os.path.join('app', 'static', 'processed_videos')
if not os.path.exists(PROCESSED_VIDEOS_DIR):
    os.makedirs(PROCESSED_VIDEOS_DIR)

# Define PHQ-8 questions and answer choices
phq8_questions = [
    "1. Little interest or pleasure in doing things",
    "2. Feeling down, depressed, or hopeless",
    "3. Trouble falling or staying asleep, or sleeping too much",
    "4. Feeling tired or having little energy",
    "5. Poor appetite or overeating",
    "6. Feeling bad about yourself — or that you are a failure or have let yourself or your family down",
    "7. Trouble concentrating on things, such as reading or watching TV",
    "8. Moving or speaking so slowly that others could have noticed? Or being so fidgety or restless?"
]

choices = {
    "0": "Not at all",
    "1": "Several days",
    "2": "More than half the days",
    "3": "Nearly every day"
}

@main.route('/')
def index():
    # Reset session data for a new assessment
    if 'phq8_responses' in session:
        session.pop('phq8_responses')
    if 'phq8_score' in session:
        session.pop('phq8_score')
    if 'current_question' in session:
        session.pop('current_question')
    
    print("\n===== APPLICATION FLOW: User accessing index page =====")
    print("Session reset for new assessment")
    return render_template('index.html')

@main.route('/phq8', methods=['GET', 'POST'])
def phq8_questionnaire():
    print("\n===== APPLICATION FLOW: PHQ-8 Questionnaire =====")
    # Initialize session variables if not present
    if 'phq8_responses' not in session:
        session['phq8_responses'] = []
    
    if 'current_question' not in session:
        session['current_question'] = 0
    
    # If all questions are answered, calculate score and redirect to results
    if session['current_question'] >= len(phq8_questions):
        # Calculate score
        total_score = sum(session['phq8_responses'])
        session['phq8_score'] = total_score
        print(f"PHQ-8 completed. Total score: {total_score}")
        return redirect(url_for('main.phq8_result'))
    
    # Handle POST request (answer submission)
    if request.method == 'POST':
        if 'skip' in request.form:
            # Skip directly to the result page for testing
            print("User skipped PHQ-8 questionnaire")
            return redirect(url_for('main.phq8_result'))
        
        answer = request.form.get('answer')
        if answer in choices:
            # Add response and move to next question
            responses = session['phq8_responses']
            responses.append(int(answer))
            session['phq8_responses'] = responses
            session['current_question'] += 1
            current_q = session['current_question']
            print(f"PHQ-8 question {current_q} answered: {answer} - {choices[answer]}")
            
            # Redirect to handle the next question or show results
            return redirect(url_for('main.phq8_questionnaire'))
    
    # Render the current question
    current_q = session['current_question']
    progress_percent = (current_q / len(phq8_questions)) * 100
    
    return render_template(
        'phq8.html', 
        question=phq8_questions[current_q], 
        choices=choices,
        progress=progress_percent,
        question_number=current_q + 1,
        total_questions=len(phq8_questions)
    )

@main.route('/phq8_result')
def phq8_result():
    print("\n===== APPLICATION FLOW: PHQ-8 Results Page =====")
    # If user tries to access results without completing the questionnaire
    if 'phq8_score' not in session:
        # For testing purposes, generate a random score
        if 'skip' in request.args:
            import random
            session['phq8_score'] = random.randint(0, 24)
            print(f"Generated random PHQ-8 score for testing: {session['phq8_score']}")
        else:
            print("Redirecting - PHQ-8 not completed")
            flash('Please complete the questionnaire first')
            return redirect(url_for('main.phq8_questionnaire'))
    
    total_score = session['phq8_score']
    
    # Interpret score
    if total_score <= 4:
        severity = "None/minimal"
        message = "You're likely doing okay, but check in with yourself regularly."
    elif 5 <= total_score <= 9:
        severity = "Mild"
        message = "You may be experiencing mild symptoms of depression."
    elif 10 <= total_score <= 14:
        severity = "Moderate"
        message = "Consider talking with a mental health professional about your symptoms."
    elif 15 <= total_score <= 19:
        severity = "Moderately severe"
        message = "It's recommended to consult with a mental health professional."
    else:
        severity = "Severe"
        message = "It's highly recommended to seek help from a mental health professional."
    
    return render_template(
        'phq8_result.html', 
        score=total_score, 
        severity=severity, 
        message=message
    )

@main.route('/game')
def game():
    print("\n===== APPLICATION FLOW: Game Page =====")
    # If PHQ-8 is not completed and not skipped, redirect to PHQ-8
    if 'phq8_score' not in session and 'skip' not in request.args:
        print("Redirecting - PHQ-8 not completed")
        flash('Please complete the PHQ-8 questionnaire first')
        return redirect(url_for('main.phq8_questionnaire'))
    
    print("Rendering game page")
    return render_template('game.html')

@main.route('/video_analysis')
def video_analysis():
    print("\n===== APPLICATION FLOW: Video Analysis Page =====")
    print("Rendering video analysis page")
    return render_template('video_analysis.html')

@main.route('/save_webcam_recording', methods=['POST'])
def save_webcam_recording():
    print("\n===== APPLICATION FLOW: Processing Webcam Recording =====")
    try:
        # Log request information
        print(f"Received webcam recording request, Content-Length: {request.content_length}")
        
        # Check if the post request has the file part
        if 'webcam_video' not in request.files:
            print("No webcam_video in request.files")
            return jsonify({"status": "error", "message": "No file part"}), 400
        
        webcam_file = request.files['webcam_video']
        
        # If user does not select file, browser also submit an empty part without filename
        if webcam_file.filename == '':
            print("Empty filename in webcam_file")
            return jsonify({"status": "error", "message": "No selected file"}), 400
        
        # Create a filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"webcam_recording_{timestamp}.webm"
        filepath = os.path.join(WEBCAM_RECORDINGS_DIR, filename)
        
        # Save the file with explicit flush to ensure it's written
        print(f"Saving webcam recording to {filepath}")
        webcam_file.save(filepath)
        
        # Initialize default return values
        dominant_emotion_code = 0
        dominant_emotion_label = "neutral"
        emotion_counts = {emotion: 0 for emotion in ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']}
        blink_count = 0
        blink_rate = 0.0
        looking_left_count = 0
        looking_right_count = 0
        looking_center_count = 0
        ratio_gaze_on_roi = 0.0
        pupil_dilation_delta = 0.0
        avg_pupil_size = 0.0
        status = "success"
        message = "Webcam recording saved and analyzed successfully"
        
        # Verify the file was saved
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath)
            print(f"File saved successfully, size: {file_size} bytes")
            
            # Only process if the file exists and has content
            if file_size > 0:
                # Store the recording path in the session
                session['webcam_recording'] = filepath
                
                # Process the video for emotion analysis
                try:
                    # Create an optional output path for debugging
                    output_path = os.path.join(PROCESSED_VIDEOS_DIR, f"processed_{timestamp}.avi")
                    
                    # Initialize emotion analyzer and process the video
                    print("Starting emotion analysis...")
                    emotion_analyzer = EmotionAnalyzer()
                    
                    try:
                        dominant_emotion_code = emotion_analyzer.process_video(
                            video_path=filepath,
                            output_path=output_path,
                            fps=10  # Process at 10 FPS for smoother analysis
                        )
                        
                        # Store the dominant emotion in the session
                        session['dominant_emotion'] = dominant_emotion_code
                        session['dominant_emotion_label'] = emotion_analyzer.dominant_emotion
                        session['emotion_counts'] = emotion_analyzer.emotion_counts
                        
                        # Update return values
                        dominant_emotion_label = emotion_analyzer.dominant_emotion
                        emotion_counts = emotion_analyzer.emotion_counts
                        
                        print(f"Dominant emotion stored in session: {dominant_emotion_code} ({dominant_emotion_label})")
                    except Exception as e:
                        print(f"Error in emotion analysis: {str(e)}")
                        status = "partial_success"
                        message = f"Webcam recording saved but emotion analysis failed: {str(e)}"
                    
                    # Process the same video for blink detection
                    print("Starting blink detection analysis...")
                    blink_detector = BlinkDetector()
                    
                    try:
                        # Process video for blink detection
                        # Open the video file
                        cap = cv2.VideoCapture(filepath)
                        
                        if not cap.isOpened():
                            raise Exception("Failed to open video file for blink detection")
                        
                        # Get video properties for processing
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        if fps <= 0:
                            fps = 30  # Default to 30fps if detection fails
                        
                        # Get frame count for duration calculation
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        video_duration = frame_count / fps if frame_count > 0 and fps > 0 else 53  # Default to 53 seconds for the hack.mp4 video
                        
                        print(f"Video properties - FPS: {fps}, Frames: {frame_count}, Duration: {video_duration:.2f} seconds")
                        
                        # Update FPS in blink detector
                        blink_detector.fps = fps
                        
                        # For tracking last blink data
                        last_blink_data = None
                        
                        # Process each frame for blink detection
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break
                                
                            # Detect blinks in this frame
                            blink_data = blink_detector.detect_blink(frame)
                            if blink_data['success']:
                                last_blink_data = blink_data
                        
                        # Release the video capture
                        cap.release()
                        
                        # If using the Stimulus-in-Page approach (web-generated webcam recording while watching hack.mp4)
                        # The actual video duration would be closer to 53 seconds
                        # Adjust blink rate if necessary
                        if last_blink_data and 'blink_rate' in last_blink_data:
                            # For videos with similar duration to hack.mp4 (53 seconds),
                            # we can check if the duration seems wrong and fix it
                            if video_duration < 40 or video_duration > 70:
                                # If the estimated duration is very different from expected 53 seconds,
                                # use the known duration and recalculate the blink rate
                                adjusted_blink_rate = (blink_detector.blink_counter / 53) * 60
                                print(f"Adjusting blink rate from {last_blink_data['blink_rate']:.2f} to {adjusted_blink_rate:.2f} based on known 53-second duration")
                                last_blink_data['blink_rate'] = adjusted_blink_rate
                        
                        # Store blink data in session
                        session['blink_count'] = blink_detector.blink_counter
                        session['blink_rate'] = last_blink_data.get('blink_rate', 0.0) if last_blink_data else 0.0
                        
                        # Update return values
                        blink_count = blink_detector.blink_counter
                        blink_rate = last_blink_data.get('blink_rate', 0.0) if last_blink_data else 0.0
                        
                        print(f"Blink data stored in session: count={blink_count}, rate={blink_rate}")
                    except Exception as e:
                        print(f"Error in blink detection: {str(e)}")
                        if status == "success":
                            status = "partial_success"
                            message = f"Webcam recording saved but blink detection failed: {str(e)}"
                        else:
                            message += f". Blink detection also failed: {str(e)}"
                    
                    # Process the same video for gaze tracking
                    print("Starting gaze tracking analysis...")
                    gaze_tracker = GazeEstimator()
                    
                    try:
                        # Process video for gaze tracking
                        # Open the video file
                        cap = cv2.VideoCapture(filepath)
                        
                        if not cap.isOpened():
                            raise Exception("Failed to open video file for gaze tracking")
                        
                        # Process each frame for gaze tracking
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break
                                
                            # Analyze gaze in this frame
                            _, _ = gaze_tracker.analyze_frame(frame)
                        
                        # Release the video capture
                        cap.release()
                        
                        # Get final gaze tracking metrics
                        gaze_metrics = gaze_tracker.get_last_metrics()
                        
                        # Store gaze data in session
                        session['looking_left_count'] = gaze_metrics['looking_left_count']
                        session['looking_right_count'] = gaze_metrics['looking_right_count']
                        session['looking_center_count'] = gaze_metrics['looking_center_count']
                        session['total_gaze_frames'] = gaze_metrics['total_frames_processed']
                        session['ratio_gaze_on_roi'] = gaze_metrics['ratio_gaze_on_roi']
                        
                        # Update return values
                        looking_left_count = gaze_metrics['looking_left_count']
                        looking_right_count = gaze_metrics['looking_right_count']
                        looking_center_count = gaze_metrics['looking_center_count']
                        ratio_gaze_on_roi = gaze_metrics['ratio_gaze_on_roi']
                        
                        print(f"Gaze data stored in session: left={looking_left_count}, right={looking_right_count}, center={looking_center_count}, ratio={ratio_gaze_on_roi}")
                    except Exception as e:
                        print(f"Error in gaze tracking: {str(e)}")
                        if status == "success":
                            status = "partial_success"
                            message = f"Webcam recording saved but gaze tracking failed: {str(e)}"
                        else:
                            message += f". Gaze tracking also failed: {str(e)}"
                    
                    # Process the same video for iris tracking and pupil dilation
                    print("Starting iris tracking and pupil dilation analysis...")
                    iris_tracker = IrisTracker()
                    
                    try:
                        # Process video for iris tracking
                        # Open the video file
                        cap = cv2.VideoCapture(filepath)
                        
                        if not cap.isOpened():
                            raise Exception("Failed to open video file for iris tracking")
                        
                        # Process each frame for iris tracking
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break
                                
                            # Detect iris in this frame
                            iris_data = iris_tracker.detect_iris(frame)
                        
                        # Release the video capture
                        cap.release()
                        
                        # Get final iris tracking metrics
                        iris_metrics = iris_tracker.get_metrics()
                        
                        # Store iris data in session
                        session['pupil_dilation_delta'] = iris_metrics['pupil_dilation_delta']
                        session['avg_pupil_size'] = iris_metrics['avg_pupil_size']
                        session['min_pupil_size'] = iris_metrics.get('min_pupil_size', 0)
                        session['max_pupil_size'] = iris_metrics.get('max_pupil_size', 0)
                        
                        # Update return values
                        pupil_dilation_delta = iris_metrics['pupil_dilation_delta']
                        avg_pupil_size = iris_metrics['avg_pupil_size']
                        
                        print(f"Iris data stored in session: pupil_dilation_delta={pupil_dilation_delta}, avg_pupil_size={avg_pupil_size}")
                    except Exception as e:
                        print(f"Error in iris tracking: {str(e)}")
                        if status == "success":
                            status = "partial_success"
                            message = f"Webcam recording saved but iris tracking failed: {str(e)}"
                        else:
                            message += f". Iris tracking also failed: {str(e)}"
                    
                    return jsonify({
                        "status": status,
                        "message": message,
                        "filename": filename,
                        "filesize": file_size,
                        "dominant_emotion": dominant_emotion_code,
                        "emotion_label": dominant_emotion_label,
                        "emotion_counts": emotion_counts,
                        "blink_count": blink_count,
                        "blink_rate": blink_rate,
                        "looking_left_count": looking_left_count,
                        "looking_right_count": looking_right_count,
                        "looking_center_count": looking_center_count,
                        "ratio_gaze_on_roi": ratio_gaze_on_roi,
                        "pupil_dilation_delta": pupil_dilation_delta,
                        "avg_pupil_size": avg_pupil_size
                    })
                    
                except Exception as e:
                    print(f"Error analyzing video: {str(e)}")
                    # Still return success but indicate analysis failed
                    session['dominant_emotion'] = dominant_emotion_code  # Default to neutral
                    session['dominant_emotion_label'] = dominant_emotion_label
                    session['blink_count'] = blink_count
                    session['blink_rate'] = blink_rate
                    session['looking_left_count'] = looking_left_count
                    session['looking_right_count'] = looking_right_count
                    session['looking_center_count'] = looking_center_count
                    session['ratio_gaze_on_roi'] = ratio_gaze_on_roi
                    session['pupil_dilation_delta'] = pupil_dilation_delta
                    session['avg_pupil_size'] = avg_pupil_size
                    session['emotion_counts'] = emotion_counts
                    
                    return jsonify({
                        "status": "partial_success",
                        "message": f"Webcam recording saved but analysis failed: {str(e)}",
                        "filename": filename,
                        "filesize": file_size
                    })
            else:
                print("File was created but is empty")
                return jsonify({"status": "error", "message": "File was saved but is empty"}), 500
        else:
            print(f"Failed to save file at {filepath}")
            return jsonify({"status": "error", "message": "Failed to save file"}), 500
            
    except Exception as e:
        print(f"Error in save_webcam_recording: {str(e)}")
        return jsonify({"status": "error", "message": f"Server error: {str(e)}"}), 500

@main.route('/final_result')
def final_result():
    print("\n===== APPLICATION FLOW: Final Results Page =====")
    # Debug information
    print("Accessing final_result route")
    print(f"Session variables: {str(session.keys())}")
    
    # Check if we have all necessary data, but use defaults if not available
    # instead of redirecting, so the user can see results even if some data is missing
    if 'phq8_score' not in session:
        print("Missing phq8_score in session - using default")
        # Use a default value instead of redirecting
        session['phq8_score'] = 5  # Default mild score
    
    if 'game_data' not in session:
        print("Missing game_data in session - using default")
        # Use default game data instead of redirecting
        session['game_data'] = {
            'score': 0,
            'features': {
                'avg_reaction_time': 1000,
                'accuracy': 75.0
            },
            'emotional_indicators': []
        }
    
    if 'dominant_emotion' not in session:
        print("Missing dominant_emotion in session - using default")
        # Use default values instead of redirecting
        session['dominant_emotion'] = 0  # Neutral
        session['dominant_emotion_label'] = 'neutral'
        session['emotion_counts'] = {
            'angry': 0, 'disgust': 0, 'fear': 0, 
            'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 1
        }
        session['blink_count'] = 0
        session['blink_rate'] = 0.0
    
    # Get data from session
    phq8_score = session.get('phq8_score')
    game_data = session.get('game_data', {})
    dominant_emotion = session.get('dominant_emotion')
    dominant_emotion_label = session.get('dominant_emotion_label', 'neutral')
    emotion_counts = session.get('emotion_counts', {})
    blink_count = session.get('blink_count', 0)
    blink_rate = session.get('blink_rate', 0.0)
    
    # Get gaze tracking data with defaults
    looking_left_count = session.get('looking_left_count', 0)
    looking_right_count = session.get('looking_right_count', 0)
    looking_center_count = session.get('looking_center_count', 0)
    total_gaze_frames = session.get('total_gaze_frames', 0)
    ratio_gaze_on_roi = session.get('ratio_gaze_on_roi', 0.0)
    
    # Get pupil dilation data with defaults
    pupil_dilation_delta = session.get('pupil_dilation_delta', 0.0)
    avg_pupil_size = session.get('avg_pupil_size', 0.0)
    min_pupil_size = session.get('min_pupil_size', 0.0)
    max_pupil_size = session.get('max_pupil_size', 0.0)
    
    # Load depression prediction model and make prediction
    try:
        print("\n----- PREDICTION MODEL DETAILS -----")
        predictor = DepressionPredictor()
        
        # Get variables being passed to prediction model
        blink_count = session.get('blink_count', 0)
        pupil_dilation_delta = session.get('pupil_dilation_delta', 0.0)
        ratio_gaze_on_roi = session.get('ratio_gaze_on_roi', 0.0)
        dominant_emotion = session.get('dominant_emotion', 0)
        phq8_score = session.get('phq8_score', 0)
        
        # Game data features for prediction
        game_data = session.get('game_data', {})
        features_dict = game_data.get('features', {})
        avg_reaction_time = features_dict.get('avg_reaction_time', 0.0)
        accuracy = features_dict.get('accuracy', 0.0)
        emotional_bias = features_dict.get('emotional_bias', 0.0)
        distraction_recovery = features_dict.get('distraction_recovery', 0.0)
        distraction_response = features_dict.get('distraction_response', 0.0)
        emotional_response_ratio = features_dict.get('emotional_response_ratio', 0.0)
        
        # Log all input variables
        print(f"Input variables for prediction model:")
        print(f"  blink_count = {blink_count}")
        print(f"  pupil_dilation_delta = {pupil_dilation_delta}")
        print(f"  ratio_gaze_on_roi = {ratio_gaze_on_roi}")
        print(f"  dominant_emotion = {dominant_emotion} ({dominant_emotion_label})")
        print(f"  phq8_score = {phq8_score}")
        print(f"  avg_reaction_time = {avg_reaction_time}")
        print(f"  accuracy = {accuracy}")
        print(f"  emotional_bias = {emotional_bias}")
        print(f"  distraction_recovery = {distraction_recovery}")
        print(f"  distraction_response = {distraction_response}")
        print(f"  emotional_response_ratio = {emotional_response_ratio}")
        
        # Extract features and make prediction
        features = predictor.extract_features_from_session(session)
        print(f"Feature array passed to model: {features}")
        print(f"Using blink_count={blink_count} in depression prediction model (replaced blink_rate)")
        
        # Make prediction
        print("Making prediction with model...")
        is_depressed, depression_confidence = predictor.predict(features)
        print(f"Prediction result: is_depressed={is_depressed}, confidence={depression_confidence:.4f}")
        
        # Store prediction results in session
        session['is_depressed'] = is_depressed
        session['depression_confidence'] = depression_confidence
        print("----- END PREDICTION MODEL DETAILS -----\n")
    except Exception as e:
        print(f"Error in prediction model: {str(e)}")
        # Use defaults if prediction fails
        is_depressed = False
        depression_confidence = 0.3
        session['is_depressed'] = is_depressed
        session['depression_confidence'] = depression_confidence
    
    print(f"Rendering final_result template with data: PHQ-8={phq8_score}, Game={game_data.get('score', 0)}, Emotion={dominant_emotion_label}")
    
    # For now, just pass these to the template
    return render_template(
        'final_result.html',
        phq8_score=phq8_score,
        game_data=game_data,
        dominant_emotion=dominant_emotion,
        dominant_emotion_label=dominant_emotion_label,
        emotion_counts=emotion_counts,
        blink_count=blink_count,
        blink_rate=blink_rate,
        looking_left_count=looking_left_count,
        looking_right_count=looking_right_count,
        looking_center_count=looking_center_count,
        total_gaze_frames=total_gaze_frames,
        ratio_gaze_on_roi=ratio_gaze_on_roi,
        pupil_dilation_delta=pupil_dilation_delta,
        avg_pupil_size=avg_pupil_size,
        min_pupil_size=min_pupil_size,
        max_pupil_size=max_pupil_size,
        is_depressed=is_depressed,
        depression_confidence=depression_confidence
    )

@main.route('/ai_report')
def ai_report():
    print("\n===== APPLICATION FLOW: AI Report Page =====")
    """Display the markdown report received from webhook"""
    # No need to check for data completeness since the webhook handling is done client-side
    # Just provide a basic template for the JavaScript to fill in
    
    print("Rendering AI report template for webhook content")
    return render_template('ai_report.html')

@main.route('/save_game_data', methods=['POST'])
def save_game_data():
    print("\n===== APPLICATION FLOW: Processing Game Data =====")
    # Get data from request
    data = request.json
    
    # Add timestamp and PHQ-8 score
    data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    data['phq8_score'] = session.get('phq8_score', None)
    
    print(f"Game data received - Score: {data.get('score', 0)}")
    
    # Extract features for emotion prediction
    features = extract_features(data)
    data['extracted_features'] = features
    
    # Analyze emotional indicators
    emotional_indicators = analyze_emotional_indicators(features)
    data['emotional_indicators'] = emotional_indicators
    
    # Save data to file
    filename = f"game_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(GAME_DATA_DIR, filename)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Store game data in session for final analysis
    session['game_data'] = {
        'score': data.get('score', 0),
        'features': features,
        'emotional_indicators': emotional_indicators
    }
    
    print("Game data saved successfully to session")
    return jsonify({
        "status": "success",
        "features": features,
        "emotional_indicators": emotional_indicators,
        "message": "Game data saved successfully"
    })

def extract_features(data):
    """Extract relevant features from game data"""
    return {
        "score": data.get("score", 0),
        "stars_collected": data.get("starsCollected", 0),
        "blocks_hit": data.get("blocksHit", 0),
        "blocks_dodged": data.get("blocksDodged", 0),
        "positive_emojis": data.get("positiveEmojiInteractions", 0),
        "negative_emojis": data.get("negativeEmojiInteractions", 0),
        "neutral_emojis": data.get("neutralEmojiInteractions", 0),
        "movement_changes": data.get("movementDirectionChanges", 0),
        "hesitations": data.get("hesitations", 0),
        "avg_reaction_time": calculate_avg_reaction_time(data.get("reactionTimes", [])),
        "reaction_time_variability": data.get("reactionTimeVariability", 0),
        "distraction_recovery": calculate_distraction_recovery(data),
        "emotional_bias": calculate_emotional_bias(data),
        "emotional_response_ratio": data.get("emotionalResponseRatio", 0),
        "movement_variability": data.get("movementVariability", 0),
        "avg_response_to_positive": data.get("avgResponseToPositive", 0),
        "avg_response_to_negative": data.get("avgResponseToNegative", 0),
        "accuracy": data.get("accuracy", 0),
        "performance_degradation": data.get("performanceDegradation", 0),
        "positive_emoji_percentage": data.get("positiveEmojiPercentage", 0),
        "distraction_accuracy_delta": data.get("distractionAccuracyDelta", 0),
        "pre_distraction_accuracy": data.get("preDistractionAccuracy", 0),
        "post_distraction_accuracy": data.get("postDistractionAccuracy", 0)
    }

def analyze_emotional_indicators(features):
    """Analyze features to determine emotional indicators"""
    indicators = []
    
    # Anxiety indicator
    anxiety_score = calculate_anxiety_score(features)
    if anxiety_score > 0.3:  # Lowered threshold to detect more subtle indicators
        indicators.append({
            "emotion": "Anxiety",
            "confidence": anxiety_score,
            "indicators": ["Frequent direction changes", "High hesitation count", "High movement variability"]
        })
    
    # Depression indicator
    depression_score = calculate_depression_score(features)
    if depression_score > 0.3:  # Lowered threshold
        indicators.append({
            "emotion": "Depression",
            "confidence": depression_score,
            "indicators": ["Low engagement", "Negative emoji preference", "Slower reaction times"]
        })
    
    # Emotional stability
    stability_score = calculate_stability_score(features)
    if stability_score > 0.6:  # Lowered threshold
        indicators.append({
            "emotion": "Emotional Stability",
            "confidence": stability_score,
            "indicators": ["Consistent performance", "Balanced responses", "Good distraction recovery"]
        })
        
    # Attention deficit
    attention_score = calculate_attention_score(features)
    if attention_score > 0.4:
        indicators.append({
            "emotion": "Attention Deficit",
            "confidence": attention_score,
            "indicators": ["High reaction time variability", "Low accuracy", "Performance degradation over time"]
        })
    
    return indicators

def calculate_avg_reaction_time(reaction_times):
    """Calculate average reaction time"""
    if not reaction_times or len(reaction_times) == 0:
        return 0
    return sum(reaction_times) / len(reaction_times)

def calculate_distraction_recovery(data):
    """Calculate recovery rate after distractions"""
    pre_distraction = data.get("preDistractionSpeed", 0)
    post_distraction = data.get("postDistractionSpeed", 0)
    
    # If we have valid speed data
    if pre_distraction and post_distraction and pre_distraction > 0:
        return min(post_distraction / pre_distraction, 1)
    
    # Alternative calculation if speeds aren't available but we have delta
    distraction_delta = data.get("distractionResponseDelta", 0)
    if distraction_delta is not None:
        # Normalize to a 0-1 scale (higher is better recovery)
        normalized_delta = max(0, min(1, 0.5 + distraction_delta / 2))
        return normalized_delta
    
    return 0.5  # Default neutral value

def calculate_emotional_bias(data):
    """Calculate emotional bias (preference for positive/negative stimuli)"""
    # Try to get from direct data first
    emotional_response_ratio = data.get("emotionalResponseRatio")
    if emotional_response_ratio is not None:
        return emotional_response_ratio
    
    # Otherwise calculate from emoji interactions
    positive = data.get("positiveEmojiInteractions", 0)
    negative = data.get("negativeEmojiInteractions", 0)
    total = positive + negative
    
    if total == 0:
        return 0
    
    return (positive - negative) / total

def calculate_anxiety_score(features):
    """Calculate anxiety score based on movement patterns"""
    # More sophisticated calculation using multiple metrics
    movement_factor = min(features.get("movement_changes", 0) / 50, 1)
    hesitation_factor = min(features.get("hesitations", 0) / 15, 1)
    
    # Higher movement variability can indicate anxiety
    variability_factor = min(features.get("movement_variability", 0) / 100, 1)
    
    # Reaction time to negative stimuli - faster reactions might indicate anxiety
    negative_response_time = features.get("avg_response_to_negative", 0)
    reaction_factor = 0.5
    if negative_response_time > 0:
        # Normalize reaction time (faster = higher score)
        reaction_factor = max(0, min(1, 1 - (negative_response_time / 2000)))
    
    # High reaction time variability can indicate anxiety
    rt_variability = features.get("reaction_time_variability", 0)
    rt_variability_factor = min(rt_variability / 500, 1)
    
    # Weighted average of factors
    return (movement_factor * 0.25 + hesitation_factor * 0.25 + 
            variability_factor * 0.2 + reaction_factor * 0.15 + rt_variability_factor * 0.15)

def calculate_depression_score(features):
    """Calculate depression score based on engagement and preferences"""
    # Lower score = higher stars collected (higher engagement)
    engagement_factor = 1 - min(features.get("stars_collected", 0) / 15, 1)
    
    # Higher score = more negative emojis collected
    emoji_preference = 0.5  # Neutral default
    positive = features.get("positive_emojis", 0)
    negative = features.get("negative_emojis", 0)
    total_emojis = positive + negative
    
    if total_emojis > 0:
        emoji_preference = negative / total_emojis
    
    # Lower positive emoji percentage may indicate depression
    positive_emoji_pct = features.get("positive_emoji_percentage", 50)
    emoji_pct_factor = 1 - (positive_emoji_pct / 100)
    
    # Slower reaction time may indicate depression
    avg_reaction_time = features.get("avg_reaction_time", 1000)
    reaction_time_factor = min(avg_reaction_time / 2000, 1)
    
    # Performance degradation may indicate depression
    perf_degradation = features.get("performance_degradation", 0)
    degradation_factor = 0.5
    if perf_degradation < 0:
        degradation_factor = min(abs(perf_degradation) / 50, 1)
    
    # Weighted calculation
    return (engagement_factor * 0.3 + emoji_preference * 0.25 + 
            emoji_pct_factor * 0.15 + reaction_time_factor * 0.15 + degradation_factor * 0.15)

def calculate_stability_score(features):
    """Calculate emotional stability score"""
    # Higher dodge rate = more stable
    blocks_dodged = features.get("blocks_dodged", 0)
    blocks_hit = features.get("blocks_hit", 0)
    total_blocks = blocks_dodged + blocks_hit
    dodge_rate = 0.5  # Default neutral
    
    if total_blocks > 0:
        dodge_rate = blocks_dodged / total_blocks
    
    # Lower hesitation = more stable
    hesitation_factor = 1 - min(features.get("hesitations", 0) / 20, 1)
    
    # Balanced emoji interactions = more stable
    emoji_balance = 0.5  # Default neutral
    positive = features.get("positive_emojis", 0)
    negative = features.get("negative_emojis", 0)
    total_emojis = positive + negative
    
    if total_emojis > 3:  # Only consider if enough emoji interactions
        # 0.5 = perfectly balanced, 0 or 1 = imbalanced
        emoji_balance = 0.5 + abs(0.5 - (positive / total_emojis)) * -1
    
    # Better distraction recovery indicates stability
    distraction_recovery = features.get("distraction_recovery", 0.5)
    
    # Lower reaction time variability indicates stability
    rt_variability = features.get("reaction_time_variability", 300)
    rt_stability_factor = 1 - min(rt_variability / 600, 1)
    
    # Weighted calculation
    return (dodge_rate * 0.3 + hesitation_factor * 0.2 + emoji_balance * 0.2 + 
            distraction_recovery * 0.15 + rt_stability_factor * 0.15)

def calculate_attention_score(features):
    """Calculate attention deficit score based on performance metrics"""
    # Higher reaction time variability may indicate attention issues
    rt_variability = features.get("reaction_time_variability", 0)
    rt_variability_factor = min(rt_variability / 500, 1)
    
    # Lower accuracy may indicate attention issues
    accuracy = features.get("accuracy", 100)
    accuracy_factor = 1 - (accuracy / 100)
    
    # Negative performance degradation (getting worse) may indicate attention issues
    perf_degradation = features.get("performance_degradation", 0)
    degradation_factor = 0.5
    if perf_degradation < 0:
        degradation_factor = min(abs(perf_degradation) / 50, 1)
    
    # Greater post-distraction accuracy drop may indicate attention issues
    distraction_accuracy_delta = features.get("distraction_accuracy_delta", 0)
    distraction_factor = 0.5
    if distraction_accuracy_delta < 0:
        distraction_factor = min(abs(distraction_accuracy_delta) / 50, 1)
    
    # Weighted calculation
    return (rt_variability_factor * 0.3 + accuracy_factor * 0.3 + 
            degradation_factor * 0.2 + distraction_factor * 0.2)

@main.route('/debug_session')
def debug_session():
    """Debug route to check session state - only for development"""
    print("\n===== APPLICATION FLOW: Debug Session Route =====")
    # Create a summary of the session for debugging
    session_summary = {
        'session_keys': list(session.keys()),
        'phq8_completed': 'phq8_score' in session,
        'game_completed': 'game_data' in session,
        'video_completed': 'dominant_emotion' in session,
        'current_state': {}
    }
    
    # Add details about each major component
    if 'phq8_score' in session:
        session_summary['current_state']['phq8_score'] = session['phq8_score']
    
    if 'game_data' in session:
        game_data = session['game_data']
        if isinstance(game_data, dict):
            session_summary['current_state']['game_score'] = game_data.get('score', 'unknown')
    
    if 'dominant_emotion' in session:
        session_summary['current_state']['dominant_emotion_code'] = session['dominant_emotion']
        session_summary['current_state']['dominant_emotion_label'] = session.get('dominant_emotion_label', 'unknown')
    
    print(f"Debug session summary: {session_summary}")
    
    # Return as both JSON (for API calls) and render HTML page
    if request.headers.get('Content-Type') == 'application/json':
        return jsonify(session_summary)
    else:
        return render_template('debug_session.html', 
                               session_summary=session_summary,
                               session_keys=list(session.keys()),
                               session_data=session) 