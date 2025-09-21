# Comprehensive Mental Health Assessment Platform (Technical Deep Dive)

This document provides an exhaustive technical breakdown of the Mental Health Assessment Platform. It is intended for developers and contributors to understand the specific implementation details of each component, the data flow between them, and the technologies involved.

## Core Technologies

- **Backend**: Flask
- **Machine Learning**: TensorFlow/Keras
- **Computer Vision**: OpenCV, DeepFace, MediaPipe
- **Frontend**: HTML/CSS, Bootstrap, JavaScript

---

## Level 1: Application Entrypoint & Configuration

### `run.py`
This is the main entry point for the application. Its sole responsibility is to import the `create_app` factory function from the `app` module and run the Flask application instance. It runs the server in debug mode, which enables features like the interactive debugger and automatic reloader.

### `app/__init__.py`
The application package is initialized here using the factory pattern.
- **`create_app()`**: This function creates and configures the Flask application instance.
    - **Configuration**: It sets the `SECRET_KEY` for session management and `MAX_CONTENT_LENGTH` to limit video upload sizes.
    - **Blueprint Registration**: It registers the `main` blueprint, which contains all the core application routes and logic.
    - **Dynamic Video Routes**: A notable feature is the dynamic creation of multiple URL rules (`/hack.mp4`, `/video.mp4`, etc.). All these routes point to the same physical video file (`app/static/vedio/hack.mp4`). This is a robustness measure to ensure the stimulus video loads correctly regardless of how a browser resolves the path.

---

## Level 2: The Application Core (Routes & Logic)

### `app/main/routes.py`
This is the largest and most critical file, acting as the central controller for the entire application. It defines the logic for each step of the user's journey.

- **Route: `/` (`index`)**
    - **Purpose**: Displays the welcome page (`index.html`).
    - **Logic**: It explicitly clears any pre-existing data from the user's `session` to ensure each assessment starts fresh.

- **Route: `/phq8` (`phq8_questionnaire`)**
    - **Purpose**: Manages the PHQ-8 questionnaire.
    - **Logic**: It uses the `session` to track the `current_question` index and store the list of `phq8_responses`. On a `POST` request, it appends the user's answer to the list and increments the question index. Once all questions are answered, it calculates the `phq8_score` and redirects to the results page.

- **Route: `/game` (`game`)**
    - **Purpose**: Displays the interactive game page (`game.html`).
    - **Logic**: It performs a check to ensure the user has completed the PHQ-8 phase (by checking for `phq8_score` in the session) before allowing access.

- **Route: `/save_game_data` (`save_game_data`)**
    - **Purpose**: An API endpoint that receives the results of the cognitive game from the frontend.
    - **Logic**: 
        1.  Receives a JSON payload from `game.js`.
        2.  Calls `extract_features()` to create a standardized dictionary of over 20 distinct gameplay metrics.
        3.  Calls `analyze_emotional_indicators()` which uses the extracted features to calculate confidence scores for potential emotional states like Anxiety and Depression based on heuristics (e.g., high movement variability, preference for negative emojis).
        4.  Saves the complete, processed data to a timestamped `.json` file in the `app/game_data/` directory for archival.
        5.  Stores the essential features and indicators in the `session` under the `game_data` key for the final prediction phase.

- **Route: `/save_webcam_recording` (`save_webcam_recording`)**
    - **Purpose**: An API endpoint to receive the user's webcam recording from the video analysis phase.
    - **Logic**: This is a major data processing pipeline.
        1.  Receives the `.webm` file and saves it to the `app/webcam_recordings/` directory with a unique timestamp.
        2.  Instantiates the four video analysis classes: `EmotionAnalyzer`, `BlinkDetector`, `GazeEstimator`, and `IrisTracker`.
        3.  It then processes the *entire* saved video file with each analyzer.
        4.  **Emotion Analysis**: Calls `emotion_analyzer.process_video()` to get the dominant emotion and emotion distribution.
        5.  **Blink Detection**: Opens the video with OpenCV, iterates through frames, and passes them to `blink_detector.detect_blink()` to get the final `blink_count` and `blink_rate`.
        6.  **Gaze & Iris Tracking**: Similarly, it iterates through the video frames for the `GazeEstimator` and `IrisTracker` to calculate metrics like `ratio_gaze_on_roi` and `pupil_dilation_delta`.
        7.  All resulting metrics are stored in the `session` and also returned as a JSON response to the frontend.

- **Route: `/final_result` (`final_result`)**
    - **Purpose**: The final step that synthesizes all data and runs the ML prediction.
    - **Logic**:
        1.  Aggregates all necessary data points from the session (`phq8_score`, `game_data`, `dominant_emotion`, `blink_count`, etc.), using default values if any data is missing to prevent crashes.
        2.  Instantiates the `DepressionPredictor` from `app/models/`.
        3.  Calls the crucial `predictor.extract_features_from_session()` method, which carefully gathers the 12 required metrics into a correctly ordered feature vector.
        4.  Calls `predictor.predict()` with this vector.
        5.  The returned `is_depressed` and `depression_confidence` values are passed to the `final_result.html` template for display.

---

## Level 3: Component Deep Dive

### Frontend Game Logic (`app/static/js/game.js`)
This script manages the entire cognitive game. It is not just a game, but a sophisticated data collection tool.
- **State Management**: It maintains dozens of variables to track every aspect of the gameplay, including `playerPositions`, `idlePeriods`, `emotionalStimulusResponses`, `reactionTimes`, `movementDirectionChanges`, and `hesitations`.
- **Game Loop (`gameLoop`)**: The main loop updates object positions, checks for collisions, and re-renders the canvas on each frame using `requestAnimationFrame`.
- **Data Collection**: Player movement is tracked via `mousemove` and `touchmove` events. The `updatePlayerPosition` function is particularly important, as it calculates movement speed, direction changes, and idle time.
- **Metric Calculation**: At the end of the game, the `endGame` function calls numerous helper functions (`calculateMovementVariability`, `calculateEmotionalResponseRatio`, etc.) to compute the final metrics from the raw data collected during gameplay.
- **Backend Communication**: It uses the `fetch` API to `POST` the final `gameData` object to the `/save_game_data` endpoint.

### Video Processing Engine (`app/video_processor/`)

- **`EmotionAnalyzer`**: 
    - **Technology**: `deepface` library.
    - **Implementation**: To provide robust emotion detection, it iterates through multiple detector backends (`opencv`, `retinaface`, `mtcnn`) until one succeeds. It also applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to the grayscale version of the frame to improve detection accuracy. The final emotion is an average of the results from the original and enhanced frames.

- **`BlinkDetector`**:
    - **Technology**: `mediapipe.solutions.face_mesh`.
    - **Implementation**: It calculates the Eye Aspect Ratio (EAR) by using the 3D coordinates of specific facial landmarks around the eyes (indices `[362, 385, ...]` for the left eye, etc.). A blink is registered when the EAR drops below a `0.2` threshold for at least 2 consecutive frames. This provides the final `blink_count`.

- **`GazeEstimator`**:
    - **Technology**: `mediapipe.solutions.face_mesh` and OpenCV.
    - **Implementation**: It first identifies the bounding boxes of the eyes using facial landmarks. It then applies a binary threshold to the grayscale eye region to isolate the pupil. Gaze direction is estimated by comparing the number of non-zero pixels in the left vs. right half of the eye box, determining if the pupil is positioned more to one side.

- **`IrisTracker`**:
    - **Technology**: `mediapipe.solutions.face_mesh` with iris refinement (`refine_landmarks=True`).
    - **Implementation**: This module uses the high-fidelity iris landmarks (indices 468-477) provided by MediaPipe. Pupil size is calculated as the Euclidean distance between opposing iris landmarks. It establishes a baseline pupil size during the initial `30` frames of the video and then calculates `pupil_dilation_delta` by comparing the baseline to the average size during subsequent frames.

### AI Prediction Engine (`app/models/`)

- **`DepressionPredictor` Class**: 
    - **Functionality**: This class orchestrates the loading of the ML model and the prediction process. It is designed for high reliability.
    - **Robust Loading**: It features a multi-step loading mechanism to handle potential Keras versioning or file corruption issues:
        1.  **Custom Optimizer**: Attempts to load with a `CustomAdamOptimizer` that strips the `weight_decay` argument, a common source of errors in older Keras models.
        2.  **Legacy Optimizer**: If the first attempt fails, it retries using `tf.keras.optimizers.legacy.Adam`.
        3.  **Safe Mode**: A third attempt is made using `tf.saved_model.LoadOptions`.
        4.  **Fallback Model**: If all loading attempts fail, the `_create_fallback_model` method is called. This programmatically defines and compiles a simple, shallow neural network with the correct input/output shape. This ensures the application never crashes due to a model loading error, even if the predictions are neutral (it's initialized to output a confidence of 0.5).

- **Prediction Model (`depression_prediction_model.h5`)**: 
    - **Architecture**: This is a pre-trained TensorFlow/Keras model.
    - **Input**: It expects a vector of exactly 12 floating-point features.
    - **Output**: It has a single output neuron with a sigmoid activation function, producing a value between 0 and 1 representing the confidence of a positive depression classification.

- **Feature Vector**: The model's prediction is critically dependent on a correctly ordered 12-feature vector. The `extract_features_from_session` method is responsible for creating this vector from the session data:
    1.  `blink_count` (from video)
    2.  `pupil_dilation_delta` (from video)
    3.  `ratio_gaze_on_roi` (from video)
    4.  `dominant_emotion` (from video, integer-coded)
    5.  `phq8_score` (from questionnaire)
    6.  `avg_reaction_time` (from game)
    7.  `accuracy` (from game)
    8.  `emotional_bias` (from game)
    9.  `distraction_recovery` (from game)
    10. `distraction_response` (from game)
    11. `emotional_response_ratio` (from game)
    12. `emoji_collection_ratio` (from game)

---

## How to Run

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the Application**:
    ```bash
    python run.py
    ```
3.  Open your web browser and navigate to `http://127.0.0.1:5000`.
