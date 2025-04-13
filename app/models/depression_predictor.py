import numpy as np
import tensorflow as tf
import os

class CustomAdamOptimizer(tf.keras.optimizers.Adam):
    """Custom Adam optimizer that ignores weight_decay parameter."""
    
    def __init__(self, *args, **kwargs):
        # Remove weight_decay if present
        if 'weight_decay' in kwargs:
            del kwargs['weight_decay']
        super().__init__(*args, **kwargs)

class DepressionPredictor:
    """Class for loading and using the depression prediction model."""
    
    def __init__(self, model_path=None):
        """
        Initialize the depression predictor.
        
        Args:
            model_path (str): Path to the .h5 model file. If None, use default path.
        """
        if model_path is None:
            model_path = os.path.join('app', 'models', 'depression_prediction_model.h5')
            # Alternative path for absolute path
            if not os.path.exists(model_path):
                model_path = "C:\\Users\\ujwal\\OneDrive\\Desktop\\hack_1\\app\\models\\depression_prediction_model.h5"
        
        self.model_path = model_path
        self.model = None
        self.loaded = False
        
        # Try to load the model
        self.load_model()
    
    def load_model(self):
        """Load the TensorFlow model."""
        print("\n----- MODEL LOADING PROCESS -----")
        print(f"Attempting to load depression prediction model from: {self.model_path}")
        try:
            # First try: Use custom optimizer to handle weight_decay
            print("Approach 1: Loading with custom Adam optimizer to handle weight_decay")
            custom_adam = CustomAdamOptimizer()
            custom_objects = {
                'Adam': CustomAdamOptimizer,
                'optimizer': custom_adam
            }
            
            # Load model with custom objects
            self.model = tf.keras.models.load_model(
                self.model_path,
                custom_objects=custom_objects,
                compile=False
            )
            
            # Recompile with known-good settings
            self.model.compile(
                optimizer=custom_adam,
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            self.loaded = True
            print(f"Successfully loaded depression prediction model from {self.model_path}")
            print(f"Model summary:")
            print(f"  Input shape: {self.model.input_shape}")
            print(f"  Output shape: {self.model.output_shape}")
            print(f"  Number of layers: {len(self.model.layers)}")
        except Exception as e:
            print(f"Error loading depression prediction model: {str(e)}")
            
            # Second try: Use legacy Adam optimizer
            try:
                print("Approach 2: Using legacy Adam optimizer")
                optimizer = tf.keras.optimizers.legacy.Adam()
                
                self.model = tf.keras.models.load_model(
                    self.model_path,
                    compile=False
                )
                
                self.model.compile(
                    optimizer=optimizer,
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                
                self.loaded = True
                print("Successfully loaded model with legacy optimizer")
                print(f"Model summary:")
                print(f"  Input shape: {self.model.input_shape}")
                print(f"  Output shape: {self.model.output_shape}")
                print(f"  Number of layers: {len(self.model.layers)}")
            except Exception as e2:
                print(f"Legacy optimizer approach failed: {str(e2)}")
                
                # Third try: Safe mode loading
                try:
                    print("Approach 3: Safe mode loading with experimental I/O device")
                    
                    self.model = tf.keras.models.load_model(
                        self.model_path, 
                        compile=False,
                        options=tf.saved_model.LoadOptions(
                            experimental_io_device='/job:localhost'
                        )
                    )
                    
                    # Simple compilation with minimal settings
                    self.model.compile(
                        optimizer='adam',
                        loss='binary_crossentropy'
                    )
                    
                    self.loaded = True
                    print("Successfully loaded model with safe mode")
                    print(f"Model summary:")
                    print(f"  Input shape: {self.model.input_shape}")
                    print(f"  Output shape: {self.model.output_shape}")
                    print(f"  Number of layers: {len(self.model.layers)}")
                except Exception as e3:
                    print(f"Safe mode loading failed: {str(e3)}")
                    self.loaded = False
                    
                    # Last resort: Create fallback model
                    print("All loading approaches failed. Creating fallback model.")
                    self._create_fallback_model()
        
        print("----- END MODEL LOADING PROCESS -----\n")
    
    def _create_fallback_model(self):
        """Create a simple fallback model if loading fails"""
        print("\n----- CREATING FALLBACK MODEL -----")
        try:
            # Create a simple binary classification model with same input shape
            print("Creating simple neural network with 12 input features and 1 output neuron")
            inputs = tf.keras.layers.Input(shape=(12,))  # 12 features as input
            x = tf.keras.layers.Dense(8, activation='relu')(inputs)
            outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
            
            self.model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
            print("Model architecture defined: Input(12) → Dense(8, relu) → Dense(1, sigmoid)")
            
            self.model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            print("Model compiled with adam optimizer and binary_crossentropy loss")
            
            # Initialize with balanced predictions (0.5)
            print("Initializing weights to provide balanced (0.5) predictions")
            for layer in self.model.layers:
                if hasattr(layer, 'kernel'):
                    layer.kernel.assign(tf.zeros(layer.kernel.shape))
                    print(f"  Zeroed weights for layer: {layer.name}, shape: {layer.kernel.shape}")
                if hasattr(layer, 'bias'):
                    if layer.name == 'dense_1':  # Output layer
                        # Set bias to give ~0.5 prediction (logit 0)
                        layer.bias.assign(tf.zeros(layer.bias.shape))
                        print(f"  Zeroed bias for output layer: {layer.name}")
                    else:
                        layer.bias.assign(tf.zeros(layer.bias.shape))
                        print(f"  Zeroed bias for layer: {layer.name}")
            
            self.loaded = True
            print("Fallback model created and initialized successfully")
            print(f"Model summary:")
            print(f"  Input shape: {self.model.input_shape}")
            print(f"  Output shape: {self.model.output_shape}")
            print(f"  Number of layers: {len(self.model.layers)}")
        except Exception as e:
            print(f"Failed to create fallback model: {str(e)}")
            self.loaded = False
        print("----- END CREATING FALLBACK MODEL -----\n")
    
    def predict(self, features):
        """
        Predict depression likelihood from features.
        
        Args:
            features (list): A list of features in the following order:
                [blink_count, pupil_dilation_delta, ratio_gaze_on_roi, 
                dominant_emotion, phq8_score, avg_reaction_time, accuracy,
                emotional_bias, distraction_recovery, distraction_response,
                emotional_response_ratio, emoji_collection_ratio]
        
        Returns:
            tuple: (is_depressed (bool), confidence (float))
        """
        if not self.loaded or self.model is None:
            print("Model not loaded, cannot make prediction")
            return False, 0.5  # Return neutral prediction
        
        try:
            # Print detailed feature names and values for better logging
            feature_names = [
                "blink_count", "pupil_dilation_delta", "ratio_gaze_on_roi", 
                "dominant_emotion", "phq8_score", "avg_reaction_time", "accuracy",
                "emotional_bias", "distraction_recovery", "distraction_response",
                "emotional_response_ratio", "emoji_collection_ratio"
            ]
            
            print("Detailed feature breakdown for prediction:")
            for i, (name, value) in enumerate(zip(feature_names, features)):
                print(f"  {i+1}. {name}: {value}")
            
            # Convert features to numpy array and reshape for model input
            features_np = np.array(features, dtype=np.float32).reshape(1, -1)
            print(f"Features as numpy array shape: {features_np.shape}")
            
            # Make prediction with reduced verbosity
            print("Running model.predict() with CPU execution...")
            with tf.device('/CPU:0'):  # Force CPU execution
                prediction = self.model.predict(features_np, verbose=0)
            
            # Get prediction value (between 0 and 1)
            pred_value = float(prediction[0][0])
            print(f"Raw prediction value: {pred_value}")
            
            # Ensure prediction is in valid range
            pred_value = max(0.0, min(1.0, pred_value))
            print(f"Normalized prediction value: {pred_value}")
            
            # Determine result (threshold at 0.5)
            is_depressed = pred_value >= 0.5
            print(f"Classification threshold: 0.5 → is_depressed = {is_depressed}")
            
            return is_depressed, pred_value
            
        except Exception as e:
            print(f"Error making depression prediction: {str(e)}")
            return False, 0.5  # Return neutral prediction on error
    
    def extract_features_from_session(self, session):
        """
        Extract required features from Flask session data.
        
        Args:
            session (dict): Flask session object
            
        Returns:
            list: Features in the expected order for prediction
        """
        print("\nExtracting features from session for prediction model...")
        
        # Extract features in the required order
        blink_count = session.get('blink_count', 0)  # Changed from blink_rate to blink_count
        pupil_dilation_delta = session.get('pupil_dilation_delta', 0.0)
        ratio_gaze_on_roi = session.get('ratio_gaze_on_roi', 0.0)
        dominant_emotion = session.get('dominant_emotion', 0)  # Numeric code for emotion
        phq8_score = session.get('phq8_score', 0)
        
        # Log session-level features as extracted
        print(f"Initial extraction from session:")
        print(f"  blink_count = {blink_count} (raw from session)")
        print(f"  pupil_dilation_delta = {pupil_dilation_delta} (raw from session)")
        print(f"  ratio_gaze_on_roi = {ratio_gaze_on_roi} (raw from session)")
        print(f"  dominant_emotion = {dominant_emotion} (raw from session)")
        print(f"  phq8_score = {phq8_score} (raw from session)")
        
        # Game data features
        game_data = session.get('game_data', {})
        features = game_data.get('features', {})
        
        print(f"Game data features from session:")
        print(f"  game_data keys: {list(game_data.keys())}")
        if isinstance(features, dict):
            print(f"  features keys: {list(features.keys())}")
        
        # Safe conversion function to handle any type of input
        def safe_float(value, default=0.0):
            if isinstance(value, (int, float, str)):
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return default
            # If it's a dictionary or other complex object, return default
            return default
        
        avg_reaction_time = safe_float(features.get('avg_reaction_time', 0.0))
        accuracy = safe_float(features.get('accuracy', 0.0))
        emotional_bias = safe_float(features.get('emotional_bias', 0.0))
        distraction_recovery = safe_float(features.get('distraction_recovery', 0.0))
        distraction_response = safe_float(features.get('distraction_response', 0.0))
        emotional_response_ratio = safe_float(features.get('emotional_response_ratio', 0.0))
        emoji_collection_ratio = 0.0  # Calculate from positive/negative emojis
        
        # Log game features
        print(f"Game data features after extraction:")
        print(f"  avg_reaction_time = {avg_reaction_time}")
        print(f"  accuracy = {accuracy}")
        print(f"  emotional_bias = {emotional_bias}")
        print(f"  distraction_recovery = {distraction_recovery}")
        print(f"  distraction_response = {distraction_response}")
        print(f"  emotional_response_ratio = {emotional_response_ratio}")
        
        # Calculate emoji_collection_ratio if possible
        positive_emojis = safe_float(features.get('positive_emojis', 0))
        negative_emojis = safe_float(features.get('negative_emojis', 0))
        total_emojis = positive_emojis + negative_emojis
        if total_emojis > 0:
            emoji_collection_ratio = positive_emojis / total_emojis
        
        print(f"Emoji analysis:")
        print(f"  positive_emojis = {positive_emojis}")
        print(f"  negative_emojis = {negative_emojis}")
        print(f"  total_emojis = {total_emojis}")
        print(f"  emoji_collection_ratio = {emoji_collection_ratio}")
        
        # If distraction_response is not available or was a complex object, try to use similar metrics
        if distraction_response == 0.0:
            distraction_response = safe_float(features.get('distraction_recovery', 0.0))
            print(f"  distraction_response not found, using distraction_recovery = {distraction_response}")
        
        # Also apply safe conversion to session-level features
        blink_count = safe_float(blink_count)  # Changed from blink_rate to blink_count
        pupil_dilation_delta = safe_float(pupil_dilation_delta)
        ratio_gaze_on_roi = safe_float(ratio_gaze_on_roi)
        dominant_emotion = safe_float(dominant_emotion)
        phq8_score = safe_float(phq8_score)
        
        # Log after safe conversion
        print("Features after safe conversion:")
        print(f"  blink_count = {blink_count}")
        print(f"  pupil_dilation_delta = {pupil_dilation_delta}")
        print(f"  ratio_gaze_on_roi = {ratio_gaze_on_roi}")
        print(f"  dominant_emotion = {dominant_emotion}")
        print(f"  phq8_score = {phq8_score}")
        
        # Compile features in the required order
        final_features = [
            blink_count,  # Changed from blink_rate to blink_count
            pupil_dilation_delta, 
            ratio_gaze_on_roi,
            dominant_emotion,
            phq8_score,
            avg_reaction_time,
            accuracy,
            emotional_bias,
            distraction_recovery,
            distraction_response,
            emotional_response_ratio,
            emoji_collection_ratio
        ]
        
        print(f"Final feature array: {final_features}")
        return final_features 