import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# Load and prepare the data
print("Loading dataset...")
data = pd.read_csv('enhanced_depression_dataset.csv')

# Separate features and target
X = data.drop('is_depressed', axis=1)
y = data['is_depressed']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the model
def create_model():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(16, activation='relu'),
        BatchNormalization(),
        Dropout(0.1),
        
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    return model

# Create and train the model
print("Training neural network...")
model = create_model()

# Add early stopping to prevent overfitting
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Train the model
history = model.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate the model
test_loss, test_accuracy, test_auc = model.evaluate(X_test_scaled, y_test)
print(f"\nTest Accuracy: {test_accuracy:.4f}")
print(f"Test AUC: {test_auc:.4f}")

# Make predictions on the test set
y_pred_proba = model.predict(X_test_scaled)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Display confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Plot training history
plt.figure(figsize=(12, 4))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.savefig('training_history.png')
print("Training history saved as 'training_history.png'")

# Save the model
model.save('depression_prediction_model.h5')
print("Model saved as 'depression_prediction_model.h5'")

# Save the scaler for future use
import joblib
joblib.dump(scaler, 'feature_scaler.pkl')
print("Scaler saved as 'feature_scaler.pkl'")

# Create a function for making predictions on new data
def predict_depression(new_data):
    """
    Predict depression status for new data.
    
    Parameters:
    - new_data: DataFrame with the same features as the training data
    
    Returns:
    - Numpy array of predictions (0 or 1)
    """
    # Scale the features
    new_data_scaled = scaler.transform(new_data)
    
    # Make predictions
    predictions_proba = model.predict(new_data_scaled)
    predictions = (predictions_proba > 0.5).astype(int).flatten()
    
    return predictions

# Example usage:
print("\nExample of using the model for prediction:")
print("To predict for new data, use the following code:")
print("from depression_prediction_model import predict_depression")
print("import pandas as pd")
print("# Prepare your data with the same features")
print("new_data = pd.DataFrame({...})  # Your new data here")
print("predictions = predict_depression(new_data)") 