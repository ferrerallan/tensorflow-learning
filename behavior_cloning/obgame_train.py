"""
Behavior Cloning Training Script for Obstacle Game

Trains a neural network to predict ship movements by learning from
human gameplay demonstrations (behavior cloning approach).
"""

from typing import Tuple

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ============================= GAME CONFIGURATION =============================

# Grid dimensions (must match the game configuration)
GRID_ROWS = 3
GRID_COLUMNS = 5

# Normalization constant for neural network input
NORMALIZATION_MAX = 10.0

# Model architecture parameters
INPUT_FEATURES = 7  # obstacle1_x, obstacle1_y, obstacle2_x, obstacle2_y, ship_x, ship_y, timestep
HIDDEN_LAYER_SIZES = [32, 16, 8]
NUM_ACTION_CLASSES = 3  # up (-1), stay (0), down (1)

# Training parameters
DEFAULT_EPOCHS = 200
DEFAULT_BATCH_SIZE = 64
TEST_SIZE_RATIO = 0.2
RANDOM_SEED = 42

# File paths
DEFAULT_DATASET_PATH = "behavior_cloning/dataset_actions.csv"
DEFAULT_MODEL_PATH = "behavior_cloning/model.h5"

# Action mapping
ACTION_UP = -1
ACTION_STAY = 0
ACTION_DOWN = 1


# ============================= DATA LOADING ===================================

def normalize_features(features: np.ndarray) -> np.ndarray:
    """
    Normalize feature values while preserving negative values.
    
    Only normalizes values >= 0, keeps -1 values intact (used for off-screen positions).
    
    Args:
        features: Raw feature array
        
    Returns:
        Normalized feature array
    """
    positive_mask = features >= 0
    normalized_features = features.copy()
    normalized_features[positive_mask] = features[positive_mask] / NORMALIZATION_MAX
    return normalized_features


def map_action_to_class_index(action: int) -> int:
    """
    Map action value to class index for one-hot encoding.
    
    Mapping:
    - Action -1 (up) -> Class 0
    - Action  0 (stay) -> Class 1
    - Action  1 (down) -> Class 2
    
    Args:
        action: Action value (-1, 0, or 1)
        
    Returns:
        Class index (0, 1, or 2)
    """
    return action + 1


def load_and_prepare_data(
    dataset_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load dataset from CSV and prepare training/test splits.
    
    Features:
    - obstaculo1_x: First obstacle column position
    - obstaculo1_y: First obstacle row position
    - obstaculo2_x: Second obstacle column position
    - obstaculo2_y: Second obstacle row position
    - nave_x: Ship column position
    - nave_y: Ship row position
    - tempo: Timestep in episode
    
    Output:
    - movimento_correto: Correct movement action (-1, 0, or 1)
      Converted to one-hot encoding with 3 classes
    
    Args:
        dataset_path: Path to CSV file containing training data
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    # Load dataset
    dataframe = pd.read_csv(dataset_path)
    
    # Extract feature columns
    feature_columns = [
        "obstaculo1_x",
        "obstaculo1_y",
        "obstaculo2_x",
        "obstaculo2_y",
        "nave_x",
        "nave_y",
        "tempo",
    ]
    features = dataframe[feature_columns].values.astype(np.float32)
    
    # Normalize features
    normalized_features = normalize_features(features)
    
    # Extract and convert actions to one-hot encoding
    actions = dataframe["movimento_correto"].values.astype(int)
    
    # Map actions (-1, 0, 1) to class indices (0, 1, 2)
    class_indices = np.array(
        [map_action_to_class_index(action) for action in actions],
        dtype=int
    )
    
    # Convert to one-hot encoding
    one_hot_labels = tf.keras.utils.to_categorical(
        class_indices,
        num_classes=NUM_ACTION_CLASSES
    )
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        normalized_features,
        one_hot_labels,
        test_size=TEST_SIZE_RATIO,
        random_state=RANDOM_SEED,
    )
    
    return X_train, X_test, y_train, y_test


# ============================= MODEL CREATION =================================

def create_mlp_model() -> tf.keras.Model:
    """
    Create Multi-Layer Perceptron model for action prediction.
    
    Architecture:
    - Input layer: 7 features
    - Hidden layers: 32 -> 16 -> 8 neurons with ReLU activation
    - Output layer: 3 neurons with softmax (up, stay, down)
    
    Returns:
        Compiled Keras Sequential model
    """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(INPUT_FEATURES,)),
            tf.keras.layers.Dense(HIDDEN_LAYER_SIZES[0], activation="relu"),
            tf.keras.layers.Dense(HIDDEN_LAYER_SIZES[1], activation="relu"),
            tf.keras.layers.Dense(HIDDEN_LAYER_SIZES[2], activation="relu"),
            tf.keras.layers.Dense(NUM_ACTION_CLASSES, activation="softmax"),
        ]
    )
    return model


def compile_model(model: tf.keras.Model) -> None:
    """
    Compile model with optimizer, loss function, and metrics.
    
    Uses:
    - Optimizer: Adam (adaptive learning rate)
    - Loss: Categorical cross-entropy (multi-class classification)
    - Metrics: Accuracy
    
    Args:
        model: Keras model to compile
    """
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


# ============================= TRAINING =======================================

def train_and_save_model(
    dataset_path: str = DEFAULT_DATASET_PATH,
    model_path: str = DEFAULT_MODEL_PATH,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tf.keras.Model:
    """
    Train neural network model from dataset and save trained model.
    
    Process:
    1. Load and prepare data from CSV
    2. Create and compile model
    3. Train with training/validation split
    4. Save trained model to disk
    
    Args:
        dataset_path: Path to CSV file with training data
        model_path: Path where to save the trained model
        epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        Trained Keras model
    """
    print(f"Loading data from {dataset_path}...")
    X_train, X_test, y_train, y_test = load_and_prepare_data(dataset_path)
    
    print(f"Training set size: {len(X_train)} samples")
    print(f"Test set size: {len(X_test)} samples")
    
    print("\nCreating model...")
    model = create_mlp_model()
    compile_model(model)
    
    print("\nModel architecture:")
    model.summary()
    
    print(f"\nTraining for {epochs} epochs...")
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=1,
    )
    
    # Display final metrics
    final_loss = history.history['loss'][-1]
    final_accuracy = history.history['accuracy'][-1]
    final_val_loss = history.history['val_loss'][-1]
    final_val_accuracy = history.history['val_accuracy'][-1]
    
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"{'='*60}")
    print(f"Final training loss: {final_loss:.4f}")
    print(f"Final training accuracy: {final_accuracy:.4f}")
    print(f"Final validation loss: {final_val_loss:.4f}")
    print(f"Final validation accuracy: {final_val_accuracy:.4f}")
    print(f"{'='*60}")
    
    print(f"\nSaving model to {model_path}...")
    model.save(model_path)
    print("Model saved successfully!")
    
    return model


# ============================= MAIN ENTRY POINT ===============================

if __name__ == "__main__":
    trained_model = train_and_save_model()
