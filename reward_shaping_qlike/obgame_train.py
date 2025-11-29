"""
Q-Value Training Script for Obstacle Game

Trains a neural network to predict Q-values for different actions
using reward shaping approach (distance-based rewards).
"""

import os
import datetime
from typing import Tuple

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# ============================= CONFIGURATION ==================================

# Normalization constant (must match game configuration)
NORMALIZATION_MAX = 10.0

# File paths
DATASET_PATH = "reward_shaping_qlike/dataset_qvalues.csv"
MODEL_PATH = "reward_shaping_qlike/model_qvalues.h5"
LOG_DIR_BASE = "reward_shaping_qlike/logs"

# Model architecture parameters
INPUT_FEATURES = 7  # obstaculo1_x, obstaculo1_y, obstaculo2_x, obstaculo2_y, nave_x, nave_y, tempo
HIDDEN_LAYER_SIZES = [32, 16, 8]
NUM_Q_VALUES = 3  # Q_up, Q_stay, Q_down

# Training parameters
DEFAULT_EPOCHS = 500
DEFAULT_BATCH_SIZE = 32
TEST_SIZE_RATIO = 0.1
RANDOM_SEED = 42
LEARNING_RATE = 1e-3


# ============================= DATA LOADING ===================================

def normalize_features(dataframe: pd.DataFrame) -> np.ndarray:
    """
    Extract and normalize feature columns from dataframe.
    
    Only normalizes values >= 0, keeps -1 values intact (used for off-screen positions).
    
    Args:
        dataframe: DataFrame containing feature columns
        
    Returns:
        Normalized feature array
    """
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
    
    # Normalize only non-negative values
    positive_mask = features >= 0
    normalized_features = features.copy()
    normalized_features[positive_mask] = features[positive_mask] / NORMALIZATION_MAX
    
    return normalized_features


def load_and_prepare_data(dataset_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset from CSV and prepare features and Q-value targets.
    
    Args:
        dataset_path: Path to CSV file containing training data
        
    Returns:
        Tuple of (features, q_values)
    """
    dataframe = pd.read_csv(dataset_path)
    
    # Extract and normalize features
    features = normalize_features(dataframe)
    
    # Extract Q-value targets
    q_value_columns = ["Q_up", "Q_stay", "Q_down"]
    q_values = dataframe[q_value_columns].values.astype(np.float32)
    
    return features, q_values


# ============================= MODEL CREATION =================================

def create_q_network() -> tf.keras.Model:
    """
    Create neural network model for Q-value prediction.
    
    Architecture:
    - Input layer: 7 features (game state)
    - Hidden layers: 32 -> 16 -> 8 neurons with ReLU activation
    - Output layer: 3 Q-values (one for each action) with linear activation
    
    Returns:
        Compiled Keras Sequential model
    """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(INPUT_FEATURES,)),
            tf.keras.layers.Dense(HIDDEN_LAYER_SIZES[0], activation="relu"),
            tf.keras.layers.Dense(HIDDEN_LAYER_SIZES[1], activation="relu"),
            tf.keras.layers.Dense(HIDDEN_LAYER_SIZES[2], activation="relu"),
            tf.keras.layers.Dense(NUM_Q_VALUES, activation="linear"),
        ]
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=["mae"],  # Mean Absolute Error for monitoring
    )
    
    return model


# ============================= TRAINING =======================================

def train_and_save_model(
    dataset_path: str = DATASET_PATH,
    model_path: str = MODEL_PATH,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tf.keras.Model:
    """
    Train Q-network from dataset and save trained model.
    
    Process:
    1. Load and prepare data from CSV
    2. Split into training and test sets
    3. Create and compile model
    4. Train with TensorBoard logging
    5. Save trained model to disk
    
    Args:
        dataset_path: Path to CSV file with training data
        model_path: Path where to save the trained model
        epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        Trained Keras model
    """
    # Ensure log directory exists
    os.makedirs(LOG_DIR_BASE, exist_ok=True)
    
    print(f"Loading data from {dataset_path}...")
    features, q_values = load_and_prepare_data(dataset_path)
    
    print(f"Dataset size: {len(features)} samples")
    print(f"Feature shape: {features.shape}")
    print(f"Q-value shape: {q_values.shape}")
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        q_values,
        test_size=TEST_SIZE_RATIO,
        random_state=RANDOM_SEED,
    )
    
    print(f"\nTraining set size: {len(X_train)} samples")
    print(f"Test set size: {len(X_test)} samples")
    
    # Create model
    print("\nCreating Q-network model...")
    model = create_q_network()
    
    print("\nModel architecture:")
    model.summary()
    
    # Setup TensorBoard callback
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(LOG_DIR_BASE, run_id)
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,      # Log weight histograms every epoch
        write_graph=True,      # Log model graph
        write_images=False,
    )
    
    # Train model
    print(f"\nTraining for {epochs} epochs...")
    print("=" * 60)
    
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=[tensorboard_callback],
    )
    
    # Display final metrics
    final_loss = history.history['loss'][-1]
    final_mae = history.history['mae'][-1]
    final_val_loss = history.history['val_loss'][-1]
    final_val_mae = history.history['val_mae'][-1]
    
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"{'='*60}")
    print(f"Final training loss (MSE): {final_loss:.4f}")
    print(f"Final training MAE: {final_mae:.4f}")
    print(f"Final validation loss (MSE): {final_val_loss:.4f}")
    print(f"Final validation MAE: {final_val_mae:.4f}")
    print(f"{'='*60}")
    
    # Save model
    print(f"\nSaving model to {model_path}...")
    model.save(model_path)
    print("Model saved successfully!")
    
    # Display TensorBoard instructions
    print(f"\nTensorBoard logs saved to: {log_dir}")
    print("\nTo visualize training with TensorBoard, run:")
    print(f"  tensorboard --logdir {LOG_DIR_BASE}")
    print("\nThen open your browser to: http://localhost:6006")
    
    return model


# ============================= MAIN ENTRY POINT ===============================

def main() -> None:
    """Main entry point for training script."""
    print("=" * 60)
    print("Q-Value Network Training - Reward Shaping Approach")
    print("=" * 60)
    print()
    
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}")
        print("Please generate training data first by running obgame.py")
        return
    
    trained_model = train_and_save_model()


if __name__ == "__main__":
    main()
