# TensorFlow Learning Projects

A collection of machine learning projects demonstrating different approaches to neural network training using TensorFlow and Keras.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Projects](#projects)
  - [1. MNIST Digit Recognition](#1-mnist-digit-recognition)
  - [2. Obstacle Game - Behavior Cloning](#2-obstacle-game---behavior-cloning)
  - [3. Obstacle Game - Q-Learning Reward Shaping](#3-obstacle-game---q-learning-reward-shaping)
- [Models](#models)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This repository contains three machine learning projects that demonstrate different training approaches:

1. **Supervised Learning**: MNIST handwritten digit classification
2. **Behavior Cloning**: Learning from human demonstrations
3. **Reward Shaping**: Q-value approximation with distance-based rewards

All projects use TensorFlow/Keras for neural network implementation and Pygame for game visualization.

## 📁 Project Structure

```
tensorflow-learning/
├── mnist/                          # MNIST digit recognition
│   ├── mnist_train.py             # Training script
│   ├── mnist_use.py               # Inference script
│   └── mnist_test.png             # Test image
│
├── behavior_cloning/              # Behavior cloning obstacle game
│   ├── obgame.py                  # Game with data collection
│   ├── obgame_train.py            # Model training script
│   ├── dataset_actions.csv        # Collected training data
│   ├── model.h5                   # Trained model
│   ├── nave.png                   # Ship sprite
│   └── enemy.png                  # Enemy sprite
│
├── reward_shaping_qlike/          # Q-learning obstacle game
│   ├── obgame.py                  # Game with Q-value recording
│   ├── obgame_train.py            # Q-network training script
│   ├── dataset_qvalues.csv        # Q-value dataset
│   ├── model_qvalues.h5           # Trained Q-network
│   ├── logs/                      # TensorBoard logs
│   ├── nave.png                   # Ship sprite
│   └── enemy.png                  # Enemy sprite
│
├── models/                        # Saved models directory
│   ├── collision_model.keras
│   ├── model.keras
│   └── obgame_model.keras
│
├── venv/                          # Python virtual environment
└── README.md                      # This file
```

## 🔧 Requirements

- Python 3.11+
- TensorFlow 2.x
- Pygame
- NumPy
- Pandas
- scikit-learn
- Pillow

## 📦 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd tensorflow-learning
```

2. Create and activate virtual environment:
```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install tensorflow pygame numpy pandas scikit-learn pillow
```

## 🚀 Projects

### 1. MNIST Digit Recognition

A classic supervised learning project for recognizing handwritten digits (0-9).

#### Features
- Convolutional Neural Network (CNN) architecture
- 99%+ accuracy on test set
- Image preprocessing and normalization
- Model inference on custom images

#### Usage

**Training:**
```bash
python mnist/mnist_train.py
```

**Testing on custom image:**
```bash
python mnist/mnist_use.py
```

#### Model Architecture
- Conv2D (32 filters, 3x3) + ReLU
- MaxPooling2D (2x2)
- Conv2D (64 filters, 3x3) + ReLU
- MaxPooling2D (2x2)
- Flatten
- Dense (128) + ReLU + Dropout(0.5)
- Dense (10) + Softmax

---

### 2. Obstacle Game - Behavior Cloning

Learn to play an obstacle avoidance game by imitating human gameplay.

#### Concept
The AI learns by observing human players navigate a ship through obstacles. This is an example of **imitation learning** where the model learns the mapping from game states to actions.

#### Game Mechanics
- **Grid**: 3 rows × 5 columns
- **Objective**: Avoid enemies moving from right to left
- **Controls**: 
  - ↑ Arrow: Move up
  - ↓ Arrow: Move down
- **Scoring**: +1 for each round survived, reset to 0 on collision

#### Visual Features
- 🚀 Ship sprite (nave.png)
- 👾 Enemy sprites (enemy.png)
- 🌅 Sky blue background
- 📊 Score display

#### Usage

**1. Generate Training Data (Manual Play):**
```bash
python behavior_cloning/obgame.py
# Select mode: 1 (manual play)
# Play the game to collect training data
```

**2. Train the Model:**
```bash
python behavior_cloning/obgame_train.py
```

**3. Watch AI Play:**
```bash
python behavior_cloning/obgame.py
# Select mode: 2 (AI play)
```

**4. Generate Random Training Data:**
```bash
python behavior_cloning/obgame.py
# Select mode: 3 (random play)
```

#### Model Architecture
- Input: 7 features (obstacle positions, ship position, timestep)
- Hidden: 32 → 16 → 8 neurons (ReLU)
- Output: 3 actions (up, stay, down) with Softmax
- Loss: Categorical Cross-Entropy
- Optimizer: Adam

#### Training Data Format (CSV)
```
obstaculo1_x, obstaculo1_y, obstaculo2_x, obstaculo2_y, nave_x, nave_y, tempo, movimento_correto
```

---

### 3. Obstacle Game - Q-Learning Reward Shaping

An alternative approach using Q-value approximation with distance-based reward shaping.

#### Concept
Instead of learning from demonstrations, this approach learns Q-values (expected future rewards) for each action. The reward is based on distance to nearest obstacle, providing a continuous learning signal.

#### Key Differences from Behavior Cloning
- **Learning Signal**: Distance-based rewards vs. human actions
- **Training Data**: Q-values computed from game dynamics
- **Model Output**: Q-values for each action (regression)
- **Action Selection**: Choose action with highest Q-value

#### Reward Function
```python
if collision:
    reward = -100.0
else:
    reward = min(manhattan_distance_to_obstacle1, manhattan_distance_to_obstacle2)
```

#### AI Visualization
When AI is playing, the terminal displays:
```
=======================================================
🔵 IA THINKING...
=======================================================

STATE VECTOR (normalized):
[0.7  0.2  0.6  0.1  0.0  0.333  0.5]

Q-VALUES predicted:
  UP   :  3.245
  STAY :  4.891
  DOWN :  2.156

👉 ACTION CHOSEN → STAY
=======================================================
```

#### Usage

**1. Generate Training Data (Manual Play):**
```bash
python reward_shaping_qlike/obgame.py
# Comment line 540, uncomment line 537
# Play manually to generate Q-value dataset
```

**2. Train the Q-Network:**
```bash
python reward_shaping_qlike/obgame_train.py
```

**3. Watch AI Play with Q-Values:**
```bash
python reward_shaping_qlike/obgame.py
# AI will display Q-value predictions in terminal
```

**4. View Training with TensorBoard:**
```bash
tensorboard --logdir reward_shaping_qlike/logs
# Open browser to http://localhost:6006
```

#### Model Architecture
- Input: 7 features (obstacle positions, ship position, timestep)
- Hidden: 32 → 16 → 8 neurons (ReLU)
- Output: 3 Q-values (one per action) with Linear activation
- Loss: Mean Squared Error (MSE)
- Optimizer: Adam (lr=0.001)
- Epochs: 500

#### Training Data Format (CSV)
```
obstaculo1_x, obstaculo1_y, obstaculo2_x, obstaculo2_y, nave_x, nave_y, tempo, Q_up, Q_stay, Q_down
```

#### Q-Value Computation
For each state, the Q-values are computed by:
1. Simulating each possible action (up, stay, down)
2. Moving obstacles one step left
3. Computing reward based on resulting state
4. Recording [Q_up, Q_stay, Q_down] as training targets

---

## 📊 Models

### Saved Models Directory

The `models/` directory contains various trained models:

- **collision_model.keras**: Collision detection model
- **model.keras**: General purpose model
- **obgame_model.keras**: Obstacle game model

### Model Loading

```python
import tensorflow as tf

# Load a trained model
model = tf.keras.models.load_model('models/model.keras')

# Or from specific project
behavior_model = tf.keras.models.load_model('behavior_cloning/model.h5')
qlearning_model = tf.keras.models.load_model('reward_shaping_qlike/model_qvalues.h5')
```

## 🎓 Learning Approaches Comparison

| Aspect | Behavior Cloning | Q-Learning Reward Shaping |
|--------|-----------------|---------------------------|
| **Data Source** | Human demonstrations | Game dynamics simulation |
| **Learning Signal** | Correct actions | Distance-based rewards |
| **Model Output** | Action probabilities | Q-values |
| **Training Target** | One-hot actions | Computed Q-values |
| **Pros** | Quick to learn from expert | No expert needed |
| **Cons** | Requires good demonstrations | Requires reward engineering |

## 🔍 Key Features

### Clean Code Practices
- ✅ Descriptive variable names
- ✅ Type hints throughout
- ✅ Comprehensive docstrings
- ✅ Modular function design
- ✅ Section organization
- ✅ No linting errors

### Neural Network Features
- Normalized inputs for stable training
- Appropriate loss functions for each task
- Dropout for regularization (MNIST)
- TensorBoard integration (Q-Learning)
- Model checkpointing
- Training/validation splits

### Game Features
- Pygame visualization
- Real-time AI decision display
- Score tracking
- Sprite-based graphics
- Keyboard controls
- Data recording to CSV

## 🛠️ Configuration

### Adjustable Parameters

**Behavior Cloning:**
```python
GRID_ROWS = 3
GRID_COLUMNS = 5
DEFAULT_EPOCHS = 200
DEFAULT_BATCH_SIZE = 64
GAME_SPEED_FPS = 2000
```

**Q-Learning:**
```python
GRID_ROWS = 3
GRID_COLUMNS = 7
DEFAULT_EPOCHS = 500
DEFAULT_BATCH_SIZE = 32
GAME_SPEED_FPS = 5
COLLISION_PENALTY = -100.0
```

## 📈 Training Tips

1. **Behavior Cloning:**
   - Play manually to generate 5000+ training samples
   - Try to play optimally (avoid collisions)
   - Use mode 3 (random) to add diversity

2. **Q-Learning:**
   - Generate diverse state-action pairs
   - Monitor validation loss in TensorBoard
   - Adjust reward shaping for better learning

3. **General:**
   - Use GPU for faster training (automatic if available)
   - Monitor overfitting via validation metrics
   - Save models at regular intervals

## 🐛 Troubleshooting

**Column name errors:**
- The code supports both Portuguese and English column names for backward compatibility
- Existing datasets use Portuguese names: `obstaculo1_x`, `nave_x`, `tempo`

**Pygame display issues:**
- Make sure X server is available (Linux)
- Install pygame correctly: `pip install pygame`

**TensorFlow warnings:**
- CPU optimization warnings are informational only
- For GPU support, install `tensorflow-gpu`

## 📚 References

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Keras Guide](https://keras.io/guides/)
- [Pygame Documentation](https://www.pygame.org/docs/)
- [Behavior Cloning Paper](https://arxiv.org/abs/1606.01540)
- [Q-Learning Introduction](https://en.wikipedia.org/wiki/Q-learning)

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Add new game modes
- Implement new learning algorithms
- Improve model architectures
- Add visualization features
- Fix bugs or improve documentation

## 📝 License

This project is for educational purposes.

---

**Happy Learning! 🚀🤖**

