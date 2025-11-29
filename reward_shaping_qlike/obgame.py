"""
Obstacle Game with Q-Learning Reward Shaping

A simple game where a ship navigates vertically to avoid obstacles moving horizontally.
Uses Q-value approximation for AI training and decision making.
"""

import os
import random
import csv
import sys
from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf
import pygame

# ============================= GAME CONFIGURATION =============================

# Grid dimensions
GRID_ROWS = 3
GRID_COLUMNS = 7

# Display settings
CELL_SIZE = 80
WINDOW_WIDTH = GRID_COLUMNS * CELL_SIZE
WINDOW_HEIGHT = GRID_ROWS * CELL_SIZE + 80

# Normalization constant for neural network input
NORMALIZATION_MAX = 10.0

# File paths
DATASET_PATH = "reward_shaping_qlike/dataset_qvalues.csv"
MODEL_PATH = "reward_shaping_qlike/model_qvalues.h5"
SHIP_IMAGE_PATH = "reward_shaping_qlike/nave.png"
ENEMY_IMAGE_PATH = "reward_shaping_qlike/enemy.png"

# Game settings
SHIP_INITIAL_ROW = 1
GAME_SPEED_FPS = 5

# Reward values
COLLISION_PENALTY = -100.0

# Actions
ACTION_MOVE_UP = -1
ACTION_STAY = 0
ACTION_MOVE_DOWN = 1
ALL_ACTIONS = [ACTION_MOVE_UP, ACTION_STAY, ACTION_MOVE_DOWN]

# Ship position (always in first column)
SHIP_COLUMN = 0

# Colors
COLOR_SKY_BLUE = (135, 206, 250)
COLOR_BLACK = (0, 0, 0)
COLOR_SHIP = (50, 200, 255)
COLOR_OBSTACLE = (255, 50, 80)

# Display offsets
SCORE_POSITION = (10, 10)
BOARD_OFFSET_TOP = 80
CELL_PADDING = 10

# Sprite settings
SPRITE_BORDER_SIZE = 10
SPRITE_BORDER_X = 5
SPRITE_BORDER_Y = 5

# Additional colors for sprites
COLOR_WHITE = (255, 255, 255)
COLOR_LIGHT_GRAY = (200, 200, 200)


# ============================= INPUT FEATURES =================================

def build_input_vector(
    obstacle1_column: int,
    obstacle1_row: int,
    obstacle2_column: int,
    obstacle2_row: int,
    ship_column: int,
    ship_row: int,
    timestep: int,
) -> np.ndarray:
    """
    Build normalized input vector for the neural network.
    
    Args:
        obstacle1_column: Column position of first obstacle
        obstacle1_row: Row position of first obstacle
        obstacle2_column: Column position of second obstacle
        obstacle2_row: Row position of second obstacle
        ship_column: Column position of ship
        ship_row: Row position of ship
        timestep: Current timestep in the episode
        
    Returns:
        Normalized numpy array representing the game state
    """
    return np.array(
        [
            obstacle1_column / NORMALIZATION_MAX,
            obstacle1_row / NORMALIZATION_MAX,
            obstacle2_column / NORMALIZATION_MAX,
            obstacle2_row / NORMALIZATION_MAX,
            ship_column / NORMALIZATION_MAX,
            ship_row / NORMALIZATION_MAX,
            timestep / NORMALIZATION_MAX,
        ],
        dtype=np.float32,
    )


# ============================= REWARD CALCULATION =============================

def calculate_manhattan_distance(
    from_column: int, from_row: int, to_column: int, to_row: int
) -> int:
    """Calculate Manhattan distance between two grid positions."""
    return abs(from_column - to_column) + abs(from_row - to_row)


def is_collision(
    obstacle_column: int, obstacle_row: int, ship_column: int, ship_row: int
) -> bool:
    """Check if obstacle collides with ship."""
    return obstacle_column == ship_column and obstacle_row == ship_row


def compute_reward(
    obstacle1_column: int,
    obstacle1_row: int,
    obstacle2_column: int,
    obstacle2_row: int,
    ship_column: int,
    ship_row: int,
) -> float:
    """
    Calculate reward based on distance to nearest obstacle.
    
    Higher reward means safer position (further from obstacles).
    Collision results in large negative reward.
    
    Args:
        obstacle1_column: Column position of first obstacle
        obstacle1_row: Row position of first obstacle
        obstacle2_column: Column position of second obstacle
        obstacle2_row: Row position of second obstacle
        ship_column: Column position of ship
        ship_row: Row position of ship
        
    Returns:
        Reward value (negative for collision, positive distance otherwise)
    """
    # Check for collision with either obstacle
    if is_collision(obstacle1_column, obstacle1_row, ship_column, ship_row):
        return COLLISION_PENALTY
    
    if is_collision(obstacle2_column, obstacle2_row, ship_column, ship_row):
        return COLLISION_PENALTY

    # Calculate distance to each obstacle
    distance_to_obstacle1 = calculate_manhattan_distance(
        obstacle1_column, obstacle1_row, ship_column, ship_row
    )
    distance_to_obstacle2 = calculate_manhattan_distance(
        obstacle2_column, obstacle2_row, ship_column, ship_row
    )

    # Return distance to nearest obstacle as reward
    return float(min(distance_to_obstacle1, distance_to_obstacle2))


# ============================= Q-VALUE SIMULATION =============================

def simulate_next_state(
    obstacle_column: int, ship_row: int, action: int
) -> Tuple[int, int]:
    """
    Simulate the next state given current state and action.
    
    Args:
        obstacle_column: Current column of obstacle
        ship_row: Current row of ship
        action: Action to take (-1=up, 0=stay, 1=down)
        
    Returns:
        Tuple of (new_obstacle_column, new_ship_row)
    """
    new_ship_row = ship_row
    
    if action == ACTION_MOVE_UP and ship_row > 0:
        new_ship_row = ship_row - 1
    elif action == ACTION_MOVE_DOWN and ship_row < GRID_ROWS - 1:
        new_ship_row = ship_row + 1
    
    # Obstacles move one step to the left
    new_obstacle_column = obstacle_column - 1
    
    return new_obstacle_column, new_ship_row


def compute_q_values_for_all_actions(
    obstacle1_column: int,
    obstacle1_row: int,
    obstacle2_column: int,
    obstacle2_row: int,
    ship_column: int,
    ship_row: int,
    timestep: int,
) -> List[float]:
    """
    Compute Q-values for all possible actions in current state.
    
    Simulates taking each action (up, stay, down) and calculates
    the reward after:
    - Moving the ship according to action
    - Moving obstacles one step to the left
    
    Args:
        obstacle1_column: Column position of first obstacle
        obstacle1_row: Row position of first obstacle
        obstacle2_column: Column position of second obstacle
        obstacle2_row: Row position of second obstacle
        ship_column: Column position of ship
        ship_row: Row position of ship
        timestep: Current timestep in the episode
        
    Returns:
        List of Q-values [Q_up, Q_stay, Q_down]
    """
    q_values = []

    for action in ALL_ACTIONS:
        # Simulate ship movement
        new_obstacle1_column, new_ship_row = simulate_next_state(
            obstacle1_column, ship_row, action
        )
        new_obstacle2_column, _ = simulate_next_state(
            obstacle2_column, ship_row, action
        )

        # Calculate reward for this state-action pair
        reward = compute_reward(
            new_obstacle1_column,
            obstacle1_row,
            new_obstacle2_column,
            obstacle2_row,
            ship_column,
            new_ship_row,
        )
        q_values.append(reward)

    return q_values


# ============================= DATA PERSISTENCE ===============================

def append_to_dataset(file_path: str, data_rows: List[List]) -> None:
    """
    Append data rows to CSV dataset file.
    
    Creates file with header if it doesn't exist.
    
    Args:
        file_path: Path to CSV file
        data_rows: List of data rows to append
    """
    file_exists = os.path.exists(file_path)
    
    with open(file_path, "a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        
        if not file_exists:
            header = [
                "obstaculo1_x",
                "obstaculo1_y",
                "obstaculo2_x",
                "obstaculo2_y",
                "nave_x",
                "nave_y",
                "tempo",
                "Q_up",
                "Q_stay",
                "Q_down",
            ]
            writer.writerow(header)
        
        writer.writerows(data_rows)


# ============================= SPRITE CREATION ================================

def load_ship_sprite(size: int) -> pygame.Surface:
    """
    Load ship image from file and resize.
    
    Falls back to creating a simple sprite if image file doesn't exist.
    
    Args:
        size: Desired sprite size
        
    Returns:
        Pygame surface with ship sprite
    """
    try:
        ship_image = pygame.image.load(SHIP_IMAGE_PATH)
        ship_image = pygame.transform.scale(ship_image, (size, size))
        ship_image = ship_image.convert_alpha()
        return ship_image
    except pygame.error:
        # Fallback: create simple ship sprite
        sprite = pygame.Surface((size, size), pygame.SRCALPHA)
        
        # Draw triangle (rocket shape)
        points = [
            (size // 2, 0),              # Top point
            (size // 4, size),           # Bottom left
            (3 * size // 4, size),       # Bottom right
        ]
        pygame.draw.polygon(sprite, COLOR_SHIP, points)
        
        # Draw window/cockpit
        pygame.draw.circle(
            sprite,
            (200, 240, 255),
            (size // 2, size // 3),
            size // 6
        )
        
        return sprite


def load_enemy_sprite(size: int) -> pygame.Surface:
    """
    Load enemy image from file and resize.
    
    Falls back to creating a simple sprite if image file doesn't exist.
    
    Args:
        size: Desired sprite size
        
    Returns:
        Pygame surface with enemy sprite
    """
    try:
        enemy_image = pygame.image.load(ENEMY_IMAGE_PATH)
        enemy_image = pygame.transform.scale(enemy_image, (size, size))
        enemy_image = enemy_image.convert_alpha()
        return enemy_image
    except pygame.error:
        # Fallback: create simple obstacle sprite (red square with border)
        sprite = pygame.Surface((size, size), pygame.SRCALPHA)
        
        # Draw red rectangle as fallback
        pygame.draw.rect(sprite, COLOR_OBSTACLE, (0, 0, size, size))
        pygame.draw.rect(sprite, (200, 30, 60), (0, 0, size, size), 3)
        
        return sprite


# ============================= AI DECISION MAKING =============================

def choose_action_from_model(
    model: tf.keras.Model, state_vector: np.ndarray
) -> int:
    """
    Choose action based on Q-value predictions from neural network model.
    
    Args:
        model: Trained Keras model
        state_vector: Normalized state vector
        
    Returns:
        Action to take (-1=up, 0=stay, 1=down)
    """
    # Get Q-value predictions for all actions
    predicted_q_values = model.predict(state_vector[None, :], verbose=0)[0]
    
    # Choose action with highest Q-value
    best_action_index = int(np.argmax(predicted_q_values))

    # ---------------- IMPRESSÃO ESTILO "IA EM AÇÃO" ----------------
    os.system('cls' if os.name == 'nt' else 'clear')  # limpa terminal pra efeito bonito

    print("\n" + "="*55)
    print("🔵 IA THINKING...".center(55))
    print("="*55)

    print("\nSTATE VECTOR (normalized):")
    print(np.round(state_vector, 3))

    print("\nQ-VALUES predicted:")
    print(f"  UP   : {predicted_q_values[0]: .3f}")
    print(f"  STAY : {predicted_q_values[1]: .3f}")
    print(f"  DOWN : {predicted_q_values[2]: .3f}")

    action_name = ["UP", "STAY", "DOWN"][best_action_index]
    print(f"\n👉 ACTION CHOSEN → {action_name}")

    print("="*55 + "\n")
    # --------------------------------------------------------------
    
    return ALL_ACTIONS[best_action_index]


    # ============================= RENDERING ======================================

def draw_game_board(
    screen: pygame.Surface,
    font: pygame.font.Font,
    score: int,
    ship_row: int,
    obstacle1_column: int,
    obstacle1_row: int,
    obstacle2_column: int,
    obstacle2_row: int,
    ship_sprite: pygame.Surface,
    enemy_sprite: pygame.Surface,
) -> None:
    """
    Render the game board with all game elements.
    
    Args:
        screen: Pygame screen surface
        font: Font for rendering text
        score: Current score to display
        ship_row: Row position of ship
        obstacle1_column: Column position of first obstacle
        obstacle1_row: Row position of first obstacle
        obstacle2_column: Column position of second obstacle
        obstacle2_row: Row position of second obstacle
        ship_sprite: Sprite for ship
        enemy_sprite: Sprite for obstacles (enemies)
    """
    # Clear screen with sky blue background
    screen.fill(COLOR_SKY_BLUE)

    # Render score
    score_text = font.render(f"Score: {score}", True, COLOR_BLACK)
    screen.blit(score_text, SCORE_POSITION)

    # Draw ship (always in first column)
    ship_x = SHIP_COLUMN * CELL_SIZE + SPRITE_BORDER_X
    ship_y = ship_row * CELL_SIZE + BOARD_OFFSET_TOP + SPRITE_BORDER_Y
    screen.blit(ship_sprite, (ship_x, ship_y))

    # Draw first obstacle (if visible on screen)
    if 0 <= obstacle1_column < GRID_COLUMNS:
        enemy_x = obstacle1_column * CELL_SIZE + SPRITE_BORDER_X
        enemy_y = obstacle1_row * CELL_SIZE + BOARD_OFFSET_TOP + SPRITE_BORDER_Y
        screen.blit(enemy_sprite, (enemy_x, enemy_y))

    # Draw second obstacle (if visible on screen)
    if 0 <= obstacle2_column < GRID_COLUMNS:
        enemy_x = obstacle2_column * CELL_SIZE + SPRITE_BORDER_X
        enemy_y = obstacle2_row * CELL_SIZE + BOARD_OFFSET_TOP + SPRITE_BORDER_Y
        screen.blit(enemy_sprite, (enemy_x, enemy_y))

    pygame.display.flip()


# ============================= GAME LOOP ======================================

def run_game(
    record_data: bool = True,
    use_ai: bool = False,
    model: Optional[tf.keras.Model] = None,
) -> None:
    """
    Main game loop for obstacle avoidance game.
    
    Args:
        record_data: If True, record Q-values to dataset file
        use_ai: If True, use AI to control ship; otherwise manual control
        model: Trained Keras model for AI decision making (required if use_ai=True)
    """
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Obstacles Game - Q-like")
    font = pygame.font.SysFont("Arial", 24)
    clock = pygame.time.Clock()
    
    # Create sprites
    sprite_size = CELL_SIZE - SPRITE_BORDER_SIZE
    ship_sprite = load_ship_sprite(sprite_size)
    enemy_sprite = load_enemy_sprite(sprite_size)

    # Game state
    ship_row = SHIP_INITIAL_ROW
    score = 0
    is_running = True

    # Main game loop
    while is_running:
        # Start new episode/round
        timestep = 0

        # Initialize obstacles at random rows, starting from rightmost column
        obstacle1_row, obstacle2_row = random.sample(range(GRID_ROWS), 2)
        obstacle1_column = GRID_COLUMNS - 1
        obstacle2_column = GRID_COLUMNS - 1

        episode_data = []
        has_collided = False
        episode_finished = False

        # Episode loop
        while is_running and not episode_finished:
            current_action = ACTION_STAY

            # =============== HANDLE INPUT EVENTS ===============
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    is_running = False
                    episode_finished = True

                # Manual control via keyboard
                if not use_ai and event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP and ship_row > 0:
                        current_action = ACTION_MOVE_UP
                    elif event.key == pygame.K_DOWN and ship_row < GRID_ROWS - 1:
                        current_action = ACTION_MOVE_DOWN

            # Continuous movement while holding key (manual control only)
            if not use_ai and current_action == ACTION_STAY:
                pressed_keys = pygame.key.get_pressed()
                if pressed_keys[pygame.K_UP] and ship_row > 0:
                    current_action = ACTION_MOVE_UP
                elif pressed_keys[pygame.K_DOWN] and ship_row < GRID_ROWS - 1:
                    current_action = ACTION_MOVE_DOWN

            # =============== AI DECISION MAKING ===============
            if use_ai and model is not None:
                state_vector = build_input_vector(
                    obstacle1_column,
                    obstacle1_row,
                    obstacle2_column,
                    obstacle2_row,
                    SHIP_COLUMN,
                    ship_row,
                    timestep,
                )
                current_action = choose_action_from_model(model, state_vector)

            # =============== RECORD Q-VALUES ===============
            if record_data:
                q_up, q_stay, q_down = compute_q_values_for_all_actions(
                    obstacle1_column,
                    obstacle1_row,
                    obstacle2_column,
                    obstacle2_row,
                    SHIP_COLUMN,
                    ship_row,
                    timestep,
                )
                episode_data.append(
                    [
                        obstacle1_column,
                        obstacle1_row,
                        obstacle2_column,
                        obstacle2_row,
                        SHIP_COLUMN,
                        ship_row,
                        timestep,
                        q_up,
                        q_stay,
                        q_down,
                    ]
                )

            # =============== APPLY SHIP ACTION ===============
            if current_action == ACTION_MOVE_UP and ship_row > 0:
                ship_row -= 1
            elif current_action == ACTION_MOVE_DOWN and ship_row < GRID_ROWS - 1:
                ship_row += 1

            # =============== MOVE OBSTACLES ===============
            obstacle1_column -= 1
            obstacle2_column -= 1

            # =============== CHECK COLLISION ===============
            if is_collision(obstacle1_column, obstacle1_row, SHIP_COLUMN, ship_row):
                has_collided = True
                episode_finished = True
            elif is_collision(obstacle2_column, obstacle2_row, SHIP_COLUMN, ship_row):
                has_collided = True
                episode_finished = True
            # Episode ends when both obstacles have passed the ship
            elif obstacle1_column < 0 and obstacle2_column < 0:
                episode_finished = True

            # =============== RENDER GAME BOARD ===============
            draw_game_board(
                screen,
                font,
                score,
                ship_row,
                obstacle1_column,
                obstacle1_row,
                obstacle2_column,
                obstacle2_row,
                ship_sprite,
                enemy_sprite,
            )

            clock.tick(GAME_SPEED_FPS)
            timestep += 1

        # =============== UPDATE SCORE ===============
        if has_collided:
            score = 0
        else:
            score += 1

        # =============== SAVE EPISODE DATA ===============
        if record_data and episode_data:
            append_to_dataset(DATASET_PATH, episode_data)

    pygame.quit()
    sys.exit()


# ============================= MAIN ENTRY POINT ===============================

if __name__ == "__main__":
    # Load trained model if it exists
    trained_model = None
    if os.path.exists(MODEL_PATH):
        trained_model = tf.keras.models.load_model(MODEL_PATH)
        print(f"Loaded model from {MODEL_PATH}")
    else:
        print(f"No model found at {MODEL_PATH}")

    # Mode 1: Generate training data (manual play with Q-value recording)
    # run_game(record_data=True, use_ai=False, model=trained_model)

    # Mode 2: Test AI after training model
    run_game(record_data=True, use_ai=True, model=trained_model)
