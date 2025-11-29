"""
Obstacle Game with Behavior Cloning

A game where a ship navigates vertically to avoid obstacles moving horizontally.
Uses behavior cloning to train a neural network to play by learning from human demonstrations.
"""

import os
import random
import csv
import sys
from typing import List, Dict, Optional, Tuple

import numpy as np
import tensorflow as tf
import pygame

# ============================= GAME CONFIGURATION =============================

# Grid dimensions
GRID_ROWS = 3
GRID_COLUMNS = 4

# Display settings
CELL_SIZE = 80
SCORE_PANEL_HEIGHT = 80
WINDOW_WIDTH = GRID_COLUMNS * CELL_SIZE
WINDOW_HEIGHT = GRID_ROWS * CELL_SIZE + SCORE_PANEL_HEIGHT

# Normalization constant for neural network input
NORMALIZATION_MAX = 10.0

# File paths
DATASET_PATH = "behavior_cloning/dataset_actions.csv"
MODEL_PATH = "behavior_cloning/model.h5"
SHIP_IMAGE_PATH = "behavior_cloning/nave.png"

# Colors
COLOR_SKY_BLUE = (135, 206, 250)
COLOR_GRID = (100, 150, 200)
COLOR_SHIP = (50, 200, 255)
COLOR_OBSTACLE = (255, 50, 80)
COLOR_TEXT = (20, 20, 20)
COLOR_WHITE = (255, 255, 255)
COLOR_LIGHT_GRAY = (200, 200, 200)
COLOR_BLACK = (0, 0, 0)

# Actions
ACTION_MOVE_UP = -1
ACTION_STAY = 0
ACTION_MOVE_DOWN = 1
ALL_ACTIONS = [ACTION_MOVE_UP, ACTION_STAY, ACTION_MOVE_DOWN]

# Ship position (always in first column)
SHIP_COLUMN = 0

# Game settings
SHIP_INITIAL_ROW = 1
GAME_SPEED_FPS = 2000
NUM_OBSTACLES = 2

# Sprite settings
SPRITE_BORDER_SIZE = 10
SPRITE_BORDER_X = 5
SPRITE_BORDER_Y = 5

# Score display settings
SCORE_TEXT_PADDING = 10
SCORE_TEXT_OFFSET_X = 10
SCORE_TEXT_OFFSET_Y = 10
SCORE_BG_ALPHA = 100
SCORE_BG_PADDING = 5


# ============================= ACTION ENCODING ================================

def encode_action_to_one_hot(action: int) -> List[float]:
    """
    Convert action to one-hot encoding.
    
    Args:
        action: Action value (-1=up, 0=stay, 1=down)
        
    Returns:
        One-hot encoded action [up, stay, down]
    """
    if action == ACTION_MOVE_UP:
        return [1.0, 0.0, 0.0]
    if action == ACTION_STAY:
        return [0.0, 1.0, 0.0]
    return [0.0, 0.0, 1.0]


def decode_action_from_index(action_index: int) -> int:
    """
    Convert action index to action value.
    
    Args:
        action_index: Index from model prediction (0=up, 1=stay, 2=down)
        
    Returns:
        Action value (-1=up, 0=stay, 1=down)
    """
    action_mapping = {
        0: ACTION_MOVE_UP,
        1: ACTION_STAY,
        2: ACTION_MOVE_DOWN,
    }
    return action_mapping.get(action_index, ACTION_STAY)


# ============================= INPUT FEATURES =================================

def normalize_value(value: int) -> float:
    """
    Normalize a single value.
    
    Uses fixed normalization value for grid-size independence.
    Returns -1.0 for negative values (off-screen positions).
    
    Args:
        value: Raw value to normalize
        
    Returns:
        Normalized value
    """
    if value < 0:
        return -1.0
    return value / NORMALIZATION_MAX


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
    
    Features are normalized using fixed NORMALIZATION_MAX for grid-size independence.
    Negative values (off-screen) are represented as -1.0.
    
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
            normalize_value(obstacle1_column),
            normalize_value(obstacle1_row),
            normalize_value(obstacle2_column),
            normalize_value(obstacle2_row),
            normalize_value(ship_column),
            normalize_value(ship_row),
            normalize_value(timestep),
        ],
        dtype=np.float32,
    )


# ============================= ACTION SELECTION ===============================

def choose_random_action(ship_row: int) -> int:
    """
    Choose random valid action for the current ship position.
    
    Args:
        ship_row: Current row position of ship
        
    Returns:
        Random valid action (-1=up, 0=stay, 1=down)
    """
    valid_actions = [ACTION_STAY]
    
    if ship_row > 0:
        valid_actions.append(ACTION_MOVE_UP)
    
    if ship_row < GRID_ROWS - 1:
        valid_actions.append(ACTION_MOVE_DOWN)
    
    return random.choice(valid_actions)


def is_action_valid(action: int, ship_row: int) -> bool:
    """
    Check if action is valid for current ship position.
    
    Args:
        action: Action to validate
        ship_row: Current row position of ship
        
    Returns:
        True if action is valid, False otherwise
    """
    if action == ACTION_MOVE_UP and ship_row == 0:
        return False
    if action == ACTION_MOVE_DOWN and ship_row == GRID_ROWS - 1:
        return False
    return True


def choose_action_from_model(
    model: tf.keras.Model,
    obstacle1_column: int,
    obstacle1_row: int,
    obstacle2_column: int,
    obstacle2_row: int,
    ship_column: int,
    ship_row: int,
    timestep: int,
) -> int:
    """
    Use trained model to predict the correct action for current state.
    
    Args:
        model: Trained Keras model
        obstacle1_column: Column position of first obstacle
        obstacle1_row: Row position of first obstacle
        obstacle2_column: Column position of second obstacle
        obstacle2_row: Row position of second obstacle
        ship_column: Column position of ship
        ship_row: Row position of ship
        timestep: Current timestep in the episode
        
    Returns:
        Predicted action (-1=up, 0=stay, 1=down)
    """
    # Build input vector
    input_vector = build_input_vector(
        obstacle1_column,
        obstacle1_row,
        obstacle2_column,
        obstacle2_row,
        ship_column,
        ship_row,
        timestep,
    )
    
    # Get model prediction: [prob_up, prob_stay, prob_down]
    prediction = model.predict(input_vector[None, :], verbose=0)[0]
    
    # Get action with highest probability
    action_index = int(np.argmax(prediction))
    action = decode_action_from_index(action_index)
    
    # Validate and correct action if necessary
    if not is_action_valid(action, ship_row):
        action = ACTION_STAY
    
    return action


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
                "movimento_correto",
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


def create_cloud_sprite(size: int) -> pygame.Surface:
    """
    Create cloud sprite using overlapping circles.
    
    Args:
        size: Desired sprite size
        
    Returns:
        Pygame surface with cloud sprite
    """
    sprite = pygame.Surface((size, size), pygame.SRCALPHA)
    
    # Main cloud circles
    pygame.draw.circle(sprite, COLOR_WHITE, (size // 3, size // 2), size // 3)
    pygame.draw.circle(sprite, COLOR_WHITE, (size // 2, size // 3), size // 3)
    pygame.draw.circle(sprite, COLOR_WHITE, (2 * size // 3, size // 2), size // 3)
    pygame.draw.circle(sprite, COLOR_WHITE, (size // 2, 2 * size // 3), size // 4)
    
    # Subtle shadow effect
    pygame.draw.circle(
        sprite,
        COLOR_LIGHT_GRAY,
        (size // 3, size // 2 + 2),
        size // 3 - 2
    )
    pygame.draw.circle(
        sprite,
        COLOR_LIGHT_GRAY,
        (size // 2, size // 3 + 2),
        size // 3 - 2
    )
    pygame.draw.circle(
        sprite,
        COLOR_LIGHT_GRAY,
        (2 * size // 3, size // 2 + 2),
        size // 3 - 2
    )
    
    return sprite


# ============================= PYGAME INITIALIZATION ==========================

def initialize_pygame() -> Tuple[
    pygame.Surface,
    pygame.font.Font,
    pygame.time.Clock,
    pygame.Surface,
    pygame.Surface,
]:
    """
    Initialize Pygame and create game resources.
    
    Returns:
        Tuple of (screen, font, clock, ship_sprite, cloud_sprite)
    """
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Obstacles Game - Deep Learning")
    font = pygame.font.SysFont("consolas", 24)
    clock = pygame.time.Clock()
    
    # Create sprites
    sprite_size = CELL_SIZE - SPRITE_BORDER_SIZE
    ship_sprite = load_ship_sprite(sprite_size)
    cloud_sprite = create_cloud_sprite(sprite_size)
    
    return screen, font, clock, ship_sprite, cloud_sprite


# ============================= RENDERING ======================================

def draw_game_board(
    screen: pygame.Surface,
    font: pygame.font.Font,
    score: int,
    ship_row: int,
    obstacle_columns: List[int],
    obstacle_rows: List[int],
    ship_sprite: pygame.Surface,
    cloud_sprite: pygame.Surface,
) -> None:
    """
    Render the game board with all game elements.
    
    Args:
        screen: Pygame screen surface
        font: Font for rendering text
        score: Current score to display
        ship_row: Row position of ship
        obstacle_columns: List of obstacle column positions
        obstacle_rows: List of obstacle row positions
        ship_sprite: Sprite for ship
        cloud_sprite: Sprite for obstacles (clouds)
    """
    # Clear screen with sky blue background
    screen.fill(COLOR_SKY_BLUE)
    
    # Optional: Draw grid lines (commented out for cleaner look)
    # for row in range(GRID_ROWS):
    #     for col in range(GRID_COLUMNS):
    #         rect = pygame.Rect(
    #             col * CELL_SIZE,
    #             row * CELL_SIZE + SCORE_PANEL_HEIGHT,
    #             CELL_SIZE,
    #             CELL_SIZE
    #         )
    #         pygame.draw.rect(screen, COLOR_GRID, rect, 1)
    
    # Draw ship (rocket)
    ship_x = SHIP_COLUMN * CELL_SIZE + SPRITE_BORDER_X
    ship_y = ship_row * CELL_SIZE + SCORE_PANEL_HEIGHT + SPRITE_BORDER_Y
    screen.blit(ship_sprite, (ship_x, ship_y))
    
    # Draw obstacles (clouds)
    for obstacle_row, obstacle_column in zip(obstacle_rows, obstacle_columns):
        if 0 <= obstacle_column < GRID_COLUMNS:
            cloud_x = obstacle_column * CELL_SIZE + SPRITE_BORDER_X
            cloud_y = obstacle_row * CELL_SIZE + SCORE_PANEL_HEIGHT + SPRITE_BORDER_Y
            screen.blit(cloud_sprite, (cloud_x, cloud_y))
    
    # Draw score with semi-transparent background
    score_text = font.render(f"Score: {score}", True, COLOR_TEXT)
    
    # Create background for better text visibility
    text_bg = pygame.Surface((
        score_text.get_width() + SCORE_TEXT_PADDING,
        score_text.get_height() + 4
    ))
    text_bg.set_alpha(SCORE_BG_ALPHA)
    text_bg.fill(COLOR_BLACK)
    
    screen.blit(text_bg, (SCORE_BG_PADDING, SCORE_BG_PADDING))
    screen.blit(score_text, (SCORE_TEXT_OFFSET_X, SCORE_TEXT_OFFSET_Y))
    
    pygame.display.flip()


# ============================= GAME STATE =====================================

def is_collision(
    obstacle_columns: List[int],
    obstacle_rows: List[int],
    ship_column: int,
    ship_row: int,
) -> bool:
    """
    Check if any obstacle collides with the ship.
    
    Args:
        obstacle_columns: List of obstacle column positions
        obstacle_rows: List of obstacle row positions
        ship_column: Column position of ship
        ship_row: Row position of ship
        
    Returns:
        True if collision detected, False otherwise
    """
    for obstacle_row, obstacle_column in zip(obstacle_rows, obstacle_columns):
        if obstacle_column == ship_column and obstacle_row == ship_row:
            return True
    return False


def initialize_episode() -> Tuple[List[int], List[int], int]:
    """
    Initialize a new episode with random obstacle positions.
    
    Returns:
        Tuple of (obstacle_rows, obstacle_columns, timestep)
    """
    # Randomly select rows for obstacles (no duplicates)
    obstacle_rows = random.sample(range(GRID_ROWS), NUM_OBSTACLES)
    
    # Start obstacles at rightmost column
    obstacle_columns = [GRID_COLUMNS - 1] * NUM_OBSTACLES
    
    timestep = 0
    
    return obstacle_rows, obstacle_columns, timestep


def convert_episode_records_to_rows(
    episode_records: List[Dict],
) -> List[List]:
    """
    Convert episode records to CSV-compatible rows.
    
    Args:
        episode_records: List of recorded game state dictionaries
        
    Returns:
        List of rows ready for CSV writing
    """
    rows = []
    for record in episode_records:
        row = [
            record['obstacle1_column'],
            record['obstacle1_row'],
            record['obstacle2_column'],
            record['obstacle2_row'],
            record['ship_column'],
            record['ship_row'],
            record['timestep'],
            record['action'],
        ]
        rows.append(row)
    return rows


# ============================= GAME LOOP ======================================

def run_game(
    record_data: bool = True,
    use_ai: bool = False,
    model: Optional[tf.keras.Model] = None,
    max_episodes: Optional[int] = None,
    dataset_path: str = DATASET_PATH,
    use_random_actions: bool = False,
) -> None:
    """
    Main game loop for obstacle avoidance game.
    
    Can operate in multiple modes:
    - Manual play: User controls ship with arrow keys
    - AI play: Trained model controls ship
    - Random play: Random actions (useful for generating diverse training data)
    
    Args:
        record_data: If True, record training data to CSV
        use_ai: If True, use AI to control ship
        model: Trained Keras model (required if use_ai=True and not using random)
        max_episodes: Maximum number of episodes to play (None = unlimited)
        dataset_path: Path to save training data
        use_random_actions: If True, use random actions instead of model
    """
    # Initialize Pygame
    screen, font, clock, ship_sprite, cloud_sprite = initialize_pygame()
    
    # Game state
    score = 0
    episodes_played = 0
    ship_row = SHIP_INITIAL_ROW
    is_running = True
    
    # Main game loop
    while is_running:
        # Initialize new episode
        obstacle_rows, obstacle_columns, timestep = initialize_episode()
        
        episode_records = []
        has_collided = False
        episode_finished = False
        
        # Episode loop
        while not episode_finished and is_running:
            current_action = ACTION_STAY
            
            # =============== HANDLE INPUT EVENTS ===============
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    is_running = False
                    episode_finished = True
                elif event.type == pygame.KEYDOWN and not use_ai:
                    # Manual control: arrow key press
                    if event.key == pygame.K_UP and ship_row > 0:
                        current_action = ACTION_MOVE_UP
                    elif event.key == pygame.K_DOWN and ship_row < GRID_ROWS - 1:
                        current_action = ACTION_MOVE_DOWN
            
            if not is_running:
                break
            
            # Manual control: continuous movement while holding key
            if not use_ai and current_action == ACTION_STAY:
                pressed_keys = pygame.key.get_pressed()
                if pressed_keys[pygame.K_UP] and ship_row > 0:
                    current_action = ACTION_MOVE_UP
                elif pressed_keys[pygame.K_DOWN] and ship_row < GRID_ROWS - 1:
                    current_action = ACTION_MOVE_DOWN
            
            # =============== AI DECISION MAKING ===============
            if use_ai:
                if use_random_actions:
                    # Generate random actions for diverse training data
                    current_action = choose_random_action(ship_row)
                elif model is not None:
                    # Use trained model to decide action
                    current_action = choose_action_from_model(
                        model=model,
                        obstacle1_column=obstacle_columns[0],
                        obstacle1_row=obstacle_rows[0],
                        obstacle2_column=obstacle_columns[1],
                        obstacle2_row=obstacle_rows[1],
                        ship_column=SHIP_COLUMN,
                        ship_row=ship_row,
                        timestep=timestep,
                    )
                else:
                    # No model available, stay still
                    current_action = ACTION_STAY
            
            # =============== RECORD GAME STATE ===============
            if record_data:
                # Record state BEFORE applying action (for training)
                episode_records.append({
                    'obstacle1_column': obstacle_columns[0],
                    'obstacle1_row': obstacle_rows[0],
                    'obstacle2_column': obstacle_columns[1],
                    'obstacle2_row': obstacle_rows[1],
                    'ship_column': SHIP_COLUMN,
                    'ship_row': ship_row,
                    'timestep': timestep,
                    'action': current_action,
                })
            
            # =============== APPLY SHIP ACTION ===============
            if current_action == ACTION_MOVE_UP and ship_row > 0:
                ship_row -= 1
            elif current_action == ACTION_MOVE_DOWN and ship_row < GRID_ROWS - 1:
                ship_row += 1
            
            # =============== MOVE OBSTACLES ===============
            obstacle_columns = [col - 1 for col in obstacle_columns]
            
            # =============== CHECK COLLISION ===============
            if is_collision(obstacle_columns, obstacle_rows, SHIP_COLUMN, ship_row):
                has_collided = True
                episode_finished = True
            
            # Episode ends when any obstacle reaches the ship's column
            if not episode_finished and any(col == 0 for col in obstacle_columns):
                episode_finished = True
            
            # =============== RENDER GAME BOARD ===============
            draw_game_board(
                screen,
                font,
                score,
                ship_row,
                obstacle_columns,
                obstacle_rows,
                ship_sprite,
                cloud_sprite,
            )
            
            clock.tick(GAME_SPEED_FPS)
            timestep += 1
            
            # Safety: end episode if timestep exceeds grid width
            if timestep >= GRID_COLUMNS and not episode_finished:
                episode_finished = True
        
        if not is_running:
            break
        
        # =============== UPDATE SCORE ===============
        if has_collided:
            score = 0
        else:
            score += 1
        
        # =============== SAVE EPISODE DATA ===============
        # Only save successful episodes (no collision) for behavior cloning
        if record_data and episode_records and not has_collided:
            rows = convert_episode_records_to_rows(episode_records)
            append_to_dataset(dataset_path, rows)
        
        episodes_played += 1
        if max_episodes is not None and episodes_played >= max_episodes:
            is_running = False
    
    pygame.quit()
    sys.exit()


# ============================= MAIN ENTRY POINT ===============================

def main() -> None:
    """Main entry point with user mode selection."""
    print("=" * 60)
    print("Obstacle Game - Behavior Cloning")
    print("=" * 60)
    print("\nSelect mode:")
    print("1) Manual play (generate training data)")
    print("2) AI play (use trained model)")
    print("3) Random play (generate diverse training data)")
    print()
    
    mode = input("Enter mode (1-3): ").strip()
    
    if mode == "1":
        print("\nStarting manual play mode...")
        print("Use UP/DOWN arrow keys to control the ship")
        print("Avoid the clouds!")
        run_game(record_data=True, use_ai=False, max_episodes=None)
    
    elif mode == "2":
        if not os.path.exists(MODEL_PATH):
            print(f"\nError: Model not found at {MODEL_PATH}")
            print("Please train the model first using obgame_train.py")
            sys.exit(1)
        
        print(f"\nLoading model from {MODEL_PATH}...")
        loaded_model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully!")
        print("\nStarting AI play mode...")
        run_game(record_data=True, use_ai=True, model=loaded_model, max_episodes=None)
    
    elif mode == "3":
        print("\nStarting random play mode...")
        print("AI will play randomly to generate diverse training data")
        run_game(
            record_data=True,
            use_ai=True,
            use_random_actions=True,
            max_episodes=None
        )
    
    else:
        print("\nInvalid mode selected. Exiting.")
        sys.exit(1)


if __name__ == "__main__":
    main()
