import os
import pickle
import random
import torch
import torch.nn as nn
import torch.optim as optim
import heapq
import numpy as np
from collections import deque

import matplotlib.pyplot as plt

from .models import DuelingDQNAgent as Agent



ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Ensure the directories exist
    if not os.path.exists("savestates"):
        os.makedirs("savestates")

    if not os.path.exists("scores"):
        os.makedirs("scores")

    # Relative paths starting from DQL_agent directory
    datafile = os.path.join("savestates", "DuelingDQN_chests_only_ab10000_ich5000_SB.pt")
    scorefile = os.path.join("scores", "Dueling_against_rule_based_5000.txt")  # File to store the scores

    # Initialize score file if it doesn't exist
    if not os.path.isfile(scorefile):
        with open(scorefile, "w") as f:
            pass

    # Add a new header to scorefile (happens after every training session)
    with open(scorefile, "a") as f:
        f.write("Score,Steps,Bombs_placed,Invalid_moves,Epsilon\n")

    if not os.path.isfile("Res_DuelDQN.pt"):  # If there is a file to load from, do it
        self.logger.info("Setting up model from scratch.")
        self.logger.info(os.listdir("savestates"))  # Check savestates directory
        self.model = Agent(self.logger)       
    else:
        self.logger.info("Loading model from saved state.")
        self.model = Agent(self.logger)
        with open("Res_DuelDQN.pt", "rb") as file:
            self.model.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    if np.random.rand() <= self.model.epsilon: ## very naive epsilon greedy approach
            self.logger.info("Model is Exploring.")
            # Update epsilon for exploration decay
            if self.model.epsilon > self.model.epsilon_min:
                self.model.epsilon *= self.model.epsilon_decay
            return ACTIONS[random.randrange(self.model.action_size)] 
        
    else:
        dist_to_coin, _, path_to_objective = dijkstra_shortest_path_to_targets(game_state, ['coins'])
        if not path_to_objective or dist_to_coin > 15:
            _, _, path_to_objective = dijkstra_shortest_path_to_targets(game_state, ['chests'])
        if not path_to_objective:
            _, _, path_to_objective = dijkstra_shortest_path_to_targets(game_state, ['others'])
            if len(path_to_objective) > 1:
                path_to_objective = path_to_objective[:-1]
            else:
                path_to_objective = []

        frames = create_multiple_local_frames(game_state, path_to_objective)
       
        
        # 2. multiply frame
        symmetric_frames = [[trans(frame).copy() for frame in frames] for trans in SYMMETRIES]
        q_values = []
        possible_actions = []
        for frames in symmetric_frames:
            move, q_value = self.model.forward(frames)
            possible_actions.append(move)
            q_values.append(q_value)

        best_index = np.argmax(q_values)
        pre_transform_move = possible_actions[best_index]
        inverse_transform = INVERSE_SYMMETRIES[best_index]
        move = matrix_to_action(inverse_transform(action_to_matrix(pre_transform_move)))
        self.logger.info(move)
        return move



def game_state_to_frame(game_state):
    # Extract game state information
    field = game_state['field']
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']
    coins = game_state['coins']
    self_agent = game_state['self']
    others = game_state['others']
    
    # Define the grayscale values
    GRAYSCALE_WALL = 50
    GRAYSCALE_CHEST = 100
    GRAYSCALE_FREE_TILE = 0
    GRAYSCALE_COIN = 150
    GRAYSCALE_ENEMY = 200
    GRAYSCALE_BOMB = 175
    GRAYSCALE_EXPLOSION = 225
    GRAYSCALE_SELF = 255
    GRAYSCALE_DANGER_ZONE = 125
    
    # Create an empty frame filled with the value for free tiles
    frame = np.full(field.shape, GRAYSCALE_FREE_TILE, dtype=np.uint8)
    
    # Place walls and chests on the frame
    frame[field == -1] = GRAYSCALE_WALL
    frame[field == 1] = GRAYSCALE_CHEST
    
    # Place bombs on the frame
    for (x, y), _ in bombs:
        frame[x, y] = GRAYSCALE_BOMB
        # Add danger zones around the bomb
        for i in range(1, 4):
            if x + i < frame.shape[0] and field[x + i, y] == 0:  # Down
                frame[x + i, y] = GRAYSCALE_DANGER_ZONE
            if x - i >= 0 and field[x - i, y] == 0:  # Up
                frame[x - i, y] = GRAYSCALE_DANGER_ZONE
            if y + i < frame.shape[1] and field[x, y + i] == 0:  # Right
                frame[x, y + i] = GRAYSCALE_DANGER_ZONE
            if y - i >= 0 and field[x, y - i] == 0:  # Left
                frame[x, y - i] = GRAYSCALE_DANGER_ZONE
    
    # Place explosion zones on the frame
    frame[explosion_map > 0] = GRAYSCALE_EXPLOSION
    
    # Place coins on the frame
    for x, y in coins:
        frame[x, y] = GRAYSCALE_COIN
    
    # Place enemies on the frame
    for _, _, _, (x, y) in others:
        frame[x, y] = GRAYSCALE_ENEMY
    
    # Place self-agent on the frame
    _, _, _, (x, y) = self_agent
    frame[x, y] = GRAYSCALE_SELF

    return frame

def create_local_frame(game_state, path=None, frame_size=9):
    # Define the grayscale values
    GRAYSCALE_FREE_TILE = 255
    GRAYSCALE_COIN = 225
    GRAYSCALE_CHEST = 200
    GRAYSCALE_PATH = 175 
    GRAYSCALE_DANGER_ZONE = 150
    GRAYSCALE_BOMB = 125
    GRAYSCALE_ENEMY = 100
    GRAYSCALE_EXPLOSION = 50
    GRAYSCALE_WALL = 0
    #print(path)

    # Unpack game state
    field = game_state['field']
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']
    coins = game_state['coins']
    self_x, self_y = game_state['self'][3]
    others = game_state['others']
    #print(self_x, self_y)
    
    # Calculate the bounds of the local frame
    half_size = frame_size // 2
    x_min = max(self_x - half_size, 0)
    x_max = min(self_x + half_size + 1, field.shape[1])
    y_min = max(self_y - half_size, 0)
    y_max = min(self_y + half_size + 1, field.shape[0])
    
    # Create local frame for the field
    local_frame = np.full((frame_size, frame_size), GRAYSCALE_WALL, dtype=np.uint8)
    
    # Map global coordinates to local frame coordinates
    local_x_range = slice(half_size - (self_x - x_min), half_size + (x_max - self_x))
    local_y_range = slice(half_size - (self_y - y_min), half_size + (y_max - self_y))
    
    # Process field tiles
    tile_mapping = {
        0: GRAYSCALE_FREE_TILE,
        1: GRAYSCALE_CHEST,
        -1: GRAYSCALE_WALL
    }
    local_field = np.zeros((frame_size, frame_size), dtype=np.uint8)
    converted_field =  np.transpose(np.vectorize(tile_mapping.get)(field[x_min:x_max, y_min:y_max]))
    local_field[local_y_range, local_x_range] = converted_field
    local_frame = np.maximum(local_frame, local_field)

    
    # Process bombs
    for (bomb_x, bomb_y), countdown in bombs:
        if x_min <= bomb_x < x_max and y_min <= bomb_y < y_max:
            local_bomb_x = bomb_x - self_x + half_size
            local_bomb_y = bomb_y - self_y + half_size
            # Check bounds before assigning
            if 0 <= local_bomb_x < frame_size and 0 <= local_bomb_y < frame_size:
                local_frame[local_bomb_y, local_bomb_x] = GRAYSCALE_BOMB
    
    # Process explosions
    local_explosion = np.zeros((frame_size, frame_size), dtype=np.uint8)
    local_explosion_slice = np.transpose(explosion_map[x_min:x_max, y_min:y_max])
    local_explosion[local_y_range, local_x_range] = np.clip(local_explosion_slice, 0, 1) * GRAYSCALE_EXPLOSION
    local_frame = np.maximum(local_frame, local_explosion)
    
    
    # Add danger zones (expanding the danger zones around bombs)
    # Calculate danger zones based on bomb positions
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for (bomb_x, bomb_y), countdown in bombs:
        if countdown > 0:  # only consider active bombs
            for dx, dy in directions:
                for i in range(1, 4):  # expand danger zone up to 3 tiles away
                    nx, ny = bomb_x + dx * i, bomb_y + dy * i
                    if x_min <= nx < x_max and y_min <= ny < y_max:
                        if field[nx, ny] == -1:  # Stop if we hit a wall
                            break
                        local_ny = ny - self_y + half_size
                        local_nx = nx - self_x + half_size
                        local_frame[local_ny, local_nx] = GRAYSCALE_DANGER_ZONE
    
    
    # Process coins
    for (coin_x, coin_y) in coins:
        if x_min <= coin_x < x_max and y_min <= coin_y < y_max:
            local_coin_x = coin_x - self_x + half_size
            local_coin_y = coin_y - self_y + half_size
            # Check bounds before assigning
            if 0 <= local_coin_x < frame_size and 0 <= local_coin_y < frame_size:
                local_frame[local_coin_y, local_coin_x] = GRAYSCALE_COIN
    
    # Highlight the path
    if path:
        for (x, y) in path:
            if x_min <= x < x_max and y_min <= y < y_max:
                local_path_x = x - self_x + half_size
                local_path_y = y - self_y + half_size
                # Check bounds before assigning
                if 0 <= local_path_x < frame_size and 0 <= local_path_y < frame_size:
                    local_frame[local_path_y, local_path_x] = GRAYSCALE_PATH

    # Place enemies on the frame
    for _, _, _, (x, y) in others:
        if x_min <= x < x_max and y_min <= y < y_max:
                local_path_x = x - self_x + half_size
                local_path_y = y - self_y + half_size
                # Check bounds before assigning
                if 0 <= local_path_x < frame_size and 0 <= local_path_y < frame_size:
                    local_frame[local_path_y, local_path_x] = GRAYSCALE_ENEMY
    
    return local_frame

def create_multiple_local_frames(game_state, path=None, frame_size=9):
    # Define the grayscale values
    GRAYSCALE_FREE_TILE = 100
    GRAYSCALE_CHEST = 50
    GRAYSCALE_WALL = 0
    GRAYSCALE_COIN = 255
    GRAYSCALE_PATH = 200

    GRAYSCALE_DANGER_ZONE = 150
    GRAYSCALE_BOMB = 125
    GRAYSCALE_EXPLOSION = 255
    GRAYSCALE_ENEMY = 50   

    # Unpack game state
    field = game_state['field']
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']
    coins = game_state['coins']
    self_x, self_y = game_state['self'][3]
    others = game_state['others']
    
    # Calculate the bounds of the local frame
    half_size = frame_size // 2
    x_min = max(self_x - half_size, 0)
    x_max = min(self_x + half_size + 1, field.shape[1])
    y_min = max(self_y - half_size, 0)
    y_max = min(self_y + half_size + 1, field.shape[0])
    
    # Create local frames
    field_frame = np.zeros((frame_size, frame_size), dtype=np.uint8)
    danger_bomb_frame = np.zeros((frame_size, frame_size), dtype=np.uint8)

    
    # Map global coordinates to local frame coordinates
    local_x_range = slice(half_size - (self_x - x_min), half_size + (x_max - self_x))
    local_y_range = slice(half_size - (self_y - y_min), half_size + (y_max - self_y))
    
    # Process field tiles
    tile_mapping = {
        0: GRAYSCALE_FREE_TILE,
        1: GRAYSCALE_CHEST,
        -1: GRAYSCALE_WALL
    }

    converted_field =  np.transpose(np.vectorize(tile_mapping.get)(field[x_min:x_max, y_min:y_max]))
    field_frame[local_y_range, local_x_range] = converted_field


    # Process coins
    for coin in coins:
        coin_x, coin_y = coin
        if x_min <= coin_x < x_max and y_min <= coin_y < y_max:
            local_coin_x = coin_x - self_x + half_size
            local_coin_y = coin_y - self_y + half_size
            field_frame[local_coin_y, local_coin_x] = GRAYSCALE_COIN
    
    
    # Process bombs and danger zones
    for (bomb_x, bomb_y), countdown in bombs:
        if x_min <= bomb_x < x_max and y_min <= bomb_y < y_max:
            local_bomb_x = bomb_x - self_x + half_size
            local_bomb_y = bomb_y - self_y + half_size
            danger_bomb_frame[local_bomb_y, local_bomb_x] = GRAYSCALE_BOMB
            
            # Add danger zones
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                for dist in range(1, 4):
                    danger_x = bomb_x + dx * dist
                    danger_y = bomb_y + dy * dist
                    if x_min <= danger_x < x_max and y_min <= danger_y < y_max and field[danger_y, danger_x] != -1:
                        local_danger_x = danger_x - self_x + half_size
                        local_danger_y = danger_y - self_y + half_size
                        danger_bomb_frame[local_danger_y, local_danger_x] = GRAYSCALE_DANGER_ZONE
                    else:
                        break
    
    # Process explosions
    local_explosion = np.zeros((frame_size, frame_size), dtype=np.uint8)
    local_explosion_slice = np.transpose(explosion_map[x_min:x_max, y_min:y_max])
    local_explosion[local_y_range, local_x_range] = local_explosion[local_y_range, local_x_range] = np.clip(local_explosion_slice, 0, 1) * GRAYSCALE_EXPLOSION

    danger_bomb_frame = np.maximum(local_explosion, danger_bomb_frame)

    # Process path
    if danger_bomb_frame[half_size,half_size] == 0:
        for px, py in path:
            if x_min <= px < x_max and y_min <= py < y_max:
                local_px = px - self_x + half_size
                local_py = py - self_y + half_size
                if field_frame[local_py, local_px] == GRAYSCALE_FREE_TILE:
                    field_frame[local_py, local_px] = GRAYSCALE_PATH
                else:
                    break
    else:
        _, _, path = dijkstra_next_free_tile(game_state)
        for px, py in path:
            if x_min <= px < x_max and y_min <= py < y_max:
                local_px = px - self_x + half_size
                local_py = py - self_y + half_size
                if field_frame[local_py, local_px] == GRAYSCALE_FREE_TILE:
                    field_frame[local_py, local_px] = GRAYSCALE_PATH
                else:
                    break

    # Place enemies on the danger frame
    for _, _, _, (x, y) in others:
        if x_min <= x < x_max and y_min <= y < y_max:
                local_path_x = x - self_x + half_size
                local_path_y = y - self_y + half_size
                # Check bounds before assigning
                if 0 <= local_path_x < frame_size and 0 <= local_path_y < frame_size:
                    danger_bomb_frame[local_path_y, local_path_x] = GRAYSCALE_ENEMY


    return np.array(field_frame), np.array(danger_bomb_frame)


# Define the mapping from actions to matrices
ACTION_TO_MATRIX = {
    'UP': np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]),
    'RIGHT': np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]]),
    'DOWN': np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]]),
    'LEFT': np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]]),
    'WAIT': np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
    'BOMB': np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]])
}

# Helper functions
# Define the mapping from matrices to actions
MATRIX_TO_ACTION = {v.tobytes(): k for k, v in ACTION_TO_MATRIX.items()}

def action_to_matrix(action):
    """Maps an action to its corresponding matrix."""
    return ACTION_TO_MATRIX.get(action, None)

def matrix_to_action(matrix):
    """Maps a matrix to its corresponding action."""
    return MATRIX_TO_ACTION.get(matrix.tobytes(), None)

SYMMETRIES = [
    lambda img: img,
    np.rot90,  # Rotate 90 degrees counterclockwise
    lambda img: np.rot90(img, 2),  # Rotate 180 degrees counterclockwise
    lambda img: np.rot90(img, 3),  # Rotate 270 degrees counterclockwise
    np.flipud,  # Flip upside down
    np.fliplr,  # Flip left to right
    np.transpose,  # Transpose
    lambda img: np.transpose(np.fliplr(np.flipud(img)))  # Mirror over y=-x
]

INVERSE_SYMMETRIES = [
    lambda img: img,  # Identity function (inverse of identity is identity)
    lambda img: np.rot90(img, 3),  # Inverse of 90 degrees counterclockwise
    lambda img: np.rot90(img, 2),  # Inverse of 180 degrees counterclockwise
    np.rot90,  # Inverse of 270 degrees counterclockwise
    np.flipud,  # Inverse of flip upside down
    np.fliplr,  # Inverse of flip left to right
    np.transpose,  # Inverse of transpose
    lambda img: np.transpose(np.fliplr(np.flipud(img)))  # Inverse of mirror over y=-x
]

def get_symmetric_images(image):
    # Original image
    original = image
    
    # Rotate 90 degrees to the left (counter-clockwise)
    rotate_90_left = np.rot90(image)
    
    # Rotate 180 degrees to the left (counter-clockwise)
    rotate_180_left = np.rot90(image, 2)
    
    # Rotate 270 degrees to the left (counter-clockwise)
    rotate_270_left = np.rot90(image, 3)
    
    # Mirror image at y=0 (flip vertically)
    mirror_y0 = np.flipud(image)
    
    # Mirror image at x=0 (flip horizontally)
    mirror_x0 = np.fliplr(image)
    
    # Mirror image at x=y (transpose)
    mirror_y_equals_x = np.transpose(image)
    
    # Mirror image at x=-y (flip both axes and transpose)
    mirror_y_equals_minus_x = np.transpose(np.fliplr(np.flipud(image)))
    
    # Return the list of images
    return [
        original,
        rotate_90_left,
        rotate_180_left,
        rotate_270_left,
        mirror_y0,
        mirror_x0,
        mirror_y_equals_x,
        mirror_y_equals_minus_x
    ]

def dijkstra_shortest_path_to_targets(game_state, targets=['coins', 'others', 'bombs']):
    # Extract game state information
    field = game_state['field']
    bombs = game_state['bombs']
    self_agent = game_state['self']
    coins = game_state['coins'] if 'coins' in targets else []
    others = [pos for _, _, _, pos in game_state['others']] if 'others' in targets else []
    chests = np.argwhere(field == 1) if 'chests' in targets else []
    bomb_positions = [pos for (pos, _) in bombs] if 'bombs' in targets else []
    # Convert chests to a list of tuples 
    chests = [(x, y) for  x, y in chests]

    # Define the target positions
    target_positions = coins + others + chests + bomb_positions

    # Initialize the Dijkstra's algorithm structures
    height, width = field.shape
    start = self_agent[-1]  # self_agent[-1] is the position (x, y)
    distances = np.full((height, width), np.inf)
    distances[start] = 0
    visited = np.zeros((height, width), dtype=bool)
    pq = [(0, start)]  # priority queue with (distance, position)
    previous = {start: None}  # To store the path

    # Define movement directions (up, down, left, right)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    if start in target_positions:
        return 0, start, []
    
    # Calculate danger zones based on bomb positions
    danger_zones = np.zeros((height, width), dtype=bool)
    for (bomb_x, bomb_y), countdown in bombs:
        if countdown > 0:  # only consider active bombs
            danger_zones[bomb_x, bomb_x] = True
            for dx, dy in directions:
                for i in range(1, 4):  # expand danger zone up to 3 tiles away
                    nx, ny = bomb_x + dx * i, bomb_y + dy * i
                    if 0 <= nx < height and 0 <= ny < width:
                        if field[nx, ny] == -1:  # Stop if we hit a wall
                            break
                        danger_zones[nx, ny] = True
    
    while pq:
        current_distance, current_position = heapq.heappop(pq)
        x, y = current_position
        
        if visited[x, y]:
            continue
        
        visited[x, y] = True
        
        # Explore neighbors
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < height and 0 <= ny < width and not visited[nx, ny]:
                # Check if we have reached a target
                if (nx, ny) in target_positions:
                    previous[(nx, ny)] = (x, y)
                    current_position = (nx, ny)
                    current_distance += 1
                    # Reconstruct the path
                    path = []
                    while current_position != start:
                        current_position = previous[current_position]
                        if current_position != start and current_position != (nx, ny):
                            path.append(current_position)
                    path.reverse()
                    return current_distance, (nx, ny), path
                
                # Make sure the position is passable 
                if field[nx, ny] == 0 and ((nx, ny) not in bomb_positions) and ((nx,ny) not in others) : # 0 means free space 
                    if ('coins' in targets or 'chests' in targets) and danger_zones[nx,ny]:
                        continue

                    new_distance = current_distance + 1
                    if new_distance < distances[nx, ny]:
                        distances[nx, ny] = new_distance
                        previous[(nx, ny)] = (x, y)
                        heapq.heappush(pq, (new_distance, (nx, ny)))

    # If no path is found to any target
    return np.inf, None, []

def dijkstra_next_free_tile(game_state):
    # Extract game state information
    field = game_state['field']
    bombs = game_state['bombs']
    self_agent = game_state['self']
    others = [pos for _, _, _, pos in game_state['others']]
    bomb_positions = [pos for (pos, _) in bombs] 

    # Initialize the Dijkstra's algorithm structures
    height, width = field.shape
    start = self_agent[-1]  # self_agent[-1] is the position (x, y)
    distances = np.full((height, width), np.inf)
    distances[start] = 0
    visited = np.zeros((height, width), dtype=bool)
    pq = [(0, start)]  # priority queue with (distance, position)
    previous = {start: None}  # To store the path

    # Define movement directions (up, down, left, right)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Calculate danger zones based on bomb positions
    danger_zones = np.zeros((height, width), dtype=bool)
    for (bomb_x, bomb_y), countdown in bombs:
        if countdown > 0:  # only consider active bombs
            danger_zones[bomb_x, bomb_y] = True
            for dx, dy in directions:
                for i in range(1, 4):  # expand danger zone up to 3 tiles away
                    nx, ny = bomb_x + dx * i, bomb_y + dy * i
                    if 0 <= nx < height and 0 <= ny < width:
                        if field[nx, ny] == -1:  # Stop if we hit a wall
                            break
                        danger_zones[nx, ny] = True
    
    while pq:
        current_distance, current_position = heapq.heappop(pq)
        x, y = current_position
        
        if visited[x, y]:
            continue
        
        visited[x, y] = True

        # Check if we have reached a free tile outside the danger zone
        if not danger_zones[x, y]:  # 0 means free tile
            # Reconstruct the path
            path = []
            while current_position != start:
                current_position = previous[current_position]
                if current_position != start:
                    path.append(current_position)
            path.reverse()
            return current_distance, current_position, path
        
        # Explore neighbors
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            if 0 <= nx < height and 0 <= ny < width and not visited[nx, ny]:
                # Make sure the position is passable (not a wall)               
                if field[nx, ny] == 0 and ((nx, ny) not in bomb_positions) and ((nx,ny) not in others):  # 0 means free space
                    new_distance = current_distance + 1
                    if new_distance < distances[nx, ny]:
                        distances[nx, ny] = new_distance
                        previous[(nx, ny)] = (x, y)
                        heapq.heappush(pq, (new_distance, (nx, ny)))

    # If no free tile is found outside the danger zone
    return np.inf, None, []