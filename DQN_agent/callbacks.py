import os
import pickle
import random
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from collections import deque

import matplotlib.pyplot as plt


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

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



# Neural network model with 3 CNN layers followed by 3 fully connected layers
class DQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        
        conv_output_size = self._get_conv_output(state_size)
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_size)
    
    def _get_conv_output(self, shape):
        with torch.no_grad():
            x = torch.zeros(1, 1, shape, shape)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            return int(np.prod(x.size()))

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# DQN Agent
class DQNAgent:
    def __init__(self):
        #Hyperparameters
        self.gamma = 0.99        # Discount factor
        self.epsilon = 0.20       # Exploration rate
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.995 # Exploration rate decay
        self.learning_rate = 0.001
        self.batch_size = 64
        self.training_start = 1000
        self.memory_size = 1000000
        self.state_size = 17 #
        self.action_size = 6
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.step = 0

        # Experience Replay memory
        self.memory = deque(maxlen= self.memory_size)

        self.model = DQNetwork(self.state_size, self.action_size).to(self.device)
        self.target_model = DQNetwork(self.state_size, self.action_size).to(self.device)
        self.update_target_model()
        self.epsilon = self.epsilon
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.SmoothL1Loss() #Huber Loss
        self.train = False

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def forward(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
        q_values = self.model(state)
        return ACTIONS[torch.argmax(q_values).item()], torch.max(q_values).item()

    def replay(self, logger):
        logger.info("We are in replay")
        if len(self.memory) < self.training_start:
            logger.info("Model tried to train unsuccessfully.")
            return
        
        minibatch = random.sample(self.memory, self.batch_size)

        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
            if not done:
                next_state = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0).to(self.device)
            reward = torch.FloatTensor([reward]).to(self.device)
            action = action_to_int(action)
            done = torch.tensor(done).to(self.device)

            target = self.model(state)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model(next_state)
                target[0][action] = reward + self.gamma * torch.max(t).item()

            output = self.model(state)
            loss = self.criterion(output, target.detach())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        logger.info("Model has trained.")

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay



    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)


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

    if not os.path.isfile("Coin_heaven_first_try.pt"): # If there is a file to load from do it 
        self.logger.info("Setting up model from scratch.")
        self.model = DQNAgent()
        
    else:
        self.logger.info("Loading model from saved state.")
        self.model = DQNAgent()
        with open("Coin_heaven_first_try.pt", "rb") as file:
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
            return ACTIONS[random.randrange(self.model.action_size -1)] ## To exclude randomly bombing himself
        
    else:
        frame = game_state_to_frame(game_state)

        # 2. multiply frame
        symmetric_frames = [trans(frame).copy() for trans in SYMMETRIES]
        q_values = []
        possible_actions = []
        for frame in symmetric_frames:
            move, q_value = self.model.forward(frame)
            possible_actions.append(move)
            q_values.append(q_value)

        best_index = np.argmax(q_values)
        pre_transform_move = possible_actions[best_index]
        inverse_transform = INVERSE_SYMMETRIES[best_index]

        return matrix_to_action(inverse_transform(action_to_matrix(pre_transform_move)))


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ...
    channels = []
    channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)

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


# Helper functions to transform Actions to strings, ints, or matrices. This can be cleaned up  
def int_to_action(i):
    return ACTIONS[i]

def action_to_int(action):
    if action == 'UP' : return 0
    if action == 'RIGHT' : return 1
    if action == 'DOWN' : return 2
    if action == 'LEFT' : return 3
    if action == 'WAIT' : return 4
    if action == 'BOMB' : return 5

# Define the mapping from actions to matrices
ACTION_TO_MATRIX = {
    'UP': np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]),
    'RIGHT': np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]]),
    'DOWN': np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]]),
    'LEFT': np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]]),
    'WAIT': np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
    'BOMB': np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]])
}

# Define the mapping from matrices to actions
MATRIX_TO_ACTION = {v.tobytes(): k for k, v in ACTION_TO_MATRIX.items()}

def action_to_matrix(action):
    """Maps an action to its corresponding matrix."""
    return ACTION_TO_MATRIX.get(action, None)

def matrix_to_action(matrix):
    """Maps a matrix to its corresponding action."""
    return MATRIX_TO_ACTION.get(matrix.tobytes(), None)