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

class PrioritizedReplayBuffer:
    def __init__(self, memory_size, batch_size, alpha=0.6, device ='cpu'):
        self.memory = []
        self.priorities = []
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.alpha = alpha  # Controls how much prioritization is used (α=0 is uniform)
        self.device = device

    def add(self, experience):
        # Add new experience with its corresponding priority
        max_priority = float(max(self.priorities, default=1.0))  # Set highest priority for new experience
        self.memory.append(experience)
        self.priorities.append(max_priority)

        if len(self.memory) > self.memory_size:
            # Remove oldest experience if buffer is full
            self.memory.pop(0)
            self.priorities.pop(0)

    def sample(self, beta=0.4):
        # Compute priorities raised to alpha (to control prioritization strength)
        scaled_priorities = np.array(self.priorities) ** self.alpha
        sample_probs = scaled_priorities / sum(scaled_priorities)

        # Sample experiences according to priority probabilities
        indices = np.random.choice(len(self.memory), self.batch_size, p=sample_probs)
        experiences = [self.memory[i] for i in indices]

        # Compute importance-sampling weights for bias correction
        total_samples = len(self.memory)
        weights = (total_samples * sample_probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalize for stability

        # Return sampled experiences and weights
        return experiences, indices, torch.FloatTensor(weights)

    def update_priorities(self, indices, td_errors):
        for idx, error in zip(indices, td_errors):
            # Update the priority of the corresponding experience
            self.priorities[idx] = abs(error) + 1e-5  # Avoid zero priority

#
    

# Deep Q Network with Batchnormalization and dropout
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        
        # Shortcut connection to match dimensions
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

# Dueling DQN Network with Residual Blocks
class Res_DuelingDQNetwork(nn.Module):
    def __init__(self, state_size, action_size, num_frames=2):
        super(Res_DuelingDQNetwork, self).__init__()
        
        # Residual blocks
        self.layer1 = ResidualBlock(num_frames, 16)
        self.layer2 = ResidualBlock(16, 32)
        self.layer3 = ResidualBlock(32, 64)
        
        conv_output_size = self._get_conv_output(state_size)
        
        # Shared fully connected layers
        self.fc_shared_1 = nn.Linear(conv_output_size, 256)
        self.bn_fc = nn.LayerNorm(256)
        self.fc_shared_2 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(p=0.5)
        
        # Value stream
        self.fc_value = nn.Linear(128, 64)
        self.value = nn.Linear(64, 1)
        
        # Advantage stream
        self.fc_advantage = nn.Linear(128, 64)
        self.advantage = nn.Linear(64, action_size)
        
    def _get_conv_output(self, shape):
        with torch.no_grad():
            x = torch.zeros(1, 2, shape, shape)  # The 2 is num_frames
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            return int(np.prod(x.size()))
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.bn_fc(self.fc_shared_1(x)))
        x = torch.relu(self.fc_shared_2(x))
        x = self.dropout(x)
        
        # Value stream
        value = torch.relu(self.fc_value(x))
        value = self.value(value)
        
        # Advantage stream
        advantage = torch.relu(self.fc_advantage(x))
        advantage = self.advantage(advantage)
        
        # Combine value and advantage streams
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

# Rest of your code remains the same
class Res_DuelingDQNAgent:
    def __init__(self, logger):
        # Hyperparameters
        self.gamma = 0.99        # Discount factor
        self.epsilon = 0       # Exploration rate (whilst not in training)
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.995 # Exploration rate decay
        self.learning_rate = 0.001
        self.batch_size = 128
        self.training_start = 500
        self.memory_size = 100000
        self.state_size = 9  # Example state size
        self.action_size = 6
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.counter_to_update_target_model = 0
        self.counter_to_reset_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epsilon = 0

        # Experience Replay memory
        self.memory = PrioritizedReplayBuffer(self.memory_size, self.batch_size, alpha=0.6, device=self.device)

        self.model = Res_DuelingDQNetwork(self.state_size, self.action_size, num_frames=2).to(self.device)
        self.target_model = Res_DuelingDQNetwork(self.state_size, self.action_size, num_frames=2).to(self.device)
        self.update_target_model()  # Initialize target model
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.SmoothL1Loss()  # Huber Loss
        self.train = False

        self.logger = logger

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))

    def forward(self, state):
        state = torch.FloatTensor(np.array(state)).unsqueeze(0).to(self.device)
        q_values = self.model(state)
        return ACTIONS[torch.argmax(q_values).item()], torch.max(q_values).item()

    def replay(self, beta=0.4):
        self.counter_to_update_target_model += 1
        self.counter_to_reset_epsilon += 1
        self.logger.info("We are in replay")
        if len(self.memory.memory) < self.training_start:
            self.logger.info("Model tried to train unsuccessfully.")
            return

        # Sample experiences with prioritized replay and importance sampling weights
        minibatch, indices, weights = self.memory.sample(beta=beta)
        weights = torch.FloatTensor(weights).to(self.device)

        # Convert minibatch to numpy arrays first
        states_list = [m[0] for m in minibatch]
        actions_list = [action_to_int(m[1]) for m in minibatch]
        rewards_list = [m[2] for m in minibatch]
        next_states_list = [
            m[3] if m[3] is not None else np.zeros_like(m[0])
            for m in minibatch
        ]

        # Convert lists to numpy arrays
        states_np = np.array(states_list)
        actions_np = np.array(actions_list)
        rewards_np = np.array(rewards_list)
        next_states_np = np.array(next_states_list)

        # Convert numpy arrays to tensors
        states = torch.from_numpy(states_np).float().to(self.device)
        actions = torch.from_numpy(actions_np).long().to(self.device)
        rewards = torch.from_numpy(rewards_np).float().to(self.device)
        next_states = torch.from_numpy(next_states_np).float().to(self.device)

        # Set up targets
        targets = torch.zeros_like(rewards)

        # Mask for non-final states
        non_final_mask = torch.tensor([not m[4] for m in minibatch], dtype=torch.bool).to(self.device)

        # Separate the next states into final and non-final
        non_final_next_states = next_states[non_final_mask]

        # Ensure next_states and states have the correct number of dimensions
        if next_states.ndimension() == 5:
            next_states = next_states.squeeze(1)  # Remove extra dimension if present

        # Get the current Q values for the selected actions
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Get the max Q values for the non-final next states using the main model for action selection
        if non_final_next_states.size(0) > 0:
            next_q_values = self.model(non_final_next_states)
            max_next_q_value_indices = next_q_values.max(1)[1]
            target_q_values = self.target_model(non_final_next_states)
            max_next_q_values = target_q_values.gather(1, max_next_q_value_indices.unsqueeze(1)).squeeze(1)
            targets[non_final_mask] = rewards[non_final_mask] + self.gamma * max_next_q_values
        else:
            targets = rewards

        # Set target values for final frames 
        targets[~non_final_mask] = rewards[~non_final_mask]

        # Compute the loss and apply importance sampling weights
        loss = (weights * self.criterion(q_values, targets.detach())).mean()

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update priorities in the replay buffer
        td_errors = (targets - q_values).detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors)
        self.logger.info("Updated Priorities in Prioritized Replay Buffer")

        # Update the target model every 100 episodes
        if self.counter_to_update_target_model == 100:
            self.update_target_model()
            self.counter_to_update_target_model = 0

        #$if self.train and self.counter_to_reset_epsilon == 250: 
        #    self.counter_to_reset_epsilon = 0
        #    self.epsilon = 0.7

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)

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
