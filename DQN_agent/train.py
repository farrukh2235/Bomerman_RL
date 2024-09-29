from collections import namedtuple, deque

import pickle
import heapq
from typing import List

import events as e
from .callbacks import state_to_features, game_state_to_frame


import numpy as np

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
SURVIVED = "Survived"
GOT_CLOSER_TO_COIN = "closer_to_coin"
GOT_FURTHER_AWAY_FROM_COIN = "further_from_coin"
MADE_A_VALID_MOVE = "valid_move"



def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.model.train = True


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """

    # Survived
    events.append(SURVIVED)

    # Got closer to Coin event:
    old_coin_distance = dijkstra_shortest_path_to_targets(old_game_state, ['coins'])
    new_coin_distance = dijkstra_shortest_path_to_targets(new_game_state, ['coins'])
    if (old_coin_distance < new_coin_distance):
        events.append(GOT_CLOSER_TO_COIN)
    elif (old_coin_distance > new_coin_distance):
        events.append(GOT_FURTHER_AWAY_FROM_COIN)

    # Made a valid move 
    if e.INVALID_ACTION not in events:
        events.append(MADE_A_VALID_MOVE)

    old_frame = game_state_to_frame(old_game_state)
    new_frame = game_state_to_frame(new_game_state)


    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    reward = reward_from_events(self, events)
    self.transitions.append(Transition(old_game_state, self_action, new_game_state, reward))


    # Save the Transition in the Train Buffer
    self.model.remember(old_frame, self_action, reward, new_frame, False)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    last_frame = game_state_to_frame(last_game_state)
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    # Collect Final Rewards <- They could be different than standard rewards
    reward = reward_from_events(self, events)
    self.transitions.append(Transition(last_frame, last_action, None, reward))

    # Add Transition to Memory 
    self.model.remember(last_frame, last_action, reward, None, True)

    # Train the model - Right now this happens after every round if the train buffer has at least batch size many elements
    self.model.replay(self.logger)

    # Store the model
    with open("Coin_heaven_first_try.pt", "wb") as file:
        self.model.save(file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.GOT_KILLED: -20,
        e.MOVED_DOWN: 1,
        e.MOVED_LEFT: 1,
        e.MOVED_RIGHT: 1,
        e.MOVED_UP: 1,
        e.WAITED: 0,
        e.COIN_COLLECTED: 20,
        e.KILLED_OPPONENT: 5,
        e.INVALID_ACTION: -10,
        e.CRATE_DESTROYED: 5,
        e.COIN_FOUND: 10,
        e.SURVIVED_ROUND: 100,
        e.KILLED_SELF: -30,
        e.BOMB_DROPPED: -30, #### This is for Coinheaven only
        MADE_A_VALID_MOVE: 5,
        SURVIVED: 1,
        GOT_CLOSER_TO_COIN: 15,
        GOT_FURTHER_AWAY_FROM_COIN: -5,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum



def dijkstra_shortest_path_to_targets(game_state, targets=['coins', 'others']):
    # Extract game state information
    field = game_state['field']
    bombs = game_state['bombs']
    self_agent = game_state['self']
    coins = game_state['coins'] if 'coins' in targets else []
    others = [pos for _, _, _, pos in game_state['others']] if 'others' in targets else []
    chests = np.argwhere(field == 1) if 'chests' in targets else []
    
    # Define the target positions
    target_positions = coins + others + [(x, y) for x, y in chests]

    # Initialize the Dijkstra's algorithm structures
    height, width = field.shape
    start = self_agent[-1]  # self_agent[-1] is the position (x, y)
    distances = np.full((height, width), np.inf)
    distances[start] = 0
    visited = np.zeros((height, width), dtype=bool)
    pq = [(0, start)]  # priority queue with (distance, position)

    # Define movement directions (up, down, left, right)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while pq:
        current_distance, current_position = heapq.heappop(pq)
        x, y = current_position
        
        if visited[x, y]:
            continue
        
        visited[x, y] = True

        # Check if we have reached a target
        if (x, y) in target_positions:
            return current_distance
        
        # Explore neighbors
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            if 0 <= nx < height and 0 <= ny < width and not visited[nx, ny]:
                # Make sure the position is passable (not a wall or bomb)
                if field[nx, ny] == 0:  # 0 means free tile
                    new_distance = current_distance + 1
                    if new_distance < distances[nx, ny]:
                        distances[nx, ny] = new_distance
                        heapq.heappush(pq, (new_distance, (nx, ny)))

    # If no path is found to any target
    return np.inf