from collections import namedtuple, deque

import pickle

from typing import List

import events as e
from .callbacks import create_multiple_local_frames, dijkstra_shortest_path_to_targets, dijkstra_next_free_tile

import os
import numpy as np


# Events
SURVIVED = "Survived"
GOT_CLOSER_TO_COIN = "closer_to_coin"
GOT_FURTHER_AWAY_FROM_COIN = "further_from_coin"
MADE_A_VALID_MOVE = "valid_move"
SPEND_TIME_IN_DANGER_ZONE = "spend time in danger Zone"
ESCAPED_DANGERZONE = "escaped danger zone"
WENT_INTO_DANGERZONE = "went into Dangerzone"
USEFULL_BOMB = "planted usefull bomb"
USELESS_BOMB = "planted useless bomb"
GOT_CLOSER_TO_CHEST = "got closer to chest"
GOT_FURTHER_AWAY_FROM_CHEST = "got further away from chest"
MOVE_TO_CERTAIN_DEATH = "moved into a certain death scenario"
SUICIDE_BOMB = "layed bomb that will kill him"
ESCAPING_BOMB = "got further away from bomb in dangerzone"
ROUND_WON = "Round won!"
ROUND_LOST = "Round lost!"
LOOP_DETECTED="loop"
LOOP_BOMBING="loop bomb"



def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.model.train = True
    self.model.epsilon = 0.9 #only when training we do exploration

    self.bomb_counter=0         #counts the number of bombs placed 
    self.invalid_move_counter=0 #counts the number of invalid moves
    self.position_history = []   # Track agent's positions to punish loops
    self.max_history_length = 4 # Max length of history to detect loops
    self.loop_counter=0
    self.loop_explore_counter=0
    self.enemy_kill_counter=0



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
    _, old_score, _, (old_x, old_y) = old_game_state['self']
    _, new_score, _, (new_x, new_y) = new_game_state['self']

    current_pos= new_game_state['self'][-1]
    # Add current position to the history
    self.position_history.append(current_pos)


    # Limit the history to the max length to avoid memory bloat
    if len(self.position_history) > self.max_history_length:
        self.position_history.pop(0)  #removes the oldest entry from the list

    if self.position_history.count(current_pos) > 1 and not e.BOMB_DROPPED in events:
        events.append(LOOP_DETECTED)  # Custom event for loop detection
    

    # Got closer to Coin event
    old_coin_distance, old_coin_coords, shortest_path_to_old_coin = dijkstra_shortest_path_to_targets(old_game_state, ['coins'])
    new_coin_distance, new_coin_coords, shortest_path_to_new_coin = dijkstra_shortest_path_to_targets(new_game_state, ['coins'])

    old_frames = create_multiple_local_frames(old_game_state, shortest_path_to_old_coin)
    new_frames = create_multiple_local_frames(new_game_state, shortest_path_to_new_coin)
    _, old_dangerzone = old_frames
    _, new_dangerzone = new_frames
    self_pos = (old_dangerzone.shape[0] // 2, old_dangerzone.shape[1] // 2)

    if (old_coin_distance > new_coin_distance and old_coin_coords == new_coin_coords and old_dangerzone[self_pos] == 0 and new_dangerzone[self_pos] == 0 and shortest_path_to_old_coin):
        events.append(GOT_CLOSER_TO_COIN)
    elif (old_coin_distance <= new_coin_distance and old_coin_coords == new_coin_coords and new_coin_coords != None and e.BOMB_DROPPED not in events and old_dangerzone[self_pos] == 0 and new_dangerzone[self_pos] == 0 and e.COIN_COLLECTED not in events):
        events.append(GOT_FURTHER_AWAY_FROM_COIN)

    # Got closer to chest
    if old_coin_distance == np.inf and old_dangerzone[self_pos] == 0 and new_dangerzone[self_pos] == 0:      
        old_chest_distance, _, old_path_to_chest = dijkstra_shortest_path_to_targets(old_game_state, ['chests'])
        new_chest_distance, _, _ = dijkstra_shortest_path_to_targets(new_game_state, ['chests'])

        if old_coin_distance > 0 and e.BOMB_DROPPED not in events:
            if old_chest_distance > new_chest_distance:
                events.append(GOT_CLOSER_TO_CHEST)
            else:
                events.append(GOT_FURTHER_AWAY_FROM_CHEST)

    # Made a valid move 
    if e.INVALID_ACTION not in events:
        events.append(MADE_A_VALID_MOVE)

    # Spend time in DangerZone
    if old_dangerzone[self_pos] !=0 and new_dangerzone[self_pos] != 0:
        old_dist_to_bomb, _, _ = dijkstra_shortest_path_to_targets(old_game_state, targets= ['bombs'])
        new_dist_to_bomb, _, _ = dijkstra_shortest_path_to_targets(new_game_state, targets= ['bombs'])
        if (old_dist_to_bomb >= new_dist_to_bomb):
            events.append(SPEND_TIME_IN_DANGER_ZONE)
        else:
            events.append(ESCAPING_BOMB)


    # Escaped DangerZone
    if old_dangerzone[self_pos] != 0 and new_dangerzone[self_pos] == 0:
        events.append(ESCAPED_DANGERZONE)

    # Entered DangerZone
    if old_dangerzone[self_pos] == 0 and new_dangerzone[self_pos] != 0 and e.BOMB_DROPPED not in events:
        events.append(WENT_INTO_DANGERZONE)
        
    # Move to certain Death
    if new_dangerzone[self_pos] != 0:
        _, bomb_pos, _ = dijkstra_shortest_path_to_targets(new_game_state, targets= ['bombs'])
        bomb_countdown_new = get_bomb_countdown(new_game_state['bombs'], bomb_pos)
        dist_to_free_tile_new, _, _ = dijkstra_next_free_tile(new_game_state)

        bomb_countdown_old = get_bomb_countdown(old_game_state['bombs'], bomb_pos)
        if bomb_countdown_old:
            dist_to_free_tile_old, _, _ = dijkstra_next_free_tile(old_game_state)
        else:
            dist_to_free_tile_old = 1 # this is dumb but works
            bomb_countdown_old = 1
        
        if bomb_countdown_new != None:
            if dist_to_free_tile_new > bomb_countdown_new and dist_to_free_tile_old <= bomb_countdown_old:
                events.append(MOVE_TO_CERTAIN_DEATH)

    # Planted Usefull Bomb (Naive - simple distance to others)
    if e.BOMB_DROPPED in events:
        dist_to_enemy, _, _ = dijkstra_shortest_path_to_targets(old_game_state, targets= ['others'])
        if dist_to_enemy <= 1 or bomb_destroys_chest(old_game_state):
            events.append(USEFULL_BOMB)
        else:
            events.append(USELESS_BOMB)

     # Suicide Bomb event
    dist_to_free_tile_new, _, _ = dijkstra_next_free_tile(new_game_state)
    if e.BOMB_DROPPED in events and dist_to_free_tile_new == np.inf:
        events.append(SUICIDE_BOMB)

    if e.KILLED_OPPONENT in events:
        self.enemy_kill_counter += 1

    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    reward = reward_from_events(self, new_score, events)
    
    # Save the Transition in the Train Buffer
    self.model.remember(old_frames, self_action, reward, new_frames, False)


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
    _, score, _, _ = last_game_state['self']
    _, _, shortest_path_to_coin = dijkstra_shortest_path_to_targets(last_game_state, ['coins'])
    last_frame = create_multiple_local_frames(last_game_state, shortest_path_to_coin)

    # Round won or lost Event
    if last_game_state['others']:
        others_score = [scores for _, scores, _, _, in last_game_state['others']]
        if score > np.max(others_score):
            events.append(ROUND_WON)
        else:
            events.append(ROUND_LOST)
    
    # Send events to logger
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    # Collect Final Rewards <- They could be different than standard rewards
    reward = reward_from_events(self, score, events)

    # Add Transition to Memory 
    self.model.remember(last_frame, last_action, reward, None, True)

    # Train the model - Right now this happens after every round if the train buffer has at least batch size many elements
    self.model.replay()

    datafile = os.path.join("savestates", "Res4DQL.pt")
    scorefile = os.path.join("scores", "Res4DQL.txt")

    # Store the model
    with open(datafile, "wb") as file:
        self.model.save(file)

    # Append the score to the file
    with open(scorefile, "a") as f:
        f.write(f"{score},{last_game_state['step']},{self.bomb_counter},{round(1-(self.invalid_move_counter/last_game_state['step']),3)},{self.enemy_kill_counter},{round(self.model.epsilon,2)}\n")

    # Reset event counters
    self.bomb_counter=0         
    self.invalid_move_counter=0
    self.loop_explore_counter=0
    self.enemy_kill_counter=0

def reward_from_events(self, score, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        ROUND_LOST: -200,
        e.GOT_KILLED: -150,
        e.KILLED_SELF: -100,
        SUICIDE_BOMB: -100,
        MOVE_TO_CERTAIN_DEATH: -75,
        WENT_INTO_DANGERZONE:-70, 
        USELESS_BOMB: -35,
        SPEND_TIME_IN_DANGER_ZONE: -30, 
        e.INVALID_ACTION: -25,
        GOT_FURTHER_AWAY_FROM_COIN: -25,
        GOT_FURTHER_AWAY_FROM_CHEST: -15,
        e.WAITED: 0,
        e.MOVED_DOWN: 0,
        e.MOVED_LEFT: 0,
        e.MOVED_RIGHT: 0,
        e.MOVED_UP: 0,
        e.SURVIVED_ROUND: 0,
        MADE_A_VALID_MOVE: 0,
        SURVIVED: 0,
        e.BOMB_DROPPED: 0, 
        ESCAPING_BOMB: 5,
        GOT_CLOSER_TO_CHEST: 10,
        GOT_CLOSER_TO_COIN: 20,
        e.COIN_FOUND: 20,
        USEFULL_BOMB: 75,
        e.CRATE_DESTROYED: 30,
        ESCAPED_DANGERZONE: 40,
        e.COIN_COLLECTED:125,
        e.KILLED_OPPONENT: 100,
        ROUND_WON:200,
        LOOP_DETECTED:0,
        LOOP_BOMBING:0
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")

    if e.BOMB_DROPPED in events:
        self.bomb_counter += 1
    if e.INVALID_ACTION in events:
        self.invalid_move_counter +=1

    return reward_sum

def get_bomb_countdown(bombs, coordinates):
    """
    Given a list of bombs and a pair of coordinates, return the countdown 't' of the bomb
    at those coordinates. If there is no bomb at the given coordinates, return None.
    
    :param bombs: List of tuples ((x, y), t) representing bomb positions and their countdowns.
    :param coordinates: Tuple (x, y) representing the coordinates to check.
    :return: The countdown 't' if the bomb exists at the given coordinates, otherwise None.
    """
    for (x, y), countdown in bombs:
        if (x, y) == coordinates:
            return countdown
    return None

def bomb_destroys_chest(gamestate):
    _, _, _, (self_x, self_y) = gamestate['self']
    field = gamestate['field']

    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        for dist in range(1, 4):
            danger_x = self_x + dx * dist
            danger_y = self_y + dy * dist
            if field[danger_x, danger_y] == -1: # We stop following this direction as soon as we hit a wall
                break
            if field[danger_x, danger_y] == 1: # Return True if we find a chest
                return True
    
    return False