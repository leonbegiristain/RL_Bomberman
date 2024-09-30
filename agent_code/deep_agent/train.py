import os
import pickle
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import events as e
from .features import get_features, features_to_tensor, get_tile_type, count_nearby_crates, count_nearby_opponents
from ..base_agent.main import get_standarized_state_dict, forward_move_transformations, get_safe_moves_from_state, get_safe_moves
from ..base_agent import train as base_train

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

TRANSITION_HISTORY_SIZE = 10000  # keep only ... last transitions
BATCH_SIZE = 64
LEARNING_RATE = 0.001
TARGET_UPDATE_FREQUENCY = 1 # In rounds
GAMMA = 0.9
HIDDEN_DIM = 64

# Events
CLOSER_TO_COIN = "closer_to_coin"
CLOSER_TO_CRATE = "closer_to_crate"
BOMB_SCORE = "bomb_score"
TARGETED_OPPONENTS = "targeted_opponents"



class DQN_MPL(nn.Module):

    def __init__(
            self, 
            feature_shapes: list, 
            n_actions: int  = 6, 
            hidden_dim: int = 64,
            device: str = "cpu"
        ):
        super(DQN_MPL, self).__init__()
        self.n_features = len(feature_shapes)

        self.fc1 = nn.Linear(sum(feature_shapes), hidden_dim, device=device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, device=device)
        self.fc3 = nn.Linear(hidden_dim, n_actions, device=device)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()


    def forward(self, x):

        #embeddings = [self.relu(embedding(feature)) for embedding, feature in zip(self.feature_embeddings, features)]

        #x = torch.cat(features, dim=0)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))

        return x

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    #self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


    if (self.resume_training or self.eval_mode) and os.path.isfile("mlp_save.pt"):
        print("loading model")
        with open("mlp_save.pt", "rb") as file:
            self.online_net = pickle.load(file).to(self.device)
        with open("mlp_save.pt", "rb") as file:
            self.target_net = pickle.load(file).to(self.device)

    else:
        print("Starting to train new model")
        self.online_net = DQN_MPL(self.feature_shapes, hidden_dim=HIDDEN_DIM, device=self.device)
        self.target_net = DQN_MPL(self.feature_shapes, hidden_dim=HIDDEN_DIM, device=self.device)

    self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=LEARNING_RATE)
    self.transitions = []
    self.rounds = 0
    self.scores = []
    self.losses = []
    self.bombs = []
    self.crates = 0
    self.coins = 0
    self.kills = 0
    self.deaths = 0
    self.exp_dir = None

    
def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):

    if self.eval_mode:
        base_train.collect_data(self, events)
        if e.BOMB_DROPPED in events:
            self.bombs.append(old_game_state['self'][3])
        return



    old_game_state, trasformation = get_standarized_state_dict(old_game_state)
    new_game_state, _ = get_standarized_state_dict(new_game_state)

    old_features = get_features(old_game_state)
    new_features = get_features(new_game_state)

    self_action = forward_move_transformations(self_action, trasformation)
    action_id = ACTIONS.index(self_action)

    reward = compute_reward(self, old_game_state, old_features, self_action, new_game_state, new_features, events)

    old_tensor_features = features_to_tensor(old_features, device=self.device)
    new_tensor_features = features_to_tensor(new_features, device=self.device)

    # Store transition
    if len(self.transitions) == 0:
        print("Initial transition")
        self.transitions = [
            old_tensor_features.unsqueeze(0), 
            torch.tensor([action_id]).to(self.device), 
            new_tensor_features.unsqueeze(0), 
            torch.tensor([reward]).to(self.device)
        ]

    else:
        begin = 0 if len(self.transitions) < TRANSITION_HISTORY_SIZE else 1

        action_id = torch.tensor([action_id]).to(self.device)
        reward = torch.tensor([reward]).to(self.device)


        self.transitions[0] = torch.cat((self.transitions[0], old_tensor_features.unsqueeze(0)))[begin:]
        self.transitions[1] = torch.cat((self.transitions[1], action_id))[begin:]
        self.transitions[2] = torch.cat((self.transitions[2], new_tensor_features.unsqueeze(0)))[begin:]
        self.transitions[3] = torch.cat((self.transitions[3], reward))[begin:]

    #print([t.shape for t in self.transitions])
    #print([len(t) for t in self.transitions])

    #self.transitions.append((old_tensor_features, action_id, new_tensor_features, reward))



    if len(self.transitions[1]) > 1:
        batch_idx = random.sample(range(len(self.transitions[1])), min(BATCH_SIZE, len(self.transitions[1])))
        batch_idx = torch.tensor(batch_idx).to(self.device)

        old_tensor_features = torch.index_select(self.transitions[0], 0, batch_idx)
        action_id = torch.index_select(self.transitions[1], 0, batch_idx)
        new_tensor_features = torch.index_select(self.transitions[2], 0, batch_idx)
        reward = torch.index_select(self.transitions[3], 0, batch_idx)
    else:
        old_tensor_features, action_id, new_tensor_features, reward = self.transitions


    current_output = self.online_net(old_tensor_features)
    with torch.no_grad():
        new_output = self.target_net(new_tensor_features)

    expected_new = reward + GAMMA * torch.max(new_output, dim=1)[0]

    current_output = current_output.gather(1, action_id.unsqueeze(1)).squeeze(1)
        
    #loss = F.mse_loss(current_output[:,action_id], expected_new)/len(batch)
    loss = (current_output - expected_new).pow(2).mean()

    self.losses.append(loss.item())
    # Update model

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()


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
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    #self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))
    self.rounds += 1

    if self.eval_mode:
        self.scores.append(last_game_state['self'][1])
        if e.GOT_KILLED in events or e.KILLED_SELF in events:
            self.deaths += 1
        if self.rounds % 100 == 0:
            store_experiment_data(self)

        return


    if self.rounds % TARGET_UPDATE_FREQUENCY == 0:
        self.target_net.load_state_dict(self.online_net.state_dict())



    if e.KILLED_SELF in events:
        print("SUICIDE")
    elif e.GOT_KILLED in events:
        print("DIED")

    last_step = last_game_state['step']
    print(f"Game ended at step: {last_step}")
    user_score = last_game_state['self'][1]
    others_score = [agent[1] for agent in last_game_state["others"]]
    print(f"Score: {user_score}. Enemy scores: {others_score}")
    print("--------------------------------------------------------------------\n")


    self.scores.append(user_score)

    if self.rounds % 100 == 0:
        with open("mlp_save.pt", "wb") as file:
            pickle.dump(self.online_net, file)


    if self.rounds % 100 == 0:
        # Scores
        np.savetxt(f"scores.txt", self.scores)

        plt.plot(self.scores)
        avg_scores = np.convolve(np.array(self.scores), np.ones(100)/100, mode='valid')
        plt.plot(avg_scores)
        plt.yticks(range(16))
        plt.grid(axis='y')
        plt.savefig("scores.png")
        plt.close()
        # Losses
        plt.plot(self.losses)
        plt.yscale("log")
        #avg_losses = np.convolve(np.array(self.scores), np.ones(50)/50, mode='valid')
        #plt.plot(avg_losses)
        plt.savefig("losses.png")
        plt.close()

        


def reward_from_events(self, events: List[str], custom_rewards) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """ 

    game_rewards = {
        e.COIN_COLLECTED: 15,
        CLOSER_TO_COIN: custom_rewards[CLOSER_TO_COIN], # Got closer to a coin
        CLOSER_TO_CRATE: custom_rewards[CLOSER_TO_CRATE], # Got closer to a crate
        TARGETED_OPPONENTS: custom_rewards[TARGETED_OPPONENTS],
        BOMB_SCORE: custom_rewards[BOMB_SCORE], # Quality of the placed bomb
    }
    
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
            # print(f"Awarded {game_rewards[event]} for event {event}")

    # print(f"Awarded {reward_sum} in total")
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def collect_data(self, events: List[str]) -> None:
    if e.GOT_KILLED in events or e.KILLED_SELF in events:
        self.deaths += 1
    if e.CRATE_DESTROYED in events:
        self.crates += events.count(e.CRATE_DESTROYED)
    if e.COIN_COLLECTED in events:
        self.coins += 1
    if e.KILLED_OPPONENT in events:
        self.kills += events.count(e.KILLED_OPPONENT)
    

def store_experiment_data(self): 
    if self.exp_dir is None:
        self.exp_dir = "evaluation/experiment" + str(len(os.listdir("evaluation")))
        os.mkdir(self.exp_dir)
    # Training scores
    with open(f"{self.exp_dir}/data.txt", "w") as f:
        f.write(f"Tested agent for {self.rounds} rounds \n")
        f.write(f"Score: {np.array(self.scores).sum()}. Average: {np.array(self.scores).mean()}. Std: {np.array(self.scores).std()}\n")
        f.write(f"Kills: {self.kills}. Average: {self.kills/self.rounds}\n")
        f.write(f"Coins: {self.coins}. Average: {self.coins/self.rounds}\n")
        f.write(f"Deaths: {self.deaths} Average: {self.deaths/self.rounds}\n")
        f.write(f"Crates: {self.crates} Average: {self.crates/self.rounds}\n")
    # Score histogram
    plt.hist(self.scores, bins = max(self.scores)+1, range=(0, max(self.scores)+1))
    plt.savefig(f"{self.exp_dir}/scores.png")
    plt.close()
    np.savetxt(f"{self.exp_dir}/scores.txt", self.scores)


def compute_reward(self, old_state, old_features, action, new_state, new_features, events):

    old_safe_moves = self.new_safe_moves
    self.new_safe_moves = get_safe_moves_from_state(new_state)
    bomb_score, targeted_opponents = calculate_bomb_score(self, old_state, new_state, old_safe_moves, action == "BOMB")
    coin_score = moved_to_coin(self, old_features[1], new_features[1], e.COIN_COLLECTED in events)
    crate_score = moved_to_crate(self, old_features[-1], action)


    custom_rewards = {
        CLOSER_TO_COIN: coin_score,
        CLOSER_TO_CRATE: crate_score,
        BOMB_SCORE: bomb_score,
        TARGETED_OPPONENTS: targeted_opponents,
    }

    return reward_from_events(self, events, custom_rewards)


def moved_to_coin(self, old_coin_features, new_coin_features, coin_collected):
    """
    Checks if the agent moved closer to a coin.
    Returns 1 if it did, 0 if it stayed the same and -1 if it moved further away.
    If the coin was collected by the agent, returns 0.
    If no coin was present, returns 0.
    If the coin is too far away, returns 0.
    """
    old_coin_distance = np.abs(old_coin_features).sum()
    new_coin_distance = np.abs(new_coin_features).sum()

    if old_coin_distance > 10 and new_coin_distance > 10:
        return 0

    if new_coin_distance > old_coin_distance:
        if coin_collected:
            return 0
        else:
            return -1
    elif new_coin_distance == old_coin_distance:
        return 0
    else:
        return 4


def moved_to_crate(self, old_direction: np.ndarray, action: str) -> int:
    """
    Checks if the agent moved closer the dense spot.
    Returns 1 if it did, 0 if it stayed the same and -1 if it moved further away.
    If no crate was present, returns 0.
    """

    if action == "BOMB" or action == "WAIT":
        return 0
    
    good_actions = []
    neutral_actions = []

    if old_direction[1] == -1:
        good_actions.append("UP")
    elif old_direction[1] == 1:
        good_actions.append("DOWN")
    elif old_direction[1] == 0:
        neutral_actions += ["UP", "DOWN"]

    if old_direction[0] == -1:
        good_actions.append("LEFT")
    elif old_direction[0] == 1:
        good_actions.append("RIGHT")
    elif old_direction[0] == 0:
        neutral_actions += ["LEFT", "RIGHT"]

    if action in good_actions:
        return 1
    if action in neutral_actions:
        return 0
    return -1
    

def estimate_bomb_effectiveness(field: np.ndarray, user_coords: list, others: list = []) -> tuple[int, int]:
    """
    Returns the score of placing a bomb in the current user position.
    Rewards placing bombs close to crates and opponents.
    Rewards placing bombs in intersections.
    Penalizes placing bombs in corners.
    Penalizes placing bombs in empty spaces.
    """

    user_coords = np.array(user_coords)
    opponent_coords = np.array([other[3] for other in others])

    _, blast_mask = get_tile_type(field, user_coords)
    tiles_hit = blast_mask.sum()
    crates_hit = count_nearby_crates(field, user_coords, blast_mask)
    opponents_targeted = count_nearby_opponents(opponent_coords, user_coords)

    if crates_hit == 0 and opponents_targeted == 0:
        score = -3
    else:
        score = (tiles_hit - 6)/2 + 3 * crates_hit

    return score, 3 * opponents_targeted


def calculate_bomb_score(self, old_game_state, new_game_state, safe_moves, placed_bomb: bool) -> tuple[int, int]:
    """
    Rewards for moving to a tile with a higher bomb score.
    """
    had_bomb = old_game_state["self"][2]
    old_bomb_effectiveness, opponents_targeted  = estimate_bomb_effectiveness(old_game_state["field"], old_game_state["self"][3], old_game_state["others"])
    new_bomb_effectiveness, _ = estimate_bomb_effectiveness(new_game_state["field"], new_game_state["self"][3], new_game_state["others"])
    
    user_coords = old_game_state["self"][3]

    is_new_bombing_safe = self.new_safe_moves[5]

    if not safe_moves[5]:
        old_bomb_effectiveness = 0
    
    if not is_new_bombing_safe:
        new_bomb_effectiveness = 0

    if old_bomb_effectiveness == 0 and new_bomb_effectiveness == 0:
        return 0, opponents_targeted

    if not placed_bomb:
        if new_bomb_effectiveness < old_bomb_effectiveness:
            return (-3 if had_bomb else -1), opponents_targeted
        if new_bomb_effectiveness == old_bomb_effectiveness and had_bomb:
            return -1, opponents_targeted

    new_positions = [
        [user_coords[0], user_coords[1] - 1], # Up
        [user_coords[0] + 1, user_coords[1]], # Right
        [user_coords[0], user_coords[1] + 1], # Down
        [user_coords[0] - 1, user_coords[1]] # Left
    ]

    old_adjacent_tiles = [new_positions[i] for i in range(4) if (safe_moves[i] and get_safe_moves({**old_game_state, "self": (0, 0, False, new_positions[i])}, [False, False, False, False, False, True])[5])]

    if (safe_moves == True).sum() < 2: # Only 2 or less valid moves 
        return 0, opponents_targeted
    old_adjacent_scores = [estimate_bomb_effectiveness(old_game_state["field"], tile, old_game_state["others"])[0] for tile in old_adjacent_tiles]
    max_adjacent_score = 0 if len(old_adjacent_scores) == 0 else max(old_adjacent_scores)
    if placed_bomb:
        if old_bomb_effectiveness >= max_adjacent_score: # Placed bomb in a local maximum
            return old_bomb_effectiveness, opponents_targeted
        return -1, opponents_targeted
    if new_bomb_effectiveness >= max_adjacent_score: # Moved to a tile with the highest bomb score
        return 2, opponents_targeted
    if new_bomb_effectiveness > old_bomb_effectiveness: # Moved to a tile with a higher bomb score
        return 1, opponents_targeted
    else:
        return 0, opponents_targeted