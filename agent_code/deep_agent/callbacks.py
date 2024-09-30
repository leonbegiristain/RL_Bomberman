import os 
import pickle
import numpy as np

import torch

from ..base_agent import main as base
from .features import get_tensor_features


def setup(self):

    self.eval_mode = True
    self.resume_training = False

    self.feature_shapes = [1,2,1,2,1,2,5,2]
        # np.zeros((0, 1)), # Step
        # np.zeros((0, 2)), # Closest coin
        # np.zeros((0, 1)), # Nearby crates
        # np.zeros((0, 2)), # Closest opponent
        # np.zeros((0, 1)), # Nearby opponents
        # np.zeros((0, 2)), # Closest bombs
        # np.zeros((0, 5)), # tile type
        # np.zeros((0, 2)) # Direction of dense crates

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #self.device = torch.device("cpu")

    print("Initializing model in ", self.device)
    print("Mode:", "train" if self.train else "play")

    if not self.train: 
        if os.path.isfile("mlp_save.pt"):
            with open("mlp_save.pt", "rb") as file:
                self.online_net = pickle.load(file).to(self.device)
        else:
            raise Exception("No model found")

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    #print(base_train.bomb_score(game_state))

    game_state, transformations = base.get_standarized_state_dict(game_state)

    if not (self.train and not self.eval_mode and game_state['step'] != 1):
        self.new_safe_moves = base.get_safe_moves_from_state(game_state)

    safe_moves = self.new_safe_moves

    play_random = False
    if play_random:
        selected_move = base.select_move(np.random.random(6), safe_moves)
    else:
        if self.train and not self.eval_mode and np.random.random() < base.epsion_greedy(self, final_eps=0.2, decay=0.01):
            selected_move = base.select_move(np.random.random(6), safe_moves)
        else:
            #features = self.new_features if (self.train and not self.eval_mode and game_state['step'] != 1) else get_tensor_features(game_state, self.device)
            
            features = get_tensor_features(game_state, self.device)
            action_values = self.online_net(features).cpu().detach().numpy()


            selected_move = base.select_move(action_values, safe_moves)
        
    return base.revert_move_transformations(selected_move, transformations)


