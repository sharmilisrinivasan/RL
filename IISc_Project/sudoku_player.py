import os.path
import pickle

import numpy as np

from mcts_wrapper import get_action

BOARD_SIZE = 4


class Player:
    def __init__(self, exp_rate=0.3):
        self.states = []  # record all positions taken
        self.lr = 0.2
        self.exp_rate = exp_rate  # Epsilon
        self.decay_gamma = 0.9
        self.states_value = {}  # state -> value
        self.load_policy('policy_agent.pkl')

    @staticmethod
    def get_hash(board):
        return str(board.reshape(BOARD_SIZE * BOARD_SIZE))

    def choose_action_epsilon(self, positions, current_board):
        action = (None, None)
        if np.random.uniform(0, 1) <= self.exp_rate:
            # take random action
            idx = np.random.choice(len(positions))
            fill_val = np.random.choice(BOARD_SIZE)+1  # To avoid 0
            action = (positions[idx], fill_val)
        else:
            value_max = -999
            for p in positions:
                for fill_val in range(1, BOARD_SIZE+1):  # To avoid 0
                    next_board = current_board.copy()
                    next_board[p] = fill_val
                    next_board_hash = self.get_hash(next_board)
                    value = 0 if self.states_value.get(
                        next_board_hash) is None else self.states_value.get(next_board_hash)
                    # print("value", value)
                    if value >= value_max:
                        value_max = value
                        action = (p, fill_val)
        # print("{} takes action {}".format(self.name, action))
        return action

    def choose_action_mcts(self, current_board):
        action = get_action(current_board)
        return (action.x, action.y), action.value

    # at the end of game, back propagate and update states value
    def feed_reward(self, reward):
        for st in reversed(self.states):
            if self.states_value.get(st) is None:
                self.states_value[st] = 0
            self.states_value[st] += self.lr * (self.decay_gamma * reward - self.states_value[st])
            reward = self.states_value[st]

    # append a hash state
    def add_state(self, state):
        self.states.append(state)

    def reset(self):
        self.states = []

    def save_policy(self):
        with open('policy_agent.pkl', 'wb') as fw:
            pickle.dump(self.states_value, fw)

    def load_policy(self, load_file):
        if not os.path.isfile(load_file):
            return

        with open(load_file, 'rb') as fr:
            self.states_value = pickle.load(fr)
