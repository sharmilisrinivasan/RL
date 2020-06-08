from copy import deepcopy
import os
import pickle

import numpy as np


class Agent:

    def __init__(self, lr=0.2, decay=0.9):
        self.memory = {}
        self.lr = lr
        self.decay_gamma = decay
        self.recall('agent_memory.pkl')

    def choose_action(self, current_board, epsilon=0.3):
        # follow some policy : e.g. epsilon greedy
        action = (None, None)

        positions = current_board.available_positions()
        if np.random.uniform(0, 1) <= epsilon:
            # take random action
            idx = np.random.choice(len(positions))
            fill_val = np.random.choice(current_board.dim)+1  # To avoid 0
            action = (positions[idx], fill_val)
        else:
            value_max = -999
            for pos in positions:
                for fill_val in range(1, current_board.dim+1):  # To avoid 0
                    next_board = deepcopy(current_board)
                    next_board.set_value(pos, fill_val)
                    value = self.memory.get(next_board.get_pos_hash(pos), 0)
                    # print("value", value)
                    if value >= value_max:
                        value_max = value
                        action = (pos, fill_val)

        return action

    def learn(self, episode_states, reward):
        for state in reversed(episode_states):
            if self.memory.get(state) is None:
                self.memory[state] = 0
            self.memory[state] += self.lr * (reward - self.memory[state])
            reward = (self.decay_gamma * reward)

    def persist(self):
        with open('agent_memory.pkl', 'wb') as fw:
            pickle.dump(self.memory, fw)

    def recall(self, load_file):
        if not os.path.isfile(load_file):
            return

        with open(load_file, 'rb') as fr:
            self.memory = pickle.load(fr)
