from abc import ABC, abstractmethod
import os
import pickle

import numpy as np

from board import Board


class BaseEpisode(ABC):

    def __init__(self, agent):
        self.board = Board()
        self.agent = agent
        self.positions_filled = []

    def get_reward(self, position):
        if not self.board.is_fill_valid(position):
            return -1

        if self.board.is_board_full():
            return 1

        return None

    @abstractmethod
    def play(self):
        # To be implemented in derived classes
        pass


class BaseRLAgent(ABC):

    def __init__(self, decay=0.9):
        self.memory = {}
        self.decay_gamma = decay
        self.recall('agent_memory.pkl')

    def choose_action(self, current_board, epsilon=0.1):
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
                current_state = self.memory.get(current_board.get_pos_hash(pos), {})

                for fill_val in range(1, current_board.dim+1):  # To avoid 0
                    value, _ = current_state.get(fill_val, (0, 0))
                    if value >= value_max:
                        value_max = value
                        action = (pos, fill_val)
        return action

    def persist(self):
        with open('agent_memory.pkl', 'wb') as fw:
            pickle.dump(self.memory, fw)

    def recall(self, load_file):
        if not os.path.isfile(load_file):
            return

        with open(load_file, 'rb') as fr:
            self.memory = pickle.load(fr)

    @abstractmethod
    def learn(self, *args):
        # To be implemented in derived classes
        pass
