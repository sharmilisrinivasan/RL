import os
import pickle

import numpy as np

from board import Board


class Episode:
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

    def play(self):

        while True:
            current_position, current_value = self.agent.choose_action(self.board)
            current_state = self.board.get_pos_hash(current_position)

            self.board.set_value(current_position, current_value)
            reward = self.get_reward(current_position)

            if reward is None:
                reward = 0
                next_states = [(pos, fill)
                               for pos in self.board.available_positions()
                               for fill in range(1, self.board.dim + 1)]
            else:
                next_states = None

            self.agent.learn(current_state=current_state,
                             current_fill_val=current_value,
                             next_possible_states=next_states,
                             reward=reward)

            if reward != 0:
                break

        # print("Game ended")
        # self.board.show_board()
        return reward


class TDAgent:

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
            fill_val = np.random.choice(current_board.dim) + 1  # To avoid 0
            action = (positions[idx], fill_val)
        else:
            value_max = -999
            for pos in positions:
                current_state = self.memory.get(current_board.get_pos_hash(pos), {})

                for fill_val in range(1, current_board.dim + 1):  # To avoid 0
                    value, _ = current_state.get(fill_val, (0, 0))
                    # print("value", value)
                    if value >= value_max:
                        value_max = value
                        action = (pos, fill_val)

        return action

    def learn(self, current_state, current_fill_val, next_possible_states, reward=0):
        # Current state values
        if self.memory.get(current_state) is None:
            self.memory[current_state] = {}

        if self.memory.get(current_state).get(current_fill_val) is None:
            self.memory[current_state][current_fill_val] = (0, 0)

        current_state_value, current_cnt = (self.memory.
                                            get(current_state, {}).
                                            get(current_fill_val, (0, 0)))

        # Next state values
        if not next_possible_states:
            next_state_value = 0
        else:
            next_state_value = -999

            for state, fill_val in next_possible_states:
                value, _ = self.memory.get(state, {}).get(fill_val, (0, 0))
                # print("value", value)
                if value >= next_state_value:
                    next_state_value = value

        # Parameters
        updated_cnt = current_cnt + 1
        learning_rate = 1.0 / updated_cnt

        # Update value
        # q(s, a) = q(s, a) + lr*(r + decay * q(s',a') - q(s, a))
        updated_value = (current_state_value +
                         (learning_rate *
                          (reward + (self.decay_gamma * next_state_value) - current_state_value)))

        self.memory[current_state][current_fill_val] = (updated_value, updated_cnt)

    def persist(self):
        with open('agent_memory.pkl', 'wb') as fw:
            pickle.dump(self.memory, fw)

    def recall(self, load_file):
        if not os.path.isfile(load_file):
            return

        with open(load_file, 'rb') as fr:
            self.memory = pickle.load(fr)
