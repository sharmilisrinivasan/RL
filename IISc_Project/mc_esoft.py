from base import BaseEpisode, BaseRLAgent


class MCEpisode(BaseEpisode):

    def play(self):
        while True:
            position, value = self.agent.choose_action(self.board)
            self.positions_filled.append((self.board.get_pos_hash(position), value))
            self.board.set_value(position, value)
            reward = self.get_reward(position)

            if reward is not None:
                # print("Game ended")
                # self.board.show_board()
                self.agent.learn(self.positions_filled, reward)
                return reward


class MCAgent(BaseRLAgent):

    def learn(self, episode_states, reward):
        for state, action_fill in reversed(episode_states):
            if self.memory.get(state) is None:
                self.memory[state] = {}

            if self.memory.get(state).get(action_fill) is None:
                self.memory[state][action_fill] = (0, 0)

            current_value, cnt = self.memory[state][action_fill]

            updated_cnt = cnt + 1
            learning_rate = 1.0 / updated_cnt
            updated_value = current_value + (learning_rate * (reward - current_value))

            self.memory[state][action_fill] = (updated_value, updated_cnt)
            reward = (self.decay_gamma * reward)
