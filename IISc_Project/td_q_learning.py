from base import BaseEpisode, BaseRLAgent


class QLearningEpisode(BaseEpisode):

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


class QLearningAgent(BaseRLAgent):

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
