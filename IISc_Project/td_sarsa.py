from base import BaseEpisode, BaseRLAgent


class SarsaEpisode(BaseEpisode):

    def play(self):
        current_position, current_value = self.agent.choose_action(self.board)
        current_state = self.board.get_pos_hash(current_position)

        while True:
            self.board.set_value(current_position, current_value)
            reward = self.get_reward(current_position)

            if reward is None:
                reward = 0
                next_position, next_value = self.agent.choose_action(self.board)
                next_state = self.board.get_pos_hash(next_position)
            else:
                next_position, next_state, next_value = None, None, None

            self.agent.learn(current_state=current_state,
                             current_fill_val=current_value,
                             next_state=next_state,
                             next_fill_val=next_value,
                             reward=reward)

            if reward != 0:
                break

            current_position, current_value = next_position, next_value
            current_state = next_state

        # print("Game ended")
        # self.board.show_board()
        return reward


class SarsaAgent(BaseRLAgent):

    def learn(self, current_state, current_fill_val, next_state, next_fill_val, reward=0):
        # Current state values
        if self.memory.get(current_state) is None:
            self.memory[current_state] = {}

        if self.memory.get(current_state).get(current_fill_val) is None:
            self.memory[current_state][current_fill_val] = (0, 0)

        current_state_value, current_cnt = (self.memory.
                                            get(current_state, {}).
                                            get(current_fill_val, (0, 0)))

        # Next state values
        if not next_state:
            next_state_value = 0
        else:
            next_state_value, _ = self.memory.get(next_state, {}).get(next_fill_val, (0, 0))

        # Parameters
        updated_cnt = current_cnt + 1
        learning_rate = 1.0 / updated_cnt

        # Update value
        # q(s, a) = q(s, a) + lr*(r + decay * q(s',a') - q(s, a))
        updated_value = (current_state_value +
                         (learning_rate *
                          (reward + (self.decay_gamma * next_state_value) - current_state_value)))

        self.memory[current_state][current_fill_val] = (updated_value, updated_cnt)
