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
            position, value = self.agent.choose_action(self.board)
            self.board.set_value(position, value)
            self.positions_filled.append(self.board.get_pos_hash(position))

            reward = self.get_reward(position)

            if reward is not None:
                print("Game ended")
                self.board.show_board()
                self.agent.learn(self.positions_filled, reward)
                return reward
