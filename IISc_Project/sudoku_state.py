from board import generate_sudoku, is_fill_valid

BOARD_SIZE = 4


class State:
    def __init__(self, agent):
        self.board = generate_sudoku(BOARD_SIZE)
        self.isEnd = False
        self.agent = agent
        self.boardHash = None
        self.last_position_filled = None

    # get unique hash of current board state
    def get_hash(self):
        self.boardHash = str(self.board.reshape(BOARD_SIZE * BOARD_SIZE))
        return self.boardHash

    def available_positions(self):
        positions = []
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i, j] == 0:
                    positions.append((i, j))  # need to be tuple
        return positions

    def update_state(self, position, value):
        self.board[position] = value
        self.last_position_filled = position

    def winner(self):
        if not is_fill_valid(self.board, self.last_position_filled):
            self.isEnd = True
            return -1

        if len(self.available_positions()) == 0:
            self.isEnd = True
            return 1

        self.isEnd = False
        return None

    # only when game ends
    def give_reward(self):
        result = self.winner()
        # back propagate reward
        self.agent.feed_reward(result)

    # board reset
    def reset(self):
        self.board = generate_sudoku(BOARD_SIZE)
        self.boardHash = None
        self.isEnd = False
        self.last_position_filled = None

    def play(self, rounds=100):
        for i in range(rounds):
            print("Rounds {}".format(i))

            while not self.isEnd:
                # positions = self.available_positions()
                # agent_action = self.agent.choose_action_epsilon(positions, self.board)
                agent_action = self.agent.choose_action_mcts(self.board)
                self.update_state(*agent_action)
                board_hash = self.get_hash()
                self.agent.add_state(board_hash)

                # check board status if it is end
                win = self.winner()
                if win is not None:
                    print("Game ended")
                    self.show_board()
                    # ended with success or wrong fill
                    self.give_reward()
                    self.agent.reset()
                    self.reset()
                    break
            print("*************************Round Ended*************************")
        print("Learning completed - Saving the learning")
        self.agent.save_policy()

    def show_board(self):
        for i in range(0, BOARD_SIZE):
            print('------------------')
            out = '| '
            for j in range(0, BOARD_SIZE):
                if self.board[i, j] == 0:
                    token = ' '
                else:
                    token = str(int(self.board[i, j]))
                out += token + ' | '
            print(out)
        print('------------------')
