from copy import deepcopy

from mcts import mcts

from board import is_fill_valid

BOARD_SIZE = 4

class Action():
    def __init__(self, value, x_pos, y_pos):
        self.value = value
        self.x = x_pos
        self.y = y_pos

    def __str__(self):
        return f"{self.x},{self.y} -> {self.value}"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.x == other.x and self.y == other.y and self.value == other.value

    def __hash__(self):
        return hash((self.x, self.y, self.value))


class State:
    def __init__(self, board):
        self.board = board
        self.last_position_filled = None

    def available_positions(self):
        positions = []
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i, j] == 0:
                    positions.append((i, j))  # need to be tuple
        return positions

    def getPossibleActions(self):
        possible_actions = []

        for x_pos in range(BOARD_SIZE):
            for y_pos in range(BOARD_SIZE):
                if self.board[x_pos, y_pos] == 0:
                    for val in range(1, BOARD_SIZE + 1):
                        possible_actions.append(Action(value=val, x_pos=x_pos, y_pos=y_pos))
        return possible_actions

    def takeAction(self, action):
        newState = deepcopy(self)
        newState.board[(action.x, action.y)] = action.value
        newState.last_position_filled = (action.x, action.y)
        return newState

    def isTerminal(self):
        if not self.last_position_filled:  # not started
            return False
        if not is_fill_valid(self.board, self.last_position_filled):
            return True
        if len(self.available_positions()) == 0:
            return True
        return False

    def getReward(self):
        if not is_fill_valid(self.board, self.last_position_filled):
            return -1
        if len(self.available_positions()) == 0:
            return 1
        return None


def get_action(current_board):
    return mcts(timeLimit=1000).search(initialState=State(current_board))
