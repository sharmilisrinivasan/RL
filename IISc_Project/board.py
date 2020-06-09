import math
from random import choices, sample

import numpy as np


class Board:
    def __init__(self, board_dim=4, clues_cnt=8):
        self.dim = board_dim
        self.clues = clues_cnt
        self.board = np.zeros((self.dim, self.dim))
        self.generate_board()

    # ------------------------ Board based ------------------------

    def generate_board(self, static=True):
        if static:
            self.board = np.zeros((self.dim, self.dim))
            self.board[1] = np.array([1, 2, 3, 4])
            self.board[2] = np.array([2, 3, 4, 1])
            self.board[3] = np.array([4, 1, 2, 3])
            return

        while True:
            self.board = np.zeros((self.dim, self.dim))
            positions = sample([(i, j) for i in range(self.dim) for j in range(self.dim)],
                               self.clues)
            vals = choices(range(1, 5), k=self.clues)
            success = True
            for position, val in zip(positions, vals):
                self.board[position] = val
                if not self.is_fill_valid(position):
                    success = False
                    break
            if success:
                break

    def is_board_full(self):
        for row in range(self.dim):
            if len(np.where(self.get_row((row, 0)) == 0.0)[0]) > 0:
                return False
        return True

    def show_board(self):
        for i in range(0, self.dim):
            print('------------------')
            out = '| '
            for j in range(0, self.dim):
                if self.board[i, j] == 0:
                    token = ' '
                else:
                    token = str(int(self.board[i, j]))
                out += token + ' | '
            print(out)
        print('------------------')

    # ------------------------ Positions based ------------------------

    def available_positions(self):
        positions = []
        for row in range(self.dim):
            for col in range(self.dim):
                if self.board[row, col] == 0:
                    positions.append((row, col))
        return positions

    def get_row(self, position):
        row_pos, _ = position
        return self.board[row_pos]

    def get_col(self, position):
        _, col_pos = position
        return np.array([self.board[(i, col_pos)] for i in range(self.dim)])

    def get_square(self, position):
        row_pos, col_pos = position

        delimit = int(math.sqrt(self.dim))
        start_row_pos = int((row_pos // delimit) * delimit)
        start_col_pos = int((col_pos // delimit) * delimit)

        box_vals_arr = []
        for i in range(start_row_pos, start_row_pos + delimit):
            for j in range(start_col_pos, start_col_pos + delimit):
                box_vals_arr.append(self.board[(i, j)])

        return np.array(box_vals_arr)

    def get_pos_hash(self, position):
        return ("".join(map(str, map(int, self.get_row(position)))) +
                "".join(map(str, map(int, self.get_col(position)))) +
                "".join(map(str, map(int, self.get_square(position)))))

    def set_value(self, position, value):
        self.board[position] = value

    # ------------------------ Validity check ------------------------

    @staticmethod
    def is_rec_valid(rec):
        non_zero_ele = len(set(rec[rec != 0.0]))
        zero_ele = len(rec[rec == 0.0])
        return (non_zero_ele + zero_ele) == len(rec)

    def is_fill_valid(self, position_filled):
        return (self.is_rec_valid(self.get_row(position_filled)) and
                self.is_rec_valid(self.get_col(position_filled)) and
                self.is_rec_valid(self.get_square(position_filled)))
