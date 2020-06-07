import math
from random import choices, sample

import numpy as np


class Board:
    def __init__(self, board_dim=4, clues_cnt=4):
        self.dim = board_dim
        self.clues = clues_cnt
        self.board = np.zeros((self.dim, self.dim))
        self.generate_board()

    # ------------------------ Board based ------------------------

    def generate_board(self):
        while True:
            self.board = np.zeros((self.dim, self.dim))
            positions = sample([(i, j) for i in range(self.dim) for j in range(self.dim)],
                               self.clues)
            vals = choices(range(1, 5), k=4)
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


def generate_sudoku(size):  # to be written later
    to_return = np.zeros((4, 4))
    to_return[1] = np.array([2,3,4,1])
    to_return[2] = np.array([3,4,1,2])
    return to_return


def is_rec_valid(rec):
    non_zero_ele = len(set(rec[rec != 0.0]))
    zero_ele = len(rec[rec == 0.0])
    return (non_zero_ele + zero_ele) == len(rec)


def is_fill_valid(board, position_filled):
    row_filled, col_filled = position_filled
    row_vals = board[row_filled]
    col_vals = np.array([board[(i, col_filled)] for i in range(4)])
    # Get box values
    delimit = int(math.sqrt(4))
    start_row_pos = int((row_filled // delimit) * delimit)
    start_col_pos = int((col_filled // delimit) * delimit)
    box_vals_arr = []
    for i in range(start_row_pos, start_row_pos + delimit):
        for j in range(start_col_pos, start_col_pos + delimit):
            box_vals_arr.append(board[(i, j)])
    box_vals = np.array(box_vals_arr)

    return is_rec_valid(row_vals) and is_rec_valid(col_vals) and is_rec_valid(box_vals)
