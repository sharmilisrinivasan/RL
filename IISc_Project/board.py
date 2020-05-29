import math

import numpy as np


def generate_sudoku(size): # to be written later
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
