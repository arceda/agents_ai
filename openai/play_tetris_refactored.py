
#!/usr/bin/python

"""
Python implementation of text-mode version of the Tetris game

Quick play instructions:

 - a (return): move piece left
 - d (return): move piece right
 - w (return): rotate piece counter clockwise
 - s (return): rotate piece clockwise
 - e (return): just move the piece downwards as is
 - q (return): quit game
"""

import os
import random
import sys
from copy import deepcopy

# Constants
BOARD_SIZE = 20
EFF_BOARD_SIZE = BOARD_SIZE + 2  # Adding walls

# Tetris pieces
PIECES = [
    [[1], [1], [1], [1]],  # I
    [[1, 0], [1, 0], [1, 1]],  # J
    [[0, 1], [0, 1], [1, 1]],  # L
    [[0, 1], [1, 1], [1, 0]],  # S
    [[1, 1], [1, 1]]  # O
]

# User inputs
MOVE_LEFT = 'a'
MOVE_RIGHT = 'd'
ROTATE_ANTICLOCKWISE = 'w'
ROTATE_CLOCKWISE = 's'
NO_MOVE = 'e'
QUIT_GAME = 'q'

def print_board(board, curr_piece, piece_pos, error_message=''):
    os.system('cls' if os.name == 'nt' else 'clear')
    print("Text mode version of the TETRIS game\n\n")

    board_copy = deepcopy(board)
    for i in range(len(curr_piece)):
        for j in range(len(curr_piece[0])):
            board_copy[piece_pos[0] + i][piece_pos[1] + j] |= curr_piece[i][j]

    for i in range(EFF_BOARD_SIZE):
        for j in range(EFF_BOARD_SIZE):
            print("[]" if board_copy[i][j] == 1 else "  ", end="")
        print()

    if error_message:
        print(f"\n{error_message}")

    print("Instructions: a (left), d (right), w (rotate ccw), s (rotate cw), e (down, default), q (quit)\n")

def get_empty_board():
    board = [[0] * EFF_BOARD_SIZE for _ in range(EFF_BOARD_SIZE)]
    for i in range(EFF_BOARD_SIZE):
        board[i][0] = board[i][EFF_BOARD_SIZE - 1] = 1  # Walls
    for j in range(EFF_BOARD_SIZE):
        board[EFF_BOARD_SIZE - 1][j] = 1  # Floor
    return board

def get_random_piece():
    return deepcopy(random.choice(PIECES))

def get_random_position(piece):
    return [0, random.randint(1, BOARD_SIZE - len(piece[0]))]

def can_move(board, piece, pos, dx, dy):
    for i in range(len(piece)):
        for j in range(len(piece[0])):
            if piece[i][j] and board[pos[0] + i + dx][pos[1] + j + dy]:
                return False
    return True

def apply_move(pos, dx, dy):
    return [pos[0] + dx, pos[1] + dy]

def rotate_piece(piece, clockwise=True):
    return [[piece[j][i] for j in range(len(piece))] for i in range(len(piece[0]) - 1, -1, -1)] if clockwise         else [[piece[j][i] for j in range(len(piece) - 1, -1, -1)] for i in range(len(piece[0]))]

def merge_piece(board, piece, pos):
    for i in range(len(piece)):
        for j in range(len(piece[0])):
            board[pos[0] + i][pos[1] + j] |= piece[i][j]

def play_game():
    board = get_empty_board()
    curr_piece = get_random_piece()
    piece_pos = get_random_position(curr_piece)
    error_msg = ''

    while True:
        do_move_down = False
        print_board(board, curr_piece, piece_pos, error_message=error_msg)
        error_msg = ''

        player_move = input().strip().lower()
        if player_move == MOVE_LEFT and can_move(board, curr_piece, piece_pos, 0, -1):
            piece_pos = apply_move(piece_pos, 0, -1)
            do_move_down = True
        elif player_move == MOVE_RIGHT and can_move(board, curr_piece, piece_pos, 0, 1):
            piece_pos = apply_move(piece_pos, 0, 1)
            do_move_down = True
        elif player_move == ROTATE_ANTICLOCKWISE and can_move(board, rotate_piece(curr_piece, False), piece_pos, 0, 0):
            curr_piece = rotate_piece(curr_piece, False)
            do_move_down = True
        elif player_move == ROTATE_CLOCKWISE and can_move(board, rotate_piece(curr_piece, True), piece_pos, 0, 0):
            curr_piece = rotate_piece(curr_piece, True)
            do_move_down = True
        elif player_move == NO_MOVE:
            do_move_down = True
        elif player_move == QUIT_GAME:
            print("Bye. Thank you for playing!")
            sys.exit(0)
        else:
            error_msg = "Invalid move!"

        if do_move_down and can_move(board, curr_piece, piece_pos, 1, 0):
            piece_pos = apply_move(piece_pos, 1, 0)

        if not can_move(board, curr_piece, piece_pos, 1, 0):
            merge_piece(board, curr_piece, piece_pos)
            curr_piece = get_random_piece()
            piece_pos = get_random_position(curr_piece)

            if not can_move(board, curr_piece, piece_pos, 0, 0):
                break

    print("GAME OVER!")

if __name__ == "__main__":
    play_game()
