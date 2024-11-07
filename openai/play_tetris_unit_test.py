
import unittest
from copy import deepcopy
from play_tetris_refactored import (
    get_empty_board,
    get_random_piece,
    get_random_position,
    can_move,
    apply_move,
    rotate_piece
)

class TestTetris(unittest.TestCase):

    def test_get_empty_board(self):
        board = get_empty_board()
        self.assertEqual(len(board), 22)  # EFF_BOARD_SIZE = 20 + 2
        self.assertEqual(len(board[0]), 22)
        # Check that the walls and floor are correctly set
        for i in range(len(board)):
            self.assertEqual(board[i][0], 1)
            self.assertEqual(board[i][-1], 1)
        for j in range(len(board[0])):
            self.assertEqual(board[-1][j], 1)

    def test_get_random_piece(self):
        piece = get_random_piece()
        self.assertIn(piece, deepcopy(list({1, 2, 3, 4, 5})))

    def test_get_random_position(self):
        piece = [[1, 1], [1, 1]]
        pos = get_random_position(piece)
        self.assertEqual(pos[0], 0)
        self.assertTrue(1 <= pos[1] < 20 - len(piece[0]) + 1)  # BOARD_SIZE - piece width + 1

    def test_can_move(self):
        board = get_empty_board()
        piece = [[1, 1], [1, 1]]
        pos = [1, 1]
        self.assertTrue(can_move(board, piece, pos, 0, 1))  # Move right
        self.assertTrue(can_move(board, piece, pos, 1, 0))  # Move down
        self.assertFalse(can_move(board, piece, pos, 0, -1))  # Move left into wall
        self.assertFalse(can_move(board, piece, [19, 1], 1, 0))  # Move down into floor

    def test_apply_move(self):
        pos = [5, 5]
        new_pos = apply_move(pos, 1, -1)
        self.assertEqual(new_pos, [6, 4])

    def test_rotate_piece(self):
        piece = [[1, 0], [1, 1]]
        rotated_piece = rotate_piece(piece, clockwise=True)
        expected_piece = [[1, 1], [1, 0]]
        self.assertEqual(rotated_piece, expected_piece)
        rotated_piece_ccw = rotate_piece(expected_piece, clockwise=False)
        self.assertEqual(rotated_piece_ccw, piece)

if __name__ == "__main__":
    unittest.main()
