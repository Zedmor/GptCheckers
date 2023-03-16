import numpy as np
from rules import Checkers


def test_initial_board():
    checkers = Checkers()
    initial_board = [
        [0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [-1, 0, -1, 0, -1, 0, -1, 0],
        [0, -1, 0, -1, 0, -1, 0, -1],
        [-1, 0, -1, 0, -1, 0, -1, 0]
        ]
    assert (checkers.board == initial_board).all()


def test_valid_move():
    checkers = Checkers()
    assert checkers.is_valid_move((5, 0), (4, 1)) == True
    assert checkers.is_valid_move((5, 0), (3, 2)) == False


def test_invalid_move():
    checkers = Checkers()
    assert checkers.is_valid_move((4, 1), (3, 2)) == False
    assert checkers.is_valid_move((5, 0), (6, 1)) == False


def test_move_piece():
    checkers = Checkers()
    assert checkers.move_piece((5, 0), (4, 1)) == True
    assert checkers.board[5, 0] == 0
    assert checkers.board[4, 1] == -1


def test_capture_move():
    checkers = Checkers()
    checkers.board[5, 0] = 0
    checkers.board[4, 1] = -1
    checkers.board[3, 2] = 1
    assert checkers.move_piece((4, 1), (2, 3)) == True
    assert checkers.board[4, 1] == 0
    assert checkers.board[2, 3] == -1
    assert checkers.board[3, 2] == 0


def test_king_promotion():
    checkers = Checkers()
    checkers.board[5, 0] = 0
    checkers.board[1, 2] = -1
    assert checkers.move_piece((1, 2), (0, 3)) == True
    assert checkers.board[0, 3] == -2


def test_is_valid_regular_move():
    checkers = Checkers()
    checkers.board[4, 1] = -1
    assert checkers.is_valid_regular_move((4, 1), (3, 2), -1) == True
    assert checkers.is_valid_regular_move((4, 1), (3, 0), -1) == True
    assert checkers.is_valid_regular_move((4, 1), (2, 3), -1) == False


def test_is_valid_capture():
    checkers = Checkers()
    checkers.board[5, 0] = 0
    checkers.board[4, 1] = -1
    checkers.board[3, 2] = 1
    assert checkers.is_valid_capture((4, 1), (2, 3), -1) == True
    assert checkers.is_valid_capture((4, 1), (3, 2), -1) == False


def test_is_valid_king_move():
    checkers = Checkers()
    checkers.board[4, 1] = -2
    assert checkers.is_valid_king_move((4, 1), (3, 2)) == True
    assert checkers.is_valid_king_move((4, 1), (3, 0)) == True


def test_perform_regular_move():
    checkers = Checkers()
    checkers.board[4, 1] = -1
    checkers.perform_regular_move((4, 1), (3, 0))
    assert checkers.board[4, 1] == 0
    assert checkers.board[3, 0] == -1


def test_perform_capture():
    checkers = Checkers()
    checkers.board[5, 0] = 0
    checkers.board[4, 1] = -1
    checkers.board[3, 2] = 1
    checkers.perform_capture((4, 1), (2, 3))
    assert checkers.board[3, 2] == 0


def test_promote_to_king():
    checkers = Checkers(init_board=False)
    checkers.board = np.zeros((8, 8), dtype=int)
    checkers.board[0, 1] = -1
    checkers.promote_to_king((0, 1), -1)
    assert checkers.board[0, 1] == -2

    checkers.board[7, 0] = 1
    checkers.promote_to_king((7, 0), 1)
    assert checkers.board[7, 0] == 2
