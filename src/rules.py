import numpy as np

class Checkers:
    def __init__(self, init_board=True):
        self.board = np.zeros((8, 8), dtype=int)
        if init_board:
            self.init_board()


    def init_board(self):
        for i in range(8):
            for j in range(8):
                if (i + j) % 2 == 1:
                    if i < 3:
                        self.board[i, j] = 1  # Player 1's regular pieces
                    elif i > 4:
                        self.board[i, j] = -1  # Player 2's regular pieces

    def move_piece(self, from_pos, to_pos):
        piece = self.board[from_pos]
        if self.is_valid_regular_move(from_pos, to_pos, piece):
            self.board[from_pos] = 0
            self.board[to_pos] = piece
            self.promote_to_king(to_pos, piece)
            return True
        elif self.is_valid_capture(from_pos, to_pos, piece):
            self.board[from_pos] = 0
            self.board[to_pos] = piece
            self.promote_to_king(to_pos, piece)
            captured_pos = self.get_captured_position(from_pos, to_pos)
            self.board[captured_pos] = 0
            return True
        return False


    def get_captured_position(self, from_pos, to_pos):
        row_diff = (to_pos[0] - from_pos[0]) // 2
        col_diff = (to_pos[1] - from_pos[1]) // 2
        return from_pos[0] + row_diff, from_pos[1] + col_diff

    def perform_regular_move(self, from_pos, to_pos):
        piece = self.board[from_pos]
        self.board[to_pos] = piece
        self.board[from_pos] = 0

    def perform_capture(self, from_pos, to_pos):
        captured_piece_pos = ((from_pos[0] + to_pos[0]) // 2, (from_pos[1] + to_pos[1]) // 2)
        self.board[captured_piece_pos] = 0

    def promote_to_king(self, to_pos, piece):
        if (piece == -1 and to_pos[0] == 0) or (piece == 1 and to_pos[0] == 7):
            self.board[to_pos] *= 2

    def is_valid_move(self, from_pos, to_pos):
        piece = self.board[from_pos]
        if piece == 0:
            return False

        if self.board[to_pos] != 0:
            return False

        if piece == 1 or piece == -1:  # Regular pieces
            if not (self.is_valid_regular_move(from_pos, to_pos, piece) or self.is_valid_capture(from_pos, to_pos, piece)):
                return False
        else:  # Kings
            if not self.is_valid_king_move(from_pos, to_pos):
                return False

        return True

    def is_valid_regular_move(self, from_pos, to_pos, piece):
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        return abs(dy) == 1 and dx == piece

    def is_valid_capture(self, from_pos, to_pos, piece):
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]

        if abs(dy) != 2 or abs(dx) != 2:
            return False

        captured_piece_pos = ((from_pos[0] + to_pos[0]) // 2, (from_pos[1] + to_pos[1]) // 2)
        return self.board[captured_piece_pos] == -piece or self.board[captured_piece_pos] == -2 * piece

    def is_valid_king_move(self, from_pos, to_pos):
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        return abs(dx) == 1 and abs(dy) == 1

    def is_game_over(self):
        p1_pieces = (self.board == 1).sum() + (self.board == 2).sum()
        p2_pieces = (self.board == -1).sum() + (self.board == -2).sum()

        return p1_pieces == 0 or p2_pieces == 0

    def __str__(self):
        return str(self.board)
