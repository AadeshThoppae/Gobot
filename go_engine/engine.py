"""
GoEngine: public interface for the test harness.
All coordinates are 0-indexed (row, col).
"""

from .board import Board, BLACK, WHITE
from .rules import is_legal, get_legal_moves
from .scoring import calculate_score


class GoEngine:
    def __init__(self):
        self.board = Board()
        self.game_over = False
        self.winner = None

    # ------------------------------------------------------------------
    # Mandatory harness interface
    # ------------------------------------------------------------------

    def initialize(self):
        """Reset to a fresh 9x9 game."""
        self.board = Board()
        self.game_over = False
        self.winner = None

    def place_stone(self, row: int, col: int) -> bool:
        """
        Place a stone for the current player.
        Returns True on success, False if the move is illegal or game is over.
        """
        if self.game_over:
            return False
        if not is_legal(self.board, row, col):
            return False
        self.board.place_stone(row, col)
        return True

    def is_legal(self, row: int, col: int) -> bool:
        """Return whether (row, col) is a legal move for the current player."""
        if self.game_over:
            return False
        return is_legal(self.board, row, col)

    def get_board_state(self):
        """
        Return the current board as a 9x9 list-of-lists.
        0 = empty, 1 = Black, 2 = White.
        """
        return [row[:] for row in self.board.grid]

    def calculate_score(self) -> dict:
        """Compute and return final scores (Chinese scoring + komi)."""
        return calculate_score(self.board)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def get_current_player(self) -> str:
        return "black" if self.board.current_player == BLACK else "white"

    def get_captured_counts(self) -> dict:
        return {
            "black": self.board.captured[BLACK],
            "white": self.board.captured[WHITE],
        }

    def get_legal_moves(self):
        return get_legal_moves(self.board)

    def resign(self):
        """Current player concedes (equivalent to passing per assignment rules)."""
        self.game_over = True
        opponent = WHITE if self.board.current_player == BLACK else BLACK
        self.winner = "white" if opponent == WHITE else "black"
