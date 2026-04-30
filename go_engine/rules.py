"""
Move validation.

This module used to do its own Ko/suicide checks by cloning the board
on every query, which was the single biggest hot spot in the old code
(most rollouts spent 80%+ of their time here). We now delegate to the
Board's incremental `is_legal` method, which does the same checks
without cloning by looking at group liberty counts directly.

The public function signatures are unchanged so the rest of the engine,
the server, and the test harness all still work.
"""

from .board import Board, SIZE


def is_legal(board: Board, r: int, c: int, player: int = None) -> bool:
    """
    Return True if placing a stone at (r,c) is legal for `player`
    (defaults to board.current_player).

    Checks: in-bounds, empty cell, not suicide, not Ko.
    """
    return board.is_legal(r, c, player)


def get_legal_moves(board: Board, player: int = None):
    """Return the list of all legal (r, c) moves for `player`."""
    if player is None:
        player = board.current_player
    moves = []
    for r in range(SIZE):
        for c in range(SIZE):
            if board.is_legal(r, c, player):
                moves.append((r, c))
    return moves
