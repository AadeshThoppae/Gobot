"""
Stress test: play many random games through the public engine API
and verify no crashes, no illegal states reached, scoring works.
"""
import sys
sys.path.insert(0, '/home/claude')
import random

from go_engine import GoEngine
from go_engine.rules import get_legal_moves


def play_random_game(seed, max_moves=200):
    random.seed(seed)
    e = GoEngine()
    e.initialize()

    for _ in range(max_moves):
        legal = e.get_legal_moves()
        if not legal:
            break
        r, c = random.choice(legal)
        ok = e.place_stone(r, c)
        assert ok, f"place_stone({r},{c}) failed though it was in legal list"

    # Score must run without error
    score = e.calculate_score()
    assert "winner" in score
    assert score["winner"] in ("black", "white")
    return True


def test_many_random_games(n=100):
    for seed in range(n):
        play_random_game(seed)
    print(f"PASS: {n} random games completed without errors")


def test_ai_game():
    """One AI vs AI game — verify it terminates cleanly."""
    from go_engine.ai import get_ai_move
    e = GoEngine()
    e.initialize()
    for move_num in range(100):
        move = get_ai_move(e.board, time_limit=0.3)
        if move is None:
            e.resign()
            break
        ok = e.place_stone(*move)
        assert ok, f"AI returned illegal move {move} at turn {move_num}"
    score = e.calculate_score()
    print(f"PASS: AI self-play game ended; winner={score['winner']}")


if __name__ == "__main__":
    test_many_random_games(50)
    test_ai_game()
    print("\nAll stress tests passed.")
