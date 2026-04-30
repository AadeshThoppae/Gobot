"""
Chinese (area) scoring for Go.
Score = stones on board + enclosed empty territory.
White receives 2.5 komi.
"""

from .board import Board, EMPTY, BLACK, WHITE, SIZE

KOMI = 2.5


def calculate_score(board: Board) -> dict:
    """
    Compute final scores using Chinese area scoring.

    Returns:
        {
            "black": float,
            "white": float,
            "winner": "black" | "white",
            "black_stones": int,
            "white_stones": int,
            "black_territory": int,
            "white_territory": int,
        }
    """
    grid = board.grid

    # Count stones
    black_stones = sum(grid[r][c] == BLACK for r in range(SIZE) for c in range(SIZE))
    white_stones = sum(grid[r][c] == WHITE for r in range(SIZE) for c in range(SIZE))

    # Flood-fill empty regions to assign territory
    visited = [[False] * SIZE for _ in range(SIZE)]
    black_territory = 0
    white_territory = 0

    for start_r in range(SIZE):
        for start_c in range(SIZE):
            if grid[start_r][start_c] != EMPTY or visited[start_r][start_c]:
                continue

            # BFS to find the empty region and its bordering colors
            region = []
            borders = set()
            queue = [(start_r, start_c)]
            visited[start_r][start_c] = True
            while queue:
                r, c = queue.pop()
                region.append((r, c))
                for nr, nc in Board.neighbors(r, c):
                    cell = grid[nr][nc]
                    if cell == EMPTY and not visited[nr][nc]:
                        visited[nr][nc] = True
                        queue.append((nr, nc))
                    elif cell != EMPTY:
                        borders.add(cell)

            # Territory is owned only if bordered exclusively by one color
            if borders == {BLACK}:
                black_territory += len(region)
            elif borders == {WHITE}:
                white_territory += len(region)
            # contested — neither player gets it

    black_score = black_stones + black_territory
    white_score = white_stones + white_territory + KOMI

    winner = "black" if black_score > white_score else "white"

    return {
        "black": black_score,
        "white": white_score,
        "winner": winner,
        "black_stones": black_stones,
        "white_stones": white_stones,
        "black_territory": black_territory,
        "white_territory": white_territory,
    }
