"""Final cross-check: new Board against a pure BFS reference board."""
import sys
sys.path.insert(0, '/home/claude')
import random

from go_engine.board import Board, SIZE, BLACK, WHITE, EMPTY, OPPONENT

class RefBoard:
    """Pure BFS implementation for cross-checking."""
    def __init__(self):
        self.grid = [[EMPTY]*SIZE for _ in range(SIZE)]
        self.current_player = BLACK
        self.prev = None
        self.captured = {BLACK: 0, WHITE: 0}

    def neighbors(self, r, c):
        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
            nr, nc = r+dr, c+dc
            if 0 <= nr < SIZE and 0 <= nc < SIZE:
                yield nr, nc

    def group(self, r, c):
        color = self.grid[r][c]
        if color == EMPTY: return frozenset()
        stack = [(r,c)]; seen=set()
        while stack:
            x = stack.pop()
            if x in seen: continue
            seen.add(x)
            for nr,nc in self.neighbors(*x):
                if (nr,nc) not in seen and self.grid[nr][nc] == color:
                    stack.append((nr,nc))
        return frozenset(seen)

    def liberties(self, grp):
        libs = set()
        for r,c in grp:
            for nr,nc in self.neighbors(r,c):
                if self.grid[nr][nc] == EMPTY:
                    libs.add((nr,nc))
        return libs

    def snapshot(self):
        return tuple(tuple(row) for row in self.grid)

    def is_legal(self, r, c, player=None):
        if player is None: player = self.current_player
        if not (0 <= r < SIZE and 0 <= c < SIZE): return False
        if self.grid[r][c] != EMPTY: return False
        opp = OPPONENT[player]
        # Simulate
        saved = [row[:] for row in self.grid]
        self.grid[r][c] = player
        captured_pts = []
        for nr,nc in self.neighbors(r,c):
            if self.grid[nr][nc] == opp:
                grp = self.group(nr,nc)
                if not self.liberties(grp):
                    for gr,gc in grp:
                        self.grid[gr][gc] = EMPTY
                        captured_pts.append((gr,gc))
        own = self.group(r,c)
        if not self.liberties(own):
            self.grid = saved
            return False
        if self.prev is not None and self.snapshot() == self.prev:
            self.grid = saved
            return False
        self.grid = saved
        return True

    def place(self, r, c):
        player = self.current_player
        opp = OPPONENT[player]
        self.prev = self.snapshot()
        self.grid[r][c] = player
        for nr,nc in self.neighbors(r,c):
            if self.grid[nr][nc] == opp:
                grp = self.group(nr,nc)
                if not self.liberties(grp):
                    for gr,gc in grp:
                        self.grid[gr][gc] = EMPTY
                    self.captured[player] += len(grp)
        self.current_player = opp


def run(seed):
    random.seed(seed)
    ref = RefBoard()
    fast = Board()
    for m in range(150):
        cells = list(range(81))
        random.shuffle(cells)
        placed = False
        for idx in cells:
            r, c = divmod(idx, SIZE)
            rl = ref.is_legal(r, c)
            fl = fast.is_legal(r, c)
            if rl != fl:
                return False, f"legal mismatch move {m} ({r},{c}): ref={rl} fast={fl}"
            if rl:
                ref.place(r, c)
                fast.place_stone(r, c)
                # Verify boards match
                for rr in range(SIZE):
                    for cc in range(SIZE):
                        if ref.grid[rr][cc] != fast.color[rr*SIZE+cc]:
                            return False, f"board mismatch at ({rr},{cc})"
                if ref.captured != fast.captured:
                    return False, f"capture count mismatch: ref={ref.captured} fast={fast.captured}"
                placed = True
                break
        if not placed:
            break
    return True, ""


n = 50
for s in range(n):
    ok, msg = run(s)
    if not ok:
        print(f"FAIL seed {s}: {msg}")
        sys.exit(1)
print(f"PASS: {n} cross-check games match reference implementation")
