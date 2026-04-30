"""
Board representation and low-level stone operations for a 9x9 Go board.

Cell values: 0 = empty, 1 = Black, 2 = White.
Coordinates are (row, col), both 0-indexed; row 0 is the top.

State:
    color[i]           flat array of 81 cell values (0/1/2)
    grid               2D view of color (lazy property)
    parent[i]          union-find parent pointer for cell i
    size[root]         number of stones in the group rooted at `root`
    libs[root]         set of empty-cell indices adjacent to that group
    captured           dict mapping player -> stones captured BY that player
    current_player     BLACK or WHITE, whose turn it is now
    last_move          (r, c) of the most recent placement, or None
    previous_hash      flat-tuple snapshot of position before last move (for ko)

Methods:
    place_stone(r, c)  place for current_player; updates everything
    is_legal(r, c)     check legality (bounds, empty, suicide, ko)
    clone()            deep copy for simulation
    snapshot()         immutable flat tuple of the position

Class method:
    neighbors(r, c)    yield neighbor (r, c) within bounds
"""

EMPTY = 0
BLACK = 1
WHITE = 2
SIZE = 9
N = SIZE * SIZE  # 81

OPPONENT = {BLACK: WHITE, WHITE: BLACK}


# Precompute a list of neighbor flat-indices for every cell. Building these
# tuples once at import is meaningfully faster than re-deriving them at every
# placement during MCTS rollouts.
_NEIGHBORS = [None] * N
for _r in range(SIZE):
    for _c in range(SIZE):
        _idx = _r * SIZE + _c
        _nbrs = []
        if _r > 0:        _nbrs.append((_r - 1) * SIZE + _c)
        if _r < SIZE - 1: _nbrs.append((_r + 1) * SIZE + _c)
        if _c > 0:        _nbrs.append(_r * SIZE + (_c - 1))
        if _c < SIZE - 1: _nbrs.append(_r * SIZE + (_c + 1))
        _NEIGHBORS[_idx] = tuple(_nbrs)

NEIGHBORS = _NEIGHBORS


def rc_to_idx(r, c):
    return r * SIZE + c


def idx_to_rc(i):
    return divmod(i, SIZE)


class Board:
    __slots__ = (
        "color",
        "parent",
        "size",
        "libs",
        "captured",
        "current_player",
        "previous_hash",
        "last_move",
        "_grid_cache",
        "_grid_cache_dirty",
    )

    def __init__(self):
        self.color = [EMPTY] * N
        self.parent = list(range(N))
        self.size = [1] * N
        self.libs = {}  # only contains entries for cells that are stones
        self.captured = {BLACK: 0, WHITE: 0}
        self.current_player = BLACK
        self.previous_hash = None
        self.last_move = None
        self._grid_cache = None
        self._grid_cache_dirty = True

    # 2D grid view (lazy — only built when something asks for it)

    @property
    def grid(self):
        if self._grid_cache_dirty or self._grid_cache is None:
            self._grid_cache = [
                [self.color[r * SIZE + c] for c in range(SIZE)]
                for r in range(SIZE)
            ]
            self._grid_cache_dirty = False
        return self._grid_cache

    def _invalidate_grid(self):
        self._grid_cache_dirty = True

    # Snapshots and cloning

    def snapshot(self):
        """Immutable snapshot of the board (used for ko comparison)."""
        return tuple(self.color)

    def clone(self):
        b = Board.__new__(Board)
        b.color = self.color[:]
        b.parent = self.parent[:]
        b.size = self.size[:]
        b.libs = {k: set(v) for k, v in self.libs.items()}
        b.captured = {BLACK: self.captured[BLACK], WHITE: self.captured[WHITE]}
        b.current_player = self.current_player
        b.previous_hash = self.previous_hash
        b.last_move = self.last_move
        b._grid_cache = None
        b._grid_cache_dirty = True
        return b

    # Adjacency

    @staticmethod
    def neighbors(r, c):
        """Yield (nr, nc) neighbors of (r, c) within board bounds."""
        for nidx in NEIGHBORS[r * SIZE + c]:
            yield divmod(nidx, SIZE)

    # Union-find

    def find(self, i):
        p = self.parent
        # Find root
        root = i
        while p[root] != root:
            root = p[root]
        # Path compression: point everything along the way directly at root
        while p[i] != root:
            nxt = p[i]
            p[i] = root
            i = nxt
        return root

    def union(self, a, b):
        """Union the groups containing stones a and b. Must be the same color."""
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return ra
        # Union by size: smaller group hangs off the larger one
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        self.libs[ra].update(self.libs[rb])
        del self.libs[rb]
        return ra

    # Legality check (no cloning — operates on incremental state)


    def is_legal(self, r, c, player=None):
        """Return True iff placing for `player` at (r, c) is legal.

        Checks: bounds, occupancy, suicide, ko.
        """
        if player is None:
            player = self.current_player
        if not (0 <= r < SIZE and 0 <= c < SIZE):
            return False
        i = r * SIZE + c
        if self.color[i] != EMPTY:
            return False

        opponent = OPPONENT[player]

        # The stone has at least one liberty after placement if any of these
        # is true:
        #   - some neighbor is empty (immediate liberty)
        #   - some friendly neighbor's group has more than 1 liberty (we
        #     keep them after losing `i`)
        #   - some opponent group has exactly 1 liberty (which must be `i`),
        #     so capturing it gives us empty cells back as liberties.
        has_empty_neighbor = False
        friendly_extra_libs = False
        captured_roots = []

        opp_roots_touched = set()
        for nj in NEIGHBORS[i]:
            c_n = self.color[nj]
            if c_n == EMPTY:
                has_empty_neighbor = True
            elif c_n == player:
                root_n = self.find(nj)
                if len(self.libs[root_n]) > 1:
                    friendly_extra_libs = True
            else:
                root_n = self.find(nj)
                if root_n not in opp_roots_touched:
                    opp_roots_touched.add(root_n)
                    if len(self.libs[root_n]) == 1:
                        captured_roots.append(root_n)

        if not (has_empty_neighbor or friendly_extra_libs or captured_roots):
            return False  # suicide

        # Ko check: only relevant when exactly one stone is captured.
        # In that case, simulate the resulting position and compare its
        # hash to the position from before the previous move.
        if self.previous_hash is not None and captured_roots:
            total_captured = sum(self.size[root] for root in captured_roots)
            if total_captured == 1:
                new_color = self.color[:]
                new_color[i] = player
                # size==1 means the root index IS the captured stone's index
                for root in captured_roots:
                    new_color[root] = EMPTY
                if tuple(new_color) == self.previous_hash:
                    return False

        return True

    # Placement and capture

    def _remove_group_at_root(self, root, capturing_player):
        """Remove every stone in the group rooted at `root`. Update neighbors' liberty sets."""
        removed = [j for j in range(N)
                   if self.color[j] != EMPTY and self.find(j) == root]

        # Clear stones
        for j in removed:
            self.color[j] = EMPTY
            self.parent[j] = j
            self.size[j] = 1

        # The newly empty cells become liberties for any group adjacent to them
        for j in removed:
            for nj in NEIGHBORS[j]:
                if self.color[nj] != EMPTY:
                    self.libs[self.find(nj)].add(j)

        self.captured[capturing_player] += len(removed)
        if root in self.libs:
            del self.libs[root]
        self._invalidate_grid()
        return removed

    def place_stone(self, r, c):
        """Place a stone for current_player at (r, c).

        Does NOT validate legality — call is_legal first if you need to check.
        Updates all incremental data structures.
        """
        player = self.current_player
        opponent = OPPONENT[player]
        i = rc_to_idx(r, c)

        # Snapshot position before mutating, for ko detection on next move
        self.previous_hash = self.snapshot()

        # Place the stone as its own one-stone group
        self.color[i] = player
        self.parent[i] = i
        self.size[i] = 1
        self.libs[i] = {nj for nj in NEIGHBORS[i] if self.color[nj] == EMPTY}

        # Tell neighbor groups they no longer have `i` as a liberty,
        # and remember which friendly groups we'll merge with.
        opp_roots_touched = set()
        friendly_roots_touched = set()
        for nj in NEIGHBORS[i]:
            c_n = self.color[nj]
            if c_n == EMPTY:
                continue
            root_n = self.find(nj)
            self.libs[root_n].discard(i)
            if c_n == player:
                friendly_roots_touched.add(root_n)
            else:
                opp_roots_touched.add(root_n)

        # Merge with all adjacent friendly groups
        for fr in friendly_roots_touched:
            self.union(i, fr)

        # Capture any opponent groups that just lost their last liberty
        for orn in opp_roots_touched:
            if orn in self.libs and len(self.libs[orn]) == 0:
                self._remove_group_at_root(orn, player)

        self.last_move = (r, c)
        self.current_player = opponent
        self._invalidate_grid()
        return True

    # Display

    def __str__(self):
        symbols = {EMPTY: ".", BLACK: "X", WHITE: "O"}
        rows = []
        for r in range(SIZE):
            rows.append(" ".join(symbols[self.color[r * SIZE + c]] for c in range(SIZE)))
        return "\n".join(rows)
