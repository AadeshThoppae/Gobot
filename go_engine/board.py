"""
Board representation and low-level stone operations for a 9x9 Go board.

Cell values: 0 = empty, 1 = Black, 2 = White.
Coordinates are (row, col), both 0-indexed, row 0 = top.


State:
    color[i]           flat array of 81 cell values (0/1/2)
    grid               2D view of the above (lazy property, for compatibility)
    current_player     BLACK or WHITE
    captured           dict mapping player -> stones captured BY that player
    last_move          (r, c) of most recent move, or None
    previous_grid      2D snapshot of board before last move, for Ko

Methods:
    place_stone(r, c)  place for current_player; updates everything
    is_legal(r, c)     check legality (bounds, empty, suicide, Ko)
    clone()            deep copy for simulation
    snapshot()         immutable flat tuple for hashing / Ko comparison
    get_group(r, c)    frozenset of (r,c) in the group; compat with old API
    get_liberties(grp) frozenset of liberty (r,c); compat with old API
    in_atari(r, c)     True if the group at (r,c) has exactly one liberty

Class method:
    neighbors(r, c)    yield neighbor (r, c) within bounds
"""

EMPTY = 0
BLACK = 1
WHITE = 2
SIZE = 9
N = SIZE * SIZE  # 81

OPPONENT = {BLACK: WHITE, WHITE: BLACK}


# Precompute neighbor lists for every point (massive speedup vs generating tuples)
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

NEIGHBORS = _NEIGHBORS  # module-level constant


def rc_to_idx(r, c):
    return r * SIZE + c


def idx_to_rc(i):
    return divmod(i, SIZE)


class Board:
    __slots__ = (
        "color",       # list[int] length 81: 0/1/2
        "parent",      # union-find parents
        "size",        # group sizes (valid at roots)
        "libs",        # dict: root_idx -> set of liberty indices
        "captured",    # dict: BLACK/WHITE -> int
        "current_player",
        "previous_hash",   # hash of position before last move (for ko)
        "last_move",
        # Lazy 2D view
        "_grid_cache",
        "_grid_cache_dirty",
    )

    def __init__(self):
        self.color = [EMPTY] * N
        self.parent = list(range(N))
        self.size = [1] * N
        self.libs = {}  # only populated for stones
        self.captured = {BLACK: 0, WHITE: 0}
        self.current_player = BLACK
        self.previous_hash = None
        self.last_move = None
        self._grid_cache = None
        self._grid_cache_dirty = True


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

    # Snapshots

    def snapshot(self):
        """Immutable snapshot of the board (for Ko comparison and equality)."""
        return tuple(self.color)

    def clone(self):
        b = Board.__new__(Board)
        b.color = self.color[:]
        b.parent = self.parent[:]
        b.size = self.size[:]
        # Deep copy libs (each value is a set)
        b.libs = {k: set(v) for k, v in self.libs.items()}
        b.captured = {BLACK: self.captured[BLACK], WHITE: self.captured[WHITE]}
        b.current_player = self.current_player
        b.previous_hash = self.previous_hash
        b.last_move = self.last_move
        b._grid_cache = None
        b._grid_cache_dirty = True
        return b

    # Adjacency helpers

    @staticmethod
    def neighbors(r, c):
        """Yield (nr, nc) neighbors of (r, c). Preserved for compatibility."""
        for nidx in NEIGHBORS[r * SIZE + c]:
            yield divmod(nidx, SIZE)

    # Union-find

    def find(self, i):
        p = self.parent
        # Path compression
        root = i
        while p[root] != root:
            root = p[root]
        while p[i] != root:
            nxt = p[i]
            p[i] = root
            i = nxt
        return root

    def union(self, a, b):
        """Union groups containing a and b. a, b must be same color stones."""
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return ra
        # Merge by size (smaller into larger)
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        self.libs[ra].update(self.libs[rb])
        del self.libs[rb]
        return ra


    # Public group/liberty queries


    def get_group(self, r, c):
        """Return frozenset of (r,c) in the group at (r,c), empty if not a stone."""
        i = rc_to_idx(r, c)
        if self.color[i] == EMPTY:
            return frozenset()
        root = self.find(i)
        # Walk all stones and collect those with same root
        result = set()
        for j in range(N):
            if self.color[j] != EMPTY and self.find(j) == root:
                result.add(idx_to_rc(j))
        return frozenset(result)

    def get_liberties(self, group):
        """Return frozenset of liberty points for given group (set of (r,c))."""
        if not group:
            return frozenset()
        # Just look at one stone's root
        r, c = next(iter(group))
        root = self.find(rc_to_idx(r, c))
        return frozenset(idx_to_rc(i) for i in self.libs[root])

    def in_atari(self, r, c):
        i = rc_to_idx(r, c)
        if self.color[i] == EMPTY:
            return False
        root = self.find(i)
        return len(self.libs[root]) == 1

    # Fast internal APIs used by AI
    def liberties_at_idx(self, i):
        """Return the liberty set of the group containing stone at idx i."""
        root = self.find(i)
        return self.libs[root]

    def lib_count_at_idx(self, i):
        root = self.find(i)
        return len(self.libs[root])

    # Legality check

    def is_legal(self, r, c, player=None):
        """
        Fast legality check without cloning the board.
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

        # Check neighbors:
        # - Any EMPTY neighbor  - the stone has a liberty immediately
        # - Any friendly group with >1 liberty  - stone gets liberties by merging
        # - Any opponent group with exactly 1 liberty (which must be `i`) - opponent will be captured; creates empty liberties for us
        has_empty_neighbor = False
        friendly_extra_libs = False
        captured_roots = []  # opp groups we will capture

        opp_roots_touched = set()
        for nj in NEIGHBORS[i]:
            c_n = self.color[nj]
            if c_n == EMPTY:
                has_empty_neighbor = True
            elif c_n == player:
                root_n = self.find(nj)
                # After placing, this group's libs lose `i` but they still had other libs
                if len(self.libs[root_n]) > 1:
                    friendly_extra_libs = True
            else:  # opponent
                root_n = self.find(nj)
                if root_n not in opp_roots_touched:
                    opp_roots_touched.add(root_n)
                    # If opponent's only liberty is `i`, this capture adds liberties to us
                    if len(self.libs[root_n]) == 1:
                        captured_roots.append(root_n)

        has_liberty = has_empty_neighbor or friendly_extra_libs or bool(captured_roots)
        if not has_liberty:
            return False  # suicide

        # Ko check: classic ko = capturing exactly one stone and returning
        # to the position before the opponent's last move.
        if self.previous_hash is not None and captured_roots:
            total_captured = sum(self.size[root] for root in captured_roots)
            if total_captured == 1:
                # Simulate the resulting color[] and compare with previous_hash.
                # For the captured singleton we know the index because size==1
                # and root equals the stone itself.
                new_color = self.color[:]
                new_color[i] = player
                for root in captured_roots:
                    new_color[root] = EMPTY  # size==1 so root IS the stone
                if tuple(new_color) == self.previous_hash:
                    return False  # ko

        return True

    def _remove_group_at_root(self, root, capturing_player):
        """Remove all stones in the group rooted at `root`. Update neighbors' liberties."""
        removed = []
        for j in range(N):
            if self.color[j] != EMPTY and self.find(j) == root:
                removed.append(j)

        for j in removed:
            self.color[j] = EMPTY
            self.parent[j] = j
            self.size[j] = 1

        # For each removed stone, add its point as a liberty to neighboring
        # friendly-of-OTHER-color stone groups.
        for j in removed:
            for nj in NEIGHBORS[j]:
                if self.color[nj] != EMPTY:
                    nroot = self.find(nj)
                    self.libs[nroot].add(j)

        self.captured[capturing_player] += len(removed)
        if root in self.libs:
            del self.libs[root]

        self._invalidate_grid()
        return removed

    def place_stone(self, r, c):
        """
        Place a stone for current_player at (r,c). Does NOT validate legality.
        Updates all incremental structures.
        """
        player = self.current_player
        opponent = OPPONENT[player]
        i = rc_to_idx(r, c)

        # Remember previous position hash (for Ko)
        self.previous_hash = self.snapshot()

        # Place stone
        self.color[i] = player
        self.parent[i] = i
        self.size[i] = 1

        # Its initial liberties = empty neighbors
        own_libs = set()
        for nj in NEIGHBORS[i]:
            if self.color[nj] == EMPTY:
                own_libs.add(nj)
        self.libs[i] = own_libs

        # Neighbors of opposite color: this point is no longer a liberty for them.
        # Friendly neighbors: union. For friendly neighbors, also remove `i`
        # from their liberty set (they used to have `i` empty and adjacent).
        opp_roots_touched = set()
        friendly_roots_touched = set()
        for nj in NEIGHBORS[i]:
            c_n = self.color[nj]
            if c_n == EMPTY:
                continue
            root_n = self.find(nj)
            if c_n == player:
                friendly_roots_touched.add(root_n)
                # Remove i from friend's libs (was empty, now filled by me)
                self.libs[root_n].discard(i)
            else:
                opp_roots_touched.add(root_n)
                # Remove i from opponent's liberty set
                self.libs[root_n].discard(i)

        # Union with friendly neighbors
        my_root = i
        for fr in friendly_roots_touched:
            my_root = self.union(my_root, fr)

        # Capture opponents that now have 0 liberties
        captured_points = []
        for orn in opp_roots_touched:
            # Root may have been merged away by captures of other groups?
            # No -- we haven't captured yet. Just verify it's still a root.
            if self.color[next(j for j in range(N) if self.find(j) == orn and self.color[j] == opponent)] == opponent \
                if False else True:
                pass
            # Simpler: check directly
            if orn in self.libs and len(self.libs[orn]) == 0:
                removed = self._remove_group_at_root(orn, player)
                captured_points.extend(removed)

        # If we captured any stones, those empty points become liberties
        # for any adjacent groups (including our own). _remove_group_at_root
        # handles this, but we must also recompute our own root.
        if captured_points:
            my_root = self.find(i)
            # The libs have already been updated by _remove_group_at_root.

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

    # Compatibility: previous_grid property mirrors previous_hash as a 2D tuple
    @property
    def previous_grid(self):
        if self.previous_hash is None:
            return None
        # Return same shape as old API: tuple of tuples
        flat = self.previous_hash
        return tuple(tuple(flat[r * SIZE + c] for c in range(SIZE)) for r in range(SIZE))

    @previous_grid.setter
    def previous_grid(self, value):
        """For clone() / tests that set this directly."""
        if value is None:
            self.previous_hash = None
        else:
            # Value is a 2D tuple of tuples
            flat = tuple(value[r][c] for r in range(SIZE) for c in range(SIZE))
            self.previous_hash = flat
