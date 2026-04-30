"""
Monte Carlo Tree Search for 9x9 Go.


Strategy and design decisions:

1. MCTS with UCB1 tree policy. UCB1 balances exploitation (child win
   rate) against exploration (less-visited children).

2. Fast incremental board representation (see board.py): groups and
   liberties are maintained via union-find, so liberty queries are O(1) instead of O(group size).

3. Rollout policy (biased, not uniform random):
      (a) If any move captures an opponent group, take it.
      (b) If any friendly group is in atari, try to save it.
      (c) Otherwise pick a random legal move, skipping real eyes
          (with false-eye detection via diagonal check) and skipping
          self-atari moves when alternatives exist.

4. Move priors: every cell gets a fixed prior score based on its
   position. Edge cells are penalized; the 3rd/4th lines and star
   points are rewarded. Priors bias UCB1 at low visit counts, so
   MCTS spends time on plausible moves rather than uniformly across
   all 81 root moves.

4a. TACTICAL FILTER AT THE ROOT. Positional priors alone aren't
    enough on a small time budget: a move might have a great position
    prior but also lose a stone next turn, and MCTS can't see that
    from the 5-10 rollouts each root move gets. So before search
    starts, the root's candidate list is pruned:
      - Self-atari moves are dropped (unless they also capture).
      - If any of our groups is in atari, ONLY moves that save it or
        capture an opponent group are considered.
      - Capture moves get an in-position prior boost to 0.95.

5. Passing policy: the AI only returns None when there are literally
   zero legal moves. It never passes voluntarily — when it has any
   legal move, it plays it.

Testing;
The rules engine was verified independently with:
  - test_correctness.py:   targeted tests for ko, suicide, captures,
    territory, bounds, and the public engine interface.
  - test_crosscheck_final.py:   50 random 150-move games where every
    placement and every legality query is cross-checked against an
    independent pure-BFS reference implementation.
  - test_stress.py:   50 random full games through the public engine
    API, verifying no crashes, no illegal states, clean scoring.
"""

import math
import random
import time

from .board import (
    BLACK, EMPTY, SIZE, N, OPPONENT, NEIGHBORS,
)
from .scoring import calculate_score

TIME_LIMIT = 5.0
C_UCB = 1.4
MAX_ROLLOUT_MOVES = 140  # hard cap on playout length

# Cells in the 3rd-line rectangle (rows 2-6, cols 2-6) — used for opening bias
_OPENING_CELLS = frozenset(
    r * SIZE + c for r in range(2, 7) for c in range(2, 7)
)
_STAR_POINTS = frozenset({
    2 * SIZE + 2, 2 * SIZE + 6, 4 * SIZE + 4, 6 * SIZE + 2, 6 * SIZE + 6
})

# Rollout helpers (operate on flat indices for speed)

def _is_eye_idx(board, i, player):
    """True if index i is a real eye for `player`.

    A real eye requires (a) all four orthogonal neighbors are friendly stones
    or off-board, and (b) at least 3 of the 4 diagonal positions are friendly
    or off-board. The diagonal check filters out "false eyes" — points that
    look like eyes but the opponent can take away.
    """
    if board.color[i] != EMPTY:
        return False
    for nj in NEIGHBORS[i]:
        if board.color[nj] != player:
            return False
    r, c = divmod(i, SIZE)
    diag_friendly = 0
    for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < SIZE and 0 <= nc < SIZE:
            if board.color[nr * SIZE + nc] == player:
                diag_friendly += 1
        else:
            # Off-board diagonals count as friendly (corner/edge eyes are real)
            diag_friendly += 1
    return diag_friendly >= 3


def _is_self_atari(board, i, player):
    """Approximate check: would playing at i leave the resulting group with
    1 or fewer liberties?

    We count the would-be liberties of the merged group without actually
    placing the stone. This is an approximation (it doesn't perfectly
    track liberties freed by captures) but it's good enough for a rollout
    heuristic and avoids the cost of a full clone.
    """
    opp = OPPONENT[player]
    libs = set()
    merged_roots = set()
    for nj in NEIGHBORS[i]:
        cn = board.color[nj]
        if cn == EMPTY:
            libs.add(nj)
        elif cn == player:
            root = board.find(nj)
            if root not in merged_roots:
                merged_roots.add(root)
                libs.update(board.libs[root])
        else:
            # Capturing an opponent group saves us — not self-atari
            root = board.find(nj)
            if len(board.libs[root]) == 1:
                return False
    libs.discard(i)
    return len(libs) <= 1


def _rollout_policy(board, move_num):
    """Pick a move for a rollout. Returns a flat index, or None to pass.

    Avoids the cost of building a full legal-moves list for every step:
       - Tactical priorities (captures, saves) iterate stones, not cells.
       - The fallback shuffles and scans empty cells, returning the
         first reasonable one.
    """
    player = board.current_player

    # Priority 1: capture opponent stones
    caps = _moves_that_capture(board, player)
    if caps:
        return random.choice(list(caps)) if len(caps) > 1 else next(iter(caps))

    # Priority 2: save our own group from atari
    saves = _atari_saves_fast(board, player)
    if saves:
        return random.choice(list(saves)) if len(saves) > 1 else next(iter(saves))

    # Priority 3: random legal non-eye, non-self-atari move
    empty = [i for i in range(N) if board.color[i] == EMPTY]
    if not empty:
        return None

    random.shuffle(empty)
    opening = move_num < 10
    fallback = None  # last-resort move if nothing better is found

    for i in empty:
        if _is_eye_idx(board, i, player):
            continue
        r, c = divmod(i, SIZE)
        if not board.is_legal(r, c, player):
            continue
        if _is_self_atari(board, i, player):
            if fallback is None:
                fallback = i
            continue
        # Soft opening bias: skip edge cells half the time during the opening
        if opening and i not in _OPENING_CELLS and random.random() < 0.5:
            if fallback is None:
                fallback = i
            continue
        return i

    return fallback


def _atari_saves_fast(board, player):
    """Approximate version of _moves_that_save_atari for use inside rollouts.

    Doesn't clone the board; just checks whether playing the group's last
    liberty would gain at least one new liberty (via empty neighbors,
    captures, or merging with another friendly group).
    """
    saves = set()
    seen_roots = set()
    for i in range(N):
        if board.color[i] != player:
            continue
        root = board.find(i)
        if root in seen_roots:
            continue
        seen_roots.add(root)
        libs = board.libs[root]
        if len(libs) != 1:
            continue
        lib_idx = next(iter(libs))
        r, c = divmod(lib_idx, SIZE)
        if not board.is_legal(r, c, player):
            continue
        # Estimate liberties gained by playing at lib_idx
        gained = 0
        opp = OPPONENT[player]
        for nj in NEIGHBORS[lib_idx]:
            cn = board.color[nj]
            if cn == EMPTY:
                gained += 1
            elif cn == opp:
                nroot = board.find(nj)
                if len(board.libs[nroot]) == 1:
                    gained += board.size[nroot] + 1   # captured stones become liberties
            elif cn == player:
                nroot = board.find(nj)
                if nroot != root:
                    gained += len(board.libs[nroot]) - 1
        if gained >= 1:
            saves.add(lib_idx)
    return saves


def _rollout(board):
    """Play out from `board` using the biased rollout policy.

    Returns 1.0 if Black wins by area scoring, 0.0 if White wins.
    """
    sim = board.clone()
    passes = 0
    for move_num in range(MAX_ROLLOUT_MOVES):
        move = _rollout_policy(sim, move_num)
        if move is None:
            passes += 1
            if passes >= 2:
                break
            sim.current_player = OPPONENT[sim.current_player]
            sim.previous_hash = None  # passing does not create a ko situation
        else:
            passes = 0
            sim.place_stone(*divmod(move, SIZE))

    return 1.0 if calculate_score(sim)["winner"] == "black" else 0.0

# Move priors
# A wide range (0.05 to 0.85) is important on 9x9 because the time budget
# only supports a few hundred MCTS iterations, so ties among equally-visited
# children are common and the prior is the best tiebreaker.
_MOVE_PRIOR = [0.0] * N
for _i in range(N):
    _r, _c = divmod(_i, SIZE)
    _edge_dist = min(_r, _c, SIZE - 1 - _r, SIZE - 1 - _c)
    if _edge_dist == 0:
        _MOVE_PRIOR[_i] = 0.05   # first line — almost never an opening move
    elif _edge_dist == 1:
        _MOVE_PRIOR[_i] = 0.25   # second line
    elif _edge_dist == 2:
        _MOVE_PRIOR[_i] = 0.70   # third line — classic strong move on 9x9
    else:
        _MOVE_PRIOR[_i] = 0.65   # fourth line / center
for _sp in _STAR_POINTS:
    _MOVE_PRIOR[_sp] = 0.85


# MCTS node
class Node:
    __slots__ = ("move", "player_to_move", "wins", "visits", "children",
                 "untried", "parent", "prior")

    def __init__(self, move, player_to_move, untried, parent=None, prior=0.5):
        self.move = move                      # flat index, or None for the root
        self.player_to_move = player_to_move  # whose turn it is AT this node
        self.wins = 0.0                       # wins for the player who moved INTO this node
        self.visits = 0
        self.children = []
        self.untried = untried                # flat indices not yet expanded
        self.parent = parent
        self.prior = prior                    # in [0, 1], a virtual prior win rate

    def ucb1_score(self, parent_visits, c=C_UCB):
        """Standard UCB1 plus a decaying prior bonus."""
        if self.visits == 0:
            # Order unvisited children by prior + a flat exploration term
            return self.prior + c * math.sqrt(math.log(parent_visits + 1))
        exploit = self.wins / self.visits
        explore = c * math.sqrt(math.log(parent_visits) / self.visits)
        bias = self.prior / (1.0 + self.visits)   # decays as visits grow
        return exploit + explore + bias

    def select_child(self):
        return max(self.children, key=lambda n: n.ucb1_score(self.visits))

    def most_visited(self):
        # Tie-break by prior so under low-budget conditions we still
        # pick the move with better positional heuristics.
        return max(self.children, key=lambda n: (n.visits, n.prior))


# Move generation helpers
def _legal_move_indices(board, player=None):
    if player is None:
        player = board.current_player
    return [i for i in range(N) if board.is_legal(*divmod(i, SIZE), player)]


def _legal_non_eye_indices(board, player=None):
    if player is None:
        player = board.current_player
    result = []
    for i in range(N):
        if board.color[i] != EMPTY:
            continue
        if _is_eye_idx(board, i, player):
            continue
        if board.is_legal(*divmod(i, SIZE), player):
            result.append(i)
    return result


def _would_be_self_atari_after(board, i, player):
    """Exact (cloning) version of self-atari check.

    Used at the root, where correctness matters more than speed.
    """
    sim = board.clone()
    r, c = divmod(i, SIZE)
    sim.place_stone(r, c)
    if sim.color[i] == EMPTY:
        # Stone got captured; treat as bad
        return True
    return len(sim.libs[sim.find(i)]) <= 1


def _our_groups_in_atari(board, player):
    """Yield (root_idx, liberty_idx) for each friendly group with 1 liberty."""
    seen = set()
    for i in range(N):
        if board.color[i] != player:
            continue
        root = board.find(i)
        if root in seen:
            continue
        seen.add(root)
        if len(board.libs[root]) == 1:
            yield root, next(iter(board.libs[root]))


def _moves_that_save_atari(board, player=None):
    """Exact version: returns moves that leave at least one of our ataried
    groups with 2+ liberties (or captures something that frees them).

    Differs from _atari_saves_fast in that it actually clones the board
    and confirms the save. Used at the root where correctness matters.
    """
    if player is None:
        player = board.current_player

    saves = set()
    opp = OPPONENT[player]

    for root, lib_idx in _our_groups_in_atari(board, player):
        # Option A: play the liberty itself and verify the group survives
        r, c = divmod(lib_idx, SIZE)
        if board.is_legal(r, c, player):
            sim = board.clone()
            sim.place_stone(r, c)
            if sim.color[lib_idx] != EMPTY and len(sim.libs[sim.find(lib_idx)]) >= 2:
                saves.add(lib_idx)
        # Option B: capture an adjacent opponent group (frees liberties for us)
        for j in range(N):
            if board.color[j] != player or board.find(j) != root:
                continue
            for nj in NEIGHBORS[j]:
                if board.color[nj] != opp:
                    continue
                opp_root = board.find(nj)
                if len(board.libs[opp_root]) == 1:
                    opp_lib = next(iter(board.libs[opp_root]))
                    r2, c2 = divmod(opp_lib, SIZE)
                    if board.is_legal(r2, c2, player):
                        saves.add(opp_lib)
    return saves


def _moves_that_capture(board, player=None):
    """Return the set of move indices that capture at least one opponent group."""
    if player is None:
        player = board.current_player
    opp = OPPONENT[player]
    moves = set()
    seen_roots = set()
    for i in range(N):
        if board.color[i] != opp:
            continue
        root = board.find(i)
        if root in seen_roots:
            continue
        seen_roots.add(root)
        libs = board.libs[root]
        if len(libs) == 1:
            lib = next(iter(libs))
            if board.is_legal(*divmod(lib, SIZE), player):
                moves.add(lib)
    return moves


def _tactical_candidates(board, player=None):
    """Return tactically-reasonable candidate moves for the root.

    Priority ladder:
      1. If any of our groups is in atari, restrict candidates to
         moves that save it (or capture an opponent group, which
         can also save it). Ignoring atari loses material.
      2. Otherwise: all legal non-eye moves except self-atari moves.
         Captures are always allowed even if they happen to be
         self-atari (throw-in tactics).

    This shrinks the root candidate set to ~10-30 meaningful moves
    so MCTS iterations are spent on real decisions, not blunders.
    """
    if player is None:
        player = board.current_player

    captures = _moves_that_capture(board, player)
    saves = _moves_that_save_atari(board, player)

    # If any of our groups is in atari, only saves and captures are sensible.
    if any(True for _ in _our_groups_in_atari(board, player)):
        urgent = saves | captures
        if urgent:
            return sorted(urgent)
        # No save available — group is dead. Fall through to normal moves.

    candidates = set(captures)  # captures always allowed
    for i in range(N):
        if board.color[i] != EMPTY:
            continue
        if _is_eye_idx(board, i, player):
            continue
        r, c = divmod(i, SIZE)
        if not board.is_legal(r, c, player):
            continue
        if i in captures:
            continue   # already in candidates
        if _would_be_self_atari_after(board, i, player):
            continue
        candidates.add(i)

    if not candidates:
        # Really constrained — fall back to anything legal
        return _legal_non_eye_indices(board, player) or _legal_move_indices(board, player)

    return sorted(candidates)


# MCTS main loop
def _run_mcts(root_board, time_limit):
    """Run MCTS until time_limit elapses; return the root node (or None)."""
    untried = _tactical_candidates(root_board)
    if not untried:
        untried = _legal_move_indices(root_board)
    if not untried:
        return None

    # Position specific prior: boost captures and atari-saves above the purely positional baseline. 
    # Used at the root only.
    captures = _moves_that_capture(root_board)
    saves = _moves_that_save_atari(root_board)

    def root_prior(m):
        p = _MOVE_PRIOR[m]
        if m in captures:
            p = max(p, 0.95)
        if m in saves:
            p = max(p, 0.90)
        return p

    root = Node(move=None,
                player_to_move=root_board.current_player,
                untried=untried,
                parent=None,
                prior=0.5)

    deadline = time.time() + time_limit
    while time.time() < deadline:
        node = root
        sim = root_board.clone()

        # --- SELECTION ---
        while not node.untried and node.children:
            node = node.select_child()
            sim.place_stone(*divmod(node.move, SIZE))

        # --- EXPANSION ---
        if node.untried:
            # Pick the untried move with the highest prior. At root we use tactical priors (which know about captures/saves)
            # deeper we use positional priors only.
            at_root = node is root
            best_idx = 0
            best_prior = -1.0
            for k, m in enumerate(node.untried):
                p = root_prior(m) if at_root else _MOVE_PRIOR[m]
                if p > best_prior:
                    best_prior = p
                    best_idx = k
            move = node.untried.pop(best_idx)
            sim.place_stone(*divmod(move, SIZE))
            child_untried = _legal_non_eye_indices(sim) or _legal_move_indices(sim)
            child = Node(
                move=move,
                player_to_move=sim.current_player,
                untried=child_untried,
                parent=node,
                prior=best_prior if at_root else _MOVE_PRIOR[move],
            )
            node.children.append(child)
            node = child

        # --- SIMULATION ---
        result = _rollout(sim)   # 1.0 if Black wins, 0.0 if White wins

        # --- BACKPROPAGATION ---
        # cur.wins counts wins for the player who moved INTO cur. The root
        # has no such player; we store from Black's perspective there.
        cur = node
        while cur is not None:
            cur.visits += 1
            if cur.parent is None:
                cur.wins += result
            else:
                mover = OPPONENT[cur.player_to_move]
                cur.wins += result if mover == BLACK else (1.0 - result)
            cur = cur.parent

    return root


# Public entry point
def get_ai_move(board, time_limit=TIME_LIMIT):
    """Choose the best move for board.current_player.

    Returns (row, col), or None if there is literally no legal move.

    Flow:
      1. If there are zero legal moves, resign (return None).
      2. Compute tactical candidates (handles urgent atari, drops self-atari).
      3. If exactly one candidate, play it without searching.
      4. Run MCTS over the candidates and return the most-visited move.
    """
    legal_all = _legal_move_indices(board)
    if not legal_all:
        return None

    candidates = _tactical_candidates(board)
    if not candidates:
        candidates = _legal_non_eye_indices(board) or legal_all

    if len(candidates) == 1:
        return divmod(candidates[0], SIZE)

    root = _run_mcts(board, time_limit)
    if root is None or not root.children:
        # MCTS produced nothing (e.g. budget too short) — pick by prior alone
        captures = _moves_that_capture(board)
        saves = _moves_that_save_atari(board)
        def fallback_prior(m):
            p = _MOVE_PRIOR[m]
            if m in captures:
                p = max(p, 0.95)
            if m in saves:
                p = max(p, 0.90)
            return p
        return divmod(max(candidates, key=fallback_prior), SIZE)

    return divmod(root.most_visited().move, SIZE)
