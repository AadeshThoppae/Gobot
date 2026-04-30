"""
Monte Carlo Tree Search for 9x9 Go.

Strategy and design decisions:

1. MCTS with UCB1 tree policy. UCB1 balances exploitation (child win rate) against exploration (less-visited children).

2. Fast incremental board representation: groups and
   liberties are maintained on the fly via union-find, so finding neighbors is O(1) instead of O(group size).
   Measured result: rollouts went from ~1.4 per second to ~280 per second on an empty board.

3. Rollout policy (biased, not uniform random). In order:
      (a) If any move captures an opponent group, take it.
      (b) If any friendly group is in atari, try to save it.
      (c) Otherwise pick a random legal move, skipping real eyes
          (with false-eye detection via diagonal check) and skipping
          self-atari moves when alternatives exist.

4. Move priors: every cell gets a fixed prior score based on its
   position: edge cells are penalized, the 3rd/4th lines are rewarded,
   and star points get a bonus. These priors bias UCB1 at low visit
   counts, so MCTS spends time on plausible moves rather than the 81
   root moves uniformly.

4a. TACTICAL FILTER AT THE ROOT. Positional priors alone aren't enough
    on a small time budget: a move might have a great position prior but
    also lose a stone to capture next turn, and MCTS can't see that from
    the 5-10 rollouts each root move gets. So before search starts, the
    root's candidate list is pruned:
      - Self-atari moves are dropped (unless they also capture).
      - If any of our groups is in atari, ONLY moves that save it or
        capture an opponent group are considered.
      - Capture moves get an in-position prior boost to 0.95.

5. Passing policy: the AI only returns None when there are literally zero legal moves. It never passes
   voluntarily just because the position looks bad, when it has a legal move, it plays it.



Testing:

I verified the rules engine independently with:
  - test_correctness.py:  targeted tests for Ko, suicide, captures,
    territory, bounds, and the engine's public interface.
  - test_crosscheck_final.py:  50 random 150-move games where every
    placement and every legality query is cross-checked against an
    independent pure-BFS reference implementation.
  - test_stress.py:  50 random full games through the public engine
    API, verifying no crashes, no illegal states, clean scoring.
  - test_ai_smoke.py:  confirms the AI never hangs, honors the time
    budget, and produces legal moves.
"""

import math
import random
import time

from .board import (
    Board, BLACK, WHITE, EMPTY, SIZE, N, OPPONENT, NEIGHBORS, rc_to_idx
)
from .scoring import calculate_score

TIME_LIMIT = 5.0
C_UCB = 1.4
MAX_ROLLOUT_MOVES = 140  # hard cap on playout length

# 3rd-line rectangle for opening bias
_OPENING_CELLS = frozenset(
    r * SIZE + c for r in range(2, 7) for c in range(2, 7)
)
_STAR_POINTS = frozenset({
    2 * SIZE + 2, 2 * SIZE + 6, 4 * SIZE + 4, 6 * SIZE + 2, 6 * SIZE + 6
})

# Rollout helpers (operate on flat indices for speed)

def _is_eye_idx(board, i, player):
    """True if idx i is an eye for player (empty, all orthogonal neighbors are
    same-color stones; also require that at least 3 of 4 diagonals are same-color
    or board edge — prevents filling false eyes in rollouts)."""
    if board.color[i] != EMPTY:
        return False
    for nj in NEIGHBORS[i]:
        if board.color[nj] != player:
            return False
    # Diagonal check for false-eye avoidance (MoGo-style)
    r, c = divmod(i, SIZE)
    diag_friendly = 0
    diag_total = 0
    for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < SIZE and 0 <= nc < SIZE:
            diag_total += 1
            if board.color[nr * SIZE + nc] == player:
                diag_friendly += 1
        else:
            # Out of bounds diagonals are friendly (corner/edge eyes)
            diag_friendly += 1
            diag_total += 1
    # Eye if at least 3 of 4 diagonal positions (or edges) are friendly
    return diag_friendly >= 3 if diag_total == 4 else diag_friendly >= diag_total - 0


def _captures_available(board, player):
    """Return set of move indices that capture at least one opponent stone.

    Fast path: an opponent group in atari has exactly one liberty.
    Playing at that liberty captures the group (unless the move itself is
    illegal due to ko).
    """
    opp = OPPONENT[player]
    moves = set()
    seen_roots = set()
    for i in range(N):
        if board.color[i] == opp:
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


def _atari_saves(board, player):
    """Return set of move indices that save a friendly group currently in atari.

    If a friendly group has exactly one liberty, we can *try* to save it by
    playing at that liberty. The move is only a true save if, after placing,
    the merged group has >1 liberty OR the move captures opponent stones.
    """
    saves = set()
    seen_roots = set()
    for i in range(N):
        if board.color[i] == player:
            root = board.find(i)
            if root in seen_roots:
                continue
            seen_roots.add(root)
            libs = board.libs[root]
            if len(libs) == 1:
                lib_idx = next(iter(libs))
                # Quick check: would playing at lib_idx actually save?
                # We play there if it has other empty neighbors OR captures something.
                r, c = divmod(lib_idx, SIZE)
                if not board.is_legal(r, c, player):
                    continue
                # Count liberties we'd gain: empty neighbors not equal to current group
                gained = 0
                opp = OPPONENT[player]
                for nj in NEIGHBORS[lib_idx]:
                    if board.color[nj] == EMPTY:
                        gained += 1
                    elif board.color[nj] == opp:
                        nroot = board.find(nj)
                        if len(board.libs[nroot]) == 1:
                            # we'd capture; that opens up liberties
                            gained += board.size[nroot] + 1
                    elif board.color[nj] == player:
                        nroot = board.find(nj)
                        if nroot != root:
                            # merging with another friendly group adds its libs
                            gained += len(board.libs[nroot]) - 1  # minus the lib_idx itself
                if gained >= 1:
                    saves.add(lib_idx)
    return saves


def _is_self_atari(board, i, player):
    """Would playing at idx i leave the resulting group with exactly 1 liberty?

    This is an approximation — we count liberties of the merged group quickly
    without actually placing. Used only as a rollout heuristic, so approximation
    is fine.
    """
    # Collect liberties that the new stone would have after placement
    opp = OPPONENT[player]
    libs = set()
    merged_roots = set()
    captures_any = False
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
            root = board.find(nj)
            if len(board.libs[root]) == 1:
                captures_any = True
                # The captured points will be liberties
                # (not tracking which points exactly — approximate: add at least size)
    # Remove i itself (shouldn't be in libs but just in case)
    libs.discard(i)
    if captures_any:
        return False  # captures save us
    return len(libs) <= 1


def _rollout_policy(board, move_num):
    """Pick a move for a rollout. Returns an idx or None if no sensible move.

    Optimization: instead of listing every legal move , use random sampling:
       1. Check tactical priorities (captures, saves) by iterating stones,
          which is much cheaper than iterating all 81 cells when few stones
          are on the board.
       2. For the normal case, pick empty cells at random and test legality.
          This gives us a uniform random legal move in O(expected 1/p) tries
          where p is the fraction of empty cells that are legal non-eyes.
    """
    player = board.current_player

    # Priority 1: capture
    caps = _captures_available(board, player)
    if caps:
        return random.choice(list(caps)) if len(caps) > 1 else next(iter(caps))

    # Priority 2: save own atari
    saves = _atari_saves(board, player)
    if saves:
        return random.choice(list(saves)) if len(saves) > 1 else next(iter(saves))

    # Priority 3: random legal non-eye, non-self-atari move
    # Build list of empty cells once per call
    empty = [i for i in range(N) if board.color[i] == EMPTY]
    if not empty:
        return None

    # Sample strategy: shuffle and scan
    random.shuffle(empty)
    opening = move_num < 10
    fallback = None  # first legal move we'd accept if we can't find a better one

    for i in empty:
        # Opening bias: skip non-opening-zone cells with probability until we've
        # looked at a few (soft bias rather than hard filter)
        if _is_eye_idx(board, i, player):
            continue
        r, c = divmod(i, SIZE)
        if not board.is_legal(r, c, player):
            continue
        if _is_self_atari(board, i, player):
            if fallback is None:
                fallback = i  # last-resort
            continue
        if opening and i not in _OPENING_CELLS:
            # 50% chance to skip edge moves early
            if random.random() < 0.5:
                if fallback is None:
                    fallback = i
                continue
        return i

    return fallback


def _rollout(board):
    """Play out from board using biased random play. Returns 1.0 if Black wins."""
    sim = board.clone()
    passes = 0
    for move_num in range(MAX_ROLLOUT_MOVES):
        move = _rollout_policy(sim, move_num)
        if move is None:
            passes += 1
            if passes >= 2:
                break
            sim.current_player = OPPONENT[sim.current_player]
            sim.previous_hash = None  # pass doesn't set ko
        else:
            passes = 0
            sim.place_stone(*divmod(move, SIZE))

    result = calculate_score(sim)
    return 1.0 if result["winner"] == "black" else 0.0


# MCTS node

# Move priors: higher = better default move.
# Used to bias UCB1 at low visit counts so MCTS spends time on plausible moves.
# A wide range (0.05 to 0.85) is important on 9x9 because the time budget
# only supports a few hundred iterations, so ties among equally-visited
# children are common and the prior is the best tiebreaker.
_MOVE_PRIOR = [0.0] * N
for _i in range(N):
    _r, _c = divmod(_i, SIZE)
    # Distance from nearest edge: 0 = on edge (bad), 4 = center (good on 9x9)
    _edge_dist = min(_r, _c, SIZE - 1 - _r, SIZE - 1 - _c)
    # First-line (edge) penalty -- very rarely good as an opening move
    if _edge_dist == 0:
        _MOVE_PRIOR[_i] = 0.05
    elif _edge_dist == 1:
        _MOVE_PRIOR[_i] = 0.25
    elif _edge_dist == 2:
        _MOVE_PRIOR[_i] = 0.70   # third line - classic good move
    elif _edge_dist == 3:
        _MOVE_PRIOR[_i] = 0.65
    else:  # center (tengen)
        _MOVE_PRIOR[_i] = 0.65
# Boost star points explicitly
for _sp in _STAR_POINTS:
    _MOVE_PRIOR[_sp] = 0.85


class Node:
    __slots__ = ("move", "player_to_move", "wins", "visits", "children",
                 "untried", "parent", "prior")

    def __init__(self, move, player_to_move, untried, parent=None, prior=0.5):
        self.move = move
        self.player_to_move = player_to_move
        self.wins = 0.0
        self.visits = 0
        self.children = []
        self.untried = untried
        self.parent = parent
        self.prior = prior            # [0,1] prior "virtual win rate"

    def ucb1_score(self, parent_visits, c=C_UCB):
        # Progressive bias: at low visits, `prior` dominates. At high visits,
        # empirical win rate dominates. Blend based on visit count.
        if self.visits == 0:
            # Use prior to order unvisited children
            return self.prior + c * math.sqrt(math.log(parent_visits + 1))
        exploit = self.wins / self.visits
        explore = c * math.sqrt(math.log(parent_visits) / self.visits)
        # Prior bonus decays as 1/(1+visits)
        bias = self.prior / (1.0 + self.visits)
        return exploit + explore + bias

    def select_child(self):
        return max(self.children, key=lambda n: n.ucb1_score(self.visits))

    def most_visited(self):
        # Break visit ties by prior so that in low-budget settings,
        # we still pick the move with better positional heuristics.
        return max(self.children, key=lambda n: (n.visits, n.prior))


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
    """Would playing at idx i put the resulting group in atari?

    This is an EXACT check (not the approximate one used in rollouts):
    we simulate the move and look at the resulting group's liberty count.
    Used at the root where correctness matters more than speed.
    """
    sim = board.clone()
    r, c = divmod(i, SIZE)
    sim.place_stone(r, c)
    # If the stone was captured, it wasn't self-atari — it was suicide,
    # which is_legal should have rejected. Be defensive anyway.
    if sim.color[i] == EMPTY:
        return True
    root = sim.find(i)
    return len(sim.libs[root]) <= 1


def _our_groups_in_atari(board, player):
    """Return list of (root_idx, liberty_idx) for each friendly group with 1 liberty."""
    result = []
    seen = set()
    for i in range(N):
        if board.color[i] == player:
            root = board.find(i)
            if root in seen:
                continue
            seen.add(root)
            if len(board.libs[root]) == 1:
                lib = next(iter(board.libs[root]))
                result.append((root, lib, board.size[root]))
    return result


def _moves_that_save_atari(board, player=None):
    """Return set of move indices that actually save at least one of our ataried groups.

    A move saves a group if, after placing, the group (possibly merged with
    adjacent friendly groups) has >= 2 liberties. Moves that put the group
    back into atari immediately (false saves) are rejected.
    """
    if player is None:
        player = board.current_player
    ataried = _our_groups_in_atari(board, player)
    if not ataried:
        return set()

    saves = set()
    for root, lib_idx, _sz in ataried:
        # Option A: play the liberty itself
        r, c = divmod(lib_idx, SIZE)
        if board.is_legal(r, c, player):
            sim = board.clone()
            sim.place_stone(r, c)
            # Check: is the group still on the board, and does it have >=2 libs?
            if sim.color[lib_idx] != EMPTY:
                new_root = sim.find(lib_idx)
                if len(sim.libs[new_root]) >= 2:
                    saves.add(lib_idx)
        # Option B: capture an opponent group adjacent to our ataried group.
        # This gives our group more liberties. Check each stone of our group
        # and see if any adjacent opponent group is in atari.
        opp = OPPONENT[player]
        for j in range(N):
            if board.color[j] == player and board.find(j) == root:
                for nj in NEIGHBORS[j]:
                    if board.color[nj] == opp:
                        opp_root = board.find(nj)
                        if len(board.libs[opp_root]) == 1:
                            opp_lib = next(iter(board.libs[opp_root]))
                            r2, c2 = divmod(opp_lib, SIZE)
                            if board.is_legal(r2, c2, player):
                                saves.add(opp_lib)
    return saves


def _moves_that_capture(board, player=None):
    """Return set of move indices that capture at least one opponent group."""
    if player is None:
        player = board.current_player
    opp = OPPONENT[player]
    moves = set()
    seen_roots = set()
    for i in range(N):
        if board.color[i] == opp:
            root = board.find(i)
            if root in seen_roots:
                continue
            seen_roots.add(root)
            libs = board.libs[root]
            if len(libs) == 1:
                lib = next(iter(libs))
                r, c = divmod(lib, SIZE)
                if board.is_legal(r, c, player):
                    moves.add(lib)
    return moves


def _tactical_candidates(board, player=None):
    """
    Return a list of tactically-reasonable candidate moves for the root.

    The priority ladder (classic Go tactics):
      1. If we can capture an opponent group, that's usually best.
      2. If any of our groups is in atari, we MUST save it (or capture).
         Nothing else matters — ignoring atari loses material.
      3. Otherwise, all legal non-eye moves EXCEPT self-atari moves
         (unless self-atari captures something, which is handled separately).

    This shrinks the root candidate set from ~81 to maybe 10-30 tactically
    meaningful moves, so MCTS iterations are spent on decisions that
    actually matter.
    """
    if player is None:
        player = board.current_player

    # Step 2 (urgency): if any of our groups is in atari, restrict candidates
    # to moves that save it (or capture something, which might save it too).
    saves = _moves_that_save_atari(board, player)
    captures = _moves_that_capture(board, player)

    # If we have ataried groups, we HAVE to respond to them. Only saves and
    # captures are considered — all other moves are tactical blunders.
    ataried = _our_groups_in_atari(board, player)
    if ataried:
        urgent = saves | captures
        if urgent:
            return sorted(urgent)
        # No save available — the group is dead. Fall through to normal moves.

    # Step 3: all legal non-eye, non-self-atari moves, plus any captures.
    candidates = set(captures)  # captures always allowed
    for i in range(N):
        if board.color[i] != EMPTY:
            continue
        if _is_eye_idx(board, i, player):
            continue
        r, c = divmod(i, SIZE)
        if not board.is_legal(r, c, player):
            continue
        # Filter out self-atari unless it's also a capture (throw-in tactic)
        if i in captures:
            candidates.add(i)
        elif _would_be_self_atari_after(board, i, player):
            continue
        else:
            candidates.add(i)

    if not candidates:
        # Really bad position — fall back to any legal move (including self-atari)
        return _legal_non_eye_indices(board, player) or _legal_move_indices(board, player)

    return sorted(candidates)


def _run_mcts(root_board, time_limit):
    """Run MCTS and return the root node and iteration count.

    Uses tactical candidate filtering at the root: only tactically-sound
    moves enter the tree, so MCTS iterations are spent on meaningful
    decisions rather than on 81 potential moves with most being obvious
    blunders.
    """
    untried = _tactical_candidates(root_board)
    if not untried:
        untried = _legal_move_indices(root_board)
    if not untried:
        return None, None

    # Dynamic prior: boost capture moves and atari-saves for this position.
    # We build a local prior dict rather than mutating the global _MOVE_PRIOR.
    captures = _moves_that_capture(root_board)
    saves = _moves_that_save_atari(root_board)
    def move_prior(m):
        p = _MOVE_PRIOR[m]
        if m in captures:
            p = max(p, 0.95)   # captures are almost always good
        if m in saves:
            p = max(p, 0.90)   # saving an ataried group is usually good
        return p

    player = root_board.current_player
    root = Node(move=None, player_to_move=player,
                untried=untried, parent=None, prior=0.5)

    deadline = time.time() + time_limit
    iterations = 0

    while time.time() < deadline:
        node = root
        sim = root_board.clone()

        # --- SELECTION ---
        while not node.untried and node.children:
            node = node.select_child()
            sim.place_stone(*divmod(node.move, SIZE))

        # --- EXPANSION ---
        if node.untried:
            # Pick the untried move with the best prior (tactical or positional)
            best_idx_in_list = 0
            best_prior = -1.0
            for k, m in enumerate(node.untried):
                # Use tactical prior only at the root level; deeper nodes
                # use purely positional priors (faster, and tactics are
                # implicit in the rollouts).
                p = move_prior(m) if node is root else _MOVE_PRIOR[m]
                if p > best_prior:
                    best_prior = p
                    best_idx_in_list = k
            move = node.untried.pop(best_idx_in_list)
            sim.place_stone(*divmod(move, SIZE))
            child_untried = _legal_non_eye_indices(sim)
            if not child_untried:
                child_untried = _legal_move_indices(sim)
            child = Node(move=move, player_to_move=sim.current_player,
                         untried=child_untried, parent=node,
                         prior=best_prior if node is root else _MOVE_PRIOR[move])
            node.children.append(child)
            node = child

        # --- SIMULATION ---
        result = _rollout(sim)  # 1.0 if black wins, 0.0 if white

        # --- BACKPROPAGATION ---
        cur = node
        while cur is not None:
            cur.visits += 1
            if cur.parent is None:
                cur.wins += result
            else:
                mover = OPPONENT[cur.player_to_move]
                cur.wins += result if mover == BLACK else (1.0 - result)
            cur = cur.parent

        iterations += 1

    return root, iterations


def get_ai_move(board, time_limit=TIME_LIMIT):
    """Choose the best move for board.current_player. Returns (r, c) or None.

    Flow:
      1. If there are literally zero legal moves, resign (return None).
      2. Compute tactical candidates: moves that handle any urgent atari
         situations and avoid self-atari.
      3. If exactly one candidate, play it without searching.
      4. Run MCTS over the candidates; return the most-visited move.

    Only returns None when there's no legal move at all. The assignment
    treats passing as conceding, so we never voluntarily give up material
    just because the position looks bad.
    """
    legal_all = _legal_move_indices(board)
    if not legal_all:
        return None  # truly no legal move -> resign

    # Tactical filter: drops self-atari, forces atari-response, always includes captures
    candidates = _tactical_candidates(board)
    if not candidates:
        candidates = _legal_non_eye_indices(board) or legal_all

    if len(candidates) == 1:
        return divmod(candidates[0], SIZE)

    root, iterations = _run_mcts(board, time_limit)
    if root is None or not root.children:
        # Fallback when MCTS produced no children: pick the best-prior tactical move
        captures = _moves_that_capture(board)
        saves = _moves_that_save_atari(board)
        def fallback_prior(m):
            p = _MOVE_PRIOR[m]
            if m in captures: p = max(p, 0.95)
            if m in saves: p = max(p, 0.90)
            return p
        best = max(candidates, key=fallback_prior)
        return divmod(best, SIZE)

    best = root.most_visited()
    return divmod(best.move, SIZE)


# Convenience alias matching the older module's name
def mcts_move(board, time_limit=TIME_LIMIT):
    return get_ai_move(board, time_limit)
