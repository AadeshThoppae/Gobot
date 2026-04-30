# 9x9 Go Engine

A Go engine for the 9x9 board with a Monte Carlo Tree Search (MCTS) AI,
built for the CS HW5 Go assignment.

## Structure

```
go_engine/
    __init__.py        exposes GoEngine
    board.py           Board class with union-find group tracking
    rules.py           is_legal / get_legal_moves (thin wrappers)
    scoring.py         Chinese (area) scoring + komi
    engine.py          public interface for the test harness
    ai.py              MCTS AI with biased rollouts and move priors
server.py              Flask HTTP server for the GUI
frontend/              TypeScript + HTML UI
```

## Running

### The tests
The harness imports `GoEngine` from `go_engine`:

```python
from go_engine import GoEngine
engine = GoEngine()
engine.initialize()
engine.place_stone(4, 4)
engine.is_legal(4, 5)   # True
engine.calculate_score()
```

### The GUI

1. Install Flask:
   ```
   pip install flask
   ```
   (flask_cors is optional — only needed if you serve the frontend from a different origin.)

2. Start the server:
   ```
   python3 server.py
   ```

3. Open **http://localhost:5001** in your browser.

**Important:** open the URL (`http://localhost:5001`), don't double-click
`index.html`. The page needs the server running so its fetch calls to
`/initialize`, `/ai_move`, etc. can reach the engine. If you open the HTML
file directly, you'll see "Loading..." forever because the browser has no
server to talk to.

## AI strategy

Monte Carlo Tree Search with the classic four phases: selection (via UCB1),
expansion, simulation (a biased random playout), and backpropagation.

What distinguishes this from a textbook MCTS:

1. **Incremental board representation.** Groups and liberty sets are
   maintained on the fly via union-find, so "is this group in atari?" is
   O(1). This lifts rollout throughput by ~200x over a naive BFS-every-
   query board, which is where most of the playing strength comes from.

2. **Biased rollouts.** During simulation, the rollout policy checks in
   order: (a) is any opponent group capturable, (b) is any friendly
   group in atari and savable, (c) otherwise pick a random legal move
   that doesn't fill an eye or put us into self-atari.

3. **Move priors.** Every cell has a fixed prior based on its position:
   edge cells are penalized, 3rd/4th line and star points are rewarded.
   UCB1 blends the prior with the empirical win rate, so at low visit
   counts (the first few hundred MCTS iterations), the tree focuses on
   plausible moves instead of uniformly exploring all 81.

4. **Conservative resignation.** The AI only returns None (which the
   engine interprets as "resign") when there are literally zero legal
   moves. It never voluntarily passes while a legal move exists.

## Testing

- `test_correctness.py` — targeted tests for Ko, suicide, captures,
  territory, bounds, and the public engine interface.
- `test_crosscheck_final.py` — 50 random 150-move games whose every
  placement and legality query is cross-checked against an independent
  pure-BFS reference implementation. This is the strongest guarantee
  of correctness: if the two implementations disagree on any move in
  any of 50 full games, the test fails.
- `test_stress.py` — 50 random full games through the public engine
  API, plus one full AI self-play game.
- `test_ai_smoke.py` — confirms the AI honors the time budget and
  never hangs.

All of these live at the repo root. Run them with `python3 test_xxx.py`.
