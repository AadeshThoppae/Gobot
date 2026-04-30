"""
Flask HTTP server — thin wrapper around GoEngine.
All game logic lives in go_engine/; this file only handles routing.

Run:
    pip install flask
    python3 server.py

Then open http://localhost:5001 in your browser. Do NOT double-click
index.html directly — the page needs the server running at
localhost:5001 so that fetch() calls to /initialize, /ai_move, etc.
reach the engine.

flask_cors is optional. The frontend is served from the same origin
as the API, so CORS isn't required.
"""

from flask import Flask, jsonify, request
try:
    from flask_cors import CORS
except ImportError:
    CORS = None

from go_engine import GoEngine
from go_engine.ai import get_ai_move

app = Flask(__name__, static_folder="frontend", static_url_path="")
if CORS is not None:
    CORS(app)

engine = GoEngine()


def game_state():
    """Serialize the full game state for any response that needs it."""
    return {
        "board": engine.get_board_state(),
        "current_player": engine.get_current_player(),
        "captured": engine.get_captured_counts(),
        "last_move": engine.board.last_move,
        "game_over": engine.game_over,
        "winner": engine.winner,
    }


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.post("/initialize")
def initialize():
    engine.initialize()
    return jsonify(game_state())


@app.get("/state")
def state():
    return jsonify(game_state())


@app.post("/place")
def place():
    data = request.get_json()
    row, col = int(data["row"]), int(data["col"])
    if not engine.place_stone(row, col):
        return jsonify({"error": "Illegal move"}), 400
    return jsonify(game_state())


@app.post("/ai_move")
def ai_move():
    if engine.game_over:
        return jsonify({"error": "Game is over"}), 400
    move = get_ai_move(engine.board)
    if move is None:
        engine.resign()
    else:
        engine.place_stone(move[0], move[1])
    return jsonify(game_state())


@app.post("/resign")
def resign():
    engine.resign()
    return jsonify(game_state())


@app.get("/score")
def score():
    return jsonify(engine.calculate_score())


if __name__ == "__main__":
    app.run(debug=False, port=5001)
