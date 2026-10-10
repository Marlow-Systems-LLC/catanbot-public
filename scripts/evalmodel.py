"""Does training help? One seat plays with the trained model (data/model.json), the other
seats with the default weights; the trained seat rotates. Prints its win rate (fair = 1/n).

    python scripts/evalmodel.py GAMES MODE [MODEL_PATH]
"""
import json
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

from catanbot.brain import Brain
from catanbot.learning import MODEL, Model
from catanbot.sim import Sim

GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 200
MODE = sys.argv[2] if len(sys.argv) > 2 else "great_canal"
PATH = Path(sys.argv[3]) if len(sys.argv) > 3 else MODEL
N = 4


def play(g):
    seat = g % N
    brains = [Brain(Model(PATH) if i == seat else Model(path=None)) for i in range(N)]
    winner, turns, _ = Sim(MODE, N, seed=9_000_000 + g, brains=brains).run()
    return winner == f"P{seat}", winner is None, turns


if __name__ == "__main__":
    # owner 2026-10-04: light on the server, same 1/4-1/2 of the cores as train_loop, low priority
    from catanbot.budget import init_worker, workers
    with Pool(workers(), initializer=init_worker) as pool:
        res = pool.map(play, range(GAMES))
    wins = sum(w for w, _, _ in res)
    out = {"mode": MODE, "games": GAMES, "trained_wins": wins, "win_rate": round(wins / GAMES, 3),
           "fair": round(1 / N, 3), "no_winner": sum(n for _, n, _ in res),
           "avg_turns": round(sum(t for *_, t in res) / GAMES, 1), "model": str(PATH)}
    print(json.dumps(out), flush=True)
