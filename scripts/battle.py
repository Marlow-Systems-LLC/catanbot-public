"""Two model files fight: 2 seats each, seating alternates (ABAB / BABA), same seeds for both.
Each model file carries its own weights and its own play ("brain" flags). Fair share = 0.5.

    python scripts/battle.py GAMES A.json B.json [MODE]
"""
import json
import os
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

from catanbot.brain import Brain
from catanbot.learning import Model
from catanbot.sim import Sim

GAMES = int(sys.argv[1])
A, B = Path(sys.argv[2]), Path(sys.argv[3])
MODE = sys.argv[4] if len(sys.argv) > 4 else "great_canal"


def play(g):
    seats = "ABAB" if g % 2 == 0 else "BABA"
    brains = [Brain(Model(A if s == "A" else B)) for s in seats]
    winner, turns, _ = Sim(MODE, 4, seed=int(os.environ.get("BATTLE_SEED", 90_000_000)) + g // 2, brains=brains).run()
    return (seats[int(winner[1:])] if winner else None), turns


if __name__ == "__main__":
    with Pool(int(os.environ.get("CATAN_WORKERS") or max(1, cpu_count() // 2))) as pool:
        res = pool.map(play, range(GAMES))
    a = sum(w == "A" for w, _ in res)
    b = sum(w == "B" for w, _ in res)
    n = a + b
    se = (0.25 / max(1, n)) ** 0.5
    print(json.dumps({"A": A.name, "B": B.name, "mode": MODE, "games": GAMES, "A_wins": a, "B_wins": b,
                      "no_winner": GAMES - n, "A_share": round(a / max(1, n), 3), "se": round(se, 3),
                      "avg_turns": round(sum(t for _, t in res) / GAMES, 1)}), flush=True)
