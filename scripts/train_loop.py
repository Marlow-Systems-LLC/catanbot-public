"""Keep training for HOURS: each round trains SHARDS parallel copies of the current best model
on fresh simulated games, averages them into a candidate, and plays the candidate (1 seat)
against the best (other seats). The candidate becomes the best only if it passes the owner-rule
strategy tests and wins more than 2 standard errors above its fair share (live_learn.passes). The best is written to data/model.json after every accepted round.

    python scripts/train_loop.py HOURS MODE [GAMES_PER_SHARD] [EVAL_GAMES]
Log: data/train/loop.jsonl (one line per round).
"""
import json
import sys
import time
import os
from multiprocessing import Pool, cpu_count
from pathlib import Path

from catanbot.brain import Brain
from catanbot.budget import init_worker, wait_for_quiet, workers
from catanbot.learning import MODEL, Model, copy_atomic
from catanbot.live_learn import ledger, passes, strategy_ok
from catanbot.sim import Sim, train

HOURS = float(sys.argv[1]) if len(sys.argv) > 1 else 8
MODE = sys.argv[2] if len(sys.argv) > 2 else "great_canal"
GAMES = int(sys.argv[3]) if len(sys.argv) > 3 else 60
EVAL = int(sys.argv[4]) if len(sys.argv) > 4 else 800
N = 4
# CATAN_TRAIN_DIR: train a side model (e.g. v2) without touching data/model.json
DIR = Path(os.environ.get("CATAN_TRAIN_DIR") or "data/train")
SIDE = "CATAN_TRAIN_DIR" in os.environ
BEST, CAND = DIR / "best.json", DIR / "candidate.json"
LOG = DIR / "loop.jsonl"
# Owner 2026-10-04: keep training light, 1/4 to 1/2 of the server, never push it to its limits.
# SHARDS is fixed so a round trains the same number of games whatever the worker count:
# fewer workers make a round slower, never smaller.
SHARDS = int(os.environ.get("CATAN_SHARDS") or 6)
def shard(job):
    rnd, k = job
    model = Model(BEST)
    _, st = train(GAMES, MODE, seed=(rnd * 1000 + k) * 10_000, model=model, share=0.0)
    return model.mode_weights.get(MODE, {}), st["turns"] / max(1, st["games"])


def duel(job):
    g, cand, best = job
    seat = g % N
    brains = [Brain(Model(Path(cand) if i == seat else Path(best))) for i in range(N)]
    winner, _, _ = Sim(MODE, N, seed=50_000_000 + g, brains=brains).run()
    return winner == f"P{seat}"


def main():
    DIR.mkdir(parents=True, exist_ok=True)
    if not BEST.exists():
        if SIDE:
            sys.exit(f"put the start model in {BEST} first")
        copy_atomic(MODEL, BEST)
    end = time.time() + HOURS * 3600
    rnd = sum(1 for _ in LOG.open()) if LOG.exists() else 0
    # CATAN_WORKERS: leave cores for the server's other work (the server also runs a VM + Docker)
    cpus = workers()
    rnd_started = False
    # maxtasksperchild: recycle workers so memory stays flat over a long run
    with Pool(cpus, initializer=init_worker, maxtasksperchild=50) as pool:
        while time.time() < end:
            wait_for_quiet(own=cpus if rnd_started else 0)
            rnd_started = True
            if not SIDE and MODEL.exists() and MODEL.stat().st_mtime > BEST.stat().st_mtime:
                copy_atomic(MODEL, BEST)    # a live-game gate promoted a better model: train on top of it
            t0 = time.time()
            res = pool.map(shard, [(rnd, k) for k in range(SHARDS)])
            keys = {f for w, _ in res for f in w}
            cand = Model(BEST)
            cand.mode_weights[MODE] = {f: sum(w.get(f, 0.0) for w, _ in res) / len(res) for f in keys}
            cand.path = CAND
            cand.save()
            ok, tests = strategy_ok(CAND)
            if ok:
                wins = sum(pool.imap_unordered(duel, [(rnd * EVAL + g, str(CAND), str(BEST)) for g in range(EVAL)], chunksize=8))
                rate = wins / EVAL
            else:                           # breaks an owner rule: no duel needed
                rate = 0.0
            se = (max(rate, 1 / N) * (1 - max(rate, 1 / N)) / EVAL) ** 0.5
            accepted = ok and passes(rate, se, 1 / N)
            if accepted:
                copy_atomic(CAND, BEST)
                if not SIDE:
                    copy_atomic(BEST, MODEL)
            row = {"round": rnd, "time": time.strftime("%H:%M"), "train_games": GAMES * SHARDS, "workers": cpus,
                   "avg_turns": round(sum(t for _, t in res) / len(res), 1),
                   "tests": tests, "cand_win_rate": round(rate, 3), "se": round(se, 3), "accepted": accepted,
                   "minutes": round((time.time() - t0) / 60, 1)}
            with LOG.open("a") as f:
                f.write(json.dumps(row) + "\n")
            if not SIDE:                    # one ledger for every stage, sim and live (catanbot/live_learn.py)
                ledger({"source": "sim", "mode": MODE, "sim_games": GAMES * SHARDS, "battle_games": EVAL,
                        "cand_share": round(rate, 3), "se": round(se, 3), "fair_share": 1 / N,
                        "tests": tests, "promoted": accepted,
                        "model_games": Model(MODEL).games})
            print(row, flush=True)
            rnd += 1


if __name__ == "__main__":
    main()
