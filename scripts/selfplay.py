"""Great Canal self-play on real-shaped boards: the same brain in all 4 seats.

Usage: PYTHONPATH=. nice -n 5 .venv/bin/python scripts/selfplay.py GAMES [SEED0] [PROCS] [VARIANT_JSON]
VARIANT_JSON (e.g. '{"third_harbour": 10}') sets Brain attributes for 2 of the 4 seats
(seats 0+2 on even seeds, 1+3 on odd) so the variant plays the baseline head to head.
Writes one JSON line per game to data/selfplay/<seed0>.jsonl and prints the summary.
SETUP_LOG=1 also appends one setup-feature row per seat to data/setup_log/sim_<date>.jsonl.
"""
import json, os, sys, time
from multiprocessing import Pool
from pathlib import Path

from catanbot.brain import Brain, set_flag
from catanbot.learning import Model
from catanbot.sim import Sim, write_setup_rows

SETUP_LOG = os.environ.get("SETUP_LOG") == "1"


VARIANT = {}


def _init(v):
    VARIANT.update(v)                   # workers re-import this module: set it there


def one(seed):
    t = time.time()
    brains = [Brain(Model()) for _ in range(4)]
    var = [i for i in range(4) if i % 2 == seed % 2] if VARIANT else []
    for i in var:
        for k, x in VARIANT.items():
            set_flag(brains[i], k, x)
    sim = Sim("great_canal", 4, seed=seed, brains=brains, record_setup=SETUP_LOG)
    winner, turns, _ = sim.run(max_turns=500)
    ps = sim.gs.players
    setup = sim.setup_rows({"phase": "selfplay", "variant": None}) if SETUP_LOG else None
    for r in setup or []:
        r["variant"] = f"P{r['seat']}" in [f"P{i}" for i in var]
    return {"setup": setup, "seed": seed, "winner": winner, "turns": turns, "secs": round(time.time() - t, 1),
            "vps": [sim.total_vp(p) for p in ps], "cities": [len(p.cities) for p in ps],
            "chips": [p.canal_chips for p in ps], "attacks": sim.attacks, "progress": [sim.progress_drawn.get(p.name, 0) for p in ps], "progress_discarded": sim.progress_discarded,
            "lost_first": [n for a, n in sim.lost if a == 0], "lost": sim.lost, "disruptions": sim.disruptions,
            "recover": sim.recover, "chases": sim.chases, "displaced": sim.displaced,
            "walls": [len(p.walls) for p in ps], "cards_lost": [sim.cards_lost.get(p.name, 0) for p in ps], "variant": [f"P{i}" for i in var]}


def summary(rows):
    n = len(rows)
    seats = 4 * n
    first = sum(len(r["lost_first"]) for r in rows if r["attacks"])
    attacked = sum(1 for r in rows if r["attacks"])
    return {"games": n, "no_winner": sum(not r["winner"] for r in rows),
            "avg_turns": round(sum(r["turns"] for r in rows) / n), "avg_secs": round(sum(r["secs"] for r in rows) / n, 1),
            "city_lost_first_attack_pct": round(100 * first / max(1, 4 * attacked), 1),
            "seats_ending_cityless_pct": round(100 * sum(c == 0 for r in rows for c in r["cities"]) / seats, 1),
            "disruptions_per_game": round(sum(r.get("disruptions", 0) for r in rows) / n, 1),
            "avg_recover_turns": round(sum(sum(r.get("recover", [])) for r in rows) / max(1, sum(len(r.get("recover", [])) for r in rows)), 1),
            "avg_chips": round(sum(sum(r["chips"]) for r in rows) / seats, 2),
            "win_by_seat": [sum(r["winner"] == f"P{i}" for r in rows) for i in range(4)],
            **({"variant": group(rows, True), "baseline": group(rows, False)} if rows[0].get("variant") else {})}


def group(rows, is_var):
    """Per-seat stats for the variant seats (is_var) or the baseline seats."""
    seats = [(r, f"P{i}") for r in rows for i in range(4) if (f"P{i}" in r["variant"]) == is_var]
    att = [(r, s) for r, s in seats if r["attacks"]]
    return {"seats": len(seats), "wins": sum(r["winner"] == s for r, s in seats),
            "lost_first_pct": round(100 * sum(s in r["lost_first"] for r, s in att) / max(1, len(att)), 1),
            "avg_vp": round(sum(r["vps"][int(s[1])] for r, s in seats) / len(seats), 2),
            **({"avg_walls": round(sum(r["walls"][int(s[1])] for r, s in seats) / len(seats), 2),
                "avg_cards_lost": round(sum(r["cards_lost"][int(s[1])] for r, s in seats) / len(seats), 1)}
               if "walls" in rows[0] else {})}


if __name__ == "__main__":
    games = int(sys.argv[1]); seed0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    procs = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    VARIANT.update(json.loads(sys.argv[4]) if len(sys.argv) > 4 else {})
    out = Path("data/selfplay"); out.mkdir(parents=True, exist_ok=True)
    rows = []
    with Pool(procs, initializer=_init, initargs=(dict(VARIANT),)) as pool, open(out / f"{seed0}.jsonl", "w") as f:
        for r in pool.imap_unordered(one, range(seed0, seed0 + games)):
            if r.get("setup"):
                write_setup_rows(r["setup"])
            r.pop("setup", None)
            rows.append(r); f.write(json.dumps(r) + "\n"); f.flush()
    print(json.dumps(summary(rows)))
