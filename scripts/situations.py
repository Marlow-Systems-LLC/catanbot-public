"""Where does a flag set win? Same games, seeds and seating as ab_policy for a SET, but every game
records the test seat's situation (setup features, VP gap to the leader at 1/3 and 2/3 of the game,
barbarian city loss, final VP) and the summary gives the win rate per feature bucket.

    python scripts/situations.py GAMES SET [MODE] [MODEL_PATH] > out.json

Set CATAN_WORKERS for the pool size. Reads the Sim; the sim is not changed.
"""
import importlib
import json
import math
import os
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catanbot.brain import Brain          # noqa: E402
from catanbot.learning import Model       # noqa: E402
from catanbot.sim import Sim              # noqa: E402

N = 4
FAIR = 0.25


def _ab():
    argv, sys.argv = sys.argv, ["ab_policy"]      # ab_policy reads sys.argv at import
    try:
        return importlib.import_module("scripts.ab_policy")
    finally:
        sys.argv = argv


class _Rec(Sim):
    """Sim that keeps every seat's VP once per round (snapshot runs at the start of each round)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.vp_series: list[list[int]] = []

    def snapshot(self):
        super().snapshot()
        self.vp_series.append([self.total_vp(p) for p in self.gs.players])


def _gap(series, seat, frac):
    if not series:
        return None
    vps = series[min(len(series) - 1, int(len(series) * frac))]
    return max(vps) - vps[seat]            # 0 = leading (or tied for the lead)


def play(args):
    g, set_name, mode, path, seed0 = args
    sets = _ab().SETS
    seat = g % N
    brains = []
    for i in range(N):
        b = Brain(Model(Path(path)))
        for k, v in sets[set_name][0 if i == seat else 1].items():
            setattr(b, k, v)
        brains.append(b)
    sim = _Rec(mode, N, seed=seed0 + g, brains=brains, record_setup=True)
    winner, turns, _ = sim.run()
    row = sim.setup_rows()[seat]
    f, o = row["features"], row["outcome"]
    return {"game": g, "seat": seat, "won": winner == f"P{seat}", "no_winner": winner is None, "turns": turns,
            "pips_total": f["pips_total"], "grain_pips": f["pips_by_res"].get("grain", 0.0),
            "harbour_3to1_or_2to1": bool(f["harb_3to1"] or f["harb_2to1"]), "distinct": f["distinct"],
            "gap_third": _gap(sim.vp_series, seat, 1 / 3), "gap_two_thirds": _gap(sim.vp_series, seat, 2 / 3),
            "lost_city": any(n == f"P{seat}" for _, n in sim.lost), "final_vp": o["final_vp"]}


def _stat(rows):
    n = len(rows)
    p = sum(r["won"] for r in rows) / n if n else 0.0
    return {"n": n, "win_rate": round(p, 3), "se": round(math.sqrt(p * (1 - p) / n), 3) if n else None}


def _terciles(vals):
    s = sorted(vals)
    return (s[len(s) // 3], s[2 * len(s) // 3]) if s else (0, 0)


def summarize(rows):
    """Win rate per bucket for each feature (n, win_rate, SE=sqrt(p(1-p)/n)); fair share 0.25."""
    def by(key, bucket):
        groups: dict[str, list] = {}
        for r in rows:
            if r[key] is not None:
                groups.setdefault(bucket(r[key]), []).append(r)
        return {k: _stat(v) for k, v in sorted(groups.items())}

    def tri(key):
        lo, hi = _terciles([r[key] for r in rows])
        return by(key, lambda x: f"low (<={lo:g})" if x <= lo else f"mid (<={hi:g})" if x <= hi else f"high (>{hi:g})")

    def gap(x):
        return "leading (0)" if x == 0 else "behind 1-2" if x <= 2 else "behind 3+"

    return {"fair": FAIR, "overall": _stat(rows),
            "buckets": {"pips_total": tri("pips_total"), "grain_pips": tri("grain_pips"),
                        "harbour_3to1_or_2to1": by("harbour_3to1_or_2to1", lambda x: "yes" if x else "no"),
                        "distinct": by("distinct", lambda x: "<=3" if x <= 3 else "4" if x == 4 else "5"),
                        "gap_third": by("gap_third", gap), "gap_two_thirds": by("gap_two_thirds", gap),
                        "lost_city": by("lost_city", lambda x: "lost a city" if x else "kept cities")}}


def run(games, set_name, mode="great_canal", path="data/train/best.json", workers=None):
    seed0 = int(os.environ.get("AB_SEED", 7_000_000))
    jobs = [(g, set_name, mode, str(path), seed0) for g in range(games)]
    workers = workers or int(os.environ.get("CATAN_WORKERS") or max(1, cpu_count() // 2))
    if workers <= 1:
        rows = [play(j) for j in jobs]
    else:
        with Pool(workers) as pool:
            rows = pool.map(play, jobs)
    return {"set": set_name, "mode": mode, "games": games, "summary": summarize(rows), "rows": rows}


if __name__ == "__main__":
    a = sys.argv
    out = run(int(a[1]) if len(a) > 1 else 200, a[2] if len(a) > 2 else "walls1004",
              a[3] if len(a) > 3 else "great_canal", a[4] if len(a) > 4 else "data/train/best.json")
    print(json.dumps(out), flush=True)
