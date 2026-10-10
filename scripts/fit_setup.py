"""Learn the setup (initial placement) weights by self-play -- Great Canal first.

    PYTHONPATH=. nice -n 10 .venv/bin/python scripts/fit_setup.py \
        [--games 600] [--accept 400] [--procs 6] [--seed0 9100000] [--sigma 0.35] \
        [--mode great_canal] [--checks 60] [--no-write]

1. explore: every seat of every game gets its own setup weights, drawn around the baseline
   (baseline = SETUP_DEFAULTS + data/config.json "setup_weights"): log-normal (sigma) for a
   positive weight, uniform [0, SCALE] for a weight that is 0 (off). One row per seat per game
   goes to data/setup_log/sim_<date>.jsonl (features + weights + outcome).
2. fit: target = final-VP rank normalised within the game (1 best, 0 last); regressors = the
   sampled weights (log ratio or w/SCALE), demeaned within each game (game fixed effects remove
   board luck), ridge. 20 % of seeds held out for an out-of-sample check. A diagnostic ridge of
   the same target on the opening FEATURES is reported too (what kind of opening wins).
3. candidate: terms with |t| >= --tmin move one sigma (log terms) or to 0 / SCALE (off terms),
   clamped to owner-rule BOUNDS. No significant term -> the owner prior (first_harbour 4.0) is
   tested instead, so the acceptance step always says something.
4. owner check: setup-only runs with the candidate in every seat: no piece on a gold corner,
   every 3rd settlement coastal, share of 3rd settlements on a 3:1 not more than 5 points below
   the baseline's.
5. acceptance: candidate in 2 seats vs baseline in 2 (seats 0+2 on even seeds, 1+3 on odd),
   fresh seeds. Paired per-game win difference with its SE. Only if the candidate wins by more
   than 2 SE AND passes the owner check is "setup_weights" written to data/config.json.
Writes data/train/setup_fit.json.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from catanbot.brain import SETUP_DEFAULTS, SETUP_OWNER_PRIOR, Brain, load_setup_weights  # noqa: E402
from catanbot.learning import Model  # noqa: E402
from catanbot.sim import Sim, setup_log_path, write_setup_rows  # noqa: E402

# sampling range for a weight that is currently 0 (off)
SCALE = {"third_harbour": 8.0, "third_grain": 3.0, "first_harbour": 6.0, "grain2": 6.0,
         "gold_route": 6.0, "deny_3to1": 4.0}
# owner-rule guards on any fitted weight (lo, hi); others: [w0/3, 3*w0] or [0, 2*SCALE]
BOUNDS = {
    "harb_3to1_canal": (6.0, 16.0),     # owner: 3:1 is essential in the canal, 4:1 cripples
    "third_harbour": (0.0, 16.0),       # never a penalty for the 3:1 on the last settlement
    "start_grain_backup": (2.0, 10.0),  # owner: start grain is the back-up when there is no 3:1
}
# terms the mode's scoring never reads: not sampled, not fitted (a 2026-10-03 smoke run gave the
# unused non-canal harb_3to1 t=-2.5 in canal -- pure noise, hence also --tmin 3)
UNUSED = {"great_canal": {"harb_3to1"}}
FEATURES = ["pips_total", "distinct", "distinct_commodities", "harb_3to1", "has_canal_hex",
            "start_total", "start_grain", "grain2", "gold_route", "gold_near_pips", "dup_pips"]
CFG = ROOT / "data" / "config.json"
OUT = ROOT / "data" / "train" / "setup_fit.json"


def sample_weights(base: dict, rng: random.Random, sigma: float, fixed=()) -> dict:
    w = {}
    for k, w0 in base.items():
        if k in fixed:
            w[k] = w0
        elif w0 > 0:
            w[k] = w0 * math.exp(rng.gauss(0.0, sigma))
        else:
            w[k] = rng.uniform(0.0, SCALE.get(k, 1.0))
    return w


def design(w: dict, base: dict) -> list[float]:
    """Regressor per term: log(w/w0) for a positive baseline, w/SCALE for an off term."""
    return [math.log(w[k] / base[k]) if base[k] > 0 else w[k] / SCALE.get(k, 1.0) for k in base]


def make_brains(ws: list[dict]) -> list[Brain]:
    return [Brain(Model(), setup_weights=w) for w in ws]


def explore_one(job):
    seed, mode, base, sigma = job
    rng = random.Random(seed * 7919 + 13)
    ws = [sample_weights(base, rng, sigma, UNUSED.get(mode, ())) for _ in range(4)]
    sim = Sim(mode, 4, seed=seed, brains=make_brains(ws), record_setup=True)
    sim.run(max_turns=500)
    rows = sim.setup_rows({"phase": "explore"})
    for r, w in zip(rows, ws):
        r["x"] = design(w, base)
        r["x_keys"] = list(base)
    return rows


def accept_one(job):
    seed, mode, cand, base = job
    var = [i for i in range(4) if i % 2 == seed % 2]
    ws = [cand if i in var else base for i in range(4)]
    sim = Sim(mode, 4, seed=seed, brains=make_brains(ws), record_setup=True)
    sim.run(max_turns=500)
    rows = sim.setup_rows({"phase": "accept"})
    for r in rows:
        r["variant"] = r["seat"] in var
    return rows


def check_one(job):
    seed, mode, w = job
    sim = Sim(mode, 4, seed=seed, brains=make_brains([w] * 4), record_setup=True)
    sim.setup()
    return [f["features"] for f in sim.setup_feats]


def owner_check(pool, mode, w, seeds) -> dict:
    feats = [f for fs in pool.map(check_one, [(s, mode, w) for s in seeds]) for f in fs]
    n = len(feats)
    return {"seats": n,
            "gold_corner_pieces": sum(f["on_gold_corner"] for f in feats),
            "last_coastal_rate": round(sum(f["last_coastal"] for f in feats) / n, 3),
            "harb_3to1_rate": round(sum(f["harb_3to1"] for f in feats) / n, 3),
            "start_grain_avg": round(sum(f["start_grain"] for f in feats) / n, 3),
            "gold_route_rate": round(sum(f["gold_route_len"] >= 0 for f in feats) / n, 3)}


def ridge_fe(rows, X, y, lam):
    """Ridge on game-demeaned X, y. Returns beta, se, residual sd."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    gid = np.array([r["seed"] for r in rows])
    for g in np.unique(gid):
        m = gid == g
        X[m] -= X[m].mean(0)
        y[m] -= y[m].mean()
    sd = X.std(0)
    sd[sd == 0] = 1.0
    Z = X / sd
    A = Z.T @ Z + lam * np.eye(Z.shape[1])
    Ainv = np.linalg.inv(A)
    b = Ainv @ Z.T @ y
    res = y - Z @ b
    dof = max(1, len(y) - len(np.unique(gid)) - Z.shape[1])
    s2 = float(res @ res) / dof
    cov = s2 * Ainv @ (Z.T @ Z) @ Ainv
    return b / sd, np.sqrt(np.diag(cov)) / sd, math.sqrt(s2), (X, y)


def feat_vec(r):
    f = r["features"]
    return [float(f["pips_total"]), f["distinct"], f["distinct_commodities"], float(f["harb_3to1"]),
            float(f["has_canal_hex"]), f["start_total"], f["start_grain"], float(f["grain_hexes_last"] >= 2),
            float(f["gold_route_len"] >= 0), f["gold_near_pips"], f["dup_pips"]]


def fit(rows, base, lam, tmin, sigma, mode="great_canal"):
    keys = [k for k in base if k not in UNUSED.get(mode, ())]
    idx = [list(base).index(k) for k in keys]
    rows = [{**r, "x": [r["x"][i] for i in idx]} for r in rows]
    seeds = sorted({r["seed"] for r in rows})
    hold = {s for s in seeds if s % 5 == 0}
    tr = [r for r in rows if r["seed"] not in hold]
    ho = [r for r in rows if r["seed"] in hold]
    y_of = lambda rs: [r["outcome"]["rank_norm"] for r in rs]
    beta, se, rsd, _ = ridge_fe(tr, [r["x"] for r in tr], y_of(tr), lam)
    out = {"n_train_rows": len(tr), "n_holdout_rows": len(ho), "lambda": lam, "resid_sd": round(rsd, 4),
           "terms": {}}
    for k, b, s in zip(keys, beta, se):
        out["terms"][k] = {"w0": base[k], "kind": "log" if base[k] > 0 else "linear/SCALE",
                           "beta": round(float(b), 4), "se": round(float(s), 4),
                           "t": round(float(b / s), 2) if s > 0 else 0.0}
    if ho:     # out-of-sample: does the fitted direction predict within-game rank on unseen seeds?
        _, _, _, (Xh, yh) = ridge_fe(ho, [r["x"] for r in ho], y_of(ho), lam)
        pred = Xh @ beta
        c = float(np.corrcoef(pred, yh)[0, 1]) if pred.std() > 0 and yh.std() > 0 else 0.0
        out["holdout_corr"] = round(c, 4)
    # diagnostic: which opening features go with a better rank (all rows)
    fb, fse, _, _ = ridge_fe(rows, [feat_vec(r) for r in rows], y_of(rows), lam)
    out["features"] = {k: {"beta": round(float(b), 4), "t": round(float(b / s), 2) if s > 0 else 0.0}
                       for k, b, s in zip(FEATURES, fb, fse)}
    # candidate: only significant terms move
    cand, moved = dict(base), []
    for k in keys:
        t = out["terms"][k]["t"]
        if abs(t) < tmin:
            continue
        if base[k] > 0:
            cand[k] = base[k] * math.exp(sigma * (1 if t > 0 else -1))
        else:
            cand[k] = SCALE.get(k, 1.0) if t > 0 else 0.0
        lo, hi = BOUNDS.get(k, (base[k] / 3, base[k] * 3) if base[k] > 0 else (0.0, 2 * SCALE.get(k, 1.0)))
        cand[k] = round(min(hi, max(lo, cand[k])), 4)
        if cand[k] != base[k]:
            moved.append(k)
    out["moved"] = moved
    return cand, out


def win_stats(rows):
    """Paired per-game comparison, variant seats vs baseline seats (2 each)."""
    games = {}
    for r in rows:
        games.setdefault(r["seed"], []).append(r)
    dw, dv, dr, wv, wb, nw = [], [], [], 0, 0, 0
    for rs in games.values():
        v = [r for r in rs if r["variant"]]
        b = [r for r in rs if not r["variant"]]
        a_w = sum(r["outcome"]["won"] for r in v)
        b_w = sum(r["outcome"]["won"] for r in b)
        wv += a_w
        wb += b_w
        nw += all(r["outcome"]["no_winner"] for r in rs)
        dw.append((a_w - b_w) / 2)            # per-seat win-rate difference in this game
        dv.append(np.mean([r["outcome"]["final_vp"] for r in v]) - np.mean([r["outcome"]["final_vp"] for r in b]))
        dr.append(np.mean([r["outcome"]["rank_norm"] for r in v]) - np.mean([r["outcome"]["rank_norm"] for r in b]))
    n = len(games)
    se = lambda d: float(np.std(d, ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float("inf")
    diff = float(np.mean(dw))
    return {"games": n, "no_winner": nw,
            "variant_win_rate": round(wv / (2 * n), 4), "baseline_win_rate": round(wb / (2 * n), 4),
            "win_diff": round(diff, 4), "win_diff_se": round(se(dw), 4),
            "z": round(diff / se(dw), 2) if se(dw) > 0 else 0.0,
            "vp_diff": round(float(np.mean(dv)), 3), "vp_diff_se": round(se(dv), 3),
            "rank_norm_diff": round(float(np.mean(dr)), 4), "rank_norm_diff_se": round(se(dr), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=600)
    ap.add_argument("--accept", type=int, default=400)
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--seed0", type=int, default=9_100_000)
    ap.add_argument("--sigma", type=float, default=0.35)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--tmin", type=float, default=3.0, help="|t| a term needs to move (about Bonferroni for 25 terms)")
    ap.add_argument("--refit", default=None, help="fit the explore rows of this setup log instead of playing new games")
    ap.add_argument("--checks", type=int, default=60, help="setup-only seeds for the owner check")
    ap.add_argument("--mode", default="great_canal")
    ap.add_argument("--no-write", action="store_true", help="never write data/config.json")
    ap.add_argument("--log", default=None, help="setup log path (default data/setup_log/sim_<date>.jsonl)")
    a = ap.parse_args()

    base = load_setup_weights()
    log = Path(a.log) if a.log else setup_log_path()
    report = {"date": datetime.datetime.now().isoformat(timespec="seconds"), "args": vars(a),
              "baseline": base, "setup_log": str(log)}
    t0 = time.time()
    with Pool(a.procs) as pool:
        rows = []
        if a.refit:
            for line in open(a.refit):
                r = json.loads(line)
                if r.get("phase") == "explore" and r.get("mode") == a.mode and "x" in r \
                        and r.get("x_keys", list(base)) == list(base):
                    rows.append(r)
        else:
            for rs in pool.imap_unordered(explore_one, [(a.seed0 + g, a.mode, base, a.sigma) for g in range(a.games)]):
                rows += rs
                write_setup_rows(rs, log)
        report["explore"] = {"games": len({r["seed"] for r in rows}), "rows": len(rows),
                             "secs": round(time.time() - t0), "refit": a.refit}
        print(json.dumps({"explore": report["explore"]}), flush=True)

        cand, fr = fit(rows, base, a.lam, a.tmin, a.sigma, a.mode)
        report["fit"] = fr
        source = "fit"
        if not fr["moved"]:
            cand = {**base, **{k: v for k, v in SETUP_OWNER_PRIOR.items() if base.get(k, 0) == 0}}
            source = "owner_prior (no significant term)"
        report["candidate"] = cand
        report["candidate_source"] = source
        report["candidate_changes"] = {k: [base[k], cand[k]] for k in base if cand[k] != base[k]}
        print(json.dumps({"fit_moved": fr["moved"], "candidate": report["candidate_changes"],
                          "holdout_corr": fr.get("holdout_corr")}), flush=True)

        chk_seeds = [a.seed0 + 2_000_000 + s for s in range(a.checks)]
        ob = owner_check(pool, a.mode, base, chk_seeds)
        oc = owner_check(pool, a.mode, cand, chk_seeds)
        ok = (oc["gold_corner_pieces"] == 0 and oc["last_coastal_rate"] >= 1.0
              and oc["harb_3to1_rate"] >= ob["harb_3to1_rate"] - 0.05)
        report["owner_check"] = {"baseline": ob, "candidate": oc, "ok": ok}
        print(json.dumps({"owner_check": report["owner_check"]}), flush=True)

        t1 = time.time()
        arows = []
        if report["candidate_changes"] and a.accept:
            for rs in pool.imap_unordered(accept_one, [(a.seed0 + 1_000_000 + g, a.mode, cand, base)
                                                       for g in range(a.accept)]):
                arows += rs
                write_setup_rows(rs, log)
        acc = win_stats(arows) if arows else {"games": 0}
        acc["secs"] = round(time.time() - t1)
        report["acceptance"] = acc
    passed = bool(arows) and ok and acc["win_diff"] > 2 * acc["win_diff_se"]
    report["accepted"] = passed
    report["written"] = False
    if passed and not a.no_write:
        cfg = json.loads(CFG.read_text()) if CFG.exists() else {}
        cfg["setup_weights"] = {k: cand[k] for k in cand if cand[k] != SETUP_DEFAULTS[k]}
        CFG.write_text(json.dumps(cfg, indent=1))
        report["written"] = True
    report["total_secs"] = round(time.time() - t0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1))
    print(json.dumps({"acceptance": acc, "accepted": passed, "written": report["written"], "out": str(OUT)}))


if __name__ == "__main__":
    main()
