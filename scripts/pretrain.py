"""Pre-training for The Great Canal (the only game marlow plays), in parallel shards.
Results merge into data/model.json as mode_weights["great_canal"]."""
import json
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

from catanbot.learning import MODEL, Model
from catanbot.sim import train
from catanbot.threat import DEFAULT_WEIGHTS

MODES = ["great_canal"]   # owner 2026-10-04: The Great Canal only
GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 300
if len(sys.argv) > 2:
    MODES = sys.argv[2].split(",")
SHARDS = int(sys.argv[3]) if len(sys.argv) > 3 else 1   # parallel runs per mode, weights averaged
OUT = Path("data/pretrain")
SHARD_DIR = OUT / "shards"


def run(job):
    mode, k = job
    model = Model(path=None)
    seed = (sum(map(ord, mode)) * 1000 + k) * 100_000      # stable across processes, unlike hash()
    _, st = train(GAMES, mode, seed=seed, model=model, share=0.0)
    pred = st["pred"]
    summary = {
        "mode": mode, "games": st["games"], "avg_turns": round(st["turns"] / st["games"], 1),
        "no_winner": st["no_winner"],
        "winner_pred_first50": round(sum(pred[:50]) / max(1, len(pred[:50])), 1),
        "winner_pred_last50": round(sum(pred[-50:]) / max(1, len(pred[-50:])), 1),
        "mode_weights": model.mode_weights.get(mode, {}),
    }
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    (SHARD_DIR / f"{mode}.s{k}.json").write_text(json.dumps(summary, indent=1))
    print(mode, k, {k2: v for k2, v in summary.items() if k2 != "mode_weights"}, flush=True)
    return summary


def average(mode, parts):
    """One mode's shards -> one summary; weights are the mean of the shard weights."""
    keys = {f for r in parts for f in r["mode_weights"]}
    n = len(parts)
    games = sum(r["games"] for r in parts)
    return {
        "mode": mode, "games": games, "shards": n,
        "avg_turns": round(sum(r["avg_turns"] * r["games"] for r in parts) / games, 1),
        "no_winner": sum(r["no_winner"] for r in parts),
        "winner_pred_first50": round(sum(r["winner_pred_first50"] for r in parts) / n, 1),
        "winner_pred_last50": round(sum(r["winner_pred_last50"] for r in parts) / n, 1),
        "mode_weights": {f: sum(r["mode_weights"].get(f, 0.0) for r in parts) / n for f in keys},
    }


if __name__ == "__main__":
    jobs = [(m, k) for m in MODES for k in range(SHARDS)]
    with Pool(min(len(jobs), cpu_count())) as pool:
        results = pool.map(run, jobs)
    OUT.mkdir(parents=True, exist_ok=True)
    for mode in MODES:
        summ = average(mode, [r for r in results if r["mode"] == mode])
        (OUT / f"{mode}.json").write_text(json.dumps(summ, indent=1))
        print("averaged", {k: v for k, v in summ.items() if k != "mode_weights"}, flush=True)
    m = Model()
    for f in sorted(OUT.glob("*.json")):          # every finished mode, not just this run's
        r = json.loads(f.read_text())
        m.mode_weights[r["mode"]] = r["mode_weights"]
    m.weights = dict(DEFAULT_WEIGHTS)
    m.save()
    print("merged into", MODEL)
