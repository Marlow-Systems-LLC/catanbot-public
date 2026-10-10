"""Every live game trains the model, and the model only changes when it gets better.

Owner 2026-10-06: "all live games and simulated games need to train the model and the model
must improve with every stage". Live games saved no feature snapshots, so model.learn never saw
them (game JSON was {winner, mode: null, log}).

1. snapshot(): once per own turn the runner logs every player's features (as sim.py does) to
   data/live_glog.json, so a runner restart mid-game keeps them.
2. finish(): at game over the log + winner becomes data/train/live_pending/<time>.json.
3. gate (run detached, niced, after the game): every pending live game is learned on top of
   data/model.json into a candidate; the candidate battles the current model (scripts/battle.py,
   same seeds, ABAB/BABA). It replaces data/model.json only if it wins MORE than half.
   Every gate, live or sim (train_loop.py), is one row in data/train/stages.jsonl.

    python -m catanbot.live_learn gate [GAMES]
    python -m catanbot.live_learn status
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .learning import DATA, MODEL, GameLog, Model

GLOG = DATA / "live_glog.json"
TRAIN = DATA / "train"
PENDING = TRAIN / "live_pending"
USED = TRAIN / "live_used"
CAND = TRAIN / "live_candidate.json"
PREV = TRAIN / "model_prev.json"
LEDGER = TRAIN / "stages.jsonl"
LOCK = TRAIN / "gate.lock"
GATE_GAMES = 200
MAX_SNAPS = 300


def _read(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as e:
        # review 0.3.1: a corrupt pending/candidate file is skipped, but never silently
        print(f"live_learn: unreadable {path}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        return None


def snapshot(gs, brain, mode_key: str | None, me: str, turn=None, min_gap: float = 20.0) -> int:
    """One snapshot per own turn (a repeated roll frame within min_gap s is the same turn);
    returns how many the game has."""
    if gs is None or not mode_key or not getattr(gs, "players", None):
        return 0
    d = _read(GLOG) or {}
    if d.get("mode") not in (None, mode_key):
        d = {}                                       # a log left from another game
    d.setdefault("mode", mode_key)
    d.setdefault("me", me)
    d["players"] = [p.name for p in gs.players]
    snaps = d.setdefault("snapshots", [])
    now = time.time()
    if snaps and (now - snaps[-1].get("at", 0) < min_gap or (turn is not None and snaps[-1].get("turn") == turn)):
        return len(snaps)
    snaps.append({"turn": len(snaps) if turn is None else turn, "at": now,
                  "features": brain.all_features(gs), "probs": brain.probs(gs)})
    del snaps[:-MAX_SNAPS]
    GLOG.write_text(json.dumps(d))
    return len(snaps)


def reset() -> None:
    GLOG.unlink(missing_ok=True)


def match_name(ocr_name: str | None, names: list[str]) -> str | None:
    """The result screen's name ("marlow.systems", "Lin") against the board names."""
    if not ocr_name:
        return None
    o = ocr_name.lower().strip()
    for n in names:
        if n.lower() == o:
            return n
    for n in names:
        a = n.lower()
        if a[:6] and (o.startswith(a[:6]) or a.startswith(o[:6])):
            return n
    return None


def finish(winner_ocr: str | None, mode_key: str | None = None) -> Path | None:
    """Turn this game's log into a pending training game. None when there is nothing to learn."""
    d = _read(GLOG)
    reset()
    if not d or not d.get("snapshots"):
        return None
    winner = match_name(winner_ocr, d.get("players") or [])
    if winner is None:
        return None
    d["winner"], d["mode"] = winner, d.get("mode") or mode_key
    PENDING.mkdir(parents=True, exist_ok=True)
    path = PENDING / f"{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(d))
    return path


def pending() -> list[Path]:
    return sorted(PENDING.glob("*.json")) if PENDING.exists() else []


def build_candidate(base: Path | None = None, out: Path | None = None) -> tuple[list[Path], str | None, list[dict]]:
    """Learn every pending live game on top of the current model."""
    base, out = base or MODEL, out or CAND
    m = Model(base)
    m.path = None
    used, results, mode = [], [], None
    for f in pending():
        d = _read(f)
        if not d or not d.get("winner"):
            continue
        gl = GameLog(d["mode"], d.get("players", []), d.get("me", ""))
        gl.snapshots = d["snapshots"]
        results.append(m.learn(gl, d["winner"]))
        used.append(f)
        mode = d["mode"]
    m.path = out
    m.save()
    return used, mode, results


def ledger(row: dict) -> dict:
    TRAIN.mkdir(parents=True, exist_ok=True)
    n = sum(1 for _ in LEDGER.open()) if LEDGER.exists() else 0
    row = {"stage": n + 1, "time": time.strftime("%Y-%m-%d %H:%M"), **row}
    with LEDGER.open("a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def battle(games: int, a: Path, b: Path, mode: str) -> dict:
    root = Path(__file__).resolve().parent.parent
    out = subprocess.run([sys.executable, str(root / "scripts" / "battle.py"), str(games), str(a), str(b), mode],
                         cwd=root, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(root)})
    lines = [x for x in out.stdout.splitlines() if x.startswith("{")]
    if not lines:
        raise RuntimeError(f"battle failed: {out.stderr[-400:]}")
    return json.loads(lines[-1])


def promote(cand: Path) -> None:
    if MODEL.exists():
        shutil.copy(MODEL, PREV)
    tmp = MODEL.with_suffix(".tmp")
    shutil.copy(cand, tmp)
    os.replace(tmp, MODEL)                           # the runner never reads half a file


def gate(games: int = GATE_GAMES) -> dict:
    TRAIN.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        if time.time() - LOCK.stat().st_mtime < 6 * 3600:
            return {"skipped": "another gate is running"}
        LOCK.touch()
    try:
        used, mode, results = build_candidate()
        if not used:
            return {"skipped": "no pending live games"}
        r = battle(games, CAND, MODEL, mode or "great_canal")
        share = r["A_share"]
        promoted = share > 0.5                      # better, not just "not worse"
        if promoted:
            promote(CAND)
            USED.mkdir(parents=True, exist_ok=True)
            for f in used:
                shutil.move(str(f), USED / f.name)
        return ledger({"source": "live", "mode": mode, "live_games": len(used),
                       "learn": [{k: x.get(k) for k in ("snapshots", "winner_predicted_pct")} for x in results],
                       "battle_games": r["games"], "cand_share": share, "se": r["se"],
                       "promoted": promoted, "model_games": Model(MODEL).games})
    finally:
        LOCK.unlink(missing_ok=True)


def spawn_gate(note=print) -> None:
    """After a live game: run the gate in the background, low priority, never blocking play."""
    root = Path(__file__).resolve().parent.parent
    TRAIN.mkdir(parents=True, exist_ok=True)
    log = (TRAIN / "gate.log").open("a")
    subprocess.Popen(["nice", "-n", "15", sys.executable, "-m", "catanbot.live_learn", "gate"], cwd=root,
                     stdout=log, stderr=log, start_new_session=True, env={**os.environ, "PYTHONPATH": str(root)})
    note("live learn: gate started (candidate vs current model, data/train/stages.jsonl)")


def status() -> dict:
    rows = [json.loads(x) for x in LEDGER.open()] if LEDGER.exists() else []
    g = _read(GLOG) or {}
    return {"this_game_snapshots": len(g.get("snapshots", [])), "pending": len(pending()),
            "stages": len(rows), "last": rows[-1] if rows else None, "model_games": Model(MODEL).games}


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "gate":
        print(json.dumps(gate(int(sys.argv[2]) if len(sys.argv) > 2 else GATE_GAMES)), flush=True)
    else:
        print(json.dumps(status(), indent=1))
