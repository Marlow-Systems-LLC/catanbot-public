"""Debug screenshots: keep what is new, drop repeats after the game (owner 2026-10-11: "we need
screenshots for each type of screen, but we don't need many duplicates ... if something new,
keep, if something it has already seen before and knows what to do, no need to keep more than
that one game it is playing").

Stage had 32 GB in 50k frames over 8 days. Every frame saved under one of the capture folders
goes through imwrite():
- its type is folder + file name without the time stamp (popups/road_after, probe/screen_unknown);
- a frame unlike every kept example of its type (difference hash, > NEW_BITS bits apart) is
  new: kept for good, up to KEEP_PER_TYPE examples a type;
- anything else is a repeat: kept for the game being played (at most GAME_PER_TYPE a type) and
  deleted with its .json/.txt/_native companions when the next game starts (new_game()).
data/shots.json holds the kept examples and this game's repeats.

    python -m catanbot.shots prune [--apply]    # the same rule over frames saved before this
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
INDEX = DATA / "shots.json"
DIRS = {"probe", "unread", "popups", "discard", "trade_wait", "lobby", "loops", "leave"}
NEW_BITS = 10
KEEP_PER_TYPE = 12
GAME_PER_TYPE = 30
STAMP = re.compile(r"^[\d_]+")
_lock = threading.Lock()


def kind(path: Path) -> str | None:
    """'popups/road_after' for data/popups/231501_road_after.png; None outside the capture folders."""
    path = Path(path)
    if path.parent.parent != DATA or path.parent.name not in DIRS:
        return None
    name = STAMP.sub("", path.stem).removesuffix("_native") or "frame"
    return f"{path.parent.name}/{name}"


def dhash(img) -> int:
    g = img if img.ndim == 2 else cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    s = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    bits = (s[:, 1:] > s[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


_cache: dict | None = None


def _load() -> dict:
    global _cache
    if _cache is None:
        try:
            _cache = json.loads(INDEX.read_text())
        except (OSError, ValueError):
            _cache = {}
        _cache.setdefault("kept", {})
        _cache.setdefault("game", [])
    return _cache


def _save(d: dict) -> None:
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    tmp = INDEX.with_name(f".{INDEX.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(d))
    os.replace(tmp, INDEX)


def _is_new(d: dict, k: str, h: int) -> bool:
    kept = d["kept"].get(k, [])
    return len(kept) < KEEP_PER_TYPE and all(bin(h ^ e["hash"]).count("1") > NEW_BITS for e in kept)


def decide(d: dict, path: Path, img) -> str:
    """'new', 'game' (a repeat kept for this game) or 'skip'. Records the choice in d."""
    k = kind(path)
    h = dhash(img)
    rel = str(Path(path).relative_to(DATA))
    if _is_new(d, k, h):
        d["kept"].setdefault(k, []).append({"hash": h, "file": rel})
        return "new"
    if sum(1 for e in d["game"] if e["kind"] == k) >= GAME_PER_TYPE:
        return "skip"
    d["game"].append({"kind": k, "file": rel})
    return "game"


def imwrite(path, img, *args) -> bool:
    """cv2.imwrite with the keep rule; frames outside the capture folders are written as before."""
    path = Path(path)
    if img is None or kind(path) is None:
        return cv2.imwrite(str(path), img, *args)
    if path.stem.endswith("_native"):                  # follows its main frame, written just before
        if not path.with_name(path.stem.removesuffix("_native") + path.suffix).exists():
            return False
        return cv2.imwrite(str(path), img, *args)
    try:
        with _lock:
            d = _load()
            if decide(d, path, img) == "skip":
                return False
            _save(d)
    except Exception:                                  # noqa: BLE001 -- capture must never stop play
        pass
    return cv2.imwrite(str(path), img, *args)


def _drop(rel: str) -> int:
    """Delete a frame and its companions (.json, .txt, _native.png)."""
    p = DATA / rel
    n = 0
    for f in p.parent.glob(p.stem + "*"):
        if f.stem in (p.stem, p.stem + "_native"):
            f.unlink(missing_ok=True)
            n += 1
    return n


def new_game() -> int:
    """A new game started: the last game's repeats go. Returns files deleted."""
    with _lock:
        d = _load()
        if not d["game"]:
            return 0
        try:
            n = sum(_drop(e["file"]) for e in d["game"])
            d["game"] = []
            _save(d)
        except OSError:                                # noqa: BLE001 -- never stop play
            return 0
    return n


def prune(apply: bool = False) -> dict:
    """Frames saved before the rule: oldest first, keep the new ones, delete the rest."""
    global _cache
    d = json.loads(json.dumps(_load()))                # a dry run changes nothing
    frames = sorted((p for name in DIRS if (DATA / name).is_dir() for p in (DATA / name).glob("*.png")
                     if not p.stem.endswith("_native")), key=lambda p: p.stat().st_mtime)
    known = {e["file"] for es in d["kept"].values() for e in es} | {e["file"] for e in d["game"]}
    kept = dropped = freed = 0
    for p in frames:
        rel = str(p.relative_to(DATA))
        if rel in known:
            continue
        img = cv2.imread(str(p), cv2.IMREAD_REDUCED_GRAYSCALE_4)
        k = kind(p)
        if img is not None and _is_new(d, k, dhash(img)):
            d["kept"].setdefault(k, []).append({"hash": dhash(img), "file": rel})
            kept += 1
            continue
        companions = [f for f in p.parent.glob(p.stem + "*") if f.stem in (p.stem, p.stem + "_native")]
        freed += sum(f.stat().st_size for f in companions)
        dropped += 1
        if apply:
            _drop(rel)
    if apply:
        _cache = d
        _save(d)
    return {"frames": len(frames), "kept": kept, "dropped": dropped, "freed_mb": round(freed / 1e6),
            "types": len(d["kept"]), "applied": apply}


if __name__ == "__main__":
    if sys.argv[1:2] == ["prune"]:
        print(json.dumps(prune("--apply" in sys.argv)))
    else:
        sys.exit(__doc__)
