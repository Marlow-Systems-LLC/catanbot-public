"""Learning from every game.

1. Every decision the bot makes is logged with the per-player features at that moment.
2. When a game ends, the winner becomes the label and the threat/value weights get a
   softmax-regression update ("given these features, who actually won?"). Later
   positions count more than early ones.
3. Per-opponent profiles track how each friend plays: how often they accept trades,
   build roads, hoard cards. The bot uses them to propose trades they'll take.

Everything is plain JSON under data/ so it survives restarts and can be inspected.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import time
from pathlib import Path

from .threat import DEFAULT_PARAMS, DEFAULT_WEIGHTS, FEATURES

DATA = Path(__file__).resolve().parent.parent / "data"
# CATANBOT_MODEL: a gate runs the strategy tests on its candidate (live_learn.strategy_ok)
MODEL = Path(os.environ.get("CATANBOT_MODEL") or DATA / "model.json")
GAMES = DATA / "games"


# features whose direction is a rule of the game, not something to learn
SIGNS = {"discard_risk": -1, "turns": 1, "vp_frac": 1}   # turns feature is -turns_to_win


def write_atomic(path: Path, text: str) -> None:
    """Review 0.3.1: the live runner polls data/model.json -- never let it read half a file."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def copy_atomic(src: Path, dst: Path) -> None:
    dst = Path(dst)
    tmp = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


class Model:
    def __init__(self, path: Path | None = MODEL):
        self.path = path
        self.weights = dict(DEFAULT_WEIGHTS)
        self.params = dict(DEFAULT_PARAMS)
        self.mode_weights: dict[str, dict[str, float]] = {}
        self.opponents: dict[str, dict] = {}
        self.games = 0
        self.brain: dict = {}           # Brain flag values this model plays with (profiles.py); {} = code defaults
        if path is not None and path.exists():
            d = json.loads(path.read_text())
            self.weights.update(d.get("weights", {}))
            self.params.update(d.get("params", {}))
            self.mode_weights = d.get("mode_weights", {})
            self.opponents = d.get("opponents", {})
            self.games = d.get("games", 0)
            self.brain = d.get("brain", {})

    def weights_for(self, mode_key: str) -> dict[str, float]:
        """Global weights plus whatever this mode has learned on top."""
        w = dict(self.weights)
        for k, v in self.mode_weights.get(mode_key, {}).items():
            w[k] = w.get(k, 0.0) + v
        for f, sign in SIGNS.items():          # learning may not flip these (hoarding runs did)
            if f in w and w[f] * sign < 0:
                w[f] = 0.0
        return w

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_atomic(self.path, json.dumps({
            "weights": self.weights, "params": self.params, "mode_weights": self.mode_weights,
            "opponents": self.opponents, "games": self.games,
            **({"brain": self.brain} if self.brain else {}),
        }, indent=1))

    # ---- opponent profiles -------------------------------------------------
    def profile(self, name: str) -> dict:
        return self.opponents.setdefault(name, {
            "games": 0, "wins": 0, "trade_offers_seen": 0, "trade_accepts": 0,
            "avg_cards": 0.0, "samples": 0,
        })

    def note_trade_response(self, name: str, accepted: bool) -> None:
        p = self.profile(name)
        p["trade_offers_seen"] += 1
        p["trade_accepts"] += int(accepted)

    def trade_accept_rate(self, name: str) -> float:
        p = self.profile(name)
        return (p["trade_accepts"] + 1) / (p["trade_offers_seen"] + 2)

    # ---- learning ----------------------------------------------------------
    def learn(self, game_log: "GameLog", winner: str, lr: float = 0.05, share: float = 0.5) -> dict:
        """Softmax-regression step: push the winner's features up and everyone else's down."""
        snaps = game_log.snapshots
        if not snaps or winner is None:
            return {"updated": False}
        mw = self.mode_weights.setdefault(game_log.mode, {})
        correct = 0
        n = len(snaps)
        for i, snap in enumerate(snaps):
            feats = snap["features"]           # name -> feature dict
            if winner not in feats:
                continue
            w = self.weights_for(game_log.mode)
            T = self.params["temperature"]
            s = {k: sum(w.get(f, 0) * x for f, x in fv.items()) / T for k, fv in feats.items()}
            m = max(s.values())
            z = sum(math.exp(v - m) for v in s.values())
            prob = {k: math.exp(v - m) / z for k, v in s.items()}
            if max(prob, key=prob.get) == winner:
                correct += 1
            weight_t = 0.3 + 0.7 * (i + 1) / n       # late positions matter more
            for f in FEATURES:
                grad = feats[winner].get(f, 0) - sum(prob[k] * feats[k].get(f, 0) for k in feats)
                step = lr * weight_t * grad
                self.weights[f] = self.weights.get(f, 0) + share * step        # shared knowledge
                mw[f] = mw.get(f, 0) + (1 - share) * step              # mode-specific
        for name in snaps[-1]["features"]:
            p = self.profile(name)
            p["games"] += 1
            p["wins"] += int(name == winner)
        self.games += 1
        if self.path is not None:
            self.save()
        return {"updated": True, "snapshots": n, "winner_predicted_pct": round(100 * correct / n)}


class GameLog:
    def __init__(self, mode: str, players: list[str], me: str):
        self.mode = mode
        self.players = players
        self.me = me
        self.started = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.snapshots: list[dict] = []
        self.decisions: list[dict] = []
        self.events: list[str] = []

    def snapshot(self, turn: int, feats: dict[str, dict], probs: dict[str, float]) -> None:
        self.snapshots.append({"turn": turn, "features": feats, "probs": probs})

    def decision(self, turn: int, kind: str, detail: dict) -> None:
        self.decisions.append({"turn": turn, "kind": kind, **detail})

    def event(self, text: str) -> None:
        self.events.append(text)

    def save(self, winner: str | None, result: dict | None = None) -> Path:
        GAMES.mkdir(parents=True, exist_ok=True)
        path = GAMES / f"{self.started}_{self.mode}.json"
        path.write_text(json.dumps({
            "mode": self.mode, "players": self.players, "me": self.me, "winner": winner,
            "learn_result": result, "snapshots": self.snapshots, "decisions": self.decisions,
            "events": self.events,
        }, indent=1))
        return path
