"""Who is going to win? Per-player features -> win probabilities.

The same model scores the bot's own options (maximise my win probability) and picks
blocking targets (the opponent with the highest win probability, which is often but
not always the points leader). The weights are learned after every game; see learning.py.
"""
from __future__ import annotations

import math

from .board import RESOURCES
from .rules import discard_limit
from .state import GameState, PlayerState

FEATURES = [
    "turns", "vp_frac", "income", "diversity", "cards", "longest_road", "largest_army",
    "ck_improve", "knights", "expansion", "harbors", "discard_risk", "hidden",
]

DEFAULT_WEIGHTS = {
    "turns": 3.0, "vp_frac": 2.5, "income": 1.0, "diversity": 0.5, "cards": 0.25,
    "longest_road": 0.3, "largest_army": 0.3, "ck_improve": 0.6, "knights": 0.25,
    "expansion": 0.5, "harbors": 0.2, "discard_risk": -0.4, "hidden": 0.4,
}
DEFAULT_PARAMS = {"cards_per_vp": 4.5, "ck_cards_per_vp": 5.5, "temperature": 1.0}


def _expansion(gs: GameState, p: PlayerState) -> float:
    """Best open settlement spots reachable within 2 road segments, by pips."""
    b = gs.board
    frontier = {v for v, _ in p.buildings()}
    for e in p.roads | p.ships:
        frontier |= set(e)
    seen = set(frontier)
    for _ in range(2):
        nxt = set()
        for v in frontier:
            for n in b.neighbors(v):
                o = gs.owner_at(n)
                if n not in seen and (o is None or o is p):
                    nxt.add(n)
        seen |= nxt
        frontier = nxt
    spots = sorted((b.vertex_pips(v) for v in seen
                    if gs.can_settle(v)), reverse=True)
    return sum(spots[:2]) / 20.0


def turns_to_win(gs: GameState, p: PlayerState, params: dict) -> float:
    """Rough number of rounds until p reaches the VP target."""
    vp = p.vp_visible + gs.hidden_vp_estimate(p)
    need = max(0.0, gs.target - vp)
    if need == 0:
        return 0.0
    # 2-VP swings that are within reach count as partially earned
    lr_best = max((q.longest_road for q in gs.players), default=0)
    if not p.has_longest_road and p.longest_road >= max(4, lr_best - 1):
        need -= 0.6
    if gs.mode.ck:
        top = max(p.improvements.values())
        if top >= 3 and not p.metropolises:
            need -= 0.5 * (top - 2)
    need = max(0.3, need)
    cpv = params["ck_cards_per_vp"] if gs.mode.ck else params["cards_per_vp"]
    inc = gs.income(p)
    n = max(2, len(gs.players))
    per_round = sum(inc.values()) * n + 0.15  # +trade/steal slack
    types = sum(1 for r in RESOURCES if inc.get(r, 0) > 0)
    per_round *= 0.6 + 0.08 * types           # missing resources cost trades
    cards = p.total_cards()
    # cards above the robber limit are at risk, not progress: no credit (stops hoarding)
    safe = discard_limit(gs.mode, p)
    cards = min(cards, safe) + 0.2 * max(0, cards - safe)
    return max(0.0, (need * cpv - 0.8 * cards) / per_round)


def features(gs: GameState, p: PlayerState, params: dict) -> dict[str, float]:
    inc = gs.income(p)
    types = sum(1 for r in RESOURCES if inc.get(r, 0) > 0)
    cards = p.total_cards()
    return {
        "turns": -turns_to_win(gs, p, params) / 10.0,
        "vp_frac": (p.vp_visible + gs.hidden_vp_estimate(p)) / gs.target,
        "income": sum(inc.values()),
        "diversity": types / 5.0,
        "cards": min(cards, 7) / 7.0,
        "longest_road": 1.0 if p.has_longest_road else 0.0,
        "largest_army": 1.0 if p.has_largest_army else 0.0,
        "ck_improve": sum(p.improvements.values()) / 15.0 if gs.mode.ck else 0.0,
        "knights": gs.defence_strength(p) / 6.0,
        "expansion": _expansion(gs, p),
        "harbors": len({gs.board.harbor_at(v) for v, _ in p.buildings()} - {None}) / 3.0,
        "discard_risk": max(0, cards - gs.mode.discard_limit) / 4.0,
        "hidden": (p.dev_cards + p.progress_cards) / 5.0,
    }


def score(f: dict[str, float], w: dict[str, float]) -> float:
    return sum(w.get(k, 0.0) * v for k, v in f.items())


def win_probs(gs: GameState, weights: dict, params: dict) -> dict[str, float]:
    s = {p.name: score(features(gs, p, params), weights) / params["temperature"] for p in gs.players}
    m = max(s.values())
    e = {k: math.exp(v - m) for k, v in s.items()}
    z = sum(e.values())
    return {k: v / z for k, v in e.items()}


def block_target(gs: GameState, weights: dict, params: dict) -> PlayerState | None:
    """Opponent most likely to win: the one every block (robber, knight, trade refusal) targets."""
    probs = win_probs(gs, weights, params)
    opps = gs.opponents()
    return max(opps, key=lambda p: probs[p.name]) if opps else None

