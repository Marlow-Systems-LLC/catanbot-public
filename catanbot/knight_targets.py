"""Which rival knight to go after (owner 2026-10-09): "largest road gives 2 points, a knight can
interrupt the road, a larger knight can move a smaller knight, when deleting a knight of another
player, go for the best position if needed or the largest one otherwise".

knight_spot_value scores the corner a rival knight stands on, from our side: it cuts our road, it
blocks a settlement corner or an undug canal corner we reach, it sits next to our city or on gold,
it guards the robber on our hex. pick_knight_target takes the best spot when one matters
(score >= SPOT_MATTERS), else the strongest knight (level, then active over inactive)."""
from __future__ import annotations

from .board import VKey, hex_corners
from .state import GameState, PlayerState

SPOT_MATTERS = 2.0
ENABLED = True       # config knight_target_by_spot (runner._brain_cfg); False = the old picks


def knight_spot_value(gs: GameState, me: PlayerState, v: VKey) -> float:
    b = gs.board
    ours = sum(1 for e in me.roads | me.ships if v in e)
    s = 0.0
    if ours >= 2:
        s += 4.0 + (1.0 if me.has_longest_road else 0.0)        # it cuts our road (2 VP at stake)
    elif ours == 1 and (me.has_longest_road or me.longest_road >= 4):
        s += 1.5                                                 # it caps our longest-road path
    if ours and not gs.owner_at(v) and gs.distance_rule_ok(v) \
            and (not gs.mode.canal or v in b.home_spots()):
        s += 2.0                                                 # a settlement corner we reach
    hexes = b.vertex_hexes(v)
    if gs.mode.canal and any(b.is_canal(h) and h.coord not in gs.canal_dug for h in hexes):
        near = ours or any(w in me.knights for h in hexes if b.is_canal(h) for w in _corners(gs, h))
        s += 2.0 if near else 0.5                                # an undug canal corner we want
    if any(n in me.cities for n in b.neighbors(v)):
        s += 1.5
    if any(h.terrain == "gold" for h in hexes):
        s += 1.0
    rob = b.robber
    if rob is not None and any(h.coord == rob for h in hexes):
        mine = {x for x, _ in me.buildings()}
        if any(c in mine for c in _corners(gs, b.hexes.get(rob))):
            s += 2.0                                             # it guards the robber on our hex
    return s


def _corners(gs, h):
    return hex_corners(*h.coord) if h is not None else []


def pick_knight_target(gs: GameState, me: PlayerState, cands) -> tuple | None:
    """cands: (player, corner). Best-valued spot when one matters, else the strongest knight."""
    cands = [(p, v) for p, v in cands if v in p.knights]
    if not cands:
        return None
    val = {(p.name, v): knight_spot_value(gs, me, v) for p, v in cands}
    best = max(cands, key=lambda c: (val[(c[0].name, c[1])], c[0].knights[c[1]].level, c[0].knights[c[1]].active, c[1]))
    if val[(best[0].name, best[1])] >= SPOT_MATTERS:
        return best
    return max(cands, key=lambda c: (c[0].knights[c[1]].level, c[0].knights[c[1]].active,
                                     val[(c[0].name, c[1])], c[1]))


def spot_matters(gs: GameState, me: PlayerState, v: VKey) -> bool:
    return knight_spot_value(gs, me, v) >= SPOT_MATTERS
