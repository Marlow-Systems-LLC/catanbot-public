"""Owner 2026-10-05 game 15: Rhea built a road just to stop our knight reaching the canal.
The brain values the same block: a road that takes a rival's only way to a free canal corner."""
import copy

from catanbot.board import ekey, hex_corners, real_canal_board
from catanbot.brain import Brain
from catanbot.rules import MODES
from catanbot.state import GameState, PlayerState


def _setup():
    b = real_canal_board(7)
    ps = [PlayerState("a", is_me=True), PlayerState("b"), PlayerState("c")]
    gs = GameState(board=b, mode=copy.copy(MODES["great_canal"]), players=ps, vp_target=18)
    canal = {v for h in b.canal_hexes for v in hex_corners(*h)}
    for c in sorted(canal):
        if not b.is_land_vertex(c):
            continue
        for u in b.neighbors(c):
            if u in canal or not b.is_land_vertex(u) or not b.is_land_edge(ekey(u, c)):
                continue
            for w in b.neighbors(u):
                if w != c and w not in canal and b.is_land_edge(ekey(w, u)):
                    return gs, c, u, w
    raise AssertionError("no test spot")


def test_road_on_rivals_last_edge_to_canal_corner_scores():
    gs, c, u, w = _setup()
    gs.players[1].roads.add(ekey(w, u))                 # rival b: one road short of corner c
    br = Brain()
    assert c in br._canal_corner_reach(gs, gs.players[1])
    g = br.clone(gs)
    g.players[0].roads.add(ekey(u, c))                  # we take the edge b needs
    assert c not in br._canal_corner_reach(g, g.players[1])
    assert br._canal_deny_value(gs, g) > 0


def test_road_elsewhere_scores_nothing():
    gs, c, u, w = _setup()
    gs.players[1].roads.add(ekey(w, u))
    br = Brain()
    g = br.clone(gs)
    assert br._canal_deny_value(gs, g) == 0
