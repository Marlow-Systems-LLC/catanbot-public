"""Live 2026-10-09 20:09: the setup ship off the city ((5,-1),(5,1)) went "nowhere" — at sea one
corner ahead has nothing to settle, so every ship option tied and the first one won."""
import copy

from catanbot.board import Board, Hex, hex_corners
from catanbot.brain import Brain
from catanbot.rules import MODES
from catanbot.state import GameState, PlayerState
from catanbot.sim import Sim


def _home(b, at):
    home, todo = {at}, [at]
    while todo:
        x = todo.pop()
        for f in b.vertex_edges(x):
            y = f[0] if f[1] == x else f[1]
            if b.is_land_edge(f) and y not in home:
                home.add(y)
                todo.append(y)
    return home


def _leads_to_spot(gs, at, e, steps=4):
    """Within `steps` free sea edges the ship reaches a spot on another landmass."""
    b = gs.board
    home = _home(b, at)
    far = e[0] if e[1] == at else e[1]
    frontier, seen = {far}, {at, far}
    for _ in range(steps):
        nxt = set()
        for x in frontier:
            for f in b.vertex_edges(x):
                if b.is_sea_edge(f) and not gs.edge_owner(f):
                    y = f[0] if f[1] == x else f[1]
                    if y not in seen:
                        seen.add(y)
                        nxt.add(y)
                        if y not in home and gs.can_settle(y):
                            return True
        frontier = nxt
    return False


def _island_board():
    """Home hex (0,0), sea all round to ring 3, one island pair at (2,0)/(3,0) to the east."""
    b = Board()
    for q in range(-3, 4):
        for r in range(-3, 4):
            if abs(q + r) <= 3:
                b.add(Hex(q, r, "sea", None))
    b.add(Hex(0, 0, "forest", 6))
    b.add(Hex(2, 0, "hills", 8))
    b.add(Hex(3, 0, "pasture", 5))
    me = PlayerState("bot", is_me=True)
    gs = GameState(board=b, mode=copy.copy(MODES["great_canal"]), players=[me, PlayerState("a")])
    return gs


def test_setup_ship_heads_for_the_island():
    gs = _island_board()
    b, br = gs.board, Brain()
    checked = 0
    for at in hex_corners(0, 0):
        opts = [e for e in b.vertex_edges(at) if b.is_sea_edge(e)]
        good = [e for e in opts if _leads_to_spot(gs, at, e, 2)]
        if not good or len(good) == len(opts):
            continue
        assert br.choose_setup_road(gs, at, opts) in good, at
        checked += 1
    assert checked >= 1


def test_flag_off_keeps_old_choice_path():
    sim = Sim("great_canal", 4, seed=0)
    gs = sim.as_me(0)
    b, br = gs.board, Brain()
    br.setup_ship_lookahead = False
    at = next(v for v in b.vertices if b.is_coastal_vertex(v) and gs.can_settle(v))
    opts = [e for e in b.vertex_edges(at) if b.is_sea_edge(e)] or list(b.vertex_edges(at))
    assert br.choose_setup_road(gs, at, opts) in opts
