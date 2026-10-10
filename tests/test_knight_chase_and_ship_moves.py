"""Owner 2026-10-09: "knights can move along the road or ships, also knights can move the robber
if active and get deactivated"."""
import types

import numpy as np

from catanbot import moves as mv
from catanbot import runner as R
from catanbot.board import ekey, hex_corners
from catanbot.brain import Action
from catanbot.sim import Sim
from catanbot.state import Knight


def _sim(mode="great_canal"):
    sim = Sim(mode, 4, seed=7)
    gs = sim.gs
    for p in gs.players:
        p.settlements.clear(); p.cities.clear(); p.metropolises.clear(); p.knights.clear()
        p.roads.clear(); p.ships.clear(); p.hand = {"wool": 2}; p.vp_visible = 9
        p.special_vp = 5                                     # above the friendly-robber 4 VP
    sim.attacks = 1
    gs.barbarians_attacked = True
    gs.barbarian_pos = 0
    return sim


def _hex(b):
    return next(h for h in b.hexes.values() if h.is_land and h.number and h.terrain not in ("desert", "gold")
                and not h.barren)


def _robber_on_our_hex(sim):
    """Our settlement and active knight on two corners of hex H; the robber sits on H; a rival
    with cards sits on another hex."""
    gs, b = sim.gs, sim.gs.board
    h = _hex(b)
    cs = sorted(hex_corners(*h.coord))
    me, rival = gs.players[0], gs.players[1]
    kv = next(v for v in cs if not any(x.terrain == "gold" for x in b.vertex_hexes(v)))
    sv = next(v for v in cs if v != kv and v not in b.neighbors(kv))
    me.settlements.add(sv)
    me.knights[kv] = Knight(1, True)
    other = next(x for x in b.hexes.values() if x.is_land and x.number and x is not h and x.terrain != "desert"
                 and not set(hex_corners(*x.coord)) & set(cs))
    rival.settlements.add(sorted(hex_corners(*other.coord))[0])
    rival.hand = {"ore": 3}
    b.robber = h.coord
    return me, kv, h


def test_knight_moves_along_ships_and_roads_only_switch():
    sim = _sim()
    gs = sim.as_me(0)
    b, me = gs.board, gs.players[0]
    src = next(v for v in b.vertices if b.is_land_vertex(v) and not gs.occupied(v))
    mid = next(w for w in b.neighbors(src) if b.is_land_vertex(w) and not gs.occupied(w))
    dst = next(w for w in b.neighbors(mid) if w != src and b.is_land_vertex(w) and not gs.occupied(w))
    me.knights[src] = Knight(1, True)
    me.roads.add(ekey(src, mid))
    me.ships.add(ekey(mid, dst))                             # road, then a ship
    assert mv.Move("move_knight", (src, dst)) in mv.knight_moves(gs)
    assert mv.Move("move_knight", (src, dst)) not in mv.knight_moves(gs, ships=False)
    assert mv.Move("move_knight", (src, mid)) in mv.knight_moves(gs, ships=False)


def test_chase_moves_need_an_active_unacted_knight_after_the_first_attack():
    sim = _sim()
    me, v, h = _robber_on_our_hex(sim)
    gs = sim.as_me(0)
    assert mv.chase_moves(gs) == [mv.Move("chase", (v, h.coord, "robber"))]
    gs.barbarians_attacked = False
    assert mv.chase_moves(gs) == []                          # robber locked before the 1st attack
    gs.barbarians_attacked = True
    gs.me().knights[v].active = False
    assert mv.chase_moves(gs) == []                          # an inactive knight can't chase
    gs.me().knights[v] = Knight(1, True, acted=True)
    assert mv.chase_moves(gs) == []                          # one knight action per turn


def test_brain_chases_the_robber_off_our_hex_and_robs_elsewhere():
    sim = _sim()
    me, v, h = _robber_on_our_hex(sim)
    gs = sim.as_me(0)
    br = sim.brains[0]
    pick = br.chase_pick(gs)
    assert pick is not None
    a, (to, victim) = pick
    assert a == Action("chase", (v, h.coord, "robber"))
    assert to is not None and to != h.coord
    g2 = br._apply(gs, a)
    k = g2.me().knights[v]
    assert not k.active and k.acted                          # the knight is deactivated
    br.knight_chase = False
    assert br.chase_pick(gs) is None                         # switch off


def test_brain_keeps_the_knight_home_near_an_attack():
    sim = _sim()
    me, v, h = _robber_on_our_hex(sim)
    city = next(x for x in sim.gs.board.vertices if sim.gs.board.is_land_vertex(x)
                and not sim.gs.occupied(x) and x not in me.knights)
    me.cities.add(city)                                      # one city, one knight: no spare
    sim.gs.barbarian_pos = sim.gs.barbarian_len - 1
    assert sim.brains[0].chase_pick(sim.as_me(0)) is None


def test_no_chase_when_the_robber_blocks_only_rivals_without_bounce():
    sim = _sim()
    me, v, h = _robber_on_our_hex(sim)
    me.settlements.clear()
    sim.brains[0].knight_bounce = False
    assert sim.brains[0].chase_pick(sim.as_me(0)) is None


def test_sim_chase_deactivates_and_respects_the_switch():
    sim = _sim()
    me, v, h = _robber_on_our_hex(sim)
    sim.brains[0].knight_chase = False
    sim.chase_robber(0)
    assert sim.chases == 0 and me.knights[v].active
    sim.brains[0].knight_chase = True
    sim.chase_robber(0)
    assert sim.chases == 1
    assert not me.knights[v].active and me.knights[v].acted
    assert sim.gs.board.robber != h.coord


def _runner(tmp_path, monkeypatch, rings, cfg=None):
    r = R.Runner.__new__(R.Runner)
    clicks = []
    r.cfg = {"knight_chase_robber": True, **(cfg or {})}
    r.turn_started = 1.0
    r.br = types.SimpleNamespace(lattice=types.SimpleNamespace(vertex_px=lambda v: (100, 100)))
    r.io = types.SimpleNamespace(snapshot=lambda: np.zeros((40, 40, 3), np.uint8), ocr=lambda img: [])
    r.click = lambda xy, why="", *a: clicks.append((tuple(xy), why))
    r.notes = []
    r.note = lambda m, *a, **k: r.notes.append(m)
    r._close_leftover = lambda *a: None
    r._popup_options = lambda img, near=None: rings
    r._probe = lambda *a, **k: None
    r.probe = types.SimpleNamespace(tried=lambda *a: None, result=lambda *a: None)
    r._save_game_flags = lambda: None
    r._glog = lambda **k: None
    r._my_knights = {}
    monkeypatch.setattr(R, "ROOT", tmp_path)
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    return r, clicks


def test_runner_taps_knight_then_chase_option_once_per_turn(tmp_path, monkeypatch):
    sim = _sim()
    me, v, h = _robber_on_our_hex(sim)
    r, clicks = _runner(tmp_path, monkeypatch, [{"x": 150, "y": 90, "kind": "chase", "colour": "green"}])
    r.brain = sim.brains[0]
    gs = sim.as_me(0)
    assert r.tap_chase_robber(gs)
    assert [c[0] for c in clicks] == [(100, 100), (150, 90)]
    assert r._my_knights == {v: False} and v in r._moved_knights
    assert not r.tap_chase_robber(gs)                        # one chase per turn
    assert any("UNVERIFIED" in n for n in r.notes)


def test_runner_chase_off_by_config_or_no_option(tmp_path, monkeypatch):
    sim = _sim()
    _robber_on_our_hex(sim)
    r, clicks = _runner(tmp_path, monkeypatch, [], cfg={"knight_chase_robber": False})
    r.brain = sim.brains[0]
    assert not r.tap_chase_robber(sim.as_me(0)) and clicks == []
    r, clicks = _runner(tmp_path, monkeypatch, [{"x": 150, "y": 90, "kind": "build", "colour": "green"}])
    r.brain = sim.brains[0]
    assert not r.tap_chase_robber(sim.as_me(0))
    assert r._chase_misses == 1 and [c[0] for c in clicks] == [(100, 100)]
