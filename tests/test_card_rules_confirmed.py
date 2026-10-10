"""Progress-card rules the owner confirmed on 2026-10-06 (one test per rule)."""
import types

import numpy as np
import pytest

import catanbot.runner as R
from catanbot import progress_cards as pc
from catanbot.board import canal_board, hex_corners
from catanbot.rules import MODES
from catanbot.sim import Sim
from catanbot.state import GameState, Knight, PlayerState, new_hand


@pytest.fixture
def gs():
    b = canal_board(seed=3)
    me = PlayerState("marlow", is_me=True, hand=new_hand(), vp_visible=6)
    a = PlayerState("Hildegard", vp_visible=10, hand_count=0)
    c = PlayerState("Nassir", vp_visible=5, hand_count=0)
    g = GameState(board=b, mode=MODES["great_canal"], players=[me, a, c])
    land = [h for h in b.hexes.values() if h.is_land and h.terrain != "canal" and h.number]
    me.cities.add(hex_corners(*land[0].coord)[0])
    return g


@pytest.fixture
def sim():
    s = Sim(seed=1, n=4)
    s.setup()
    for p in s.gs.players:
        for r in p.hand:
            p.hand[r] = 0
    s.as_me(0)
    return s


# 1. Crane: level 1 is free
def test_crane_level_one_is_free_in_sim(sim):
    me = sim.gs.players[0]
    me.improvements.clear()
    assert pc.choose_crane(sim.gs, me) is not None
    sim._play_progress(0, "crane")
    assert sim.progress_played.get("crane") == 1
    assert sum(me.improvements.values()) == 1
    assert sum(me.hand.get(c, 0) for c in ("paper", "coin", "cloth")) == 0


# 2. Deserter: the knight keeps level and active state
def test_deserter_knight_keeps_level_and_inactive_state(sim):
    me, opp = sim.gs.players[0], sim.gs.players[1]
    v = next(v for e in opp.roads for v in e if not sim.gs.occupied(v))
    opp.knights[v] = Knight(3, active=False)
    sim._play_progress(0, "deserter")
    assert [(k.level, k.active) for k in me.knights.values()] == [(3, False)]


# 3. Saboteur: a player TIED with us is a target too
def test_saboteur_hits_players_tied_with_us(gs):
    me, a, c = gs.players
    c.vp_visible, c.hand_count = me.vp_visible, 4      # tied: loses 2
    assert pc.choose_saboteur(gs, me) == {}
    c.vp_visible = me.vp_visible - 1                   # behind us: safe, nobody else holds cards
    assert pc.choose_saboteur(gs, me) is None


# 4. Diplomat and Intrigue reach ships
def _stray_edge(gs, p):
    taken = {v for q in gs.players for v, _ in q.buildings()}
    v0 = next(v for v in sorted(gs.board.vertices) if gs.can_settle(v) and v not in taken)
    v1 = min(gs.board.neighbors(v0))
    return v0, v1, next(x for x in gs.board.vertex_edges(v0) if v1 in x)


def test_diplomat_takes_a_rivals_open_ship(gs):
    me, a, _ = gs.players
    v0, _, e = _stray_edge(gs, a)
    a.settlements.add(v0)
    a.ships.add(e)
    assert e in pc._open_roads(gs, a)
    assert pc.choose_diplomat(gs, me) == {"edge": e}


def test_sim_diplomat_removes_a_ship(sim, monkeypatch):
    gs = sim.gs
    opp = gs.players[1]
    e = next(iter(opp.roads))
    opp.roads.discard(e)
    opp.ships.add(e)
    monkeypatch.setattr(pc, "choose_diplomat", lambda g, m: {"edge": e})
    sim._play_progress(0, "diplomat")
    assert e not in opp.ships


def test_diplomat_moves_our_own_ship(gs, monkeypatch):
    me = gs.players[0]
    c = next(iter(me.cities))
    v1 = min(v for v in gs.board.neighbors(c) if gs.owner_at(v) is None)
    e = next(x for x in gs.board.vertex_edges(c) if v1 in x)
    me.ships.add(e)
    other = next(x for x in gs.board.vertex_edges(c) if x != e)
    calls = []

    def legal(p, ship=False):
        calls.append(ship)
        return [other]
    monkeypatch.setattr(gs, "legal_road_edges", legal)
    monkeypatch.setattr(gs, "longest_road_len", lambda p: len(p.ships))   # any move "gains"
    spots = iter([0, 1])
    monkeypatch.setattr(gs, "legal_settlement_spots", lambda p: [None] * next(spots, 1))
    a = pc._diplomat_own_move(gs, me)
    assert a == {"edges": [e, other], "own": True, "ship": True}
    assert calls == [True] and me.ships == {e}       # ship edges only; ships restored


def test_intrigue_reaches_a_knight_next_to_our_ship(gs):
    me, a, _ = gs.players
    c = next(iter(me.cities))
    v1 = min(v for v in gs.board.neighbors(c) if gs.owner_at(v) is None)
    me.ships.add(next(x for x in gs.board.vertex_edges(c) if v1 in x))
    a.knights[v1] = Knight(1, active=True)
    assert pc.choose_intrigue(gs, me) == {"knight_at": v1}


# 5. A pillaged city loses its wall (sim: tests/test_rules_audit.py; live runner here)
def test_runner_lost_city_drops_its_wall(tmp_path, monkeypatch, gs):
    from test_probe_controls import make
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    r = make(tmp_path)
    me = gs.players[0]
    v = next(iter(me.cities))
    me.walls.add(v)
    r._my_walls = {v}
    monkeypatch.setattr(r, "ensure_state", lambda img, texts: gs)
    monkeypatch.setattr(r, "_save_game_flags", lambda: None)
    monkeypatch.setattr(r, "_rings", lambda img: [])
    r.br = types.SimpleNamespace(lattice=types.SimpleNamespace(vertex_px=lambda v: (100, 100)))
    monkeypatch.setattr(r.brain, "choose_city_to_lose", lambda g, opts: v)
    r.lose_city(np.zeros((888, 1372, 3), np.uint8), [])
    assert v not in me.cities and v not in me.walls and v not in r._my_walls


# 6. Inventor may swap barren farmland chips while the canal is dug
def test_inventor_may_move_barren_chips_not_gold(gs):
    b = gs.board
    gs.canal_progress = 0
    fld = next(h for h in b.hexes.values() if h.terrain == "fields" and h.number in (3, 4, 5, 9, 10, 11))
    fld.barren = True                                  # face-down farmland chip, canal open
    barren = [h for h in b.hexes.values() if h.barren and h.number in (3, 4, 5, 9, 10, 11)]
    assert barren and all(pc.inventor_movable(gs, h) for h in barren)
    assert not any(pc.inventor_movable(gs, h) for h in b.hexes.values()
                   if h.terrain == "gold" or h.number in (2, 12, 6, 8))
