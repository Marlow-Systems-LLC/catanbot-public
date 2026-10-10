"""Bugs traced from the live game of 2026-10-08 (18:40-19:24)."""
from types import SimpleNamespace

import numpy as np

from catanbot import config
from catanbot import runner as R
from catanbot.moves import build_moves
from catanbot.runner import ReplayIO, Runner


def _runner(monkeypatch, tmp_path):
    cfg = config.load(); cfg["dry_run"] = True
    r = Runner(ReplayIO([]), cfg)
    monkeypatch.setattr(Runner, "FLAGS", tmp_path / "game_flags.json")
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    return r


class _Me:
    def __init__(self, cities, settlements):
        self.cities, self.settlements = set(cities), set(settlements)


def test_build_city_offer_demotes_a_city_we_thought_we_had(monkeypatch, tmp_path):
    # 18:59: a reset after the pillage re-read the settlement (3,-7) as a city; 8 improve taps and
    # 3 walls failed on it. The game offering "build city" there proves it is a settlement.
    r = _runner(monkeypatch, tmp_path)
    r.mode = SimpleNamespace(start_with_city=True, key="great_canal")
    me = _Me({(3, -7)}, {(-1, 1)})
    gs = SimpleNamespace(me=lambda: me)
    r._my_cities = {(3, -7)}
    r._lost_cities = 0
    r._settlement_seen(gs, (3, -7))
    assert me.cities == set() and (3, -7) in me.settlements
    assert (3, -7) in r._not_city and (3, -7) not in r._my_cities
    assert r._lost_cities == 1                     # no city left: the pillage is remembered again


def test_not_city_survives_a_runner_restart(monkeypatch, tmp_path):
    r = _runner(monkeypatch, tmp_path)
    r._not_city = {(3, -7)}
    r._save_game_flags()
    r2 = _runner(monkeypatch, tmp_path)
    r2._load_game_flags()
    assert (3, -7) in r2._not_city


def test_rules_panel_is_asked_again_after_a_rejoin(monkeypatch, tmp_path):
    # 19:20: after the relaunch into the match it opened the rules panel once, then waited
    # 2.5 min declining every offer
    r = _runner(monkeypatch, tmp_path)
    r.mode = None
    clicks = []
    monkeypatch.setattr(Runner, "click", lambda self, xy, why, ocr=None: clicks.append(why))
    now = [1000.0]
    monkeypatch.setattr(R.time, "time", lambda: now[0])
    img = np.zeros((888, 1372, 3), np.uint8)
    assert r.ensure_state(img, []) is None
    assert r.ensure_state(img, []) is None
    assert clicks == ["open rules info"]
    now[0] += Runner.RULES_RETRY_S + 1
    r.ensure_state(img, [])
    assert clicks == ["open rules info", "open rules info"]


def test_refused_kind_leaves_the_build_list():
    # 18:56: a red knight stayed in the plan 3 more times
    import sys
    sys.path.insert(0, "tests")
    from test_canal_playbook import _gs
    from catanbot.board import hex_corners
    gs = _gs()
    me = gs.me()
    c = hex_corners(0, -1)
    me.cities.add(c[0])
    me.hand.update({"wool": 2, "ore": 2, "brick": 2, "lumber": 2, "grain": 2})
    assert any(m.kind == "road" for m in build_moves(gs))
    gs.blocked = {"road"}
    assert not any(m.kind == "road" for m in build_moves(gs))


def test_no_announcement_wait_off_live():
    cfg = config.load(); cfg["dry_run"] = True
    r = Runner(ReplayIO([]), cfg)
    assert r._wait_announcement() is False


def _live_board_gs():
    import copy, json
    from pathlib import Path
    from catanbot.board import Board, Hex
    from catanbot.rules import MODES
    from catanbot.state import GameState, PlayerState
    b = Board()
    for q, r, t, n in json.loads((Path(__file__).parent / "fixtures" / "board_20261008.json").read_text()):
        b.add(Hex(q, r, t, n))
    me = PlayerState("bot", is_me=True)
    return GameState(board=b, mode=copy.copy(MODES["great_canal"]), players=[me, PlayerState("x")],
                     vp_target=18), me


def test_road_building_prefers_roads_to_a_spot_over_coastal_ships():
    # 18:55: Road Building put two ships along the coast at the top settlement (1,-7)
    from catanbot import progress_cards as pc
    gs, me = _live_board_gs()
    me.settlements.add((1, -7))
    args = pc.choose_road_building(gs, me)
    assert args["ships"] == []
    reached = {v for e in args["edges"] for v in e} - {(1, -7)}
    assert any(gs.can_settle(v) for v in reached)
    assert me.roads == set() and me.ships == set()


def test_road_building_uses_ships_when_roads_run_out():
    from catanbot import progress_cards as pc
    gs, me = _live_board_gs()
    me.settlements.add((1, -7))
    me.roads.update({(("x", i), ("y", i)) for i in range(gs.mode.max_roads)})   # no roads left
    args = pc.choose_road_building(gs, me)
    assert args and set(args["ships"]) == set(args["edges"])                     # ships still offered


def test_knight_first_lets_an_affordable_city_go_first_when_barbarians_are_far():
    from catanbot.sim import Sim
    from catanbot.state import new_hand
    sim = Sim("great_canal", seed=5)
    sim.setup()
    g, me = sim.gs, sim.gs.players[0]
    for p in g.players:
        p.hand, p.progress_hand = new_hand(), []
    me.hand.update(ore=4, grain=2)
    b = sim.brains[0]
    g.barbarian_pos = 0
    assert [a.kind for a in b.plan_turn(g)][:1] == ["city"]
    g.barbarian_pos = g.barbarian_len - 1                    # barbarians close: knight first
    assert "knight" in [a.kind for a in b.plan_turn(g)]
