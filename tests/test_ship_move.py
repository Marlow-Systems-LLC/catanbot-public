"""Owner 2026-10-09 "build into the game the way to move ships": brain value, config switch,
runner turn memory and the (UNVERIFIED) click flow's guards."""
import copy

from catanbot import config
from catanbot import moves as mv
from catanbot.board import real_canal_board
from catanbot.brain import Action, Brain
from catanbot.rules import MODES
from catanbot.runner import Runner
from catanbot.state import GameState, PlayerState, new_hand
from tests.test_progress_wiring import FakeIO


def _gs(seed=7):
    b = real_canal_board(seed)
    ps = [PlayerState("a", is_me=True), PlayerState("b"), PlayerState("c")]
    for p in ps:
        p.hand = new_hand()
    return GameState(board=b, mode=copy.copy(MODES["great_canal"]), players=ps, vp_target=18)


def _line(gs):
    """Settlement on the coast, two ships out to sea: (settlement, inner ship, open-end ship)."""
    me, b = gs.me(), gs.board
    v = next(v for v in sorted(b.vertices) if gs.can_settle(v) and b.is_coastal_vertex(v)
             and sum(b.is_sea_edge(e) for e in b.vertex_edges(v)) >= 1)
    me.settlements = {v}
    e1 = next(e for e in gs.legal_road_edges(me, ship=True) if v in e)
    me.ships = {e1}
    w = e1[0] if e1[1] == v else e1[1]
    e2 = next(e for e in gs.legal_road_edges(me, ship=True) if w in e and v not in e)
    me.ships.add(e2)
    return v, e1, e2


def _runner(**over):
    cfg = config.load()
    cfg["dry_run"] = True
    cfg.update(over)
    return Runner(FakeIO([]), cfg)


def test_config_switch_on_by_default_and_sets_brain_flag():
    assert config.DEFAULTS["ship_move"] is True
    assert _runner().brain.ship_moves is True
    assert _runner(ship_move=False).brain.ship_moves is False


def test_candidates_follow_brain_flag():
    gs = _gs()
    _line(gs)
    br = Brain(setup_weights={})
    br.ship_moves = False
    assert not any(a.kind == "move_ship" for a in br.candidate_actions(gs))
    br.ship_moves = True
    assert any(a.kind == "move_ship" for a in br.candidate_actions(gs))


def test_only_the_open_end_ship_moves_never_the_link():
    gs = _gs()
    v, e1, e2 = _line(gs)
    srcs = {m.target[0] for m in mv.ship_moves(gs)}
    assert srcs == {e2}                    # the ship joining the settlement never moves


def test_move_to_nothing_is_not_made():
    gs = _gs()
    _line(gs)
    br = Brain(setup_weights={})
    br._prospector_value = lambda g, p, v: 0.0
    for m in mv.ship_moves(gs):
        g2 = br._apply(gs, Action("move_ship", m.target))
        assert br._ship_move_value(gs, g2) < 0


def test_move_toward_gold_or_harbour_is_valued_and_losing_it_is_not():
    gs = _gs()
    _line(gs)
    br = Brain(setup_weights={})
    ms = mv.ship_moves(gs)
    target = ms[0].target
    src, dst = target
    old_ends = {v for e in gs.me().ships for v in e}
    good = next(v for v in dst if v not in old_ends)
    gs.board.is_small_island_vertex = lambda v: True
    br._prospector_value = lambda g, p, v: 0.04 if v == good else 0.0
    g2 = br._apply(gs, Action("move_ship", target))
    assert br._ship_move_value(gs, g2) > 0
    # the other way round: the old end reached it, the move gives it up
    lost = next(v for v in src if v in old_ends and sum(v in e for e in gs.me().ships) == 1)
    br._prospector_value = lambda g, p, v: 0.04 if v == lost else 0.0
    assert br._ship_move_value(gs, g2) < 0


def test_runner_turn_memory_marks_new_ships_and_one_move():
    r = _runner()
    gs = _gs()
    _, e1, e2 = _line(gs)
    me = gs.me()
    r._turn_n = 4
    r._ships_this_turn = {e2}
    r._mark_ship_turn(me)
    assert me.ships_new == {e2} and me.ship_moved is False
    assert mv.ship_moves(gs) == []          # the open-end ship was built this turn
    r._ships_this_turn = set()
    r._ship_moved_turn = 4
    r._mark_ship_turn(me)
    assert me.ship_moved is True and mv.ship_moves(gs) == []
    r._turn_n = 5                           # next turn: may move again
    r._mark_ship_turn(me)
    assert me.ship_moved is False and mv.ship_moves(gs)


def test_record_ship_build_counts_as_new_this_turn():
    r = _runner()
    r._record_build("ship", ((1, 2), (3, 4)))
    assert ((1, 2), (3, 4)) in r._ships_this_turn


def test_tap_move_ship_guards_click_nothing():
    clicks = []
    for over, setup in ((dict(ship_move=False), {}),
                        ({}, {"_ship_move_misses": 2}),
                        ({}, {"_turn_n": 3, "_ship_moved_turn": 3})):
        r = _runner(**over)
        r.__dict__.update(setup)
        r.click = lambda *a, **k: clicks.append(a)
        a = Action("move_ship", (((0, 0), (1, 0)), ((1, 0), (2, 0))))
        assert r.tap_move_ship(a) is False
        assert ("move_ship", a.target) in r._failed_taps
    assert clicks == []
