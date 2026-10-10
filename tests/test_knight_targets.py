"""Owner 2026-10-09: "largest road gives 2 points, a knight can interrupt the road, a larger knight
can move a smaller knight, when deleting a knight of another player, go for the best position if
needed or the largest one otherwise" (catanbot/knight_targets.py)."""
import copy

import pytest

from catanbot import knight_targets as kt
from catanbot import progress_cards as pc
from catanbot.board import Board, Hex, ekey, hex_corners
from catanbot.brain import Action, Brain
from catanbot.learning import Model
from catanbot.rules import MODES
from catanbot.state import GameState, Knight, PlayerState


def _gs():
    b = Board()
    for q, r, t, n in [(0, 0, "forest", 6), (1, 0, "hills", 5), (2, 0, "pasture", 9),
                       (0, 1, "mountains", 10), (1, 1, "forest", 8), (2, 1, "hills", 4),
                       (5, 5, "pasture", 3), (8, 8, "mountains", 11)]:
        b.add(Hex(q, r, t, n))
    me = PlayerState("bot", is_me=True, vp_visible=6)
    lead = PlayerState("leader", vp_visible=8)
    cutter = PlayerState("cutter", vp_visible=5)
    gs = GameState(board=b, mode=copy.copy(MODES["great_canal"]), players=[me, lead, cutter], vp_target=18)
    gs.mode.canal = False                       # plain C&K corners: no canal/gold extras in the scores
    return gs


H = hex_corners(1, 0)
FAR = hex_corners(8, 8)
FAR2 = hex_corners(5, 5)


@pytest.fixture(autouse=True)
def _switch_on():
    kt.ENABLED = True
    yield
    kt.ENABLED = True


def _road_through(gs):
    """Our road H0-H1-H2 with the cutter's basic knight on H1 (it splits the road)."""
    me = gs.me()
    me.settlements.add(H[0])
    me.roads |= {ekey(H[0], H[1]), ekey(H[1], H[2])}
    gs.players[2].knights[H[1]] = Knight(1, False)
    gs.players[1].knights[FAR[0]] = Knight(3, True)       # the leader's mighty knight, far away
    gs.players[1].settlements.add(FAR[2])


def test_deserter_goes_for_the_knight_cutting_our_road_over_a_stronger_one():
    gs = _gs()
    _road_through(gs)
    assert kt.knight_spot_value(gs, gs.me(), H[1]) >= kt.SPOT_MATTERS
    assert not kt.spot_matters(gs, gs.me(), FAR[0])
    args = pc.choose_deserter(gs, gs.me())
    assert args == {"player": "cutter", "knight": H[1]}


def test_deserter_with_no_spot_that_matters_takes_the_strongest_knight():
    gs = _gs()
    lead = gs.players[1]
    lead.knights[FAR[0]] = Knight(1, True)
    lead.knights[FAR2[0]] = Knight(3, False)
    gs.players[2].knights[FAR[3]] = Knight(2, False)   # strength: leader 1 active, cutter 0
    args = pc.choose_deserter(gs, gs.me())
    assert args["player"] == "leader"                     # player choice unchanged (strength)
    assert args["knight"] == FAR2[0]                       # its largest knight


def test_switch_off_keeps_the_old_player_pick():
    gs = _gs()
    _road_through(gs)
    kt.ENABLED = False
    assert pc.choose_deserter(gs, gs.me()) == {"player": "leader"}
    assert pc.choose_deserter(gs, gs.me(), by_spot=False) == {"player": "leader"}


def _displace_gs():
    """Our strong knight at H3 (our settlement next to it); two basic rival knights at the ends of
    our roads: H2 splits our road H3-H2-H1, H4 sits next to a rival settlement (no settlement
    corner, no cut)."""
    gs = _gs()
    me = gs.me()
    home = [v for v in gs.board.neighbors(H[3]) if v not in H][0]
    me.settlements.add(home)
    me.knights[H[3]] = Knight(2, True)
    me.roads |= {ekey(home, H[3]), ekey(H[3], H[2]), ekey(H[2], H[1]), ekey(H[3], H[4])}
    gs.players[2].knights[H[2]] = Knight(1, False)
    gs.players[1].knights[H[4]] = Knight(1, True)
    gs.players[1].settlements.add(H[5])
    return gs, H[4]


def _brain(**kw):
    br = Brain(Model(path=None))
    br.knight_moves = True
    br.displace = False
    for k, v in kw.items():
        setattr(br, k, v)
    return br


def test_displacement_onto_a_road_cutting_knight_is_allowed_with_displace_off():
    gs, d = _displace_gs()
    gs.mode.canal = True               # brain knight moves are planned in the Great Canal only
    on = {a.target for a in _brain().candidate_actions(gs) if a.kind == "move_knight"}
    assert (H[3], H[2]) in on
    assert (H[3], d) not in on         # a spot that doesn't matter: blanket displacement stays off
    off = {a.target for a in _brain(knight_target_by_spot=False).candidate_actions(gs) if a.kind == "move_knight"}
    assert (H[3], H[2]) not in off and (H[3], d) not in off


def test_displacement_prefers_the_road_cutting_spot():
    gs, d = _displace_gs()
    gs.mode.canal = True
    assert kt.spot_matters(gs, gs.me(), H[2]) and not kt.spot_matters(gs, gs.me(), d)
    br = _brain(displace=True)
    acts = [a for a in br.candidate_actions(gs) if a.kind == "move_knight" and a.target[1] in (H[2], d)]
    assert len(acts) == 2
    best = max(acts, key=lambda a: br.action_value(gs, a))
    assert best.target == (H[3], H[2])
    # the spot bonus is what ranks it: without it the push itself is the same
    on = br.action_value(gs, Action("move_knight", (H[3], H[2])))
    br.knight_target_by_spot = False
    assert on > br.action_value(gs, Action("move_knight", (H[3], H[2])))


def test_pick_with_no_spot_that_matters_is_the_largest_knight():
    gs = _gs()
    a, b = gs.players[1], gs.players[2]
    a.knights[FAR[0]] = Knight(1, True)
    b.knights[FAR2[0]] = Knight(2, False)
    b.knights[FAR2[3]] = Knight(2, True)
    assert kt.pick_knight_target(gs, gs.me(), [(a, FAR[0]), (b, FAR2[0]), (b, FAR2[3])]) == (b, FAR2[3])
