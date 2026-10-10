"""Owner 2026-10-04: longest road = 2 VP; a rival's settlement or knight on a corner breaks it;
knights can be pushed off by stronger knights or progress cards."""
import copy
from catanbot.board import hex_corners, ekey
from catanbot.brain import Action, Brain
from catanbot.learning import Model
from catanbot.rules import MODES
from catanbot.state import GameState, Knight, PlayerState, award_longest_road
from catanbot import progress_cards as pc
from tests.test_canal_playbook import _board


def _road(n=6):
    """A path of n edges around two hexes, as a list of corners."""
    a, b = hex_corners(0, 0), hex_corners(1, 0)
    path = list(a) + [x for x in b if x not in a]
    return path[: n + 1]


def _gs():
    me = PlayerState("bot", is_me=True, vp_visible=5)
    o = PlayerState("rival", vp_visible=7)
    gs = GameState(board=_board(), mode=copy.copy(MODES["great_canal"]), players=[me, o], vp_target=19)
    return gs, me, o


def _lay(p, corners):
    for u, w in zip(corners, corners[1:]):
        if w in _board().neighbors(u):
            p.roads.add(ekey(u, w))


def test_a_knight_breaks_a_road_and_the_card_is_set_aside():
    gs, me, o = _gs()
    ring = list(hex_corners(0, 0))
    _lay(o, ring)                                       # an open line of 5 edges
    assert gs.longest_road_len(o) == 5
    award_longest_road(gs)
    assert o.has_longest_road
    me.knights[ring[2]] = Knight(1, False)              # inactive still breaks it
    assert gs.longest_road_len(o) == 3
    assert award_longest_road(gs) == {"rival": -2} and not o.has_longest_road


def test_tie_after_a_break_sets_the_card_aside():
    gs, me, o = _gs()
    third = PlayerState("third")
    gs.players.append(third)
    _lay(o, list(hex_corners(0, 0)))
    award_longest_road(gs)
    assert o.has_longest_road
    me.knights[hex_corners(0, 0)[2]] = Knight(1, False)
    me.roads |= {ekey(u, w) for u, w in zip(hex_corners(1, -1)[:6], hex_corners(1, -1)[1:6])}
    third.roads |= {ekey(u, w) for u, w in zip(hex_corners(-1, 1)[:6], hex_corners(-1, 1)[1:6])}
    assert gs.longest_road_len(me) == gs.longest_road_len(third) == 5
    assert award_longest_road(gs) == {"rival": -2}
    assert not any(p.has_longest_road for p in gs.players)


def test_intrigue_prefers_the_knight_that_breaks_our_road():
    gs, me, o = _gs()
    path = list(hex_corners(0, 0))
    _lay(me, path)
    o.knights[path[2]] = Knight(1, False)              # in the middle of our road
    o.knights[path[0]] = Knight(1, False)              # at the end: breaks nothing
    me.longest_road = gs.longest_road_len(me)
    assert pc.choose_intrigue(gs, me) == {"knight_at": path[2]}


def test_cut_corner_is_a_special_knight_spot():
    gs, me, o = _gs()
    ring = list(hex_corners(0, 0)) + [hex_corners(0, 0)[0]]
    _lay(o, ring)
    o.longest_road = gs.longest_road_len(o)
    br = Brain(Model(path=None))
    assert br._cuts_road(gs, me, ring[2])
    br.road_cuts = False
    assert br._cuts_road(gs, me, ring[2])              # the check itself; the flag gates its use


def test_road_toward_a_rivals_road_to_cut_it_with_a_knight():
    """Owner 2026-10-04: build a road to reach a rival's road with a knight, if it cuts their lead."""
    gs, me, o = _gs()
    ring = list(hex_corners(1, -1))                    # all home-island corners
    _lay(o, ring)
    award_longest_road(gs)
    assert o.has_longest_road
    mid = ring[2]
    out = next(w for w in gs.board.neighbors(mid) if w not in ring)   # the spoke off their road
    me.settlements.add(out)                            # our settlement next to their road
    br = Brain(Model(path=None))
    g = copy.deepcopy(gs)
    g.me().roads.add(ekey(out, mid))
    assert br._cut_reach_value(gs, g) == 0.02
    br.road_cuts = False
    assert br.action_value(gs, Action("road", ekey(out, mid))) < \
        (setattr(br, "road_cuts", True) or br.action_value(gs, Action("road", ekey(out, mid))))
