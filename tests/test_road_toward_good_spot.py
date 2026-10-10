"""owner 2026-10-08 "the roads he built on the right side of the map, going down, were a waste of
resources": a road only counts as progress toward a spot worth settling (Brain.ROAD_SPOT_MIN pips),
never toward a lone coast corner."""
from catanbot.board import ekey

from test_canal_playbook import _gs, brain

S = (-1, 1)          # our settlement (6 pips)



def _value(br, gs, e):
    g = brain().clone(gs)
    g.me().roads.add(e)
    return br._road_value(gs, g)


def test_road_toward_weak_corner_is_worth_nothing():
    br, gs = brain(), _gs()
    gs.me().settlements.add(S)
    assert br._raw_spot(gs, (-3, 1)) < br.ROAD_SPOT_MIN and br._raw_spot(gs, (-2, 4)) < br.ROAD_SPOT_MIN
    assert _value(br, gs, ekey(S, (-2, 2))) == 0.0      # leads only to 3-pip coast corners


def test_road_toward_good_spot_is_worth_building():
    br, gs = brain(), _gs()
    gs.me().settlements.add(S)
    assert br._raw_spot(gs, (0, -2)) >= br.ROAD_SPOT_MIN
    assert _value(br, gs, ekey(S, (-1, -1))) > 0.0      # one step from the 8-pip spot


def test_spot_a_rival_road_already_reaches_is_no_target():
    br, gs = brain(), _gs()
    gs.me().settlements.add(S)
    gs.players[1].roads.add(ekey((0, -2), (1, -1)))
    gs.players[1].roads.add(ekey((-2, -2), (-2, -4)))
    assert not br._road_target(gs, gs.me(), (0, -2)) and not br._road_target(gs, gs.me(), (-2, -2))
    assert br._road_target(gs, gs.me(), (0, -4))        # still open: rivals' roads don't touch it
