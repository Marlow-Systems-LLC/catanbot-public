"""Owner 2026-10-08: "it just lost its city because it wasted resources to build ships and roads;
when it does not have a knight, it should hold on to the resources more dearly to make one first.
the tradeoff is the 7 and too many resources and for that marlow should take into consideration
building a wall, but again, after the knight is done"."""
from test_canal_playbook import _gs, brain

from catanbot.board import ekey, hex_corners
from catanbot.state import Knight


def _city_no_knight(hand):
    gs = _gs()
    me = gs.me()
    c = hex_corners(0, -1)
    me.cities.add(c[0])
    me.roads.add(ekey(c[0], c[1]))           # a road: a free corner for the knight
    me.hand.update(hand)
    gs.barbarian_pos = gs.barbarian_len - 2
    return gs


def _kinds(plan):
    return [a.kind for a in plan]


def test_banks_for_the_knight_before_anything():
    gs = _city_no_knight({"lumber": 4, "wool": 2, "grain": 1, "brick": 1})
    plan = brain()._plan_turn(gs)
    k = _kinds(plan)
    assert "knight" in k, plan
    assert "ship" not in k and "road" not in k[:k.index("knight")], plan
    assert k.index("knight") < (k.index("activate") if "activate" in k else 99)


def test_short_of_the_knight_holds_the_cards():
    # no ore and no way to bank one: nothing that spends wool/grain, no ship, no road
    gs = _city_no_knight({"lumber": 1, "wool": 1, "grain": 1, "brick": 1})
    plan = brain()._plan_turn(gs)
    assert not any(a.kind in ("ship", "road", "settlement") for a in plan), plan


def test_with_the_knight_up_normal_play_returns():
    gs = _city_no_knight({"lumber": 1, "brick": 1})
    me = gs.me()
    me.knights[hex_corners(0, -1)[2]] = Knight(1, True)
    assert not brain()._knight_gap(gs)


def test_wall_before_the_knight_only_near_the_limit():
    big = _city_no_knight({"brick": 3, "lumber": 3, "wool": 1, "grain": 1})     # 8 cards, no ore to be had
    assert [a.kind for a in brain()._plan_turn(big)] == ["wall"]
    small = _city_no_knight({"brick": 2, "lumber": 1, "wool": 1, "grain": 1})   # 5 cards: hold them
    assert "wall" not in [a.kind for a in brain()._plan_turn(small)]
