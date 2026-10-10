"""Owner 2026-10-08: "the hunt for the canal points - marlow needs to do it better, it missed some
of the point in the top right corner and it had a ship there where it could add a knight ...
building road to get to the right place on the map ... moving an active knight onto a spot that
later gives a point when activating ... it should not deactivate a knight to move it if that would
endanger the city(es)." Board: the real canal template (today's game snapped at offset 0,0)."""
import copy

from catanbot.board import board_for_mode, ekey
from catanbot.brain import Action, Brain
from catanbot.learning import Model
from catanbot.rules import MODES
from catanbot.state import GameState, Knight, PlayerState


def _gs():
    b = board_for_mode("great_canal", seed=1)
    me = PlayerState("bot", is_me=True, vp_visible=6)
    gs = GameState(board=b, mode=copy.copy(MODES["great_canal"]),
                   players=[me, PlayerState("a", vp_visible=6), PlayerState("b", vp_visible=6)], vp_target=19)
    # today's top right: settlement (5,-5), ship to (6,-4), road to (5,-7)
    me.settlements.add((5, -5))
    me.ships.add(ekey((5, -5), (6, -4)))
    me.roads.add(ekey((5, -7), (5, -5)))
    return gs


def brain():
    return Brain(Model(path=None))


def test_ship_end_canal_corner_with_a_second_corner_in_reach_beats_a_lone_corner():
    gs, br = _gs(), brain()
    # (6,-4) is a corner of canal hex (3,-1); (4,-4), one road from (5,-5), is another
    hunt = br._canal_pair_value(gs, Action("knight", (6, -4)))
    br.canal_hunt = False
    assert hunt > br._canal_pair_value(gs, Action("knight", (6, -4))) == 0.02


def test_road_to_the_second_corner_next_to_our_knight_is_valued():
    gs, br = _gs(), brain()
    road = Action("road", ekey((5, -5), (4, -4)))
    lone = br._canal_reach_value(gs, br._apply(gs, road))
    gs.me().knights[(6, -4)] = Knight(1, True)
    paired = br._canal_reach_value(gs, br._apply(gs, road))
    assert paired > lone > 0


def test_road_to_a_dug_hex_corner_gets_no_canal_reach():
    gs, br = _gs(), brain()
    gs.canal_dug |= {(3, -2), (3, -1)}
    road = Action("road", ekey((5, -5), (4, -4)))
    assert br._canal_reach_value(gs, br._apply(gs, road)) == 0.0


def _defender_gs(grain):
    gs = _gs()
    me = gs.me()
    me.cities.add((4, 2))
    me.roads |= {ekey((4, 2), (5, 1))}
    me.knights[(5, 1)] = Knight(1, True)            # our only active knight guards the city
    me.roads.add(ekey((5, 1), (5, -1)))
    me.hand["grain"] = grain
    for o in gs.opponents():
        o.cities.add((-1, -1) if o.name == "a" else (-3, -1))
    gs.barbarian_pos = 6
    return gs


def test_never_switch_off_the_city_defender_without_grain():
    br = brain()
    move = Action("move_knight", ((5, 1), (5, -1)))
    no_grain = br.action_value(_defender_gs(0), move)
    with_grain = br.action_value(_defender_gs(1), move)
    assert with_grain - no_grain > 0.5
