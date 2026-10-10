"""Overnight 2026-10-04: the relative barbarian rule (weakest defender loses a city)."""
from catanbot.board import canal_board, hex_corners
from catanbot.brain import Brain
from catanbot.rules import MODES
from catanbot.state import GameState, Knight, PlayerState, new_hand


def _gs():
    b = canal_board(seed=3)
    land = [h for h in b.hexes.values() if h.is_land and h.number]
    me = PlayerState("marlow.systems", is_me=True, hand=new_hand())
    a = PlayerState("Lin")
    me.cities.add(hex_corners(*land[0].coord)[0])
    a.cities.add(hex_corners(*land[5].coord)[2])
    a.cities.add(hex_corners(*land[9].coord)[2])
    g = GameState(board=b, mode=MODES["great_canal"], players=[me, a])
    return g, me, a


def test_weakest_with_a_city_loses_when_the_barbarians_win():
    g, me, a = _gs()
    assert Brain.barb_loss_risk(g)                     # 0 strength vs 3 cities, we are weakest
    v = next(v for v in g.board.vertices if g.board.is_land_vertex(v) and not g.board.is_small_island_vertex(v)
             and v not in me.cities and v not in a.cities) if hasattr(g.board, "vertices") else None
    if v is None:
        return
    me.knights[v] = Knight(level=1, active=True)
    assert not Brain.barb_loss_risk(g)                  # 1 vs Lin's 0: Lin is the weakest now


def test_flag_off_by_default():
    assert Brain().barb_relative == 0.0
