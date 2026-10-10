"""Stage 2026-10-09 19:15:31: road ((-2,-4),(-2,-2)) went "up towards grain": (-2,-4) touches
fields 3 + fields 12 + mountains 9. Farmland pays nothing while the canal is shut, so that corner
pays from one hex and the 2-grain setup must-have must not make it a road goal."""
from catanbot.board import Hex, hex_corners

from test_canal_playbook import _gs, brain


def _farm_corner():
    gs = _gs()
    b = gs.board
    # on the main island: two numbered fields + one mountains hex at an outer corner, as (-2,-4) live
    b.hexes[(1, 0)].terrain, b.hexes[(1, 0)].number, b.hexes[(1, 0)].barren = "fields", 3, True
    b.hexes[(1, -1)].terrain, b.hexes[(1, -1)].number, b.hexes[(1, -1)].barren = "fields", 12, True
    b.add(Hex(2, -1, "mountains", 9))
    v = next(c for c in hex_corners(1, 0) if c in hex_corners(1, -1) and c in hex_corners(2, -1))
    return gs, v


def test_two_fields_corner_is_no_road_goal_while_canal_shut():
    br, (gs, v) = brain(), _farm_corner()
    gs.canal_progress, gs.canal_total = 0, 9
    assert br.fields_hexes(gs, v) >= 2 and br._raw_spot(gs, v) >= br.ROAD_SPOT_MIN
    assert not br._road_target(gs, gs.me(), v)


def test_flag_off_is_the_old_target():
    br, (gs, v) = brain(), _farm_corner()
    gs.canal_progress, gs.canal_total = 0, 9
    br.road_skip_farmland = False
    assert br._road_target(gs, gs.me(), v)


def test_canal_open_farmland_counts_again():
    br, (gs, v) = brain(), _farm_corner()
    gs.canal_progress, gs.canal_total = 9, 9
    assert br._road_target(gs, gs.me(), v)


def test_setup_still_keeps_the_two_grain_must_have():
    br, (gs, v) = brain(), _farm_corner()
    gs.canal_progress, gs.canal_total = 0, 9
    assert br._hex_must_have(gs, v, gs.me())
