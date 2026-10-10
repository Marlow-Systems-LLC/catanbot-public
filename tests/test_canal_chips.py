"""Great Canal: 9 chip hexes that keep producing; 2 active knights dig one; done at the 8th (owner 2026-10-03)."""
import copy

import pytest
from catanbot.board import hex_corners, real_canal_board
from catanbot.rules import MODES
from catanbot.state import GameState, Knight, PlayerState, dig_canals


def _gs():
    b = real_canal_board(7)
    ps = [PlayerState("a", is_me=True), PlayerState("b"), PlayerState("c")]
    return GameState(board=b, mode=copy.copy(MODES["great_canal"]), players=ps, vp_target=18)


def test_board_has_nine_producing_chip_hexes():
    b = real_canal_board(3)
    assert len(b.canal_hexes) == 9
    for c in b.canal_hexes:
        h = b.hexes[c]
        assert h.terrain in ("hills", "forest", "pasture", "mountains", "fields") and h.number
        assert c in b.main_island()


def test_fixed_parts_stay():
    a, b = real_canal_board(1), real_canal_board(2)
    for c, h in a.hexes.items():
        if h.terrain in ("fields", "gold", "desert", "lake", "mountains", "sea"):
            assert b.hexes[c].terrain == h.terrain
        if h.terrain in ("fields", "gold"):
            assert b.hexes[c].number == h.number
    assert [(x.edge, x.kind) for x in a.harbors] == [(x.edge, x.kind) for x in b.harbors]


def test_two_knights_dig_one_point_each_done_at_eighth():
    gs = _gs()
    a, b = gs.players[0], gs.players[1]
    assert gs.canal_total == 9
    order = sorted(gs.board.canal_hexes)
    for i, c in enumerate(order[:8]):
        cs = hex_corners(*c)
        a.knights[cs[0]] = Knight(active=True)
        b.knights[cs[3]] = Knight(active=True)
        dug = dig_canals(gs)
        assert c in {d[0] for d in dug}
        a.knights.clear(); b.knights.clear()
        if i < 7:
            assert gs.canal_progress < gs.canal_total
    assert gs.canal_progress >= gs.canal_total
    assert a.canal_chips == 8 and b.canal_chips == 8


def test_live_read_snaps_to_template():
    import cv2
    from catanbot.runner import BOARD_ROI
    from catanbot.vision.board_reader import read_board
    img = cv2.imread("data/calib/gc_game4_board.png")
    if img is None:
        pytest.skip("live frame not in this checkout")
    b = read_board(img, BOARD_ROI, canal=True).board
    assert len(b.canal_hexes) == 9 and len(b.harbors) == 8
    assert b.hexes[(2, -1)].terrain == "mountains" and b.hexes[(-2, -1)].terrain == "desert"   # hidden under pieces / sand
    assert b.hexes[(1, 1)].terrain == "lake" and b.hexes[(2, -2)].terrain == "forest"
    assert not any(h.terrain == "canal" for h in b.hexes.values())
