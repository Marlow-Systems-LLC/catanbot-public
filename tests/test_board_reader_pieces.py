"""Live 2026-10-04 (docs/analysis/live_controls_2026-10-04.md): the piece reader missed our
settlements, roads, ships and every knight, and read the sea as Jean's teal. Replays the saved
server frames through detect_pieces, on the frame and on a native-size stand-in."""
import pytest

from board_fixture import COLOURS, board_read, frame, native_of
from catanbot.vision.board_reader import detect_pieces

BEFORE = "board_045123_road_420_190.png"
AFTER = "board_045318_road_after_420_190.png"
BR = board_read()


def _read(name, native):
    img = frame(name)
    kw = dict(zip(("native", "lay"), native_of(img))) if native else {}
    return detect_pieces(img, BR, COLOURS, **kw)


@pytest.mark.parametrize("native", [False, True])
def test_buildings_all_players(native):
    p = _read(BEFORE, native)
    want = {(3, -7): "magenta", (1, -1): "magenta", (3, -1): "magenta",
            (1, -7): "green", (2, -4): "green", (-2, -4): "green", (5, -1): "green",
            (0, -4): "purple", (-1, -1): "purple", (6, -4): "purple",
            (-1, -7): "blue", (4, -4): "blue", (5, 5): "blue"}
    assert p.blds == want


@pytest.mark.parametrize("native", [False, True])
def test_knights_roads_ships(native):
    p = _read(BEFORE, native)
    assert p.knights[(2, -8)] == ("magenta", False)
    assert p.knights[(3, -5)][0] == "blue"
    assert p.knights[(0, -2)][0] == "purple"
    assert p.knights[(6, -2)][0] == "green"
    assert p.roads[((3, -7), (3, -5))] == "magenta"
    assert p.roads[((2, -4), (3, -5))] == "green"
    # Jean's two teal ships are real (checked on the frame); the open sea reads as no one's
    assert p.ships == {((2, -8), (3, -7)): "magenta", ((-1, -7), (0, -8)): "blue", ((0, -8), (1, -7)): "blue",
                       ((5, -1), (6, -2)): "green", ((5, -5), (6, -4)): "purple"}


@pytest.mark.parametrize("native", [False, True])
def test_new_road_seen(native):
    assert ((2, -4), (2, -2)) not in _read(BEFORE, native).roads
    assert _read(AFTER, native).roads.get(((2, -4), (2, -2))) == "magenta"


def test_still_unpacks_as_buildings_and_roads():
    bl, rd = _read(BEFORE, False)
    assert bl[(3, -7)] == "magenta" and rd[((3, -7), (3, -5))] == "magenta"
