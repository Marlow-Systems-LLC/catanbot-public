"""Live 2026-10-04 (docs/analysis/live_controls_2026-10-04.md): builds failed because popups
stayed open and took the next tap (04:57:20: a tap 16 px from an open "Build city" ring rebuilt
the city by accident), spots were re-tapped all turn, red (unaffordable) options were tried, and
no build was ever checked. Replays the saved server frames through Runner.tap_build, the
piece read in ensure_state and _run_builds. OCR is stubbed out: the rings + icons decide."""
import types

import numpy as np
import pytest

from board_fixture import COLOURS, board_read, frame
from catanbot import runner as R
from catanbot.board import Board
from catanbot.brain import Action
from catanbot.state import GameState, PlayerState
from catanbot.vision.screens import icon_kind, option_rings

CLEAN = np.zeros((888, 1372, 3), np.uint8)


def _runner(frames, monkeypatch, xy=None, br=None, gs=None):
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    monkeypatch.setattr(R.cv2, "imwrite", lambda *a, **k: True)
    seq = list(frames)
    r = R.Runner.__new__(R.Runner)
    r.br = br or types.SimpleNamespace(
        lattice=types.SimpleNamespace(vertex_px=lambda t: xy, edge_px=lambda t: xy), board=Board())
    r.io = types.SimpleNamespace(snapshot=lambda: seq.pop(0) if len(seq) > 1 else seq[0], ocr=lambda img: [])
    r.clicks, r.notes = [], []
    r.click = lambda p, why, *a: r.clicks.append((why, p))
    r.note = r.notes.append
    r._save_game_flags = lambda: None
    if gs is None:
        me = PlayerState("marlow.systems", is_me=True)
        me.hand["grain"] = 1
        gs = types.SimpleNamespace(me=lambda: me)
    r.gs = gs
    return r


def _near(p, q, d=8):
    return abs(p[0] - q[0]) <= d and abs(p[1] - q[1]) <= d


@pytest.mark.parametrize("name,ring", [
    ("popup_045013_road_585_176.png", [("road", "green"), ("ship", "green")]),
    ("popup_045520_knight_520_190.png", [("activate", "red"), ("promote", "green")]),
    ("popup_044802_knight_520_190.png", [("knight", "green")]),
    ("popup_045024_road_520_190.png", [("ship", "green")]),
    ("popup_045514_city_520_190.png", [("city", "red")]),
    ("popup_050302_city_520_190.png", [("wall", "green"), ("trade", "green"), ("politics", "red"), ("science", "red")]),
])
def test_option_rings_and_icons(name, ring):
    img = frame(name)
    got = [(icon_kind(img, r["x"], r["y"])[0], r["colour"]) for r in option_rings(img)]
    assert got == ring


def test_road_option_clicked_on_its_ring(monkeypatch):
    r = _runner([CLEAN, frame("popup_045013_road_585_176.png"), CLEAN], monkeypatch, xy=(785, 246))
    assert r.tap_build(Action("road", ((2, -8), (3, -7))))
    why, p = r.clicks[1]
    assert why == "popup road" and _near(p, (721, 246))


def test_promote_green_clicked_activate_red_refused(monkeypatch):
    img = frame("popup_045520_knight_520_190.png")
    r = _runner([CLEAN, img, CLEAN], monkeypatch, xy=(742, 237))
    r.tap_build(Action("promote", (2, -8)))
    assert r.clicks[1][0] == "popup promote" and _near(r.clicks[1][1], (803, 238))
    r = _runner([CLEAN, img, CLEAN], monkeypatch, xy=(742, 237))
    assert not r.tap_build(Action("activate", (2, -8)))
    assert any("cannot afford activate" in n for n in r.notes)
    assert not any(w.startswith("popup") for w, _ in r.clicks)
    assert "activate" in r._unaffordable


def test_red_city_not_clicked(monkeypatch):
    r = _runner([CLEAN, frame("popup_045514_city_520_190.png"), CLEAN], monkeypatch, xy=(771, 255))
    assert not r.tap_build(Action("city", (3, -7)))
    assert any("cannot afford city" in n for n in r.notes)


def test_leftover_popup_closed_before_the_next_tap(monkeypatch):
    """04:57:20: the next tap landed on the open popup. Close it first, away from the board."""
    r = _runner([frame("popup_044802_knight_520_190.png"), CLEAN, frame("popup_045013_road_585_176.png"), CLEAN],
                monkeypatch, xy=(785, 246))
    r.tap_build(Action("road", ((2, -8), (3, -7))))
    why, p = r.clicks[0]
    assert why == "close leftover popup" and not _near(p, (741, 238), 60)
    assert r.clicks[1][0].startswith("tap spot")


def test_ship_only_edge_learned(monkeypatch):
    t = ((1, -7), (2, -8))
    r = _runner([CLEAN, frame("popup_045024_road_520_190.png"), CLEAN], monkeypatch, xy=(726, 249))
    assert not r.tap_build(Action("road", t))
    assert t in r._sea_edges
    assert not any(w.startswith("popup") for w, _ in r.clicks)


def test_wall_popup_marks_our_city_and_clicks_wall(monkeypatch):
    r = _runner([CLEAN, frame("popup_050302_city_520_190.png"), CLEAN], monkeypatch, xy=(770, 254))
    r._lost_cities = 1
    r.tap_build(Action("wall", (3, -7)))
    assert (3, -7) in r._my_cities and r._lost_cities == 0
    assert r.clicks[1][0] == "popup wall" and _near(r.clicks[1][1], (581, 254))


def test_same_spot_not_tapped_twice_in_a_turn(monkeypatch):
    r = _runner([CLEAN, frame("popup_045024_road_520_190.png"), CLEAN], monkeypatch, xy=(726, 249))
    t = ((1, -7), (2, -8))
    r.tap_build(Action("road", t))
    n = len(r.clicks)
    assert not r.tap_build(Action("ship", t))
    assert len(r.clicks) == n and any("already tapped" in x for x in r.notes)


def _real(frames, monkeypatch):
    br = board_read()
    gs = GameState(br.board)
    for name, c in zip(("marlow.systems", "Lin", "Marianne", "Jean"), COLOURS):
        gs.players.append(PlayerState(name, c, is_me=c == "magenta"))
    return _runner(frames, monkeypatch, br=br, gs=gs)


def test_built_road_verified_on_the_board(monkeypatch):
    t = ((2, -4), (2, -2))
    r = _real([CLEAN, frame("board_045316_road_420_190.png"), frame("board_045318_road_after_420_190.png")],
              monkeypatch)
    assert r.tap_build(Action("road", t))
    assert _near(r.clicks[1][1], (741, 321))
    assert f"built road at {t}" in r.notes and t in r._my_roads


def test_build_not_seen_is_a_failed_tap(monkeypatch):
    t = ((2, -4), (2, -2))
    r = _real([CLEAN, frame("board_045316_road_420_190.png"), frame("board_045123_road_420_190.png")],
              monkeypatch)
    assert not r.tap_build(Action("road", t))
    assert ("road", t) in r._failed_taps and t not in getattr(r, "_my_roads", set())


def test_piece_read_skipped_while_a_popup_is_open(monkeypatch):
    """045316: the open road popup made our road at ((3,-7),(3,-5)) read as green."""
    r = _real([CLEAN], monkeypatch)
    br, e = r.br, ((3, -7), (3, -5))
    assert r._read_pieces(frame("board_045123_road_420_190.png"), br, COLOURS).roads[e] == "magenta"
    assert r._read_pieces(frame("board_045316_road_420_190.png"), br, COLOURS).roads[e] == "magenta"


def test_one_action_per_spot_per_pass(monkeypatch):
    r = _runner([CLEAN], monkeypatch, xy=(0, 0))
    taps = []
    r.tap_build = lambda a: taps.append(a.kind) or False
    r._run_builds(r.gs, [Action("knight", (2, -8)), Action("activate", (2, -8)), Action("road", ((1, -7), (2, -8)))])
    assert taps == ["knight", "road"]


def test_landed_build_ends_the_pass_for_a_replan(monkeypatch):
    r = _runner([CLEAN], monkeypatch, xy=(0, 0))
    taps = []
    r.tap_build = lambda a: taps.append(a.kind) or True
    r._run_builds(r.gs, [Action("road", ((1, -7), (2, -8))), Action("knight", (3, -5))])
    assert taps == ["road"]


def test_sea_only_edge_gets_no_road():
    br = board_read()
    gs = GameState(br.board)
    me = PlayerState("marlow.systems", "magenta", is_me=True)
    me.settlements.add((3, -7))
    gs.players.append(me)
    legal = gs.legal_road_edges(me)
    assert legal
    gs.sea_only_edges = {legal[0]}
    assert legal[0] not in gs.legal_road_edges(me)
