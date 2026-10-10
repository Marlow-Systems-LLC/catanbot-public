"""Live 2026-10-03 game 6: the axial origin came from the token nearest the middle, so a
restart re-keyed the board ((3,7) became (1,7)). Canal reads are anchored on the template."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from catanbot.board import hex_center
from catanbot.vision import board_reader as brd

FRAME = Path(__file__).resolve().parent.parent / "data/calib/gc_game4_board.png"
ROI = (90, 180, 1000, 820)


def _keys(br):
    return sorted(h.edge for h in br.board.harbors), sorted(br.board.hexes)


@pytest.mark.skipif(not FRAME.exists(), reason="calibration frame missing")
@pytest.mark.parametrize("shift", [(1, 0), (-1, 1), (2, -1)])
def test_origin_shift_gives_same_keys(monkeypatch, shift):
    img = cv2.imread(str(FRAME))
    base = brd.read_board(img, ROI, canal=True)
    real_fit = brd.fit_lattice

    def shifted_fit(points, W=None):                  # same pixels, origin moved by `shift` hexes
        lat, coords = real_fit(points, W)
        A = lat.A.copy()
        A[:, 2] = A[:, 2] + A[:, :2] @ np.array(hex_center(*shift))
        return brd.Lattice(A, lat.W), coords

    monkeypatch.setattr(brd, "fit_lattice", shifted_fit)
    moved = brd.read_board(img, ROI, canal=True)
    assert _keys(moved) == _keys(base)
    v = next(iter(base.board.vertices))
    assert np.allclose(moved.lattice.vertex_px(v), base.lattice.vertex_px(v), atol=1.0)


@pytest.mark.skipif(not FRAME.exists(), reason="calibration frame missing")
def test_scan_lattice_passed_in_is_anchored_then_fixed():
    """layout.scan fits its own lattice and passes it in: still re-anchored, and the
    lattice that comes back is fixed so later reads never re-key mid-game."""
    img = cv2.imread(str(FRAME))
    base = brd.read_board(img, ROI, canal=True)
    lat, _ = brd.fit_lattice([(x, y) for x, y, *_ in brd.find_tokens(img, ROI)], brd.camera_warp(img.shape))
    A = lat.A.copy()
    A[:, 2] = A[:, 2] + A[:, :2] @ np.array(hex_center(1, 0))
    first = brd.read_board(img, ROI, lattice=brd.Lattice(A, lat.W), canal=True)
    assert _keys(first) == _keys(base)
    assert getattr(first.lattice, "_fixed", False)
    again = brd.read_board(img, ROI, lattice=first.lattice, canal=True)
    assert _keys(again) == _keys(base)
