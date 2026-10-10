"""Dug canal hexes read off the board picture (live game 3, 2026-10-02)."""
import os

import cv2
import pytest

from catanbot.runner import BOARD_ROI
from catanbot.vision.board_reader import read_board
from catanbot.vision.canal_reader import canal_path, read_canal_dug

REF = "data/calib/live_g3_setup.png"
pytestmark = pytest.mark.skipif(not (os.path.exists(REF) and os.path.isdir("data/trade_wait")),
                                reason="live frames not in this checkout")


@pytest.fixture(scope="module")
def setup():
    ref = cv2.imread(REF)
    return ref, read_board(ref, BOARD_ROI, canal=True)


def dug(setup, name):
    ref, br = setup
    order = [c for c, _, _ in canal_path(br.board)]
    got = read_canal_dug(cv2.imread(f"data/trade_wait/{name}.png"), ref, br.lattice, br.board)
    return sorted(order.index(c) for c in got)          # path positions, basin = 0


def test_path_follows_the_nine_chips(setup):
    _, br = setup
    assert {c for c, _, _ in canal_path(br.board)} == br.board.canal_hexes


def test_nothing_dug_at_setup(setup):
    ref, br = setup
    assert read_canal_dug(ref, ref, br.lattice, br.board) == set()


def test_trench_grows_through_the_game(setup):
    assert dug(setup, "144828") == [4]
    assert dug(setup, "151436") == [1, 3, 4]
    assert dug(setup, "152102") == [1, 3, 4, 5, 6]


def test_lattice_kept_between_reads(setup):
    ref, br = setup
    assert read_board(ref, BOARD_ROI, lattice=br.lattice, canal=True).lattice is br.lattice
