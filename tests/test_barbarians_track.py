"""Owner 2026-10-05 game 15: the bot never read the barbarian track (danger was always 0)."""
from pathlib import Path

import cv2
import pytest

from board_fixture import canvas
from catanbot.vision import barbarians as bb

SNAP = Path("data/calib/barb_track_g15.png")


@pytest.mark.skipif(not SNAP.exists(), reason="game 15 frame not on this machine")
def test_ship_on_step_4():
    # the dark barbarian ship covers the 4th circle (the drawn ship at the start is only the anchor)
    img = cv2.imread(str(SNAP))
    pos, xy, score = bb.read_track(img)
    assert pos == 4 and score > 0.9
    small = cv2.resize(img, (1308, 736))
    assert bb.read_track(small)[0] == 4


PROBE = {"060045": 3, "060218": 5, "060311": 6, "060353": 1}


@pytest.mark.parametrize("stamp,step", sorted(PROBE.items()))
def test_solo_probe_steps(stamp, step):
    """Solo probe 2026-10-05: 3 -> 5 -> 6, the attack took our city at 06:03:21, then 1.
    The track crop of data/probe/<stamp>_*_native.png, pasted back into its 1920x1080 frame."""
    assert bb.read_track(canvas(f"barb_{stamp}_1920x1080_440_380.png"))[0] == step


def test_counts_from_ocr_boxes():
    texts = [{"text": "2", "x": 572, "y": 469}, {"text": "Bar barian strength", "x": 578, "y": 452},
             {"text": "Active knights", "x": 588, "y": 470}]
    assert bb.read_counts(texts) == {"active": 2}
    assert bb.read_counts([{"text": "3 Barbarian strength", "x": 1, "y": 1}]) == {"strength": 3}
