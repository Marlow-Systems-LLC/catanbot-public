"""Alchemist dice dialog (live game 13, 17:28): 6 white dice over 6 red dice, green check below."""
from pathlib import Path

import cv2

from catanbot.runner import Runner

FIX = Path(__file__).parent / "fixtures"


def test_alchemist_dialog_found():
    wx, wy, rx, ry, chk = Runner.alchemist_dice_boxes(cv2.imread(str(FIX / "alchemist_dice_dialog.png")))
    assert len(wx) == len(rx) == 6 and ry > wy
    assert wx == sorted(wx) and abs(wx[0] - 425) < 6 and abs(rx[5] - 882) < 6
    assert chk is not None and abs(chk[0] - 653) < 6 and abs(chk[1] - 587) < 6


def test_no_dialog_on_board_frame():
    img = cv2.imread(str(FIX / "alchemist_dice_dialog.png"))
    img[300:520, 380:930] = (40, 70, 110)          # dialog painted over: nothing to find
    assert Runner.alchemist_dice_boxes(img) is None
