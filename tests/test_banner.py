"""Banner hand counts + dice pips on saved 4-player C&K frames (values checked by eye)."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from catanbot.vision import banner

D = Path(__file__).parent / "data" / "banner"

# frame: ({name: (resource cards, progress cards)}, (white die, red die))
TRUTH = {
    "f1_discard_progress": ({"marlow.systems": (11, 5), "Lin": (12, 3), "Marianne": (10, 3),
                             "Jean": (6, 0)}, (5, 3)),
    "f2_after_drag": ({"marlow.systems": (13, 4), "Lin": (12, 2), "Marianne": (12, 3),
                       "Jean": (6, 0)}, (5, 3)),
    "f3_ship_prompt": ({"marlow.systems": (9, 0), "Lin": (5, 2), "Marianne": (2, 0),
                        "Jean": (15, 3)}, (5, 4)),
    "f4_progress": ({"marlow.systems": (6, 4), "Lin": (5, 4), "Marianne": (6, 3),
                     "Jean": (3, 2)}, (1, 1)),
}


def _ocr():
    try:
        from catanbot.mac import ocr
        ocr(np.full((40, 40, 3), 255, np.uint8))
        return ocr
    except Exception:                         # noqa: BLE001
        pytest.skip("macOS Vision OCR not available")


@pytest.fixture(scope="module")
def ocr():
    return _ocr()


@pytest.mark.parametrize("frame", sorted(TRUTH))
def test_banner_counts(frame, ocr):
    img = cv2.imread(str(D / f"{frame}.png"))
    assert img is not None
    assert banner.read_banner_counts(img, ocr) == TRUTH[frame][0]


@pytest.mark.parametrize("frame", sorted(TRUTH))
def test_dice(frame):
    img = cv2.imread(str(D / f"{frame}.png"))
    assert banner.read_dice(img) == TRUTH[frame][1]


def test_no_dice_on_blank_frame():
    assert banner.read_dice(np.zeros((888, 1372, 3), np.uint8)) is None


def test_dice_hidden_returns_none():
    img = cv2.imread(str(D / "f1_discard_progress.png"))
    x0, y0, x1, y1 = banner.DICE_ROI
    img[y0:y1, x0:x1] = (30, 60, 110)         # paint the table over the dice
    assert banner.read_dice(img) is None


def test_blank_frame_has_no_banners(ocr):
    assert banner.read_banner_counts(np.zeros((888, 1372, 3), np.uint8), ocr) == {}


def test_unreadable_digit_is_none(ocr):
    """A badge with its digit wiped reads None, not a guess."""
    img = cv2.imread(str(D / "f4_progress.png"))
    ov = banner._find_ovals(img)[0]
    x, y, w, h = ov
    img[y + 3:y + h - 3, x + 3:x + w - 3] = 255     # blank the resource oval's digit
    res, prog = banner.read_banner_counts(img, ocr)["marlow.systems"]
    assert res is None and prog == 4


def test_three_player_layout(ocr):
    """No fixed 4 slots: with one banner gone the other three still read."""
    img = cv2.imread(str(D / "f4_progress.png"))
    img[55:185, 660:1000] = (40, 70, 120)            # table brown over Marianne's banner
    got = banner.read_banner_counts(img, ocr)
    want = dict(TRUTH["f4_progress"][0])
    want.pop("Marianne")
    assert got == want
