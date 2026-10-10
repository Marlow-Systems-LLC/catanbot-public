"""Owner 2026-10-09 21:00 "marlow timed out with 15 resources in its hand": Spy then Intrigue took
a minute of the main phase, the turn timer ended the turn and only a knight activation was built."""
import numpy as np

from catanbot import runner as R
from tests.test_progress_wiring import gs, runner  # noqa: F401 (fixture)


def _offered(gs, tmp_path, hand, phase="main", on=True):
    r = runner(gs, [], tmp_path, dry=True)
    r.cfg["spend_before_slow_cards"] = on
    gs.me().hand.clear()
    gs.me().hand.update(hand)
    seen = []
    r._read_progress_hand = lambda ocr, img: [("spy", {"x": 0, "y": 0, "w": 9, "h": 9}),
                                             ("warlord", {"x": 0, "y": 0, "w": 9, "h": 9})]
    r.brain.progress_play = lambda g, ph, hand_cards=None: seen.append(list(hand_cards)) or None
    r.play_progress(phase, np.zeros((888, 1372, 3), np.uint8), [])
    return seen[0]


BIG = {"lumber": 4, "wool": 2, "grain": 2, "ore": 3, "paper": 3, "brick": 1}


def test_big_hand_spy_waits_for_the_builds(gs, tmp_path):
    assert _offered(gs, tmp_path, BIG) == ["warlord"]


def test_small_hand_spy_plays_first(gs, tmp_path):
    assert _offered(gs, tmp_path, {"lumber": 2, "ore": 1}) == ["spy", "warlord"]


def test_end_turn_and_switch_off_still_offer_the_spy(gs, tmp_path):
    assert _offered(gs, tmp_path, BIG, phase="end_turn") == ["spy", "warlord"]
    assert _offered(gs, tmp_path, BIG, on=False) == ["spy", "warlord"]
