"""Owner 2026-10-09 (live 20:39-20:40, game 3): "I offered a lot of resources to marlow, it should
have accepted when more than 3:1 and the offering party is not more than 5 points from winning".
Rhea's offer showed its cards in columns (+brick +2 wool +1 ore for -1 lumber); only the
rightmost column was read, so a 4:1 went in as 1:1 and was declined."""
import cv2
import numpy as np

from catanbot import runner as R
from catanbot.brain import Brain
from catanbot.vision import layout as L
from tests.test_progress_wiring import gs  # noqa: F401 (fixture)


def _frame():
    crop = cv2.imread("data/calib/live_offer_multi_2040.png")
    img = np.zeros((1080, 1920, 3), np.uint8)
    x0, y0 = int(R.LAY.x(1000, "r")), int(R.LAY.y(420, "c"))
    img[y0:y0 + 200, x0:x0 + 230] = crop
    return img, x0, y0


def test_inner_offer_column_is_read():
    old = R.LAY
    try:
        R.set_layout(L.Layout(1920, 1080))
        img, x0, y0 = _frame()
        r = R.Runner.__new__(R.Runner)
        assert r._offer_card(img, y0 + 32)[0] == "ore"            # rightmost "+1"
        assert r._offer_card(img, y0 + 92)[0] == "lumber"         # our "-1"
        for dx in (92, 100, 106):                                 # "+2" token, OCR jitter
            assert r._offer_card(img, y0 + 32, dx)[0] == "wool"
    finally:
        R.set_layout(old)


def _offer(gs, partner_vp, get, give):
    b = Brain()
    me = gs.me()
    me.hand.clear()
    me.hand.update({"lumber": 1, "wool": 1, "ore": 1, "paper": 1, "cloth": 2, "coin": 1})
    p = gs.opponents()[0]
    p.vp_visible = partner_vp
    me.vp_visible = 7
    return b, b.evaluate_offer(gs, p.name, give, get)


def test_four_for_one_from_a_player_far_from_winning_is_taken(gs):
    _, ok = _offer(gs, gs.target - 9, {"brick": 1, "wool": 2, "ore": 1}, {"lumber": 1})
    assert ok


def test_four_for_one_from_a_player_near_winning_is_not_forced(gs):
    b, ok = _offer(gs, gs.target - 5, {"brick": 1, "wool": 2, "ore": 1}, {"lumber": 1})
    b2 = Brain(); b2.generous_accept = False
    assert ok == b2.evaluate_offer(gs, gs.opponents()[0].name, {"lumber": 1}, {"brick": 1, "wool": 2, "ore": 1})


def test_three_for_one_is_not_enough_and_grain_still_never_goes(gs):
    b = Brain()
    gs.me().hand.update({"grain": 1, "lumber": 1})
    p = gs.opponents()[0]
    p.vp_visible = 2
    assert not b.evaluate_offer(gs, p.name, {"grain": 1}, {"brick": 2, "wool": 2, "ore": 2})


def test_two_for_one_from_the_leader_is_not_forced(gs):
    """Owner 2026-10-09: "even 2:1 if it makes sense and it does not help the winning player"."""
    b = Brain()
    me = gs.me()
    me.hand.update({"lumber": 1})
    me.vp_visible = 7
    for q in gs.opponents():
        q.vp_visible = 5
    p = gs.opponents()[0]
    p.vp_visible = 10                                    # in front, but > 5 from winning
    assert b._is_winning(gs, p)
    b2 = Brain(); b2.generous_accept = False
    give, get = {"lumber": 1}, {"wool": 1, "ore": 1}
    assert b.evaluate_offer(gs, p.name, give, get) == b2.evaluate_offer(gs, p.name, give, get)


def test_two_for_one_from_a_trailing_player_is_taken_when_it_helps(gs, monkeypatch):
    b = Brain()
    me = gs.me()
    me.hand.update({"lumber": 1})
    me.vp_visible = 7
    for q in gs.opponents():
        q.vp_visible = 4
    p = gs.opponents()[0]
    monkeypatch.setattr(b, "my_prob", lambda g: 0.10)
    monkeypatch.setattr(b, "_best_followup", lambda g: 0.12)
    monkeypatch.setattr(b, "_focus_player", lambda g: p)   # the BLOCK selector no longer vetoes it
    assert b.evaluate_offer(gs, p.name, {"lumber": 1}, {"wool": 1, "ore": 1})
    monkeypatch.setattr(b, "_best_followup", lambda g: 0.09)
    assert not b.evaluate_offer(gs, p.name, {"lumber": 1}, {"wool": 1, "ore": 1})
