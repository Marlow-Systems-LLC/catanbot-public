"""Stage 2026-10-09 07:18:41: textless ring prompts after our own card play waited 45 s."""
import numpy as np

import catanbot.runner as R
from tests.test_progress_wiring import box, gs, runner  # noqa: F401


def _step(gs, tmp_path, monkeypatch, card_ago):
    p = [box("Select position", 600, 300)]

    class LiveLikeIO:
        def snapshot(self): return np.zeros((888, 1372, 3), np.uint8)
        def ocr(self, img): return p
        def click(self, x, y, why=""): pass
        def paused(self): return False

    r = runner(gs, [], tmp_path, dry=False)
    r.io = LiveLikeIO()
    r._in_play = True
    t = [1000.0]
    monkeypatch.setattr(R.time, "time", lambda: t[0])
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    r._last_click_at = t[0] - 6
    if card_ago is not None:
        r._progress_click_at = t[0] - card_ago
    fired = []
    r._rings = lambda img: [(300, 300), (400, 300), (500, 300)]
    r._recenter_board = lambda img: False
    r._rings_only_on_our_knights = lambda img, texts: False
    r._forced_fallback = lambda img, why, *a, **k: fired.append(why)
    r.setup_watch = lambda *a, **k: None
    r.step()
    return r, fired


def test_after_card_play_ring_fallback_first_pass(gs, tmp_path, monkeypatch):
    _, fired = _step(gs, tmp_path, monkeypatch, card_ago=10)
    assert fired == ["ringed_prompt"]


def test_no_recent_card_play_waits_45(gs, tmp_path, monkeypatch):
    _, fired = _step(gs, tmp_path, monkeypatch, card_ago=None)
    assert fired == []


def test_cfg_value_and_window(gs, tmp_path):
    r = runner(gs, [], tmp_path)
    r._progress_click_at = R.time.time() - 10
    r.cfg["textless_ring_wait_s"] = 7
    assert r._ring_wait_s() == 7
    r._progress_click_at = R.time.time() - 100
    assert r._ring_wait_s() == 45
