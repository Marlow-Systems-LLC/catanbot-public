"""Stage 2026-10-10 03:10-14:58: the runner sat on the SOLO: OFFLINE GAMES menu with solo off.
From 05:19 Apple Vision read nothing there; the pixel-only turn_phase called the orange backdrop
"main" and the runner clicked "open rules info" every ~9 s (159 loop-guard blocks, never a way
out). Frames: tests/fixtures/solo_offline_menu_*.png (data/loops on the server)."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from catanbot import config
from catanbot import runner as R
from catanbot.runner import ReplayIO, Runner
from catanbot.vision.screens import classify, offline_menu

FX = Path(__file__).parent / "fixtures"
MENU = sorted(FX.glob("solo_offline_menu_*.png"))
# OCR of solo_offline_menu_145655.png when Vision works (Apple Vision, 2026-10-10)
MENU_TEXT = ["SOLO: OFFLINE GAMES", "1100 •", "THE GAME", "SEAFARERS", "CITIES &", "KNIGHTS",
             "SPECIAL", "SCENARIOS", "RISE OF THE", "INKA", "RIVALS FOR", "CATAN",
             "SCENARIO SELECTION", "Quickplay", "The First Island", "Quickplay", "The Great Canal"]


def test_menu_frames_read_by_pixels():
    assert len(MENU) == 2
    for p in MENU:
        assert offline_menu(cv2.imread(str(p))), p.name


def test_game_frames_are_not_the_menu():
    game = [p for p in sorted(FX.glob("*.png")) + sorted(FX.glob("*.jpg"))
            if not p.name.startswith(("solo_offline_menu", "solo_setup"))]
    frames = [(p, cv2.imread(str(p))) for p in game]
    frames = [(p, f) for p, f in frames if f is not None and f.shape[:2] == (888, 1372)]
    assert len(frames) >= 15
    for p, f in frames:
        assert not offline_menu(f), p.name
    assert not offline_menu(np.zeros((888, 1372, 3), np.uint8))


def test_menu_text_is_a_menu_not_home_or_main():
    assert classify(MENU_TEXT)[0] == "offline_menu"


class _IO:
    """The 2026-10-10 failure: the menu frame on screen, OCR returning nothing."""
    mac = True

    def __init__(self, frame, texts=()):
        self.frame, self.texts, self.clicks = frame, list(texts), []

    def snapshot(self):
        return self.frame.copy()

    def ocr(self, img):
        return [{"text": t, "x": 600, "y": 80, "w": 80, "h": 20} for t in self.texts]

    def click(self, x, y, why=""):
        self.clicks.append((int(x), int(y), why))

    def paused(self):
        return False


def _runner(monkeypatch, texts=(), solo=False):
    cfg = config.load(); cfg["dry_run"] = False; cfg["solo"] = solo
    r = Runner(ReplayIO([]), cfg)
    r.io = _IO(cv2.imread(str(MENU[-1])), texts)
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    monkeypatch.setattr(r, "_live_waits", lambda: False)
    return r


@pytest.mark.parametrize("texts", [(), MENU_TEXT], ids=["ocr_dead", "ocr_ok"])
def test_solo_off_presses_home_and_never_asks_for_rules(monkeypatch, texts):
    r = _runner(monkeypatch, texts)
    r.step()
    assert r._step_state == "offline_menu"
    assert [c[2] for c in r.io.clicks] == ["menu home button"]
    assert r.io.clicks[0][:2] == Runner.SOLO["home"]
    r.step()                                             # within MENU_HOME_EVERY_S: no second press
    assert len(r.io.clicks) == 1
    assert not any("rules" in c[2] for c in r.io.clicks)


def test_solo_off_falls_back_to_back_arrow(monkeypatch):
    r = _runner(monkeypatch)
    t = [1000.0]
    monkeypatch.setattr(R.time, "time", lambda: t[0])
    for _ in range(3):
        r.step()
        t[0] += Runner.MENU_HOME_EVERY_S + 1
    assert [c[2] for c in r.io.clicks] == ["menu home button", "menu home button", "menu back button"]


def test_solo_on_still_starts_a_solo_game_from_the_menu(monkeypatch):
    r = _runner(monkeypatch, solo=True)
    called = []
    monkeypatch.setattr(r, "start_solo", lambda img: called.append(1))
    r.step()
    assert called == [1] and not r.io.clicks


def test_loop_guard_escalates_to_home_out_of_a_game(monkeypatch):
    r = _runner(monkeypatch)
    r.cfg["loop_guard_in_replay"] = True
    t = [1000.0]
    monkeypatch.setattr(R.time, "time", lambda: t[0])
    r._last_img = r.io.frame
    for loop in range(Runner.LOOP_ESCALATE):
        for _ in range(Runner.LOOP_MAX + 1):
            r.click((33, 602), "open rules info")
            t[0] += 1
        t[0] += Runner.LOOP_BLOCK_S + 1                  # the block runs out, the loop starts again
    whys = [c[2] for c in r.io.clicks]
    assert whys.count("menu home button") == 1 and whys[-1] == "menu home button"


def test_loop_guard_relaunches_in_a_game(monkeypatch):
    r = _runner(monkeypatch)
    r.cfg["loop_guard_in_replay"] = True
    r._in_play = True
    r._last_img = np.zeros((888, 1372, 3), np.uint8)
    restarts = []
    monkeypatch.setattr(r, "restart_app", lambda why: restarts.append(why))
    t = [1000.0]
    monkeypatch.setattr(R.time, "time", lambda: t[0])
    for loop in range(Runner.LOOP_ESCALATE):
        for _ in range(Runner.LOOP_MAX + 1):
            r.click((1263, 747), "end turn")
            t[0] += 1
        t[0] += Runner.LOOP_BLOCK_S + 1
    assert len(restarts) == 1 and "end turn" in restarts[0]
    assert not any("menu" in c[2] for c in r.io.clicks)


def test_dead_ocr_restarts_the_runner_process(monkeypatch):
    r = _runner(monkeypatch)
    t = [1000.0]
    monkeypatch.setattr(R.time, "time", lambda: t[0])
    execs = []
    monkeypatch.setattr(r, "_reexec", lambda: execs.append(1))
    monkeypatch.setattr(r, "alert", lambda msg, halt=True: None)
    img = r.io.frame
    r._ocr_watch(img, [])
    t[0] += 300
    r._ocr_watch(img, [])
    assert not execs
    r._ocr_watch(img, [{"text": "x"}])                   # one read resets the clock
    t[0] += 500
    r._ocr_watch(img, [])
    assert not execs
    t[0] += 601
    r._ocr_watch(img, [])
    assert execs == [1]
    r._ocr_watch(np.zeros((888, 1372, 3), np.uint8), [])  # a black loading frame is not OCR failing
    assert r._ocr_empty_since is None
