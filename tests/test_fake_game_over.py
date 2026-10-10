"""Stage 2026-10-10 15:10: after a kill + relaunch the frame was black apart from the macOS toast
"Game Mode: On / Press ... to open Game Overlay". "Game Overlay" matched "game over": the runner
recorded a game with winner=None ranking=[], learned from it, posted the black shot to the group
and pressed the quit gear on nothing. The game-over path must fire only on a real end screen."""
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from catanbot import runner as R
from catanbot.vision.screens import classify

FIX = Path(__file__).parent / "fixtures"
BLACK = FIX / "black_game_mode_toast_151015.png"
END = FIX / "game_over_014609.png"
TOAST = ["Game Mode: On", "Press XS to open Game Overlay"]          # mac.ocr of BLACK
RANKING = ["RESULT", "RANKING", "PLAYER STATISTICS", "GAME STATISTICS", "1 Lin", "2 Marianne", "2 Jean",
           "3 marlow.systems", "Experience gained: +25", "Hide overlay", "BUILD COSTS", "QUIT"]   # mac.ocr of END
WON = ["You've won!", "50", "• Experience", "Ranking", "VIEW PROGRESS", "CONTINUE"]


def _frame(path):
    return R.to_frame(cv2.imread(str(path)))


def _runner(monkeypatch, tmp_path):
    monkeypatch.setattr(R.Runner, "GAME_OVER_MARK", tmp_path / "last_game_over.json")
    monkeypatch.setattr(R.cv2, "imwrite", lambda *a, **k: True)
    r = R.Runner.__new__(R.Runner)
    r.io, r.cfg, r.notes, r.calls = object(), {}, [], []
    r.note = r.notes.append
    r.save_endgame_shot = lambda *a, **k: r.calls.append("shot")
    r._handle_game_over = lambda img, texts: r.calls.append("handle")
    r.leave_game = lambda **k: r.calls.append("leave") or True
    r.restart_app = lambda why: r.calls.append("restart")
    r.probe = type("P", (), {"reset": lambda self: None, "card": None})()
    return r


def test_game_overlay_toast_is_not_game_over():
    assert classify(TOAST)[0] != "game_over"
    assert classify(RANKING)[0] == "game_over"
    assert classify(["GAME OVER"])[0] == "game_over"


def test_black_frame_is_not_an_end_screen():
    img = _frame(BLACK)
    assert R.Runner.end_screen_kind(img, TOAST) is None
    # even with end-screen words OCR'd off it, a mostly black frame is no end screen
    assert R.Runner.end_screen_kind(img, RANKING) is None
    assert R.Runner.end_screen_kind(np.zeros((888, 1372, 3), "uint8"), WON) is None


def test_valid_end_screens():
    assert R.Runner.end_screen_kind(_frame(END), RANKING) == "ranking"
    assert R.Runner.end_screen_kind(_frame(END), WON) == "result"
    assert R.Runner.end_screen_kind(_frame(END), ["PLAYER STATISTICS", "Experience gained: 0"]) is None   # no QUIT, no ranks
    assert R.Runner.end_screen_kind(_frame(END), ["1 Lin", "2 Jean"]) is None                             # no QUIT


def test_black_frame_does_not_record_share_or_leave(monkeypatch, tmp_path):
    r = _runner(monkeypatch, tmp_path)
    r.on_game_over(_frame(BLACK), [], TOAST, {})
    assert r.calls == [] and "not a real end screen" in r.notes[-1]


def test_real_ranking_screen_is_handled(monkeypatch, tmp_path):
    r = _runner(monkeypatch, tmp_path)
    r.on_game_over(_frame(END), [], RANKING, {})
    assert r.calls[:2] == ["shot", "handle"] and "restart" in r.calls


def test_black_frame_is_not_saved_as_an_endgame_shot(monkeypatch, tmp_path):
    written = []
    (tmp_path / "data").mkdir()
    monkeypatch.setattr(R, "ROOT", tmp_path)
    monkeypatch.setattr(R.cv2, "imwrite", lambda p, img: written.append(p) or True)
    r = R.Runner.__new__(R.Runner)
    r.save_endgame_shot(_frame(BLACK), "game_over")
    assert written == []
    r.save_endgame_shot(_frame(END), "game_over")
    assert len(written) == 1


@pytest.mark.skipif(not (BLACK.exists() and END.exists()), reason="frames missing")
def test_ocr_of_the_frames(monkeypatch):
    mac = pytest.importorskip("catanbot.mac")
    try:
        toast = [o["text"] for o in mac.ocr(_frame(BLACK))]
    except Exception as e:                       # noqa: BLE001  (no Vision OCR on this host)
        pytest.skip(f"no OCR: {e}")
    assert classify(toast)[0] != "game_over"
    assert R.Runner.end_screen_kind(_frame(BLACK), toast) is None
    end = [o["text"] for o in mac.ocr(_frame(END))]
    assert R.Runner.end_screen_kind(_frame(END), end) == "ranking"
