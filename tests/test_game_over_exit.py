"""Owner 2026-10-06: "the solo game never exited". With stop_after_game.flag the runner wrote STOP
and stayed on the end screen (01:46 solo). After game over it must always leave the end screen to
the front page; then idle (flag) or relaunch and wait for the next solo game / an invite."""
import json
import time
from pathlib import Path

import cv2
import pytest

from catanbot import runner as R

END = Path(__file__).parent / "fixtures" / "game_over_014609.png"   # live 01:46:09 ranking screen, QUIT at (653, 720)
HOME_OCR = [{"text": t, "x": 100, "y": 100 + 30 * i, "w": 80, "h": 20}
            for i, t in enumerate(["PLAY", "Leaderboards", "Almanac", "Guilds"])]


class FakeIO:
    """End screen until QUIT is clicked, then the front page."""
    mac = True

    def __init__(self, img, quit_works=True):
        self.img, self.quit_works, self.left = img, quit_works, False

    def snapshot(self):
        return self.img

    def ocr(self, img):
        if self.left:
            return HOME_OCR
        from catanbot import mac
        return mac.ocr(img)


def _runner(tmp_path, monkeypatch, io):
    monkeypatch.setattr(R, "STOP", tmp_path / "STOP")
    monkeypatch.setattr(R.Runner, "STOP_AFTER_GAME", tmp_path / "stop_after_game.flag")
    monkeypatch.setattr(R.Runner, "GAME_OVER_MARK", tmp_path / "last_game_over.json")
    monkeypatch.setattr(R.cv2, "imwrite", lambda *a, **k: True)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    r = R.Runner.__new__(R.Runner)
    r.io, r.cfg, r.notes, r.clicks, r.handled, r.restarts = io, {"bot_name": "MARLOW.SYSTEMS"}, [], [], [], []
    r.note = r.notes.append
    r.save_endgame_shot = lambda *a, **k: None

    def handle(img, texts):                         # the real one records, shares, then writes the mark
        r.handled.append(texts)
        r.GAME_OVER_MARK.write_text(json.dumps({"ranking": r.read_ranking(texts), "at": time.time()}))
    r._handle_game_over = handle
    r.restart_app = r.restarts.append
    r.probe = type("P", (), {"reset": lambda self: None, "card": None})()

    def click(xy, why, ocr=None):
        r.clicks.append((xy, why))
        if "QUIT" in why.upper() and io.quit_works:
            io.left = True
    r.click = click
    return r


def _game_over(r):
    from catanbot import mac
    img = r.io.snapshot()
    ocr = mac.ocr(img)
    texts = [o["text"] for o in ocr]
    assert R.classify(texts, "marlow.systems")[0] == "game_over"
    r.on_game_over(img, ocr, texts, {})


needs_frame = pytest.mark.skipif(not END.exists(), reason="end-screen frame missing")


@needs_frame
def test_stop_after_game_still_leaves_the_end_screen(tmp_path, monkeypatch):
    r = _runner(tmp_path, monkeypatch, FakeIO(R.to_frame(cv2.imread(str(END)))))
    r.STOP_AFTER_GAME.write_text("")
    _game_over(r)
    (x, y), why = r.clicks[0]
    assert "QUIT" in why.upper() and abs(x - 653) < 20 and abs(y - 720) < 20
    assert r.io.left and len(r.handled) == 1 and not r.restarts
    assert R.STOP.exists() and not r.STOP_AFTER_GAME.exists()     # idle at the front page


@needs_frame
def test_already_handled_game_over_still_exits(tmp_path, monkeypatch):
    """A restarted runner on the end screen: no second record/share, but it must still leave."""
    r = _runner(tmp_path, monkeypatch, FakeIO(R.to_frame(cv2.imread(str(END)))))
    r.STOP_AFTER_GAME.write_text("")
    r.io.quit_works = False
    _game_over(r)                                    # first runner: recorded, exit failed
    r2 = _runner(tmp_path, monkeypatch, r.io)        # a restarted runner on the same end screen
    r.io.quit_works = True
    _game_over(r2)
    assert not r2.handled and r.io.left and R.STOP.exists()


@needs_frame
def test_failed_exit_keeps_the_flag_and_tries_again(tmp_path, monkeypatch):
    r = _runner(tmp_path, monkeypatch, FakeIO(R.to_frame(cv2.imread(str(END))), quit_works=False))
    r.STOP_AFTER_GAME.write_text("")
    _game_over(r)
    assert not R.STOP.exists() and r.STOP_AFTER_GAME.exists()   # next look on the end screen retries
    assert len(r.handled) == 1
    r.io.quit_works = True
    _game_over(r)
    assert r.io.left and R.STOP.exists() and len(r.handled) == 1  # recorded once


@needs_frame
def test_without_the_flag_it_relaunches_for_the_next_game(tmp_path, monkeypatch):
    r = _runner(tmp_path, monkeypatch, FakeIO(R.to_frame(cv2.imread(str(END)))))
    _game_over(r)
    assert r.restarts == ["game over"] and not R.STOP.exists()
