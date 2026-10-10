"""Stage 2026-10-09 game 0738: "UI flow unconfirmed -- trying it once this game" came back 9 times
for flows that had already played (alchemist 2, commercial harbor 3, monopoly 3, inventor 1)."""
import sys

import catanbot.runner as R
from catanbot import progress_cards as pc

sys.path.insert(0, "tests")
from test_progress_wiring import box, gs, runner  # noqa: E402,F401  (gs is a fixture)


def _live(gs, tmp_path, monkeypatch, plays):
    """Two own turns, one alchemist each: held -> opened (no Play) -> gone -> no prompt -> empty."""
    monkeypatch.setitem(R.UI, "progress_live", True)
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    monkeypatch.setattr(pc, "can_play", lambda *a, **k: (True, ""))
    r = runner(gs, [], tmp_path, dry=False)
    io = r.io
    io.held = True
    io.snapshot = lambda: R.np.zeros((888, 1372, 3), R.np.uint8)
    io.ocr = lambda img: [box("Alchemist")] if io.held else []
    drag = io.drag

    def dragged(p0, p1, why=""):
        drag(p0, p1, why)
        io.held = False                     # the drag played it: the card left the hand
    io.drag = dragged
    r.brain.progress_play = lambda g, phase, hand_cards=None: ("alchemist", {"dice": (6, 2)}) \
        if "alchemist" in (hand_cards or []) else None
    r._alchemist_pick = lambda dice: None
    shots = []
    r._probe = lambda kind, img=None, note=None, texts=None: shots.append(kind)
    r._save_progress_step = lambda img, tag: shots.append(tag)
    r._park_pointer = lambda why="": None
    r._ledger = lambda: type("L", (), {"on_my_card": lambda *a, **k: None})()
    return r, shots


def test_second_play_of_a_confirmed_card_skips_the_captures(gs, tmp_path, monkeypatch):
    r, shots = _live(gs, tmp_path, monkeypatch, 2)
    assert r.play_progress("pre_roll") == ["alchemist"]
    first = list(shots)
    assert "progress_once_alchemist" in first
    r._progress_last_phase = "main"
    r.io.held = True                        # next own turn: another alchemist drawn
    shots.clear()
    n0 = len(r.log)
    assert r.play_progress("pre_roll") == ["alchemist"]
    second = r.log[n0:]
    assert not any("UI flow unconfirmed -- trying it once" in m for m in second)
    assert any("confirmed" in m for m in second)
    assert shots == []


def test_flag_off_is_the_old_capture(gs, tmp_path, monkeypatch):
    r, shots = _live(gs, tmp_path, monkeypatch, 2)
    r.cfg["progress_skip_confirmed"] = False
    r.play_progress("pre_roll")
    r._progress_last_phase = "main"
    r.io.held = True
    n0 = len(r.log)
    r.play_progress("pre_roll")
    assert any("UI flow unconfirmed -- trying it once" in m for m in r.log[n0:])


def test_new_game_clears_the_confirmed_set(gs, tmp_path, monkeypatch):
    r, _ = _live(gs, tmp_path, monkeypatch, 1)
    r.play_progress("pre_roll")
    assert "alchemist" in r._progress_confirmed
    assert "_progress_confirmed" in R.Runner._PER_GAME_LAZY
