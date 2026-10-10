"""Stage 2026-10-09 07:20:23: a discard on an all-unreadable hand dropped the paper science 5 needed."""
import numpy as np

import catanbot.runner as R
from catanbot.state import new_hand
from tests.test_progress_wiring import box, gs, runner  # noqa: F401 (fixture)

REAL = dict(new_hand(), brick=6, wool=2, paper=1, cloth=2)
BLANK = {k: None for k in new_hand()}


def _run(gs, tmp_path, monkeypatch, reads, last=None):
    frames = [np.full((10, 10, 3), i, np.uint8) for i in range(len(reads))]
    by_id = {id(f): h for f, h in zip(frames, reads)}
    snaps = iter(frames[1:] + [frames[-1]] * 5)

    class IO:
        def snapshot(self): return next(snaps)
        def ocr(self, img): return []
        def click(self, x, y, why=""): pass
        def paused(self): return False

    r = runner(gs, [], tmp_path, dry=False)
    r.io = IO()
    r.hand.read = lambda img: dict(by_id.get(id(img), BLANK))
    if last:
        r._last_hand = dict(last)

    def ens(img, texts):
        gs.me().hand.clear()
        gs.me().hand.update({k: (v or 0) for k, v in r.hand.read(img).items()})
        return gs
    r.ensure_state = ens
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    monkeypatch.setattr(R.cv2, "imwrite", lambda *a, **k: True)
    picks = []
    r.brain.choose_discard = lambda g, n: picks.append((dict(g.me().hand), n)) or {}
    r.on_discard(frames[0], [], ["Choose 5 more resources to hand over to the robber"], {})
    return r, picks


def test_unreadable_twice_then_real_hand(gs, tmp_path, monkeypatch):
    _, picks = _run(gs, tmp_path, monkeypatch, [BLANK, BLANK, REAL])
    hand, n = picks[0]
    assert n == 5 and hand["brick"] == 6 and hand["paper"] == 1


def test_still_unreadable_plans_from_last_hand(gs, tmp_path, monkeypatch):
    last = {"brick": 6, "wool": 2, "paper": 1, "cloth": 2}
    r, picks = _run(gs, tmp_path, monkeypatch, [BLANK] * 4, last=last)
    hand, _ = picks[0]
    assert {k: v for k, v in hand.items() if v} == last


def test_choose_discard_drops_brick_keeps_paper(gs):
    from catanbot.brain import Brain
    gs.me().hand.clear(); gs.me().hand.update(REAL)
    pick = Brain().choose_discard(gs, 5)
    assert pick.get("brick", 0) >= 1 and not pick.get("paper")


def test_cfg_off_skips_reread(gs, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(R.Runner, "_discard_reread", lambda self, img: calls.append(1) or (img, False))
    r = runner(gs, [], tmp_path)
    r.cfg["discard_reread"] = False
    r._hand_unread = lambda img: True
    r.ensure_state = lambda img, texts: None
    r._blind_discard = lambda *a: None
    monkeypatch.setattr(R.cv2, "imwrite", lambda *a, **k: True)
    r.on_discard(np.zeros((10, 10, 3), np.uint8), [], ["Choose 2 more"], {})
    assert calls == []


def test_progress_discard_retries_the_planned_card(gs, tmp_path, monkeypatch):
    r = runner(gs, [], tmp_path, dry=False)
    held = [("saboteur", box("Saboteur", 100, 600)), ("bishop", box("Bishop", 200, 600)),
            ("engineer", box("Engineer", 300, 600))]

    class IO:
        def snapshot(self): return np.zeros((10, 10, 3), np.uint8)
        def ocr(self, img): return []
    r.io = IO()
    r._read_progress_hand = lambda ocr, img: list(held)       # the drag missed: all still held
    r.drag = lambda *a, **k: None
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    import catanbot.progress_cards as pc
    monkeypatch.setattr(pc, "keep_value", lambda g, m, c, *a: {"saboteur": 0, "bishop": 1, "engineer": 2}[c])
    r.on_discard_progress(None, [], [], {})
    r.on_discard_progress(None, [], [], {})
    assert "saboteur" in r._pdiscard_tried and "bishop" not in r._pdiscard_tried
    assert any("kept saboteur, planned saboteur" in m for m in r.log)
