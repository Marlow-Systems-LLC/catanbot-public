"""Owner 2026-10-05: all computer players on VETERAN before a solo game starts."""
from catanbot import runner as R

CYCLE = ["BEGINNER", "VETERAN", "MASTER"]


def box(text, y):
    return {"text": text, "x": 550, "y": y, "w": 100, "h": 20}


class FakeIO:
    def __init__(self, levels):
        self.levels = levels

    def snapshot(self):
        return object()

    def ocr(self, img):
        return [box("MARLOW.SYSTE...", 300)] + [box(t, 350 + 43 * i) for i, t in enumerate(self.levels)]


def make(levels):
    r = R.Runner.__new__(R.Runner)
    r.io = FakeIO(levels)
    r.notes = []
    r.note = r.notes.append

    def click(xy, why=""):
        i = round((xy[1] - 10 - 350) / 43)
        r.io.levels[i] = CYCLE[(CYCLE.index(r.io.levels[i]) + 1) % 3]
    r.click = click
    return r


def test_all_bots_become_veteran(monkeypatch):
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    r = make(["VETERAN", "MASTER", "BEGINNER"])
    r._bots_to_veteran(None, r.io.ocr(None))
    assert r.io.levels == ["VETERAN"] * 3
    assert "not all veteran" not in r.notes[-1]
