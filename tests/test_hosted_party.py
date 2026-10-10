"""Live 2026-10-09 18:09: an inviter left their custom match and the party passed to us; the front
page (our name + crown + "Host", CUSTOM + LEAVE) read as home and we sat there, busy to friends."""
import numpy as np

from catanbot import config
from catanbot import runner as R
from catanbot.runner import ReplayIO, Runner

HOSTED = ["PROFILE", "LEVEL: 6", "MARLOW.SYSTEMS", "Host", "MESSAGES", "FRIENDS", "GUILDS",
          "LEADERBOARDS", "ALMANAC", "SETTINGS", "CUSTOM", "LEAVE", "PLAY"]


class _IO:
    mac = True

    def __init__(self, after):
        self.after = after

    def snapshot(self):
        return np.zeros((888, 1372, 3), np.uint8)

    def ocr(self, img):
        return [{"text": t, "x": 600, "y": 400, "w": 80, "h": 20} for t in self.after]


def _setup(monkeypatch, after=()):
    cfg = config.load(); cfg["dry_run"] = True; cfg["solo"] = False
    r = Runner(ReplayIO([]), cfg)
    r.io = _IO(list(after))
    clicks = []
    r.click = lambda xy, why, ocr=None: clicks.append(why)
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    return r, clicks


def _ocr(texts):
    return [{"text": t, "x": 100 + 40 * i, "y": 700, "w": 60, "h": 20} for i, t in enumerate(texts)]


def test_hosted_party_left_after_wait(monkeypatch):
    r, clicks = _setup(monkeypatch, after=["Leave the party?", "NO", "YES"])
    t = [1000.0]
    monkeypatch.setattr(R.time, "time", lambda: t[0])
    r.on_home(None, _ocr(HOSTED), HOSTED, {})
    assert clicks == []                                  # first sight: wait
    t[0] += 31
    r.on_home(None, _ocr(HOSTED), HOSTED, {})
    assert clicks == ["home: LEAVE the hosted party", "home: confirm LEAVE party"]


def test_plain_home_and_switch_off(monkeypatch):
    r, clicks = _setup(monkeypatch)
    plain = [x for x in HOSTED if x not in ("Host", "LEAVE")]
    t = [1000.0]
    monkeypatch.setattr(R.time, "time", lambda: t[0])
    for _ in range(3):
        r.on_home(None, _ocr(plain), plain, {}); t[0] += 40
    assert clicks == []
    r.cfg["leave_hosted_party_s"] = 0
    for _ in range(3):
        r.on_home(None, _ocr(HOSTED), HOSTED, {}); t[0] += 40
    assert clicks == []
