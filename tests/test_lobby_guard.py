"""Live 2026-10-08 19:06:59: one frame right after "send offer" read as lobby mid-game and wiped
the game (half-board rescan, a lost turn, the intro posted again). In play, a lobby reading must
hold for a few frames before the runner resets."""
import numpy as np

from catanbot import config
from catanbot import runner as R
from catanbot.runner import ReplayIO, Runner


class _IO:
    mac = True

    def __init__(self, screens):
        self.screens, self.i, self.clicks = screens, 0, []

    def snapshot(self):
        self.i = min(self.i + 1, len(self.screens))
        return np.zeros((888, 1372, 3), np.uint8)

    def ocr(self, img):
        return [{"text": t, "x": 600, "y": 400, "w": 80, "h": 20} for t in self.screens[self.i - 1]]

    def click(self, x, y, why=""):
        self.clicks.append((int(x), int(y), why))

    def paused(self):
        return False


def _runner(screens, monkeypatch):
    cfg = config.load(); cfg["dry_run"] = True
    r = Runner(ReplayIO([]), cfg)
    r.io = _IO(screens)
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    r._in_play = True
    r.game_colors = {"marlow.systems": "red"}
    return r


def test_one_lobby_frame_mid_game_keeps_the_game(monkeypatch):
    r = _runner([["Waiting for players"]], monkeypatch)
    assert r.step() == "lobby_unconfirmed"
    assert r._in_play and r.game_colors == {"marlow.systems": "red"}


def test_lobby_held_for_three_frames_resets(monkeypatch):
    r = _runner([["Waiting for players"]] * 3, monkeypatch)
    assert r.step() == "lobby_unconfirmed" and r.step() == "lobby_unconfirmed"
    r.step()
    assert not r._in_play and r.game_colors is None


def test_streak_restarts_after_a_game_frame(monkeypatch):
    r = _runner([["Waiting for players"], ["Waiting for players"], ["nothing here"],
                 ["Waiting for players"]], monkeypatch)
    r.step(); r.step(); r.step()
    assert r.step() == "lobby_unconfirmed" and r._in_play


def test_lobby_outside_play_resets_at_once(monkeypatch):
    r = _runner([["Waiting for players"]], monkeypatch)
    r._in_play = False
    assert r.step() != "lobby_unconfirmed" and r.game_colors is None


def test_our_trade_offer_panel_is_not_the_lobby():
    from catanbot.vision.screens import classify
    frame = ["Rhea", "marlow.systems", "Bramble", "Marianne", "Player trade", "= 2",
             "Waiting for players", "Other Players", "marlow.systems"]       # live 19:14:43
    assert classify(frame)[0] != "lobby"
    assert classify(["Waiting for players", "Start game"])[0] == "lobby"


def test_one_home_frame_mid_game_keeps_the_game(monkeypatch):
    """Review 0.3.1: the home screen words (leaderboards, almanac, guilds) need the same streak."""
    r = _runner([["Leaderboards", "Almanac", "Guilds"]], monkeypatch)
    assert r.step() == "home_unconfirmed"
    assert r._in_play and r.game_colors == {"marlow.systems": "red"}
