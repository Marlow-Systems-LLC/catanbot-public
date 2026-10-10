"""Stage test 2026-10-08 (crash rejoin + end screen): the six bugs it found."""
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from catanbot import config
from catanbot import runner as R
from catanbot.board import canal_board
from catanbot.brain import Brain
from catanbot.rules import MODES
from catanbot.runner import ReplayIO, Runner
from catanbot.state import GameState, PlayerState, new_hand
from catanbot.vision.screens import classify

FIX = Path(__file__).parent / "fixtures"


# ---- bug 1: a ROOKIE seat is set to VETERAN, and the log says "all veteran" only when true
CYCLE = ["ROOKIE", "VETERAN", "MASTER"]


def _box(text, y, x=550, w=100):
    return {"text": text, "x": x, "y": y, "w": w, "h": 20}


class _LevelIO:
    def __init__(self, levels):
        self.levels = levels

    def snapshot(self):
        return object()

    def ocr(self, img):
        return [_box("MARLOW.SYSTE...", 300)] + [_box(t, 350 + 43 * i) for i, t in enumerate(self.levels)]


def _level_runner(levels):
    r = Runner.__new__(Runner)
    r.io = _LevelIO(levels)
    r.notes = []
    r.note = r.notes.append

    def click(xy, why=""):
        i = round((xy[1] - 10 - 350) / 43)
        r.io.levels[i] = CYCLE[(CYCLE.index(r.io.levels[i]) + 1) % 3]
    r.click = click
    return r


def test_rookie_seat_becomes_veteran(monkeypatch):
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    r = _level_runner(["VETERAN", "ROOKIE", "VETERAN"])
    r._bots_to_veteran(None, r.io.ocr(None))
    assert r.io.levels == ["VETERAN"] * 3
    assert "not all veteran" not in r.notes[-1]


def test_two_seats_read_is_not_all_veteran(monkeypatch):
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    r = _level_runner(["VETERAN", "VETERAN"])
    r._bots_to_veteran(None, r.io.ocr(None))
    assert "not all veteran" in r.notes[-1]


def test_rookie_fixture_reads_three_seats():
    pytest.importorskip("Vision")
    from catanbot import mac
    img = cv2.imread(str(FIX / "solo_setup_rookie_214523.png"))
    levels = Runner.__new__(Runner)._bot_levels(mac.ocr(img))
    assert [o["text"].upper() for o in levels] == ["VETERAN", "ROOKIE", "VETERAN"]


# ---- bug 2: the Diplomat taps a ring that is shown and confirms; else cancels / leaves it
def _diplomat_runner(name, monkeypatch, check=True):
    from catanbot.vision import board_reader as brd
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    img = cv2.imread(str(FIX / name))
    r = Runner.__new__(Runner)
    r.br = brd.read_board(img, R.BOARD_ROI, canal=True)
    r.gs = None
    r.clicks, r.notes = [], []
    r.note = r.notes.append
    r.click = lambda xy, why="", ocr=None: r.clicks.append((tuple(map(int, xy)), why))

    class IO:
        def snapshot(self):
            return img
    r.io = IO()
    r._confirm_if_check = lambda why: check
    return r, img


def test_diplomat_taps_the_wanted_ringed_road(monkeypatch):
    r, img = _diplomat_runner("diplomat_rings_222130.png", monkeypatch)
    assert r._diplomat_pick(((4, 4), (4, 2)), img=img)
    (x, y), _ = r.clicks[0]
    assert math.hypot(x - 795, y - 420) < 8


def test_diplomat_wanted_edge_without_ring_taps_a_shown_ring(monkeypatch):
    # 22:33: the brain's edge ((-1,5),(-1,7)) at (653,471) had no ring and the bot sat
    r, img = _diplomat_runner("diplomat_rings_223710.png", monkeypatch)
    rings = [xy for _, xy in r._edge_rings(img)]
    assert len(rings) >= 5
    assert r._diplomat_pick(((-1, 5), (-1, 7)), img=img)
    (x, y), _ = r.clicks[0]
    assert math.hypot(x - 653, y - 471) > 10
    assert any(math.hypot(x - a, y - b) < 3 for a, b in rings)


def test_diplomat_no_check_cancels_with_red_x(monkeypatch):
    r, img = _diplomat_runner("diplomat_rings_222130.png", monkeypatch, check=False)
    r._red_x = lambda im: True
    assert not r._diplomat_pick(((4, 2), (4, 4)), img=img)
    assert len(r.clicks) == 5 and "cancel" in r.clicks[-1][1]


def test_diplomat_no_rings_does_not_tap_the_board(monkeypatch):
    r, img = _diplomat_runner("remove_city_prompt_224638.png", monkeypatch)
    r._red_x = lambda im: False
    assert not r._diplomat_pick(((4, 2), (4, 4)), img=img)
    assert r.clicks == []
    assert any("stall rescue" in n for n in r.notes)


# ---- bug 3: "Select position to remove City"
def test_remove_city_prompt_classified():
    assert classify(["Select position to remove City"], "marlow.systems")[0] == "remove_city"


def test_remove_city_fixture_classified():
    pytest.importorskip("Vision")
    from catanbot import mac
    img = cv2.imread(str(FIX / "remove_city_prompt_224638.png"))
    texts = [o["text"] for o in mac.ocr(img)]
    assert classify(texts, "marlow.systems")[0] == "remove_city"


def _city_runner(monkeypatch, metros=()):
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    b = canal_board(seed=3)
    me = PlayerState("marlow.systems", is_me=True, hand=new_hand())
    gs = GameState(board=b, mode=MODES["great_canal"], players=[me, PlayerState("Jean")])
    land = [v for v in sorted(b.vertices) if any(h.is_land and h.number for h in b.vertex_hexes(v))]
    a, c = land[0], land[len(land) // 2]
    me.cities |= {a, c}
    me.metropolises |= set(metros)
    r = Runner.__new__(Runner)
    r.gs, r.mode, r.brain = gs, gs.mode, Brain()

    class Lat:
        def vertex_px(self, v):
            return (100, 100)

    class Br:
        lattice = Lat()
    r.br = Br()
    r.clicks, r.notes = [], []
    r.note = r.notes.append
    r.click = lambda xy, why="", ocr=None: r.clicks.append(why)
    r.ensure_state = lambda img, texts: gs
    r._vertex_rings = lambda img: []
    r._green_check = lambda img: False
    r._confirm_if_check = lambda why: True
    r._save_game_flags = lambda: None
    r._probe = lambda *a, **k: None
    return r, gs, a, c


def test_remove_city_gives_up_the_metropolis_city(monkeypatch):
    r, gs, a, c = _city_runner(monkeypatch)
    # our metropolis sits on the better city (the brain would pillage the other one)
    c = max((a, c), key=lambda v: r.brain.city_keep_value(gs, v))
    gs.me().metropolises.add(c)
    r.on_remove_city(None, [], ["Select position to remove City"], {})
    me = gs.me()
    assert str(c) in r.clicks[0] and "metropolis" in r.clicks[0]
    assert c in me.cities and c not in me.metropolises      # the city stays
    assert getattr(r, "_lost_cities", 0) == 0


def test_remove_city_without_metropolis_loses_the_least_kept(monkeypatch):
    r, gs, a, c = _city_runner(monkeypatch)
    want = min((a, c), key=lambda v: r.brain.city_keep_value(gs, v))
    r.on_remove_city(None, [], ["Select position to remove City"], {})
    me = gs.me()
    assert want not in me.cities and want in me.settlements
    assert r._lost_cities == 1


def test_choose_city_to_lose_skips_a_metropolis(monkeypatch):
    r, gs, a, c = _city_runner(monkeypatch)
    worst = min((a, c), key=lambda v: r.brain.city_keep_value(gs, v))
    gs.me().metropolises.add(worst)
    assert r.brain.choose_city_to_lose(gs) != worst


# ---- bug 4: the 240 s rescue click must not reset the stall clock
class _StallIO:
    mac = True

    def __init__(self):
        self.texts = ["nothing here"]

    def snapshot(self):
        return np.zeros((888, 1372, 3), np.uint8)

    def ocr(self, img):
        return [{"text": t, "x": 600, "y": 400, "w": 80, "h": 20} for t in self.texts]

    def click(self, x, y, why=""):
        pass

    def move(self, x, y, why=""):
        pass

    def paused(self):
        return False


def _stall_runner(monkeypatch):
    cfg = config.load(); cfg["dry_run"] = True; cfg["solo"] = True
    r = Runner(ReplayIO([]), cfg)
    r.io = _StallIO()
    now = [10_000.0]
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    monkeypatch.setattr(R.time, "time", lambda: now[0])
    r._probe = lambda *a, **k: None
    r.restarts, r.clicked = [], []
    r.restart_app = lambda reason: r.restarts.append(reason)
    r.click = lambda xy, why="", ocr=None: (r.clicked.append(why), setattr(r, "_last_click_at", now[0]))
    return r, now


def test_stall_rescue_escalates_to_restart(monkeypatch):
    r, now = _stall_runner(monkeypatch)
    r.step()
    for t in (250, 430):
        now[0] = 10_000 + t
        r.step()
    assert sum("rescue" in c for c in r.clicked) == 2      # 240 s and 420 s both fired
    now[0] = 10_610
    assert r.step() == "restart" and r.restarts


def test_new_screen_resets_the_stall_clock(monkeypatch):
    r, now = _stall_runner(monkeypatch)
    r.step()
    now[0] = 10_500
    r.io.texts = ["Roll the dice"]
    r.step()                                               # a new screen: progress
    now[0] = 10_700
    r.io.texts = ["Roll the dice"]
    assert r.step() != "restart" and not r.restarts


# ---- bug 5: the crash relaunch watches for a rejoin like a planned restart
def test_crash_relaunch_sets_the_watch(monkeypatch):
    import subprocess
    r = Runner.__new__(Runner)
    r.notes = []
    r.note = r.notes.append
    r.alert = r.notes.append
    calls = []

    class P:
        returncode = 1                                     # pgrep: the game is not running
    monkeypatch.setattr(subprocess, "run", lambda args, **k: calls.append(args) or P())
    monkeypatch.setattr(R.time, "time", lambda: 1000.0)
    r._no_window_since = 900.0
    r._window_missing(True, "no window")
    assert any("steam://" in str(a) for a in calls)
    assert r._relaunch_watch == (1000.0, False)


# ---- bug 6: the end-game shot has no black bars
def test_endgame_shot_trims_black_bars():
    img = np.zeros((888, 1372, 3), np.uint8)
    img[54:790, 0:1308] = (40, 80, 160)
    out = Runner._trim_black(img)
    assert out.shape[:2] == (736, 1308)
