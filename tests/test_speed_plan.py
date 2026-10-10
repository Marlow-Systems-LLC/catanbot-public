"""Owner 2026-10-09: "figure out how the bot can play every move way faster. it runs out of time
almost always when we have a timer (usually 100s)". SPEED_PLAN 1-8: each change has its own
switch; these tests count the clicks and the sleeps each change leaves."""
import time
from types import SimpleNamespace

import numpy as np
import pytest

import catanbot.runner as R
from catanbot import config
from catanbot.board import canal_board, hex_corners
from catanbot.probe import Probe
from catanbot.rules import MODES
from catanbot.runner import UI, ReplayIO, Runner
from catanbot.state import GameState, PlayerState, new_hand

SHAPE = (888, 1372, 3)


class FakeIO(ReplayIO):
    """Scripted frames (arrays) and OCR lists; clicks recorded; sleeps counted (real_sleeps)."""

    def __init__(self, frames=None, ocrs=None):
        super().__init__([])
        self.frames = list(frames or [])
        self.ocrs = list(ocrs or [])
        self.cur = []
        self.ocr_at = []                      # number of clicks made when each OCR ran
        self.real_sleeps = True

    def snapshot(self):
        self.cur = self.ocrs.pop(0) if self.ocrs else []
        if self.frames:
            return self.frames.pop(0) if len(self.frames) > 1 else self.frames[0]
        return np.zeros(SHAPE, np.uint8)

    def ocr(self, img):
        self.ocr_at.append(len(self.clicks))
        return self.cur


@pytest.fixture
def sleeps(monkeypatch):
    got = []
    monkeypatch.setattr(R.time, "sleep", lambda s: got.append(round(s, 3)))
    return got


@pytest.fixture
def gs():
    b = canal_board(seed=3)
    me = PlayerState("marlow.systems", is_me=True, hand=new_hand(), vp_visible=6)
    a = PlayerState("Hildegard", vp_visible=7, hand_count=6)
    g = GameState(board=b, mode=MODES["great_canal"], players=[me, a])
    land = [h for h in b.hexes.values() if h.is_land and h.number]
    me.settlements.add(hex_corners(*land[0].coord)[0])
    me.hand.update(ore=3, grain=1)
    return g


def make(gs, tmp_path, io=None, **over):
    cfg = config.load()
    cfg.update(dry_run=False, solo=False, poll_s=0.8, click_delay_s=0.35, **over)
    r = Runner(io or FakeIO(), cfg)
    r.mode, r.gs, r.popup_dir = gs.mode, gs, tmp_path
    r.probe = Probe(tmp_path)
    r._game_file = lambda name: tmp_path / name
    r._game_save = lambda *a, **k: None
    return r


def whys(r):
    return [w for *_, w in r.io.clicks]


def test_speed_switches_default_on():
    d = config.DEFAULTS
    assert d["probe_async"] is True and d["probe_unknown_on_turn"] is False
    assert d["chat_once_per_turn"] is True and d["bank_settle_poll"] is True
    assert d["trade_wait_floor_s"] == 0 and d["offer_cutoff_s"] == 40 and d["offers_per_turn"] == 1
    assert d["click_delay_fast_s"] == 0.15 and d["poll_s_turn"] == 0.4
    assert d["fast_transitions"] is True and d["scan_after_drag"] == "check" and d["log_timing"] is True


# 1. probes off the turn ------------------------------------------------------------------------
def test_probe_returns_at_once_and_the_file_lands_after_flush(gs, tmp_path, sleeps):
    r = make(gs, tmp_path)
    img = np.random.default_rng(1).integers(0, 255, SHAPE, np.uint8)
    t0 = time.perf_counter()
    assert r._probe("bank_trade_before", img, texts=["x"]) is True
    assert time.perf_counter() - t0 < 0.05
    assert r.probe.flush(10)
    assert list(tmp_path.rglob("*bank_trade_before*"))


def test_failure_probe_still_written_inside_the_turn(gs, tmp_path, sleeps):
    r = make(gs, tmp_path)
    p = r._probe("bank_failed", np.zeros(SHAPE, np.uint8))
    assert p is not None and p is not True and list(tmp_path.rglob("*bank_failed*"))


def _unknown_step(gs, tmp_path, monkeypatch, roll_ago, **over):
    r = make(gs, tmp_path, **over)
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(R, "ROOT", tmp_path)
    r._in_play = True
    r._key_click = ("roll dice", time.time() - roll_ago)
    seats = []
    r._marker_seat = lambda img: seats.append(1) or None
    monkeypatch.setattr(R, "active_seat", lambda *a, **k: seats.append(2) or None)
    r._recenter_board = lambda img: False
    r.setup_watch = lambda *a, **k: None
    r.ensure_state = lambda img, texts: gs
    r.step()
    r.probe.flush(10)
    return r, seats


def test_dice_frame_on_our_turn_writes_no_probe(gs, tmp_path, monkeypatch, sleeps):
    """A screen=unknown frame 4 s after our roll (dice) saves no screen_unknown probe."""
    r, _ = _unknown_step(gs, tmp_path, monkeypatch, roll_ago=4)
    assert not list(tmp_path.rglob("*screen_unknown*"))
    r2, _ = _unknown_step(gs, tmp_path / "old", monkeypatch, roll_ago=4, probe_unknown_on_turn=True)
    assert list((tmp_path / "old").rglob("*screen_unknown*"))


# 7. transition frames --------------------------------------------------------------------------
def test_transition_frame_skips_the_seat_read_and_the_unread_write(gs, tmp_path, monkeypatch, sleeps):
    r, seats = _unknown_step(gs, tmp_path, monkeypatch, roll_ago=1)
    assert seats == [] and not (tmp_path / "data" / "unread").exists()
    assert r._loop_sleep_s() == 0.25
    r2, seats2 = _unknown_step(gs, tmp_path / "old", monkeypatch, roll_ago=1, fast_transitions=False)
    assert seats2 and (tmp_path / "old" / "data" / "unread").exists()


# 2. chat once per own turn ---------------------------------------------------------------------
def test_chat_opened_once_per_own_turn(gs, tmp_path, sleeps):
    r = make(gs, tmp_path, chat=True, chat_poll_s=0, chat_on_turn=True)
    if not all(UI.get(k) for k in ("chat_open", "chat_close", "chat_input", "chat_lines_region")):
        pytest.skip("no chat calibration in data/ui.json")
    r.game_colors = [("marlow.systems", "red"), ("Hildegard", "blue")]
    r.turn_started = time.time()
    img = np.zeros(SHAPE, np.uint8)
    opened = [r._chat_poll_turn(img, []) for _ in range(3)]
    assert opened == [True, False, False]
    assert whys(r) == ["chat open", "chat close"] and sleeps.count(0.6) == 1
    r.turn_started = time.time() + 1                 # our next turn polls again
    assert r._chat_poll_turn(img, []) is True and len(r.io.clicks) == 4


# 3. offers ------------------------------------------------------------------------------------
def test_offer_wait_and_one_offer_per_turn(gs, tmp_path, sleeps):
    r = make(gs, tmp_path, trade_wait_s=12)
    assert r.offer_wait_s() == 12
    r.turn_started = time.time()
    r.offers_this_turn = 0
    assert r._offer_allowed()
    r.offers_this_turn = 1
    assert not r._offer_allowed() and not r._offer_allowed(urgent=True)
    r.cfg["trade_wait_floor_s"] = 15
    assert r.offer_wait_s() == 15                    # the old floor


def test_offer_cutoff_spares_the_city_save_ask(gs, tmp_path, sleeps):
    r = make(gs, tmp_path)
    r.offers_this_turn = 0
    r.turn_started = time.time() - 45
    assert not r._offer_allowed() and r._offer_allowed(urgent=True)
    assert sum("offer cutoff" in m for m in r.log) == 1
    r.turn_started = time.time() - 10
    assert r._offer_allowed()


# 4. bank dialog polls ---------------------------------------------------------------------------
def test_bank_trade_polls_and_reads_once(gs, tmp_path, sleeps):
    a, b, c = (np.full(SHAPE, v, np.uint8) for v in (10, 80, 160))
    io = FakeIO(frames=[a, b, b, c, c])
    io.settle_poll = True
    r = make(gs, tmp_path, io=io, verify_in_replay=True)
    r._click_pause = lambda *a, **k: 0               # count only the waits for the screen
    r.brain.ratio_seen = {"ore": 2}                  # ratios known: no ratio read
    r._read_bank_ratios = lambda img=None: pytest.fail("ratios read again")
    r._read_dialog_counts = lambda img=None: {"ore": 3}
    hands = [{"ore": 3, "grain": 1}, {"ore": 1, "grain": 2}]
    r._hand_now = lambda img=None: dict(hands.pop(0)) if hands else {}
    assert r.bank_offer({"ore": 2}, {"grain": 1}) is True
    assert whys(r) == ["open trade", "bank tab", "bank want grain", "bank give ore", "bank give ore",
                       "bank trade {'ore': 2}->{'grain': 1}"]
    assert sum(s for s in sleeps if s) < 1.0         # old: 0.5 + 1.2 fixed
    assert [n for n in io.ocr_at if n == 6] == [6]   # one OCR after the OK press


def test_bank_trade_fixed_sleeps_when_switched_off(gs, tmp_path, sleeps):
    r = make(gs, tmp_path, verify_in_replay=True, bank_settle_poll=False)
    r._click_pause = lambda *a, **k: 0
    r._read_bank_ratios = lambda img=None: {}
    r._read_dialog_counts = lambda img=None: {}
    hands = [{"ore": 3, "grain": 1}, {"ore": 1, "grain": 2}]
    r._hand_now = lambda img=None: dict(hands.pop(0)) if hands else {}
    assert r.bank_offer({"ore": 2}, {"grain": 1}) is True
    assert 0.5 in sleeps and 1.2 in sleeps


# 5. click pauses -------------------------------------------------------------------------------
def test_click_pause_fast_unless_it_must_settle(gs, tmp_path, sleeps):
    r = make(gs, tmp_path)
    for i in range(10):
        r.click((100 + 7 * i, 100), f"tab {i}")
    assert sleeps == [0.15] * 10
    sleeps.clear()
    r.click((300, 120), "tab x", settle=True)
    r.click((300, 140), "confirm settlement")
    r.click((300, 160), "bank give ore")
    assert sleeps == [0.35] * 3
    sleeps.clear()
    r.cfg["click_delay_fast_s"] = None
    r.click((300, 180), "tab y")
    assert sleeps == [0.35]


# 6. loop wait ---------------------------------------------------------------------------------
def test_loop_wait_on_and_off_our_turn(gs, tmp_path, sleeps):
    r = make(gs, tmp_path)
    r._last_click_at = 0
    r.turn_started = time.time()
    assert r._loop_sleep_s() == 0.4
    r._end_turn_at = time.time() + 1
    assert r._loop_sleep_s() == 0.8
    r.turn_started = None
    assert r._loop_sleep_s() == 0.8


# 8. board check after a drag -------------------------------------------------------------------
class Lat:
    def hex_px(self, q, r):
        return 600 + 60 * q + 30 * r, 420 + 52 * r


def test_drag_keeps_the_scan_when_the_board_did_not_move(gs, tmp_path, monkeypatch, sleeps):
    from catanbot.vision import layout as L
    scans = []
    lat = Lat()
    monkeypatch.setattr(L, "scan", lambda *a, **k: scans.append(1) or SimpleNamespace(
        ok=True, lattice=lat, token_scale=1.0, summary=lambda: "ok", to_json=lambda: {}))
    monkeypatch.setattr(L, "save", lambda *a, **k: None)
    r = make(gs, tmp_path)
    img = np.random.default_rng(2).integers(0, 255, SHAPE, np.uint8)
    r.lattice, r._board_ref = lat, img
    r.scan = SimpleNamespace(ok=True, lattice=lat)
    r._after_drag_scan()
    assert r._scan_ok(img.copy()) and r.scan is not None and scans == []
    assert not any(m.startswith("SCAN") for m in r.log)
    r._after_drag_scan()
    assert r._scan_ok(np.roll(img, 25, axis=1)) and scans == [1]
    assert any("moved" in m for m in r.log)
    r.cfg["scan_after_drag"] = "full"
    r._after_drag_scan()
    assert r.scan is None and r.lattice is None
