"""Round 3 (stage game 0959, 0.3.5): a cap on any prompt, no blind confirm tap, the Inventor's
"Select position for Number Token", save-city offers under the decline cap, a knight the popup no
longer shows is forgotten, the threat ranking weights a climbing rival, ring fallbacks only on
our turn."""
import math
import types
from collections import deque

import numpy as np
import pytest

import catanbot.runner as R
from board_fixture import frame
from catanbot import outlook
from catanbot import progress_cards as pc
from catanbot.brain import Action, Brain
from catanbot.state import Knight
from catanbot.vision.screens import classify
from tests.test_progress_wiring import box, gs, runner  # noqa: F401 (fixture)
from tests.test_round2_0834 import live_layout  # noqa: F401 (fixture)


class _Lat:
    """A flat hex lattice: hex (q, r) centre at (650 + 58 (q + r/2), 420 + 50 r)."""
    size = 33.0

    def hex_px(self, q, r):
        return 650 + 58 * (q + r / 2), 420 + 50 * r

    def nearest_hex(self, x, y):
        r = round((y - 420) / 50)
        return round((x - 650) / 58 - r / 2), r

    def vertex_px(self, v):
        return 600.0, 300.0

    def edge_px(self, e):
        return 600.0, 300.0


class _LiveIO:
    """Not a ReplayIO: the step loop's live-only branches run."""

    def __init__(self, texts=("Select position",)):
        self.img = np.zeros((888, 1372, 3), np.uint8)
        self.texts, self.clicks = list(texts), []

    def snapshot(self):
        return self.img

    def ocr(self, img):
        return [box(t, 600, 300) for t in self.texts]

    def click(self, x, y, why=""):
        self.clicks.append((int(x), int(y), why))

    def paused(self):
        return False


RINGS = [(300 + 60 * k, 300) for k in range(13)]


def _live(gs, tmp_path, monkeypatch, seat=None):
    r = runner(gs, [], tmp_path, dry=False)
    r.io = _LiveIO()
    r._in_play = True
    t = [1000.0]
    monkeypatch.setattr(R.time, "time", lambda: t[0])
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    r._last_click_at = t[0] - 50
    r._rings = lambda img: list(RINGS)
    r._vertex_rings = lambda img: []
    r._recenter_board = lambda img: False
    r._rings_only_on_our_knights = lambda img, texts: False
    r._marker_seat = lambda img: seat
    r.setup_watch = lambda *a, **k: None
    return r, t


# ---- 1. prompt cap + no blind confirm ---------------------------------------------------------
def test_no_blind_confirm_then_cap_ends_the_turn(gs, tmp_path, monkeypatch, live_layout):
    r, t = _live(gs, tmp_path, monkeypatch)
    r._progress_click_at = t[0]                   # our card play: the 5 s textless ring gate
    for _ in range(3):
        r.step()
        t[0] += 10
    assert len(r._fb_tried["ringed_prompt"]) == 3
    assert not any((x, y) == (654, 645) for x, y, _ in r.io.clicks)
    assert all("ringed target" in why for *_, why in r.io.clicks)
    t[0] = 1000.0 + 91 + 60                       # past the 90 s cap from the first fallback
    n = len(r.io.clicks)
    # the turn buttons read as our main phase (the step's own screen read stays textless)
    r._prompt_exit_phase = "main"
    monkeypatch.setattr(R, "turn_phase", lambda img: getattr(r, "_prompt_exit_phase", "idle")
                        if img is r.io.img and r.__dict__.get("_in_exit") else "idle")
    orig = r._prompt_exit

    def exit_(img, state):
        r._in_exit = True
        try:
            orig(img, state)
        finally:
            r._in_exit = False
    r._prompt_exit = exit_
    r.step()
    new = r.io.clicks[n:]
    assert [w for *_, w in new] == ["prompt cap: ending the turn"]
    assert new[0][:2] == tuple(map(int, R.UI["dice_or_end_turn"]))
    t[0] += 10                                    # once per 30 s
    r.step()
    assert len(r.io.clicks) == n + 1


def test_cap_off_keeps_the_old_loop(gs, tmp_path, monkeypatch, live_layout):
    r, t = _live(gs, tmp_path, monkeypatch)
    r.cfg["prompt_cap_s"] = 0
    for _ in range(5):
        r.step()
        t[0] += 50
    assert len(r.io.clicks) == 5 and all("ringed target" in w for *_, w in r.io.clicks)


def test_blind_confirm_cfg_restores_the_bar_tap(gs, tmp_path, monkeypatch, live_layout):
    r, t = _live(gs, tmp_path, monkeypatch)
    r.cfg["fallback_blind_confirm"] = True
    r.step()
    assert any("confirm" in w for *_, w in r.io.clicks)


def test_cap_exit_prefers_the_red_x(gs, tmp_path, monkeypatch, live_layout):
    r, t = _live(gs, tmp_path, monkeypatch)
    r._prompt_since = {"place_robber": t[0] - 100}
    monkeypatch.setattr(R.Runner, "_bar_blob", staticmethod(lambda img, c: (683, 645) if c == "red" else None))
    r._prompt_exit(r.io.img, "place_robber")
    assert r.io.clicks[-1] == (683, 645, "prompt cap place_robber: red X")


def test_clock_clears_on_main(gs, tmp_path, monkeypatch):
    r, t = _live(gs, tmp_path, monkeypatch)
    r._prompt_clock("ringed_prompt")
    t[0] += 100
    assert r._prompt_over_cap("ringed_prompt")
    r._stall_progress("main")
    assert not r._prompt_over_cap("ringed_prompt")


# ---- 2. Inventor: Select position for Number Token -------------------------------------------
def test_classify_select_token():
    assert classify(["Select position for Number Token"], "marlow.systems")[0] == "select_token"
    assert classify(["Selet position fo Number Tokon"], "marlow.systems")[0] == "select_token"


def test_real_frame_rings_the_tokens():
    """Game 0959 10:34:42: the game rings the chips it allows; (654,321) 10, (741,271) 9, (769,321) 4."""
    rings = R.Runner.ring_spots(frame("inventor_tokens_103442_330_180.png"))
    assert len(rings) >= 10
    for x, y in [(654, 321), (741, 271), (769, 321)]:
        assert min(math.hypot(x - a, y - b) for a, b in rings) <= 15
    img = frame("inventor_tokens_103442_330_180.png")
    assert R.Runner._check_xy(img) is None        # no confirm check on this prompt


def _movable(gs):
    return [h for h in gs.board.hexes.values() if pc.inventor_movable(gs, h)]


def test_choose_inventor_only_among_ringed(gs):
    hs = _movable(gs)
    a = hs[0]
    b = next(h for h in hs if pc.PIPS[h.number] != pc.PIPS[a.number])
    got = pc.choose_inventor(gs, gs.me(), allowed={a.coord, b.coord}, force=True)
    assert got is not None and set(got["swap"]) == {a.coord, b.coord}
    assert pc.choose_inventor(gs, gs.me(), allowed={a.coord}, force=True) is None
    free = pc.choose_inventor(gs, gs.me(), allowed={a.coord, b.coord})
    assert free is None or set(free["swap"]) == {a.coord, b.coord}


def test_barren_chip_movable_by_default_off_by_cfg(gs):
    """Owner-confirmed 2026-10-06: barren chips may be swapped; inventor_barren False blocks them."""
    h = next((h for h in gs.board.hexes.values() if h.barren), None)
    if h is None:
        pytest.skip("no barren hex on this board")
    n0 = h.number
    h.number = 4
    try:
        gs.canal_progress = 0
        assert pc.inventor_movable(gs, h)
        pc.INVENTOR_BARREN = False
        assert not pc.inventor_movable(gs, h)
    finally:
        pc.INVENTOR_BARREN = True
        h.number = n0


def test_inventor_taps_two_rings_no_confirm(gs, tmp_path, monkeypatch, live_layout):
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    r = runner(gs, [], tmp_path, dry=False)
    r.io = _LiveIO()
    lat = _Lat()
    r.br = types.SimpleNamespace(lattice=lat)
    hs = _movable(gs)
    a = hs[0]
    b = next(h for h in hs if pc.PIPS[h.number] != pc.PIPS[a.number])
    spots = [tuple(map(int, lat.hex_px(*h.coord))) for h in (a, b)]
    r._rings = lambda img: list(spots)
    assert r._inventor_swap("inventor", planned=[(9, 9), (8, 8)])      # planned pair not ringed
    assert sorted(c[:2] for c in r.io.clicks) == sorted(spots)
    assert not any(c[:2] == (654, 645) for c in r.io.clicks)
    assert r._token_pairs == 1


def test_select_token_handler_leaves_after_two_pairs(gs, tmp_path, monkeypatch, live_layout):
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    monkeypatch.setattr(R, "turn_phase", lambda img: "main")
    r = runner(gs, [], tmp_path, dry=False)
    r.io = _LiveIO()
    r.br = types.SimpleNamespace(lattice=_Lat())
    r._rings = lambda img: []
    r._token_pairs = 2
    r.on_select_token(r.io.img, [], [], {})
    assert r.io.clicks[-1][2] == "prompt cap: ending the turn"
    r._stall_progress("main")
    assert "_token_pairs" not in r.__dict__


# ---- 3. save-the-city offers obey the decline cap; grain asks capped ---------------------------
def _save_city(gs, tmp_path):
    r = runner(gs, [], tmp_path)
    sent = []
    r.offer = lambda give, get: sent.append((dict(give), dict(get)))
    r.brain.save_city_offer = lambda g, made, order: Action("player_trade", give={"lumber": 2}, get={"grain": 1})
    r._turn_n = 1
    return r, sent


def _decline(r, give, get):
    r._pending_offer = (dict(give), dict(get))
    r._remember_decline(*r._pending_offer)


def test_save_city_offer_declined_twice_goes_to_the_bank(gs, tmp_path):
    r, sent = _save_city(gs, tmp_path)
    assert r._save_city_offer(gs)
    _decline(r, *sent[-1])
    r._offers_made = set(); r._turn_n = 7                 # the decline memory expired
    assert r._save_city_offer(gs)
    _decline(r, *sent[-1])
    r._offers_made = set(); r._turn_n = 14
    assert not r._save_city_offer(gs) and len(sent) == 2
    assert r._grain_asks_capped()


def test_save_city_cap_off_sends_again(gs, tmp_path):
    r, sent = _save_city(gs, tmp_path)
    r.cfg["grain_ask_cap"] = False
    for turn in (1, 2, 3):
        r._turn_n = turn
        assert r._save_city_offer(gs)
        _decline(r, *sent[-1])
    assert len(sent) == 3 and not r._grain_asks_capped()


def test_grain_cap_survives_the_turn_reset(gs, tmp_path):
    r, _ = _save_city(gs, tmp_path)
    for _ in range(2):
        _decline(r, {"wool": 2}, {"grain": 1})
    r._offers_made = set()
    if r._grain_asks_capped():                             # the own-turn reset's rule
        r._offers_made.add(("want", "grain"))
    assert ("want", "grain") in r._offers_made
    r.cfg["offer_max_declines"] = 0
    assert not r._grain_asks_capped()


# ---- 4. a knight the popup no longer shows is forgotten -----------------------------------------
def test_build_only_popup_forgets_the_knight(gs, tmp_path, monkeypatch):
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    r = runner(gs, [], tmp_path, dry=False)
    r.io = _LiveIO(texts=())
    r.br = types.SimpleNamespace(lattice=_Lat())
    t = (-3, -1)
    r._my_knights, r._knight_levels = {t: False}, {t: 1}
    gs.me().knights[t] = Knight(active=False)
    r._wait_announcement = lambda: False
    r._close_leftover = lambda: None
    r._settled_snapshot = lambda ref, s: r.io.img
    r._popup_options = lambda img, near: [{"kind": "knight", "colour": "red", "x": 600, "y": 240}]
    r._misclick = lambda *a: False
    r._save_game_flags = lambda: None
    assert r.tap_build(Action("activate", target=t)) is False
    assert t not in r._my_knights and t not in r._knight_levels and t not in gs.me().knights
    assert ("activate", t) in r._failed_taps


def test_knight_memory_cfg_off(gs, tmp_path):
    r = runner(gs, [], tmp_path)
    r.cfg["knight_gone_memory"] = False
    r._my_knights = {(1, 1): True}
    r._forget_knight((1, 1))
    assert (1, 1) in r._my_knights


# ---- 5. threat ranking weights the climber -----------------------------------------------------
def test_rise_flips_the_threat(gs):
    """A at 13 flat, B 7 -> 10 over 6 turns: B gains 3 x 0.08 = 0.24 > the 3/18 = 0.17 gap. (Game
    0959 at 10:34: Lin 13, Jean 10 after +3 -- this flips it; at 10:13 Jean +2 (0.16) would not.)"""
    a, b = gs.opponents()[:2]
    gs.vp_target = 18
    a.vp_visible, b.vp_visible = 13, 10
    o = outlook.Outlook([], {}, {}, set())
    rise = {a.name: 0, b.name: 3}
    assert outlook.rank_threats(gs, o, {}, rise, 0.08)[0][0] == b.name
    assert "rise +3" in outlook.rank_threats(gs, o, {}, rise, 0.08)[0][2]
    assert outlook.rank_threats(gs, o, {}, rise, 0.0)[0][0] == a.name
    rise[b.name] = 2
    assert outlook.rank_threats(gs, o, {}, rise, 0.08)[0][0] == a.name


def test_vp_rise_history(gs):
    br = Brain(setup_weights={})
    a, b = gs.opponents()[:2]
    for va, vb in zip([13] * 7, [7, 7, 8, 8, 9, 10, 10]):
        a.vp_visible, b.vp_visible = va, vb
        rise = br._vp_rise(gs)
    assert rise == {a.name: 0, b.name: 3}
    b.vp_visible = 8                             # a misread low: now = min of the last two
    assert br._vp_rise(gs)[b.name] == 1
    assert isinstance(br._vp_hist[b.name], deque) and br._vp_hist[b.name].maxlen == 8


def test_rise_flag_off_by_set_flag():
    from catanbot.brain import set_flag
    br = Brain(setup_weights={})
    assert br.threat_rise_w == 0.08
    set_flag(br, "threat_rise_w", 0.0)
    assert br.threat_rise_w == 0.0


# ---- 6. ring fallback only on our turn; rings off the board dropped -----------------------------
def test_rival_turn_rings_are_not_ours(gs, tmp_path, monkeypatch):
    r, _ = _live(gs, tmp_path, monkeypatch, seat=2)
    fired = []
    r._forced_fallback = lambda img, why, *a, **k: fired.append(why)
    r.step()
    assert fired == []
    r2, _ = _live(gs, tmp_path, monkeypatch, seat=0)
    r2._forced_fallback = lambda img, why, *a, **k: fired.append(why)
    r2.step()
    assert fired == ["ringed_prompt"]
    r3, _ = _live(gs, tmp_path, monkeypatch, seat=2)
    r3.cfg["ring_fallback_our_turn"] = False
    r3._forced_fallback = lambda img, why, *a, **k: fired.append(why)
    r3.step()
    assert fired == ["ringed_prompt", "ringed_prompt"]


def test_left_edge_ring_dropped(gs, tmp_path):
    r = runner(gs, [], tmp_path)
    r.br = types.SimpleNamespace(lattice=_Lat())
    r._rings = lambda img: [(173, 404), (650, 420)]
    r._vertex_rings = lambda img: []
    img = np.zeros((888, 1372, 3), np.uint8)
    assert r._fallback_rings(img) == [(650, 420)]
    r.cfg["ring_fallback_our_turn"] = False
    assert (173, 404) in r._fallback_rings(img)


# ---- r3 review: the clock clears on the screen the loop really reads, the tries reset per prompt --
def test_clock_clears_on_my_turn(gs, tmp_path, monkeypatch):
    r, t = _live(gs, tmp_path, monkeypatch)
    r._prompt_clock("place_knight")
    t[0] += 100
    assert r._prompt_over_cap("place_knight")
    r._stall_progress("my_turn")
    assert not r._prompt_over_cap("place_knight")
    r._prompt_clock("place_knight")               # a second knight prompt later in the same turn
    assert not r._prompt_over_cap("place_knight")


def test_ring_tries_reset_for_a_new_prompt(gs, tmp_path, monkeypatch, live_layout):
    r, t = _live(gs, tmp_path, monkeypatch)
    r._progress_click_at = t[0]
    for _ in range(3):
        r.step()
        t[0] += 10
    assert len(r._fb_tried["ringed_prompt"]) == 3
    r._stall_progress("my_turn")                  # that prompt is gone
    t[0] += 10
    r._progress_click_at = t[0]                   # a later card play opens a new textless prompt
    n = len(r.io.clicks)
    r.step()
    assert len(r.io.clicks) > n and "ringed target" in r.io.clicks[-1][2]
