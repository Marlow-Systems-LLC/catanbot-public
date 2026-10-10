"""Bot 0.4.3 "fast turn". Owner 2026-10-09: "turn can be planned before the dice is rolled but has
to be redone once the dice is rolled, it gets its resources, it uses the cards that can steal more
resources, the plan should be made once all resources are in hand, and then all resources used
quickly". Owner 2026-10-10: "the timer is a color edge around the next button ... the timer does
not change throughout a game and bot can time how much time it has".

Fixtures tests/fixtures/turn_ring/*.png: 100x100 crops round the dice / End-turn button from the
people games of 2026-10-09 (stage data/probe + data/popups, centre 1264,745 of the stage frame)."""
import dataclasses
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

import catanbot.runner as R
from catanbot import config
from catanbot.board import canal_board, hex_corners
from catanbot.brain import Action
from catanbot.probe import Probe
from catanbot.rules import MODES
from catanbot.runner import UI, ReplayIO, Runner
from catanbot.state import GameState, PlayerState, new_hand
from catanbot.turn_clock import phase_cutoffs, ring_read

FIX = Path(__file__).parent / "fixtures" / "turn_ring"
IMG = np.zeros((888, 1372, 3), np.uint8)


class FakeIO(ReplayIO):
    def __init__(self, screens=None, frames=None):
        super().__init__([])
        self.screens = list(screens or [])
        self.frames = list(frames or [])
        self.cur = []

    def snapshot(self):
        self.cur = self.screens.pop(0) if self.screens else []
        if self.frames:
            return self.frames.pop(0) if len(self.frames) > 1 else self.frames[0]
        return np.zeros((888, 1372, 3), np.uint8)

    def ocr(self, img):
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
    a.cities.add(hex_corners(*land[-1].coord)[3])
    me.hand.update(lumber=2, brick=2, wool=1, grain=1)
    return g


def make(gs, tmp_path, screens=None, frames=None, **over):
    cfg = config.load()
    cfg.update(dry_run=True, solo=False, **over)
    r = Runner(FakeIO(screens, frames), cfg)
    r.mode, r.gs, r.popup_dir = gs.mode, gs, tmp_path
    r.probe = Probe(tmp_path)
    r.ensure_state = lambda img, texts: gs
    r.bar_hidden = lambda img: False
    r._confirm_accept = lambda *a: False
    r._counter_offer = lambda *a: False
    r._handle_responses = lambda *a: False
    r._want_trade = lambda gs: None
    r._chat_poll_turn = lambda img, ocr: False
    r.play_progress = lambda phase, img=None, ocr=None: []
    r.tap_chase_robber = lambda g: False
    return r


def whys(r):
    return [w for *_, w in r.io.clicks]


def frame_with(name):
    """A blank frame with the fixture crop pasted round this window's button (UI coordinates)."""
    crop = cv2.imread(str(FIX / f"{name}.png"))
    cx, cy = int(round(UI["dice_or_end_turn"][0] + 1)), int(round(UI["dice_or_end_turn"][1] - 2))
    img = IMG.copy()
    img[cy - 50:cy + 50, cx - 50:cx + 50] = crop
    return img


def test_switches_default_on():
    d = config.DEFAULTS
    for k in ("turn_clock", "turn_ring_read", "steal_first", "build_batch", "preplan", "robber_fast", "log_phases"):
        assert d[k] is True, k
    assert d["turn_timer_default_s"] == 100


# ---- the timer border (step 7) ----

@pytest.mark.parametrize("name,frac", [("ring_040", 0.40), ("ring_063", 0.63), ("ring_089", 0.86), ("ring_100", 1.0)])
def test_ring_reads_the_used_share_of_our_border(name, frac):
    got, hue = ring_read(cv2.imread(str(FIX / f"{name}.png")), (50, 50))
    assert abs(got - frac) < 0.03
    assert 145 <= hue <= 165                         # our magenta border


def test_ring_rival_border_has_their_colour():
    got, hue = ring_read(cv2.imread(str(FIX / "ring_rival_red.png")), (50, 50))
    assert abs(got - 0.34) < 0.03 and (hue <= 10 or hue >= 170)


@pytest.mark.parametrize("name", ["ring_rival_grey", "ring_solo_none"])
def test_no_border_reads_none(name):
    assert ring_read(cv2.imread(str(FIX / f"{name}.png")), (50, 50)) is None


def test_ring_none_for_no_image_or_edge():
    assert ring_read(None, (50, 50)) is None
    assert ring_read(IMG, (5, 5)) is None
    assert ring_read(IMG, (600, 400)) is None


def test_cutoffs_scale_with_the_timer():
    c = phase_cutoffs(100, config.DEFAULTS)
    assert [c[k] for k in ("late", "offer", "chat", "end", "hard")] == pytest.approx([45, 45, 55, 70, 94])
    c = phase_cutoffs(130, config.DEFAULTS)
    assert c["offer"] == pytest.approx(58.5) and c["end"] == pytest.approx(91) and c["hard"] == 124


def test_timer_from_rules_else_default_else_off(gs, tmp_path):
    r = make(gs, tmp_path)
    assert r._turn_timer_s() == 100                  # Round Timer unread, game with people
    r.mode = dataclasses.replace(gs.mode, turn_timer_s=130)
    assert r._turn_timer_s() == 130
    r.mode = gs.mode
    r.cfg["solo"] = True
    assert r._turn_timer_s() is None                 # solo: no timer border, old budgets
    r.cfg["solo"], r.cfg["turn_clock"] = False, False
    assert r._turn_timer_s() is None


def test_border_moves_our_clock_forward_after_two_agreeing_reads(gs, tmp_path):
    r = make(gs, tmp_path)
    r.turn_started = time.time() - 10
    img = frame_with("ring_063")
    r._ring_check(img)
    assert r._turn_used_s() < 12                     # one read: not trusted yet
    r._ring_check(img)
    assert 60 <= r._turn_used_s() <= 66              # 0.63 of 100 s
    r._ring_check(frame_with("ring_040"))            # behind our clock: never moves it back
    r._ring_check(frame_with("ring_040"))
    assert r._turn_used_s() >= 60


def test_border_read_off_keeps_our_clock(gs, tmp_path):
    r = make(gs, tmp_path, turn_ring_read=False)
    r.turn_started = time.time() - 10
    for _ in range(2):
        r._ring_check(frame_with("ring_089"))
    assert r._turn_used_s() < 12


# ---- phase budget and drop order (step 1) ----

def test_offer_cutoff_runs_on_the_timer(gs, tmp_path):
    r = make(gs, tmp_path)
    r.offers_this_turn = 0
    r.turn_started = time.time() - 46
    assert not r._offer_allowed()
    r.mode = dataclasses.replace(gs.mode, turn_timer_s=130)
    assert r._offer_allowed()                        # 130 s timer: cutoff at 58.5 s
    r.mode = gs.mode
    r.turn_started = time.time() - 30
    assert r._offer_allowed()


def test_seconds_left_end_turn_now(gs, tmp_path):
    r = make(gs, tmp_path)
    r.brain.plan_turn = lambda g: pytest.fail("no plan this late")
    r.turn_started = time.time() - 95
    r._turn_begin()
    r.on_main(IMG, [], [], {})
    assert whys(r) == ["end turn (clock)"]


def test_past_end_share_builds_only_then_end_turn(gs, tmp_path):
    r = make(gs, tmp_path)
    offers, built = [], []
    trade = Action("player_trade", give={"wool": 1}, get={"ore": 1})
    road = Action("road", target=next(iter(gs.legal_road_edges(gs.me()))))
    r.brain.plan_turn = lambda g: [trade, road]
    r.offer = lambda give, get: offers.append((give, get))
    r._run_builds = lambda g, p: built.append([a.kind for a in p])
    cards = []
    r.play_progress = lambda phase, img=None, ocr=None: cards.append(phase) or []
    r.turn_started = time.time() - 72
    r.on_main(IMG, [], [], {})
    assert built == [["road"]] and offers == [] and cards == []
    r.brain.plan_turn = lambda g: [trade]
    r.on_main(IMG, [], [], {})
    assert offers == [] and whys(r)[-1] == "end turn"


def test_clock_off_keeps_the_old_45_s_budget(gs, tmp_path):
    r = make(gs, tmp_path, turn_clock=False)
    r.turn_started = time.time() - 95
    assert not r._turn_past("hard") and r._turn_late()
    r.turn_started = time.time() - 40
    assert not r._turn_late()


def test_clock_starts_at_the_roll_frame_capture(gs, tmp_path, sleeps):
    r = make(gs, tmp_path)
    r._snap_at = time.time() - 2
    r.on_roll(IMG, [], [], {})
    assert 1.5 < r._turn_used_s() < 3
    t0 = r._clock_t0
    r.on_roll(IMG, [], [], {})                       # Alchemist: a second roll frame, same turn
    assert r._clock_t0 == t0


# ---- steal first (step 2) ----

def test_robber_chase_before_the_other_cards(gs, tmp_path):
    r = make(gs, tmp_path)
    order = []
    r.tap_chase_robber = lambda g: order.append("chase") or False
    r.play_progress = lambda phase, img=None, ocr=None: order.append(phase) or []
    r.brain.plan_turn = lambda g: order.append("plan") or []
    r._shed_hand = lambda g: False
    r.turn_started = time.time()
    r.on_main(IMG, [], [], {})
    assert order[:3] == ["chase", "main", "plan"]


def test_steal_first_off_keeps_cards_then_chase(gs, tmp_path):
    r = make(gs, tmp_path, steal_first=False)
    order = []
    r.tap_chase_robber = lambda g: order.append("chase") or False
    r.play_progress = lambda phase, img=None, ocr=None: order.append(phase) or []
    r.brain.plan_turn = lambda g: order.append("plan") or []
    r._shed_hand = lambda g: False
    r.turn_started = time.time()
    r.on_main(IMG, [], [], {})
    assert order[:3] == ["main", "chase", "plan"]


# ---- plan once, act many (step 3) ----

def batch_runner(gs, tmp_path, monkeypatch, seen_hands, ok=True, **over):
    r = make(gs, tmp_path, **over)
    monkeypatch.setattr(R, "turn_phase", lambda img: "main")
    r._prompt_pending = lambda ocr: None
    hands = list(seen_hands)
    r._hand_now = lambda img=None: hands.pop(0) if hands else {}
    taps = []

    def tap(a):
        taps.append(a.kind)
        r._last_build_ok = ok
        return True
    r.tap_build = tap
    r.turn_started = time.time()
    return r, taps


def two_builds(gs):
    edges = sorted(gs.legal_road_edges(gs.me()), key=repr)
    return [Action("road", target=edges[0]), Action("road", target=edges[1])]


def test_second_build_without_a_replan_when_the_hand_matches(gs, tmp_path, monkeypatch):
    r, taps = batch_runner(gs, tmp_path, monkeypatch, [{"lumber": 1, "brick": 1, "wool": 1, "grain": 1}])
    r._run_builds(gs, two_builds(gs))
    assert taps == ["road", "road"]


def test_hand_not_as_expected_stops_the_batch(gs, tmp_path, monkeypatch):
    r, taps = batch_runner(gs, tmp_path, monkeypatch, [{"lumber": 2, "brick": 2, "wool": 1, "grain": 1}])
    r._run_builds(gs, two_builds(gs))
    assert taps == ["road"]


def test_unverified_build_stops_the_batch(gs, tmp_path, monkeypatch):
    r, taps = batch_runner(gs, tmp_path, monkeypatch, [{"lumber": 1, "brick": 1, "wool": 1, "grain": 1}], ok=False)
    r._run_builds(gs, two_builds(gs))
    assert taps == ["road"]


def test_next_build_the_hand_cannot_pay_stops_the_batch(gs, tmp_path, monkeypatch):
    gs.me().hand.update(lumber=1, brick=1)
    r, taps = batch_runner(gs, tmp_path, monkeypatch, [{"wool": 1, "grain": 1}])
    r._run_builds(gs, two_builds(gs))
    assert taps == ["road"]


def test_batch_off_one_build_per_pass(gs, tmp_path, monkeypatch):
    r, taps = batch_runner(gs, tmp_path, monkeypatch, [{"lumber": 1, "brick": 1, "wool": 1, "grain": 1}],
                           build_batch=False)
    r._run_builds(gs, two_builds(gs))
    assert taps == ["road"]


def test_prompt_after_a_build_stops_the_batch(gs, tmp_path, monkeypatch):
    r, taps = batch_runner(gs, tmp_path, monkeypatch, [{"lumber": 1, "brick": 1, "wool": 1, "grain": 1}])
    r._prompt_pending = lambda ocr: "metropolis"
    r._run_builds(gs, two_builds(gs))
    assert taps == ["road"]


# ---- pre-plans on rivals' turns (step 4) ----

def prewarmed(gs, tmp_path, **over):
    r = make(gs, tmp_path, **over)
    r._in_play, r.turn_started = True, None
    ranked = []
    r._rank_threat_before_turn = lambda: ranked.append(1)
    plans = []
    r.brain.plan_turn = lambda g: plans.append(dict(g.me().hand)) or [Action("road", target=("plan", len(plans)))]
    for _ in range(12):
        r._prewarm_turn()
    return r, ranked, plans


def test_rival_turn_plans_every_total_once(gs, tmp_path):
    r, ranked, plans = prewarmed(gs, tmp_path)
    assert ranked == [1] and len(plans) == 10
    assert sorted(r._preplan["plans"]) == [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]


def test_rolled_total_with_the_expected_hand_uses_the_preplan(gs, tmp_path):
    r, _, _ = prewarmed(gs, tmp_path)
    want, plan = r._preplan["plans"][6]
    gs.me().hand = dict(want)
    r._last_dice, r.turn_started = (3, 3), time.time()
    assert r._preplan_for(gs) == plan
    assert r._preplan_for(gs) is None                # only the turn's first plan


def test_board_change_or_other_hand_is_a_key_miss(gs, tmp_path):
    r, _, _ = prewarmed(gs, tmp_path)
    want, _ = r._preplan["plans"][6]
    gs.me().hand = dict(want)
    gs.players[1].roads.add(next(iter(gs.legal_road_edges(gs.players[1]) or [((0, 0), (1, 1))])))
    r._last_dice, r.turn_started = (3, 3), time.time()
    assert r._preplan_for(gs) is None
    r2, _, _ = prewarmed(gs, tmp_path)
    gs.me().hand = {"ore": 9}
    r2._last_dice, r2.turn_started = (3, 3), time.time()
    assert r2._preplan_for(gs) is None


def test_preplan_off_plans_nothing(gs, tmp_path):
    r, ranked, plans = prewarmed(gs, tmp_path, preplan=False)
    assert ranked == [] and plans == []


def test_on_main_takes_the_preplan(gs, tmp_path):
    r, _, _ = prewarmed(gs, tmp_path)
    want, plan = r._preplan["plans"][8]
    gs.me().hand = dict(want)
    r._last_dice, r.turn_started = (4, 4), time.time()
    r.brain.plan_turn = lambda g: pytest.fail("planned again")
    got = []
    r._run_builds = lambda g, p: got.append(p)
    r.on_main(IMG, [], [], {})
    assert got == [plan]


# ---- fast 7s (step 5) ----

def choose_player_runner(gs, tmp_path, **over):
    r = make(gs, tmp_path, **over)
    r.game_colors = [("Hildegard", "red"), ("marlow.systems", "blue")]
    hdr = {"text": "Choose a player", "x": 500, "y": 300, "w": 200, "h": 20}
    name = {"text": "Hildegard", "x": 520, "y": 360, "w": 90, "h": 20}
    return r, [hdr, name]


def test_player_pick_reads_the_settled_screen(gs, tmp_path, sleeps):
    r, ocr = choose_player_runner(gs, tmp_path)
    r.on_choose_player(IMG, ocr, [o["text"] for o in ocr], {})
    assert whys(r) == ["choose player Hildegard"] and 1.0 not in sleeps


def test_player_pick_fixed_sleep_when_off(gs, tmp_path, sleeps):
    r, ocr = choose_player_runner(gs, tmp_path, robber_fast=False)
    r.on_choose_player(IMG, ocr, [o["text"] for o in ocr], {})
    assert 1.0 in sleeps


# ---- offer polling (step 6) and the phase line (step 8) ----

def test_offer_out_polls_fast(gs, tmp_path):
    r = make(gs, tmp_path, poll_s=0.8, poll_s_turn=0.6, offer_poll_s=0.3)
    r.turn_started, r._last_click_at = time.time(), 0
    r.offer_sent = time.time() - 2
    assert r._loop_sleep_s() == 0.3
    r.offer_sent = 0
    assert r._loop_sleep_s() == 0.6


def test_phase_line_per_turn(gs, tmp_path):
    r = make(gs, tmp_path)
    notes = []
    r.note = notes.append
    r.turn_started = time.time() - 30
    r._turn_begin()
    r._phase_cur = ("roll", time.time() - 30)
    r._phase("plan")
    r._phase("build")
    r._turn_done("end turn")
    line = next(n for n in notes if n.startswith("phase:"))
    assert "roll 30" in line and "of 100" in line
    assert any(n.startswith("turn done:") for n in notes)
