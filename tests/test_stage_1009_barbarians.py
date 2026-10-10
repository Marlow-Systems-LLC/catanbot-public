"""Stage 2026-10-09 (Great Canal solo): the only city fell to the barbarians.

* The track read step 6 as 0 ("barbarians far") -- on the 1372-px frame the 3-D ship sits up and
  right of its circle and the old 5-px centre patch stayed light. Readings now only move the ship
  forward; it goes back only after a confirmed attack (or a new voyage seen moving).
* Near the limit with the ship coming, the wool went into a ship; the late turn (Master Merchant
  ate the budget) never asked the table for the grain the idle knight needed."""
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

import catanbot.runner as R
from catanbot import config
from catanbot.brain import Brain
from catanbot.learning import Model
from catanbot.runner import ReplayIO, Runner
from catanbot.sim import Sim
from catanbot.state import Knight
from catanbot.vision import barbarians as bb

CALIB = Path("data/calib")


@pytest.mark.parametrize("step", [1, 5, 6, 7])
def test_live_1372_frames(step):
    crop = cv2.imread(str(CALIB / f"barb_live_1372_step{step}.png"))
    img = np.zeros((888, 1372, 3), np.uint8)
    img[260:460, 300:500] = crop
    assert bb.read_track(img)[0] == step


def _runner():
    r = Runner.__new__(Runner)
    r.notes = []
    r.note = lambda m, *a, **k: r.notes.append(m)
    r.barbarians_attacked = False
    r._my_knights = {(0, 0, 0): True}
    return r


def _feed(r, raw):
    r._barb_read(None if raw is None else (raw, (348, 411), 0.9))
    return r._barb_pos


def test_live_sequence_never_goes_back_on_a_misread():
    """The stage readings 00:33-00:40, in order: the 0s and the lone low reads are held."""
    r = _runner()
    assert [_feed(r, x) for x in (4, 5, 0)] == [4, 5, 5]          # 0 at 6/7 -> ship stays
    assert r._my_knights[(0, 0, 0)] is True and not r.barbarians_attacked
    assert _feed(r, 2) == 5                                       # one low read: held
    assert _feed(r, 4) == 4                                       # 2 then 4: a new voyage
    assert r.barbarians_attacked and r._my_knights[(0, 0, 0)] is False
    r._my_knights[(0, 0, 0)] = True                               # activated again
    r._activated_at = {(0, 0, 0): time.time() + 1}
    assert [_feed(r, x) for x in (5, 7)] == [5, 0]                # 7 = the attack
    assert r._my_knights[(0, 0, 0)] is False
    r._my_knights[(0, 0, 0)] = True
    r._activated_at = {(0, 0, 0): time.time() + 1}
    assert [_feed(r, x) for x in (1, 2, 4, 0, None)] == [1, 2, 4, 4, 4]
    assert r._my_knights[(0, 0, 0)] is True                       # the 0 deactivated nothing


def test_false_zero_never_makes_the_barbarians_far():
    r = _runner()
    for x in (3, 4, 5, 6):
        _feed(r, x)
    for _ in range(5):
        assert _feed(r, 0) == 6
    assert r._my_knights[(0, 0, 0)] is True
    assert r._barb_tracker().near_attack()


def test_attack_text_counts_only_near_the_coast():
    trk = bb.Track()
    assert trk.near_attack()                     # never read: trust the text
    trk.update(2)
    assert not trk.near_attack()                 # ship at 2/7: "attack" text is not an attack
    trk.update(5)
    assert trk.near_attack()


def test_confirmed_attack_resets_the_ship_and_knights():
    r = _runner()
    for x in (5, 6):
        _feed(r, x)
    r._barb_attack("attack message")
    assert r._barb_pos == 0 and r._my_knights[(0, 0, 0)] is False and r.barbarians_attacked
    r._my_knights[(0, 0, 0)] = True
    r._barb_attack("attack message")             # the same popup read again: no 2nd attack
    assert r._my_knights[(0, 0, 0)] is True
    assert _feed(r, 1) == 1


def _gs():
    sim = Sim("great_canal", 4, seed=7)
    sim.setup()
    gs = sim.gs
    me = gs.me()
    if not me.cities:
        me.cities = {next(iter(me.settlements))}
    me.knights = {next(iter(me.cities)): Knight(level=1, active=False)}
    b = Brain(Model(path=None))
    b.harbor_3to1 = False                        # live: no 3:1 harbour, every bank trade 4:1
    gs.ratio_seen = {r: 4 for r in ("lumber", "brick", "wool", "grain", "ore", "cloth", "coin", "paper")}
    return gs, me, b


def test_city_in_danger_needs_the_ship_close():
    gs, me, b = _gs()
    gs.barbarian_pos = 1
    assert not b.city_in_danger(gs)
    gs.barbarian_pos = 5
    assert b.city_in_danger(gs)
    me.knights = {next(iter(me.cities)): Knight(level=1, active=True)}
    gs.barbarian_pos = 5
    assert b.city_in_danger(gs) == b.barb_loss_risk(gs)


def test_near_limit_in_danger_keeps_the_knight_cards():
    """00:37:29: hand at the limit, idle knight, ship coming -- the wool went into a ship."""
    gs, me, b = _gs()
    gs.barbarian_pos = 4
    me.hand = {"lumber": 2, "brick": 2, "wool": 1, "ore": 1, "paper": 1}
    plan = b.plan_turn(gs)
    assert not any(a.kind == "ship" for a in plan)
    spent = {}
    for a in plan:
        if a.kind == "bank_trade":
            for r, n in a.give.items():
                spent[r] = spent.get(r, 0) + n
    assert not spent.get("wool") and not spent.get("ore")
    # the 2nd knight (one more than the cities) is the build that spends them
    assert any(a.kind == "knight" for a in plan), plan


def test_save_city_offer_never_gives_grain_or_commodities():
    gs, me, b = _gs()
    gs.barbarian_pos = 5
    me.hand = {"brick": 2, "lumber": 2, "ore": 3, "paper": 2, "cloth": 2}
    a = b.save_city_offer(gs, set(), R.CARD_ORDER)
    assert a is not None and a.get == {"grain": 1}
    assert sum(a.give.values()) == 2 and set(a.give) <= {"brick", "lumber"}
    assert b.save_city_offer(gs, {("want", "grain")}, R.CARD_ORDER) is None
    gs.barbarian_pos = 1
    assert b.save_city_offer(gs, set(), R.CARD_ORDER) is None   # far: the normal trades


def test_late_turn_still_asks_the_table_for_grain(tmp_path):
    """00:39:36: turn late after Master Merchant, idle knight, ship at 5/7, no grain and no bank
    route -- only improve + roads ran. Now the grain offer goes out before the builds."""
    gs, me, b = _gs()
    gs.barbarian_pos = 5
    me.hand = {"brick": 2, "lumber": 2, "ore": 3, "paper": 2, "cloth": 2}
    cfg = config.load()
    cfg["dry_run"] = True
    cfg["harbor_3to1"] = False
    r = Runner(ReplayIO([]), cfg)
    r.brain, r.mode, r.gs, r.popup_dir = b, gs.mode, gs, tmp_path
    r.ensure_state = lambda img, texts: gs
    r.bar_hidden = lambda img: False
    r._confirm_accept = lambda *a: False
    r._counter_offer = lambda *a: False
    r._handle_responses = lambda *a: False
    r.play_progress = lambda *a, **k: []
    r.turn_started = time.time() - 60      # 0.4.3: past the 45 s offer cut, before the 70 s End-turn share
    sent, built = [], []
    r.offer = lambda give, get: sent.append((give, get))
    r._run_builds = lambda g, p: built.append([a.kind for a in p])
    r.on_main(np.zeros((888, 1372, 3), np.uint8), [], [], {})
    assert sent and sent[0][1] == {"grain": 1} and "grain" not in sent[0][0]
    assert not built
