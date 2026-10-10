"""Owner 2026-10-09 21:09: "took too much time to trade, if it can do multiple trades at once, it
should try that, build it in". A plan's bank trades the hand pays for together go in one go:
one combined bank dialog (cfg bank_combined, UNVERIFIED live), else back to back single trades."""
import numpy as np
import pytest

import catanbot.runner as R
from catanbot import config
from catanbot.board import canal_board, hex_corners
from catanbot.brain import Action
from catanbot.probe import Probe
from catanbot.rules import MODES
from catanbot.runner import ReplayIO, Runner
from catanbot.state import GameState, PlayerState, new_hand

IMG = np.zeros((888, 1372, 3), np.uint8)


class FakeIO(ReplayIO):
    def __init__(self, screens=None):
        super().__init__([])
        self.screens = list(screens or [])
        self.cur = []

    def snapshot(self):
        self.cur = self.screens.pop(0) if self.screens else []
        return np.zeros((888, 1372, 3), np.uint8)

    def ocr(self, img):
        return self.cur


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(R.time, "sleep", lambda s: None)


@pytest.fixture
def gs():
    b = canal_board(seed=3)
    me = PlayerState("marlow.systems", is_me=True, hand=new_hand(), vp_visible=6)
    a = PlayerState("Hildegard", vp_visible=7, hand_count=6)
    g = GameState(board=b, mode=MODES["great_canal"], players=[me, a])
    land = [h for h in b.hexes.values() if h.is_land and h.number]
    me.settlements.add(hex_corners(*land[0].coord)[0])
    a.cities.add(hex_corners(*land[-1].coord)[3])
    me.hand.update(lumber=3, paper=3, ore=2, grain=1, wool=1)
    return g


def make(gs, tmp_path, screens=None, **over):
    cfg = config.load()
    cfg.update(dry_run=True, **over)
    r = Runner(FakeIO(screens), cfg)
    r.mode, r.gs, r.popup_dir = gs.mode, gs, tmp_path
    r.probe = Probe(tmp_path)
    r.ensure_state = lambda img, texts: gs
    r.bar_hidden = lambda img: False
    r._confirm_accept = lambda *a: False
    r._counter_offer = lambda *a: False
    r._handle_responses = lambda *a: False
    r._want_trade = lambda gs: None
    r.play_progress = lambda phase, img=None, ocr=None: []
    return r


TWO = [Action("bank_trade", give={"lumber": 3}, get={"ore": 1}),
       Action("bank_trade", give={"paper": 3}, get={"grain": 1}),
       Action("city", target=None)]


def whys(r):
    return [w for *_, w in r.io.clicks]


def test_config_switches_default_on():
    d = config.DEFAULTS
    assert d["bank_batch"] is True and d["bank_combined"] is True


def test_two_planned_bank_trades_one_panel_session(gs, tmp_path):
    """Plan bank, bank, city: one 'open trade', every want and give, one trade press."""
    r = make(gs, tmp_path)
    r.brain.plan_turn = lambda g: list(TWO)
    r.on_main(IMG, [], [], {})
    w = whys(r)
    assert w.count("open trade") == 1 and w.count("bank tab") == 1
    presses = [x for x in w if x.startswith("bank trade ")]
    assert len(presses) == 1 and "combined" in presses[0]
    i = w.index("open trade")
    body = w[i:w.index(presses[0])]
    assert body.count("bank want ore") == 1 and body.count("bank want grain") == 1
    assert body.count("bank give lumber") == 3 and body.count("bank give paper") == 3
    # wants before gives (the single-trade order)
    assert max(body.index("bank want ore"), body.index("bank want grain")) < body.index("bank give lumber")
    assert r._sheds == 2


def test_batch_off_keeps_one_trade_per_pass(gs, tmp_path):
    r = make(gs, tmp_path, bank_batch=False)
    r.brain.plan_turn = lambda g: list(TWO)
    r.on_main(IMG, [], [], {})
    w = whys(r)
    assert w.count("open trade") == 1 and [x for x in w if x.startswith("bank trade ")] == ["bank trade {'lumber': 3}->{'ore': 1}"]


def test_back_to_back_without_replan_when_combined_off(gs, tmp_path):
    r = make(gs, tmp_path, bank_combined=False)
    calls = []
    r.brain.plan_turn = lambda g: (calls.append(1), list(TWO))[1]
    r.on_main(IMG, [], [], {})
    w = whys(r)
    assert len(calls) == 1                                   # no re-plan between the two trades
    assert w.count("open trade") == 2
    assert [x for x in w if x.startswith("bank trade ")] == [
        "bank trade {'lumber': 3}->{'ore': 1}", "bank trade {'paper': 3}->{'grain': 1}"]
    assert gs.me().hand["ore"] == 3 and gs.me().hand["grain"] == 2


def test_batch_only_trades_the_hand_pays_together(gs, tmp_path):
    r = make(gs, tmp_path)
    plan = [Action("bank_trade", give={"lumber": 3}, get={"ore": 1}),
            Action("bank_trade", give={"lumber": 3}, get={"wool": 1}),   # only 3 lumber held
            Action("bank_trade", give={"paper": 3}, get={"grain": 1}),
            Action("city", target=None)]
    got = r._bank_batch(gs, plan)
    assert [(a.give, a.get) for a in got] == [({"lumber": 3}, {"ore": 1}), ({"paper": 3}, {"grain": 1})]


def test_batch_never_pulls_a_trade_ahead_of_a_build_it_would_starve(gs, tmp_path):
    """bank ore->wool after a knight that needs the ore: pulled ahead it would leave the knight short."""
    r = make(gs, tmp_path)
    gs.me().hand.update(lumber=3, paper=0, ore=1, grain=1, wool=1)
    plan = [Action("bank_trade", give={"lumber": 3}, get={"grain": 1}),
            Action("knight", target=None),
            Action("bank_trade", give={"ore": 1}, get={"wool": 1}),     # (not a real ratio: shape test)
            Action("road", target=None)]
    got = r._bank_batch(gs, plan)
    assert [(a.give, a.get) for a in got] == [({"lumber": 3}, {"grain": 1})]


def _hand_seq(r, hands):
    seq = list(hands)
    r._hand_now = lambda img=None: dict(seq.pop(0)) if seq else {}


def test_combined_miss_falls_back_to_single_trades_for_the_game(gs, tmp_path):
    """Combined press: dialog still open, hand unchanged -> cancel, then two single trades in the
    same pass, and every later batch skips the combined dialog."""
    r = make(gs, tmp_path, verify_in_replay=True)
    r._read_bank_ratios = lambda img=None: {}
    r._read_dialog_counts = lambda img=None: {}
    h0 = {"lumber": 3, "paper": 3, "ore": 2, "grain": 1}
    h1 = {"lumber": 0, "paper": 3, "ore": 3, "grain": 1}
    h2 = {"lumber": 0, "paper": 0, "ore": 3, "grain": 2}
    open_dialog = [{"text": "Maritime trade", "x": 600, "y": 300, "w": 120, "h": 16, "conf": 1.0}]
    # combined: before, after (unchanged); single 1: before, after; single 2: before, after
    _hand_seq(r, [h0, h0, h0, h1, h1, h2])
    # snapshots: frame before the press, after the press (dialog open), then single trades (closed)
    r.io.screens = [[], [], [], open_dialog] + [[] for _ in range(10)]   # + the dialog frame (bank_settle_poll)
    n = r.bank_offer_batch([({"lumber": 3}, {"ore": 1}), ({"paper": 3}, {"grain": 1})])
    w = whys(r)
    assert n == 2 and r._bank_combined_off is True
    assert "cancel failed bank trade (combined)" in w
    assert w.count("open trade") == 3
    assert [x for x in w if x.startswith("bank trade ")] == [
        "bank trade {'lumber': 3, 'paper': 3}->{'ore': 1, 'grain': 1} (combined)",
        "bank trade {'lumber': 3}->{'ore': 1}", "bank trade {'paper': 3}->{'grain': 1}"]
    r.probe.flush()                          # probes are written on a worker (SPEED_PLAN 1)
    shots = sorted(p.name for p in tmp_path.rglob("*bank_combined*"))
    assert any("before" in s for s in shots) and any("after" in s for s in shots)
    # later in the game: no combined dialog any more
    r.io.clicks.clear()
    r.cfg["verify_in_replay"] = False
    r.bank_offer_batch([({"lumber": 3}, {"ore": 1}), ({"paper": 3}, {"grain": 1})])
    assert not any("combined" in x for x in whys(r)) and whys(r).count("open trade") == 2


def test_combined_landed_checks_the_hand(gs, tmp_path):
    r = make(gs, tmp_path, verify_in_replay=True)
    r._read_bank_ratios = lambda img=None: {"lumber": 3, "paper": 3}
    r._read_dialog_counts = lambda img=None: {}
    _hand_seq(r, [{"lumber": 3, "paper": 3, "ore": 2, "grain": 1},
                  {"lumber": 0, "paper": 0, "ore": 3, "grain": 2}])
    assert r.bank_offer_batch([({"lumber": 3}, {"ore": 1}), ({"paper": 3}, {"grain": 1})]) == 2
    assert not getattr(r, "_bank_combined_off", False)
    assert whys(r).count("open trade") == 1


def test_combined_that_does_not_fit_the_ratios_is_cancelled(gs, tmp_path):
    r = make(gs, tmp_path, verify_in_replay=True)
    r._read_bank_ratios = lambda img=None: {"lumber": 4, "paper": 3}
    r._read_dialog_counts = lambda img=None: {}
    assert r.bank_offer_batch([({"lumber": 3}, {"ore": 1}), ({"paper": 3}, {"grain": 1})]) == 0
    w = whys(r)
    assert w[-1] == "cancel bank trade (ratio)" and not any(x.startswith("bank give") for x in w)


def test_combined_skipped_in_solo_games(gs, tmp_path):
    """Owner 2026-10-09 23:20: combined bank trades only in games with friends; solo goes back to back."""
    r = make(gs, tmp_path, solo=True)
    r.brain.plan_turn = lambda g: list(TWO)
    r.on_main(IMG, [], [], {})
    presses = [x for x in whys(r) if x.startswith("bank trade ")]
    assert len(presses) == 2 and not any("combined" in x for x in presses)
