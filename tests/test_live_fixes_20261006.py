"""Solo game 2026-10-06 (owner watching): board panned 13 px, Crane unread/unused, grain traded
away in the canal, a missed knight tap opened a road popup."""
import copy
from pathlib import Path

import cv2
import pytest

import catanbot.runner as R
from catanbot import progress_cards as pc
from catanbot.board import Board, Hex, hex_corners
from catanbot.brain import Action, Brain
from catanbot.learning import Model
from catanbot.rules import MODES
from catanbot.runner import Runner
from catanbot.state import GameState, PlayerState

from test_probe_controls import make

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(R.time, "sleep", lambda s: None)


def test_13px_pan_is_dragged_back(tmp_path):
    r = make(tmp_path)
    r.mode = object()
    assert r._recenter_board(cv2.imread(str(FIX / "panned13_crane_020535.png"))) is True
    assert any(c[2] == "recenter board" for c in r.io.clicks)
    r2 = make(tmp_path)
    r2.mode = object()
    assert r2._recenter_board(cv2.imread(str(FIX / "board_normal_020436.png"))) is False


def test_hand_reader_keeps_short_crane_title(tmp_path):
    from catanbot import mac
    img = cv2.imread(str(FIX / "panned13_crane_020535.png"))
    r = make(tmp_path)
    r.io.ocr = mac.ocr
    keys = [k for k, _ in r._read_progress_hand(mac.ocr(img), img)]
    assert "crane" in keys and "medicine" in keys


def test_card_key_short_prefix_only_when_asked():
    assert pc.card_key("Cra", min_prefix=3) == "crane"
    with pytest.raises(KeyError):
        pc.card_key("Cra")


def _gs():
    b = Board()
    for q, r_, t in [(0, 0, "mountains"), (1, 0, "forest"), (0, 1, "pasture")]:
        b.add(Hex(q, r_, t, 6))
    me = PlayerState("bot", is_me=True, vp_visible=5)
    gs = GameState(board=b, mode=copy.copy(MODES["great_canal"]), players=[me, PlayerState("x", vp_visible=6)])
    me.cities.add(hex_corners(0, 0)[0])
    return gs, me


def test_crane_starts_a_development_at_level_0():
    gs, me = _gs()
    me.hand = {"paper": 0, "coin": 0, "cloth": 0}
    assert pc.choose_crane(gs, me) is not None          # level 1 is free with Crane
    me.cities.clear()
    assert pc.choose_crane(gs, me) is None


def test_canal_never_gives_grain_away():
    gs, me = _gs()
    me.hand = {"grain": 2, "brick": 2, "ore": 1, "wool": 1, "lumber": 0}
    assert Brain(Model(path=None)).evaluate_offer(gs, "x", {"grain": 1}, {"lumber": 1}) is False


def test_missed_tap_resets_and_retries_once(tmp_path):
    """Owner: "if it wants to do something and clicks on something else by mistake, it should
    reset and try again". Solo 02:05: activate at (2,-2) opened a road popup."""
    r = make(tmp_path)
    t = hex_corners(2, -2)[0]
    a = Action("activate", t)
    r._tapped_spots = {t: False}
    img = r.io.snapshot()
    assert r._misclick(a, t, {"road"}, img) is True
    assert t not in r._tapped_spots                      # the next plan may tap it again
    assert r._misclick(a, t, {"road"}, img) is False     # only once per spot per turn
    assert r._misclick(Action("knight", t), t, {"activate", "move_knight"}, img) is False
    assert r._misclick(Action("road", t), t, {"settlement"}, img) is True
    assert r._misclick(Action("road", (1, 2)), (1, 2), {"road", "ship"}, img) is False


# --- second batch: owner watching 02:30-02:34 ------------------------------------------

def test_improve_dialog_ok_found_when_ocr_misses_it():
    """02:32:19: 'Do you want to improve your city?' -- OCR never read OK, the dialog stayed
    open and ate two bank trades and three knight taps."""
    from catanbot import mac
    ocr = mac.ocr(cv2.imread(str(FIX / "improve_dialog_023219.png")))
    x, y = Runner._dialog_ok_xy(ocr)
    assert 715 < x < 775 and 455 < y < 485                # the OK button (743, 469)
    assert Runner._dialog_ok_xy([{"text": "Build knight", "x": 1, "y": 1, "w": 5, "h": 5}]) is None


def test_leftover_improve_dialog_is_cancelled(tmp_path):
    from catanbot import mac
    img = cv2.imread(str(FIX / "improve_dialog_023219.png"))
    r = make(tmp_path)
    r.io.snapshot = lambda: img
    r.io.ocr = mac.ocr
    assert r._close_leftover() is True
    (x, y, why), = r.io.clicks
    assert why == "close leftover improve dialog" and 540 < x < 590


def _canal_sim():
    from catanbot.sim import Sim
    from catanbot.state import Knight
    sim = Sim("great_canal", 4, seed=7)
    sim.setup()
    gs = sim.gs
    me = gs.me()
    v = next(iter(me.settlements | me.cities))
    me.knights = {v: Knight(level=1, active=False)}
    return gs, me


def test_canal_takes_grain_for_a_spare_card_even_from_a_leader():
    gs, me = _canal_sim()
    me.hand = {"wool": 1, "brick": 3}
    me.vp_visible = 6
    lin = gs.opponents()[0]
    b = Brain(Model(path=None))
    lin.vp_visible = 13                                 # 2x our points: used to be refused
    assert b.evaluate_offer(gs, lin.name, {"wool": 1}, {"grain": 1}) is True
    lin.vp_visible = 14                                 # owner 2026-10-08: 14+ gets no trade
    assert b.evaluate_offer(gs, lin.name, {"wool": 1}, {"grain": 1}) is False
    lin.vp_visible = gs.target - 1                      # one from winning: still refused
    assert b.evaluate_offer(gs, lin.name, {"wool": 1}, {"grain": 1}) is False


def test_bank_never_breaks_an_affordable_settlement(tmp_path):
    """02:33:47: hand paid for a settlement; 'knight one card short' banked 3 brick -> ore."""
    gs, me = _canal_sim()
    r = make(tmp_path)
    r.brain = Brain(Model(path=None))
    me.hand = {"brick": 3, "lumber": 1, "wool": 2, "grain": 1, "ore": 0, "cloth": 1}
    offers = []
    r.bank_offer = lambda give, get: offers.append((give, get))
    r._bank_into_build(gs)
    assert not any("brick" in g and g["brick"] > 2 for g, _ in offers)


def test_settlement_built_where_the_game_offered_it(tmp_path):
    gs, me = _canal_sim()
    r = make(tmp_path)
    me.hand = {"brick": 1, "lumber": 1, "wool": 1, "grain": 1}
    spot = (2, 2)
    r._offered_spots = {"settlement": {spot}}
    tapped = []
    r.tap_build = lambda a: tapped.append((a.kind, a.target)) or True
    assert r._build_offered(gs) is True and tapped == [("settlement", spot)]
    assert r._build_offered(gs) is False                # one try per spot per turn


def test_knights_survive_a_runner_restart(tmp_path):
    r = make(tmp_path)
    r._my_knights = {(-6, 2): True, (6, -2): False}
    r._knight_levels = {(-6, 2): 2}
    r._save_game_flags()
    r2 = make(tmp_path)
    r2._load_game_flags()
    assert r2._my_knights == {(-6, 2): True, (6, -2): False} and r2._knight_levels == {(-6, 2): 2}


def test_refused_smith_marks_its_knights_strong(tmp_path):
    gs, me = _canal_sim()
    r = make(tmp_path)
    r.gs = gs
    me.improvements["politics"] = 2
    r._read_progress_hand = lambda ocr, img: [("smith", {})]
    r._smith_check("smith", [("smith", {})], [(-6, 2)])
    assert r._knight_levels[(-6, 2)] == 2


def test_one_grain_short_city_comes_before_the_settlement(tmp_path):
    """02:33:37: brick 3, wool 2, grain 1, ore 3 -- the city is one grain away; it banked 3 ore
    for lumber (settlement) and then 3 brick for ore (knight)."""
    gs, me = _canal_sim()
    if not me.settlements:
        me.settlements = {next(iter(me.cities))}
    r = make(tmp_path)
    r.brain = Brain(Model(path=None))
    r.brain._trade_targets = lambda g: ["activate", "settlement", "city", "knight"]
    me.hand = {"brick": 3, "wool": 2, "grain": 1, "ore": 3, "cloth": 1}
    offers = []
    r.bank_offer = lambda give, get: offers.append((give, get))
    r._bank_into_build(gs)
    assert offers and offers[0][1] == {"grain": 1} and "ore" not in offers[0][0]


def test_no_buy_back_of_a_resource_sold_this_turn(tmp_path):
    """02:33: 3 ore -> lumber, then 3 brick -> ore in the same turn."""
    gs, me = _canal_sim()
    r = make(tmp_path)
    r.brain = Brain(Model(path=None))
    r.brain._trade_targets = lambda g: ["knight"]
    me.hand = {"brick": 3, "wool": 1, "grain": 1}
    offers = []
    r.bank_offer = lambda give, get: offers.append((give, get))
    r._sold_turn = {"ore"}
    assert not r._bank_into_build(gs) and not offers


def test_no_window_with_game_running_activates_instead_of_launching(tmp_path, monkeypatch):
    """Owner 02:47: the bot launched the game while it was open -> "already running" error."""
    import subprocess
    import time as _t
    r = make(tmp_path)
    calls = []

    class P:
        returncode = 0
    monkeypatch.setattr(subprocess, "run", lambda cmd, **k: calls.append(cmd) or P())
    r._no_window_since = _t.time() - 90
    r._relaunch_at = 0
    r._window_missing(True, "no window")
    assert calls[0][0] == "pgrep" and not any("steam://" in " ".join(c) for c in calls)


def test_landed_improve_allows_the_next_level_same_turn(tmp_path):
    """02:53: 3 paper, plan improve+improve; science 1 landed, the 2nd was 'already tried'."""
    r = make(tmp_path)
    r._failed_taps = {("improve", "science")}
    r._improved("science")
    assert ("improve", "science") not in r._failed_taps and r._improve_levels()["science"] == 1


def test_setup_first_harbour_then_numbers_last():
    """Owner 02:51 board: first settlement 8/9 on the 3:1 (5,-1), last settlement the 5/6 (3,7)."""
    import copy
    from pathlib import Path
    import cv2
    import pytest
    from catanbot.runner import BOARD_ROI
    from catanbot.vision.board_reader import read_board
    from catanbot.state import GameState, PlayerState
    from catanbot.rules import MODES
    f = Path(__file__).parent / "fixtures" / "canal_board_025119.png"
    if not f.exists():
        pytest.skip("fixture missing")
    b = read_board(cv2.imread(str(f)), BOARD_ROI, canal=True).board
    me = PlayerState(name="marlow.systems", color="magenta", is_me=True)
    ps = [PlayerState(name=n, color=c) for n, c in (("Lin", "brown"), ("Marianne", "orange"), ("Jeani", "red"))]
    gs = GameState(board=b, mode=copy.copy(MODES["great_canal"]), players=[me] + ps, vp_target=18)
    ps[2].settlements.add((0, -4))
    br = Brain(Model(path=None), setup_weights={})
    assert br.choose_setup_settlement(gs) == (5, -1)
    me.settlements.add((5, -1)); me.cities.add((-3, 1))
    ps[0].settlements |= {(-1, -1), (5, 5)}; ps[1].settlements |= {(2, -2), (5, -5)}
    assert br.choose_setup_settlement(gs) == (3, 7)


def test_idle_knight_puts_activate_first_and_new_knight_last():
    """02:57: one inactive knight; it offered paper for wool to build a 2nd knight."""
    from catanbot.state import Knight
    gs, me = _canal_sim()
    if not me.cities:
        me.cities = {next(iter(me.settlements))}
    me.knights = {next(iter(me.cities)): Knight(level=1, active=False)}
    t = Brain(Model(path=None))._trade_targets(gs)
    assert t[0] == "activate" and t[-1] == "knight"


def test_paper_is_not_offered_while_science_can_use_it(tmp_path):
    from catanbot.state import Knight
    gs, me = _canal_sim()
    if not me.cities:
        me.cities = {next(iter(me.settlements))}
    me.knights = {next(iter(me.cities)): Knight(level=1, active=True)}
    me.hand = {"lumber": 1, "ore": 4, "paper": 2}
    r = make(tmp_path)
    r.brain = Brain(Model(path=None))
    assert r._improve_commodities(gs) == {"paper", "coin", "cloth"}
    a = r._want_trade(gs)
    assert a is None or "paper" not in (a.give or {})


def test_trade_landing_clears_the_red_activate(tmp_path):
    """02:59: activate red (no grain yet) -> 4 ore -> grain landed -> plan empty, city lost."""
    gs, me = _canal_sim()
    r = make(tmp_path)
    r.gs = gs
    r._unaffordable = {"activate"}
    r._failed_taps = {("activate", (-3, -1)), ("road", (1, 1))}
    r._tapped_spots = {(-3, -1): False}
    r._expect_trade({"ore": 4}, {"grain": 1})
    assert not r._unaffordable and ("activate", (-3, -1)) not in r._failed_taps
    assert ("road", (1, 1)) in r._failed_taps and (-3, -1) not in r._tapped_spots


def test_leave_presses_the_green_check_not_quit_game_again(tmp_path, monkeypatch):
    """Owner 03:07: "fails to click the green confirm button" -- QUIT GAME clicked 13x."""
    import numpy as np
    import catanbot.runner as R
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    r = make(tmp_path)
    img = np.zeros((888, 1372, 3), np.uint8)
    img[620:670, 630:680] = (60, 200, 60)                  # the green check (BGR)
    quit_btn = {"text": "QUIT GAME", "x": 60, "y": 550, "w": 90, "h": 16}
    screens = [[quit_btn], [quit_btn], [{"text": "PLAY", "x": 600, "y": 400, "w": 60, "h": 20}]]

    class IO:
        mac = object()
        def snapshot(self):
            self.cur = screens.pop(0) if screens else []
            return img
        def ocr(self, im):
            return self.cur
    r.io = IO()
    clicks = []
    r.click = lambda xy, why="", *a, **k: clicks.append((xy, why))
    monkeypatch.setattr(R, "classify", lambda texts, me: ("home" if "PLAY" in texts else "unknown", None))
    assert r.leave_game(tries=4)
    assert clicks[0][1] == "leave game: QUIT GAME"
    (x, y), why = clicks[1]
    assert why == "leave game: green check" and abs(x - 655) < 3 and abs(y - 645) < 3
