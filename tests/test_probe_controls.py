"""HANDOFF_v02 Part A: probe capture, per-action scorecard, forced-prompt fallbacks, bank-trade
verification (frame replays where a live frame exists)."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import catanbot.runner as R
from catanbot import config
from catanbot.probe import Probe
from catanbot.runner import ReplayIO, Runner

DATA = Path(__file__).parent / "data"


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


def make(tmp_path, screens=None, **cfg_over):
    cfg = config.load()
    cfg.update(dry_run=False, **cfg_over)
    r = Runner(FakeIO(screens), cfg)
    r.probe = Probe(tmp_path)
    return r


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(R.time, "sleep", lambda s: None)


def test_scorecard_counts_and_dumps(tmp_path):
    p = Probe(tmp_path)
    p.tried("roll"); p.result("roll", True)
    p.tried("bank_trade"); p.result("bank_trade", False, "dialog still open")
    p.tried("build_road"); p.result("build_road", None)
    assert not p.clean()
    out = json.loads(p.dump(final=True).read_text())
    assert out["actions"]["roll"] == {"tries": 1, "landed": 1, "failed": 0, "unverified": 0}
    assert out["actions"]["bank_trade"]["failed"] == 1 and out["failures"][0]["why"] == "dialog still open"
    assert out["final"] is True and out["clean"] is False


def test_probe_save_caps_frames(tmp_path):
    p = Probe(tmp_path)
    p.CAP = 2
    img = np.zeros((10, 10, 3), np.uint8)
    assert p.save("a", img, note={"x": 1}) and p.save("b", img) and p.save("c", img) is None
    j = json.loads(next(tmp_path.glob("data/probe/*_a.json")).read_text())
    assert j["kind"] == "a" and j["note"] == {"x": 1}


def test_prompt_lands_when_the_screen_changes_and_fails_after_8_rounds(tmp_path):
    r = make(tmp_path)
    r._after_handler("steal", True, None)
    r._track_pending("steal")                      # still up: nothing yet
    r._track_pending("main")
    assert r.probe.card["steal"] == {"tries": 1, "landed": 1, "failed": 0, "unverified": 0}
    for _ in range(8):
        r._after_handler("discard", True, None)
    assert r.probe.card["discard"]["failed"] == 1
    r._track_pending("main")                       # a failed prompt is not also counted landed
    assert r.probe.card["discard"]["landed"] == 0


def test_unknown_frame_right_after_a_prompt_click_is_not_a_landing(tmp_path):
    r = make(tmp_path)
    r._after_handler("place_robber", True, None)
    r._track_pending("unknown")                    # animation between click and the prompt
    assert r.probe.card["place_robber"]["landed"] == 0


def test_forced_prompt_clicks_each_ringed_target_in_turn(tmp_path):
    """Pirate / knight placement / Smith: no planner, click what the game rings."""
    r = make(tmp_path, screens=[[], [], []])
    r._rings = lambda img: [(500, 400), (600, 450)]
    img = np.zeros((888, 1372, 3), np.uint8)
    r.on_place_pirate(img, [], [], {})
    r.on_place_pirate(img, [], [], {})
    clicks = [(x, y) for x, y, _ in r.io.clicks]
    assert clicks == [(500, 400), (600, 450)]
    assert r.probe.card["fallback_place_pirate"]["landed"] == 2


def test_forced_prompt_confirms_when_the_prompt_stays(tmp_path):
    up = [{"text": "Select position for knight", "x": 600, "y": 230, "w": 120, "h": 16}]
    r = make(tmp_path, screens=[up, []])
    r.cfg["fallback_blind_confirm"] = True        # the old path, kept behind its switch (round 3)
    r._rings = lambda img: [(500, 400)]
    r.on_place_knight(np.zeros((888, 1372, 3), np.uint8), [], [], {})
    whys = [w for *_, w in r.io.clicks]
    assert whys[0].startswith("fallback place_knight: ringed target") and whys[1] == "fallback place_knight: confirm"


def test_forced_prompt_no_blind_confirm_without_a_check(tmp_path):
    """Round 3 (game 0959): no green check on screen -> no tap in the middle of the bar."""
    up = [{"text": "Select position for knight", "x": 600, "y": 230, "w": 120, "h": 16}]
    r = make(tmp_path, screens=[up, []])
    r._rings = lambda img: [(500, 400)]
    r.on_place_knight(np.zeros((888, 1372, 3), np.uint8), [], [], {})
    whys = [w for *_, w in r.io.clicks]
    assert whys[0].startswith("fallback place_knight: ringed target")
    assert "fallback place_knight: confirm" not in whys


def test_strong_knight_prompt_clicks_the_ringed_knight(tmp_path):
    """Audit gap 2: 211 misses guessing knight spots; the game rings the knights it accepts."""
    r = make(tmp_path, screens=[[]])
    r._rings = lambda img: [(710, 390)]
    r.on_promote_knight(np.zeros((888, 1372, 3), np.uint8), [], ["Select position for Strong knight"], {})
    assert [(x, y) for x, y, _ in r.io.clicks] == [(710, 390)]


def test_bank_ratios_read_from_the_dialog():
    """Probe game 1 2026-10-04 07:36: every ratio was 4:1 and the bot offered 3 wool."""
    from catanbot import mac
    from catanbot.vision.layout import Layout
    native = cv2.imread(str(DATA / "bank_dialog_4to1_native.png"))
    lay = Layout(native.shape[1], native.shape[0])
    saved = {k: (dict(v) if isinstance(v, dict) else v) for k, v in R.UI.items()}
    glob = {k: getattr(R, k) for k in ("LAY", "HAND_XS", "HAND_Y", "BOARD_ROI", "BAR_BOX")}
    resp = R.Runner.RESP_XS
    try:
        R.set_layout(lay)
        frame = lay.frame(native, R.FRAME)

        class IO:
            pass
        io = IO()
        io.native, io.lay, io._frame, io.ocr = native, lay, frame, mac.ocr
        r = Runner.__new__(Runner)
        r.io = io
        seen = r._read_bank_ratios(frame)
    finally:
        R.UI.clear(); R.UI.update(saved)
        for k, v in glob.items():
            setattr(R, k, v)
        R.Runner.RESP_XS = resp
    assert len(seen) >= 6 and set(seen.values()) == {4}


def test_bank_trade_that_does_not_fit_the_ratio_is_cancelled(tmp_path):
    r = make(tmp_path, verify_in_replay=True, harbor_3to1=True)
    assert r._cfg_3to1()
    r._read_bank_ratios = lambda img=None: {"wool": 4, "ore": 4}
    assert r.bank_offer({"wool": 3}, {"grain": 1}) is False
    whys = [w for *_, w in r.io.clicks]
    assert whys[-1] == "cancel bank trade (ratio)" and not any(w.startswith("bank give") for w in whys)
    assert r.brain.ratio_seen["wool"] == 4 and not r._cfg_3to1()
    assert r.probe.card["bank_trade"]["tries"] == 0 and r.probe.card["bank_trade"]["refused"] == 1


def test_bank_trade_still_open_after_confirm_is_retried_once_then_blocked(tmp_path):
    dialog = [{"text": "Maritime trade", "x": 600, "y": 380, "w": 120, "h": 16}]
    r = make(tmp_path, screens=[dialog] * 12, verify_in_replay=True)
    r._read_bank_ratios = lambda img=None: {}
    assert r.bank_offer({"wool": 4}, {"grain": 1}) is False
    whys = [w for *_, w in r.io.clicks]
    assert whys.count("bank trade {'wool': 4}->{'grain': 1}") == 2
    assert whys.count("cancel failed bank trade") == 2
    assert r.probe.card["bank_trade"]["failed"] == 1
    n = len(r.io.clicks)
    assert r.bank_offer({"wool": 4}, {"grain": 1}) is False and len(r.io.clicks) == n   # blocked this turn


def test_defender_deck_prompt_picks_a_deck(tmp_path):
    """Probe game 1 07:41: 'Defender of Catan ... you receive a progress card' + 3 decks; read
    as progress_card and the board scans failed under it."""
    from catanbot import mac
    from catanbot.vision.layout import Layout
    from catanbot.vision.screens import classify
    native = cv2.imread(str(DATA / "defender_choose_deck_native.png"))
    frame = cv2.imread(str(DATA / "defender_choose_deck.png"))
    lay = Layout(native.shape[1], native.shape[0])
    ocr = mac.ocr(frame)
    assert classify([o["text"] for o in ocr])[0] == "choose_deck"
    r = make(tmp_path, screens=[[]])
    r.gs = None
    r.on_choose_deck(frame, ocr, [o["text"] for o in ocr], {})
    x, y, why = r.io.clicks[0]
    assert why == "progress deck science" and 780 < x < 940 and 330 < y < 540


def test_unread_bank_ratio_takes_the_common_rate(tmp_path):
    """Probe game 1 07:47: ore unread among five 4:1 labels -> 3 ore offered at a 4:1 twice."""
    r = make(tmp_path)
    r.gs = None
    reads = [{"text": "4:1", "x": x - 10, "y": R.UI["trade_give_row"]["y"] + 52, "w": 20, "h": 12}
             for x in R.UI["trade_give_row"]["x"][:3]]
    r.io.ocr = lambda img: [dict(o, x=(o["x"] - (R.UI["trade_give_row"]["x"][0] - 50)) * 2,
                                 y=(o["y"] - (R.UI["trade_give_row"]["y"] + 41)) * 2, w=40, h=24) for o in reads]
    seen = r._read_bank_ratios(np.zeros((888, 1372, 3), np.uint8))
    assert seen["ore"] == 4 and len(seen) == len(R.CARD_ORDER)


def test_deserter_player_dialog_clicks_the_check_under_the_name(tmp_path):
    """Probe game 1 07:49: Deserter played by drag, then 'Choose a player to remove a knight.'
    The old code clicked the name in the top banner and sat 10 min."""
    from catanbot import mac
    from catanbot.vision.screens import classify
    frame = cv2.imread(str(DATA / "deserter_choose_player.png"))
    ocr = mac.ocr(frame)
    assert classify([o["text"] for o in ocr])[0] == "choose_player"
    r = make(tmp_path, screens=[[]])
    r.game_colors = [("marlow.systems", "magenta"), ("Lin", "green"), ("Marianne", "red"), ("Jean", "orange")]
    r._want_player = "Jean"
    r.on_choose_player(frame, ocr, [o["text"] for o in ocr], {})
    x, y, why = r.io.clicks[0]
    assert why == "choose player Jean" and 740 < x < 820 and 430 < y < 490


def test_progress_banner_in_the_roll_phase_rolls_instead_of_trading(tmp_path):
    """Probe game 2 08:11: 'receives 1 progress card' in our roll phase -> on_main bank-traded
    with the trade button grey (failed twice)."""
    from catanbot.vision.layout import Layout
    frame = cv2.imread(str(DATA / "progress_banner_roll_phase.png"))
    saved = {k: (dict(v) if isinstance(v, dict) else v) for k, v in R.UI.items()}
    glob = {k: getattr(R, k) for k in ("LAY", "HAND_XS", "HAND_Y", "BOARD_ROI", "BAR_BOX")}
    resp = R.Runner.RESP_XS
    try:
        R.set_layout(Layout(1920, 1080))
        r = make(tmp_path)
        called = []
        r.on_main = lambda *a: called.append("main")
        r.on_roll = lambda *a: called.append("roll")
        r._save_lobby_frame = lambda *a: None
        r.on_progress_card(frame, [], ["marlow.systems receives 1 progress card."], {})
    finally:
        R.UI.clear(); R.UI.update(saved)
        for k, v in glob.items():
            setattr(R, k, v)
        R.Runner.RESP_XS = resp
    assert called == ["roll"]


def test_player_trade_panel_is_not_a_player_pick():
    """Probe game 3 08:41: 'Player trade -- Choose a player to trade with' read as choose_player."""
    from catanbot.vision.screens import classify
    assert classify(["Lin", "Player trade", "Choose a player to trade with"])[0] != "choose_player"
    assert classify(["Player trade", "Choose a player"])[0] != "choose_player"
    assert classify(["Deserter", "Choose a player to remove a knight.", "Lin", "Hide overlay"])[0] == "choose_player"


def _with_live_layout(fn):
    from catanbot.vision.layout import Layout
    saved = {k: (dict(v) if isinstance(v, dict) else v) for k, v in R.UI.items()}
    glob = {k: getattr(R, k) for k in ("LAY", "HAND_XS", "HAND_Y", "BOARD_ROI", "BAR_BOX")}
    resp = R.Runner.RESP_XS
    try:
        R.set_layout(Layout(1920, 1080))
        return fn()
    finally:
        R.UI.clear(); R.UI.update(saved)
        for k, v in glob.items():
            setattr(R, k, v)
        R.Runner.RESP_XS = resp


def test_two_digit_count_and_the_slots_after_it():
    """Probe game 3 09:12: '10 wool' read as unknown (assumed 5) and every later count shifted
    8 px: grain/ore unknown, so the bot planned with 5 grain it didn't have."""
    from catanbot import mac
    global img
    img = cv2.imread(str(DATA / "hand_ten_wool.png"))

    def run():
        h = R.HandReader()
        h.ocr_fn, h.save = mac.ocr, lambda: None
        return h.read(img)
    assert _with_live_layout(run) == {"lumber": 0, "brick": 1, "wool": 10, "grain": 0, "ore": 1,
                                      "cloth": 0, "coin": 0, "paper": 0}
    img = cv2.imread(str(DATA / "hand_ten_wool_b.png"))          # bar shifted: same counts
    assert _with_live_layout(run)["wool"] == 10 and _with_live_layout(run)["ore"] == 1
    img = cv2.imread(str(DATA / "progress_banner_roll_phase.png"))  # a "2" next to its icon is one digit
    assert _with_live_layout(run) == {"lumber": 4, "brick": 0, "wool": 2, "grain": 0, "ore": 1,
                                      "cloth": 1, "coin": 0, "paper": 0}


def test_enlarged_card_is_not_a_player_pick():
    """Probe game 3 09:14: the enlarged Master Merchant card's text ('Choose an opponent...')
    read as a player dialog for 10 min. Real dialogs carry 'Hide overlay'."""
    from catanbot import mac
    from catanbot.vision.screens import classify
    ocr = mac.ocr(cv2.imread(str(DATA / "hand_ten_wool.png")))
    assert classify([o["text"] for o in ocr])[0] != "choose_player"
    assert Runner._enlarged_card(ocr)[0] == "master_merchant"
    dlg = mac.ocr(cv2.imread(str(DATA / "deserter_choose_player.png")))
    assert Runner._enlarged_card(dlg) is None


def test_ringed_prompt_without_text_clicks_a_ring_then_the_check(tmp_path):
    """Final probe game 10:03: Diplomat rings the open roads, no prompt text (screen unknown);
    the bot tapped one and sat 10 min. A ring + the green check clears it."""
    r = make(tmp_path, screens=[[], []])
    # one ring read per frame (review 0.3.1: the after-frame is read once, like the before-frame)
    r.cfg["fallback_blind_confirm"] = True        # black test frame has no green check (round 3)
    calls = iter([[(668, 295), (720, 330)], []])
    r._rings = lambda img: next(calls, [])
    r._forced_fallback(np.zeros((888, 1372, 3), np.uint8), "ringed_prompt")
    whys = [w for *_, w in r.io.clicks]
    assert whys == ["fallback ringed_prompt: ringed target (2 rings)", "fallback ringed_prompt: confirm"]
    assert r.probe.card["fallback_ringed_prompt"]["landed"] == 1


def test_panned_board_is_dragged_back(tmp_path):
    """Final probe game 10:18: the board was panned up-left and every scan failed; 10:23 the
    recenter drag brought it back (31 land hexes)."""
    r = make(tmp_path)
    r.mode = object()
    img = cv2.imread(str(DATA / "board_panned.png"))
    assert r._recenter_board(img) is True
    (x, y, why), = [c for c in r.io.clicks if "recenter" in c[2]]
    assert why == "recenter board"
    assert Runner._sea_box(cv2.imread(str(DATA / "hand_ten_wool.png")))[0] == 365
    assert r._recenter_board(cv2.imread(str(DATA / "hand_ten_wool.png"))) is False


def test_spy_pick_clicks_a_card_under_its_title(tmp_path):
    """Final probe game 10:36: Spy's 'Select progress card' read as progress_card -> 10-min stall."""
    from catanbot import mac
    from catanbot.vision.screens import classify
    frame = cv2.imread(str(DATA / "spy_select_progress.png"))
    ocr = mac.ocr(frame)
    assert classify([o["text"] for o in ocr])[0] == "pick_progress"
    r = make(tmp_path, screens=[[]])
    r.gs = None
    r.on_pick_progress(frame, ocr, [o["text"] for o in ocr], {})
    x, y, why = r.io.clicks[0]
    assert why.startswith("pick progress card ") and 360 < y < 420 and 330 < x < 1000


def test_green_check_waiting_is_seen():
    """Game 6 11:34: Smith's knight selected, the green check waited 10 min (prompt OCR garbled)."""
    from catanbot.vision.screens import classify
    assert classify(["Selet position focstrong kniy D"])[0] == "promote_knight"

    def run():
        return (Runner._green_check(cv2.imread(str(DATA / "smith_waiting_check.png"))),
                Runner._green_check(cv2.imread(str(DATA / "hand_ten_wool.png"))))
    assert _with_live_layout(run) == (True, False)


def test_prompt_text_letters_are_not_rings(tmp_path, monkeypatch):
    """Game 8 13:01: the 'o' in 'Select position for Settlement' (native ring read at (714,238))
    was clicked 13x as the city to lose; our city's real ring was (828,388)."""
    from catanbot import mac
    frame = cv2.imread(str(DATA / "lose_city_prompt.png"))
    r = make(tmp_path)
    monkeypatch.setattr(Runner, "ring_spots", staticmethod(lambda *a, **k: [(714, 238), (828, 388)]))
    r._last_img, r._last_ocr = frame, mac.ocr(frame)
    assert r._rings(frame) == [(828, 388)]


def test_stuck_strong_knight_prompt_ends_the_turn(tmp_path, monkeypatch):
    """Games 8/9: Smith's 2nd promotion rang a knight the game would not take; End turn skips it."""
    r = make(tmp_path, screens=[[]] * 40)
    r._rings = lambda img: [(567, 405)]
    monkeypatch.setattr(R, "turn_phase", lambda img: "main")
    img = np.zeros((888, 1372, 3), np.uint8)
    for _ in range(8):
        r._forced_fallback(img, "promote_knight")
    r._after_handler("promote_knight", True, img)
    assert any("skip the stuck strong-knight prompt" in w for *_, w in r.io.clicks)


def test_bank_dialog_shows_our_real_counts(tmp_path):
    """Game 9 14:08: hand bar 5 lumber, dialog 2 -> 4-lumber trade failed twice."""
    from catanbot import mac
    from catanbot.vision.layout import Layout
    native = cv2.imread(str(DATA / "bank_dialog_counts_native.png"))

    def run():
        lay = Layout(native.shape[1], native.shape[0])
        frame = lay.frame(native, R.FRAME)
        r = make(tmp_path)
        r.io.native, r.io.lay, r.io._frame, r.hand.ocr_fn = native, lay, frame, mac.ocr
        return r._read_dialog_counts(frame)
    got = _with_live_layout(run)
    assert got["lumber"] == 2 and got["brick"] == 1 and got["wool"] == 2 and got["grain"] == 1


def test_bank_trade_repicks_a_dropped_want(tmp_path):
    """Final game 14:42: at a 2:1 the want click was dropped ('Choose 1 more resources that you
    would like to receive from the supply'); pick it again and confirm."""
    ask = [{"text": "Maritime trade", "x": 600, "y": 380, "w": 120, "h": 16},
           {"text": "Choose 1 more resources that you would like to receive from the supply (bank).",
            "x": 450, "y": 420, "w": 500, "h": 16}]
    r = make(tmp_path, screens=[[], [], ask, ask, [], []], verify_in_replay=True)   # 2nd: the dialog frame
    r._read_bank_ratios = lambda img=None: {}
    r._read_dialog_counts = lambda img=None: {}
    assert r.bank_offer({"ore": 2}, {"grain": 1}) is True
    whys = [w for *_, w in r.io.clicks]
    assert "bank want grain (again)" in whys and r.probe.card["bank_trade"]["landed"] == 1


def test_five_progress_titles_read_for_the_discard():
    """Game 11 15:41: 'Discard progress card' with 5 cards, titles at y~714 below the old strip."""
    from catanbot import mac

    def run():
        r = Runner.__new__(Runner)

        class IO:
            pass
        r.io = IO()
        r.io.ocr = mac.ocr
        img = cv2.imread(str(DATA / "discard_progress_unread.png"))
        return [k for k, _ in r._read_progress_hand(mac.ocr(img), img)]
    assert _with_live_layout(run) == ["resource_monopoly", "saboteur", "crane", "deserter", "mining"]


def test_science_bonus_pick_clicks_the_top_row_and_the_check(tmp_path):
    """Game 13 16:30: 'Science bonus -- Choose 1 more resources that you'll gain.' read as main;
    End turn was clicked 13x behind the dialog."""
    from catanbot import mac
    from catanbot.vision.screens import classify
    frame = cv2.imread(str(DATA / "science_bonus.png"))
    ocr = mac.ocr(frame)
    texts = [o["text"] for o in ocr]
    assert classify(texts)[0] == "choose_resource"
    r = make(tmp_path, screens=[[]])
    r.gs = None
    r.on_choose_resource(frame, ocr, texts, {})
    whys = [w for *_, w in r.io.clicks]
    assert whys == ["gain grain", "gain grain: confirm"]


def test_bank_trade_landed_with_dialog_still_open_is_not_traded_again(tmp_path):
    # solo 2026-10-06 03:35: 3 ore -> 1 grain landed, the dialog stayed open ("choose more resources
    # that you would like to receive"), the re-pick traded again: 6 ore -> 2 grain, no city
    dialog = [{"text": "Maritime trade", "x": 600, "y": 380, "w": 120, "h": 16},
              {"text": "Choose more resources that you would like to receive", "x": 520, "y": 420, "w": 300, "h": 16}]
    r = make(tmp_path, screens=[dialog] * 12, verify_in_replay=True)
    r._read_bank_ratios = lambda img=None: {}
    hands = iter([{"ore": 6, "grain": 0}] + [{"ore": 3, "grain": 1}] * 10)
    r._hand_now = lambda img=None: next(hands)
    assert r.bank_offer({"ore": 3}, {"grain": 1}) is True
    whys = [w for *_, w in r.io.clicks]
    assert whys.count("bank trade {'ore': 3}->{'grain': 1}") == 1
    assert not any("again" in w or "re-picking" in w for w in whys)
    assert whys[-1] == "close bank dialog (trade already landed)"


def test_side_icons_found_by_colour():
    import cv2
    import numpy as np
    img = np.full((888, 1372, 3), (40, 60, 90), np.uint8)
    hsv_orange = cv2.cvtColor(np.uint8([[[20, 220, 240]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
    for y in (436, 477, 518, 559, 599, 641, 681, 722):
        cv2.rectangle(img, (22, y - 10), (42, y + 10), hsv_orange, -1)
    r = Runner.__new__(Runner)
    icons = r._side_icons(img)
    assert len(icons) == 8 and abs(icons[2][1] - 518) <= 1 and abs(icons[2][0] - 32) <= 1
