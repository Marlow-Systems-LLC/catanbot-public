"""Replay the calibration captures through the runner (dry run) and check what it clicks."""
import pytest

from catanbot.runner import ReplayIO, Runner, UI, turn_phase, to_frame
from catanbot import config
import cv2
import numpy as np

C = "data/calib/"
RULES = ['(C&K+Sf) The Great Canal', '18 Victory points', 'Balanced']   # frames are an old Greater Catan game; only the clicks are tested


def _frame(path):
    import os
    if not os.path.exists(path):
        pytest.skip("calibration frame not in this checkout")
    return path


def runner(paths):
    cfg = config.load(); cfg["dry_run"] = True
    paths = [_frame(C + p)[len(C):] for p in paths]
    r = Runner(ReplayIO([C + p for p in paths]), cfg)
    r.players = [("marlow.systems", "red"), ("Hildegard", "purple"), ("Nassir", "brown")]
    return r


@pytest.mark.parametrize("img,phase", [("gc_turn1_roll_button", "roll"), ("gc_turn1_rolled", "main"),
                                       ("gc_other_turn", "idle"), ("gc_setup1", "idle"),
                                       ("gc_trade_declined", "main")])
def test_turn_phase(img, phase):
    assert turn_phase(to_frame(cv2.imread(_frame(C + img + ".jpg")))) == phase


def test_no_rules_no_play():
    r = runner(["gc_setup1.jpg"])
    r.step()
    whys = [w for *_, w in r.io.clicks]
    assert whys == ["open rules info"]          # asks for rules instead of guessing


def test_setup_after_rules():
    r = runner(["gc_setup1.jpg", "gc_setup3_road.jpg"])
    assert r.read_rules(RULES) and r.mode.key == "great_canal"
    assert r.step() == "setup_settlement"
    (x, y, why), (cx, cy, cw) = r.io.clicks[:2]
    assert why.startswith("setup settlement") and 270 < x < 950 and 230 < y < 720
    assert (cx, cy) == tuple(UI["setup_confirm"])
    assert r.step() == "setup_road"
    assert r.io.clicks[2][2].startswith("setup road")


def test_roll_then_end_or_trade():
    r = runner(["gc_turn1_roll_button.jpg", "gc_turn1_rolled.jpg"])
    r.read_rules(RULES)
    assert r.step() == "roll"
    assert r.io.clicks[-1][:2] == tuple(UI["dice_or_end_turn"])
    assert r.step() == "main"
    whys = [w for *_, w in r.io.clicks[1:]]
    assert whys[-1] in ("end turn",) or whys[-1].startswith("send offer"), whys
    if whys[0] == "open trade":
        assert sum(w.startswith("want") for w in whys) >= 1 and sum(w.startswith("give") for w in whys) >= 1


def test_declined_trade_closes():
    r = runner(["gc_trade_declined.jpg"])
    r.read_rules(RULES)
    assert r.step() == "trade_panel"
    assert r.io.clicks[-1][:2] == tuple(UI["trade_cancel"])


def test_stop_file(tmp_path, monkeypatch):
    import catanbot.runner as R
    stop = tmp_path / "STOP"; stop.write_text("")
    monkeypatch.setattr(R, "STOP", stop)
    r = runner(["gc_turn1_roll_button.jpg"])
    assert r.step() == "stopped" and r.io.clicks == []


def test_incoming_offer_declined():
    r = runner(["gc_trade_offer_in.jpg"] * 3)
    r.read_rules(RULES)
    assert r.step() == "trade_offer_in" and r.io.clicks == []     # first read: no click yet
    assert r.step() == "trade_offer_in"
    n = len(r.io.clicks)
    assert r.step() == "trade_offer_in" and len(r.io.clicks) == n  # answered once only
    # 2026-10-03: buttons are found by colour and the bot now sees its own pieces, so it may
    # take this fair 1:1 -- assert it answered on one of the panel buttons
    assert any(abs(r.io.clicks[-1][0] - UI[k][0]) < 8 and abs(r.io.clicks[-1][1] - UI[k][1]) < 30
               for k in ("offer_in_decline", "offer_in_accept"))


def test_owned_popup_marks_spot_taken():
    """Live 2026-10-03: the knight was aimed twice at Vale's unseen settlement."""
    import re as _re
    who = _re.sub(r".*owned by\s*|\)\s*$", "", "Settlement (owned by Vale)").strip()
    assert who == "Vale"


def test_hand_bar_found_when_it_shifts():
    """Live 2026-10-03 full screen: a held progress card widened the bar and every count was
    read one card to the right (ore as grain, wool lost)."""
    import cv2
    from catanbot import runner as R
    from catanbot.vision import layout as L
    old = R.LAY
    try:
        R.set_layout(L.Layout(1920, 1080))
        h = R.HandReader()
        img = cv2.imread("data/calib/hand_bar_shifted.png")
        if img is None:
            pytest.skip("live frame not in this checkout")
        assert abs(h.bar_left(img) - 350) <= 3
        got = h.read(img)
        assert (got["wool"], got["grain"], got["paper"]) == (1, 0, 0)
    finally:
        R.set_layout(old)


def test_setup_prompt_with_rules_panel_is_not_robber():
    """Solo 2026-10-03: the rules panel ("...the robber may be moved only...") was still up under
    "Select position for Settlement"; place_robber matched first and the settlement went on a
    random ring."""
    from catanbot.vision.screens import classify
    texts = ["Select position for Settlement", "The Great Canal",
             "the special rule for the robber applies (the robber may be moved only upon"]
    assert classify(texts)[0] == "setup_settlement"
    assert classify(["Select a hex for the robber"])[0] == "place_robber"


def test_solo_mode_never_chats_and_home_idles_without_it():
    """Solo 2026-10-03: offline games have no chat box; the intro text went to the board as
    camera keys. Without cfg solo the home screen is left alone (owner invites us)."""
    from catanbot import config
    cfg = config.load(); cfg["solo"] = True; cfg["chat"] = True
    r = Runner(ReplayIO([]), cfg)
    assert r._chat_on() is False
    cfg2 = config.load(); cfg2["solo"] = False
    r2 = Runner(ReplayIO([]), cfg2)
    r2.on_home(None, [], [], {})
    assert not getattr(r2, "_solo_tried", 0)


def test_city_prompt_read_as_cify():
    from catanbot.vision.screens import classify
    assert classify(["Select position for Cify"])[0] == "setup_city"


def test_knight_popup_read_as_baild_khiai():
    """Solo 2026-10-03: the knight popup OCR'd as 'Baild Khiai' and the knight was not built."""
    import re
    from catanbot.runner import Runner
    pat = Runner.POPUP_WORDS["knight"]
    for t in ("Baild Khiai", "Build knighit", "Build Knigfit.", "Build Knight"):
        assert re.search(pat, t, re.I), t


def _tap(texts, kind, monkeypatch, grain=0):
    """Drive Runner.tap_build against one scripted popup; return (clicks, notes, taken)."""
    import types
    import numpy as np
    from catanbot import runner as R
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    ocr = [{"text": t, "x": 900, "y": 280 + 20 * i, "w": 80, "h": 16, "conf": 1.0} if isinstance(t, str)
           else {"text": t[0], "x": t[1], "y": t[2], "w": 60, "h": 16, "conf": 1.0} for i, t in enumerate(texts)]
    clicks, notes = [], []
    r = R.Runner.__new__(R.Runner)
    r.br = types.SimpleNamespace(lattice=types.SimpleNamespace(vertex_px=lambda t: (770, 288),
                                                                 edge_px=lambda t: (770, 288)))
    r.io = types.SimpleNamespace(snapshot=lambda: np.zeros((10, 10, 3), np.uint8), ocr=lambda img: ocr)
    r.click = lambda xy, why, *a: clicks.append(why)
    r.note = notes.append
    from catanbot.brain import Action
    from catanbot.state import PlayerState
    me = PlayerState("marlow.systems", is_me=True); me.hand["grain"] = grain
    r.gs = types.SimpleNamespace(me=lambda: me)
    r.tap_build(Action(kind, (1, -5)))
    return clicks, notes, getattr(r, "_taken_spots", {})


def test_activate_never_clicks_on_another_players_knight(monkeypatch):
    """Solo 2026-10-03: 'activ' matched 'Strong knight (inactive) (owned by Jean)'."""
    clicks, notes, taken = _tap(["Strong knight (inactive) (owned by Jean)"], "activate", monkeypatch)
    assert clicks == ["tap spot for activate at (1, -5)"]
    assert taken == {(1, -5): "Jean"}


def test_knight_popup_read_as_puildkaias(monkeypatch):
    clicks, _, _ = _tap(["puildkaias"], "knight", monkeypatch)
    assert "popup knight" in clicks


def test_own_knight_activate_still_works(monkeypatch):
    clicks, _, taken = _tap(["Knight (inactive) (owned by marlow.systems)", "Activate"], "activate", monkeypatch)
    assert "popup activate" in clicks and taken == {}


def test_offer_lumber_card_read_full_screen():
    """Solo 2026-10-03: Marianne's brick-for-lumber offer read lumber as unknown and stalled."""
    import cv2
    from catanbot import runner as R
    from catanbot.vision import layout as L
    old = R.LAY
    try:
        R.set_layout(L.Layout(1920, 1080))
        img = cv2.imread("data/calib/solo_offer_brick_for_lumber.png")
        r = R.Runner.__new__(R.Runner)
        assert r._offer_card(img, 422)[0] == "brick"
        assert r._offer_card(img, 483)[0] == "lumber"
    finally:
        R.set_layout(old)


def test_three_letter_player_name_gets_a_colour():
    """Solo 2026-10-03: 'Lin' (3 letters) was dropped, so the 15-VP leader was never tracked."""
    import cv2
    from catanbot import runner as R
    from catanbot.vision import layout as L
    old = R.LAY
    try:
        R.set_layout(L.Layout(1920, 1080))
        img = cv2.imread("data/calib/solo_offer_brick_for_lumber.png")
        ocr = [{"text": t, "x": x, "y": 85, "w": w, "h": 18, "conf": 1.0}
               for t, x, w in (("marlow.systems", 97, 120), ("Lin", 431, 28), ("Marianne", 765, 70), ("Jean", 1099, 36))]
        r = R.Runner.__new__(R.Runner)
        cols = dict(r._banner_colors(img, ocr))
        assert cols.get("Lin") == "red", cols
    finally:
        R.set_layout(old)


def test_knight_tap_on_our_inactive_knight_activates_it(monkeypatch):
    """Solo 2026-10-03: the popup offered "Activate knight" (OCR "Olvete Kmighi"); the bot kept
    trying to build a knight on its own unseen knight."""
    clicks, notes, _ = _tap(["Olvete Kmighi"], "knight", monkeypatch, grain=1)
    assert "popup activate" in clicks
    clicks, notes, _ = _tap(["Olvete Kmighi"], "knight", monkeypatch, grain=0)
    assert clicks == ["tap spot for knight at (1, -5)"] and any("no grain" in n for n in notes)


def test_other_scenario_is_refused():
    """Owner 2026-10-04: marlow plays only The Great Canal."""
    from catanbot import runner as R
    r = R.Runner.__new__(R.Runner)
    r.log, r.mode = [], None
    r.note = r.log.append
    assert r.read_rules(["(C&K+SF) GREATER CATAN (3)", "20 Victory points"]) is False
    assert r.mode is None and "not The Great Canal" in r.log[-1]


def test_thinner_rules_read_never_replaces_the_mode():
    """Solo 2026-10-03: great_canal (2 unconfirmed) was overwritten by a 'base' read (5)."""
    from catanbot.rules import MODES
    from catanbot import runner as R
    r = R.Runner.__new__(R.Runner)
    r.log, r.mode, r._rules_unconf = [], MODES["great_canal"], 2
    r.note = r.log.append
    r._save_game_flags = lambda: None
    assert r.read_rules(["Some panel", "10 Victory Points"]) is True
    assert r.mode is MODES["great_canal"]


def test_build_costs_label_is_not_a_popup_option(monkeypatch):
    """Solo 2026-10-03: 'Settleme' on the BUILD COSTS card (371,757) was clicked as the option."""
    clicks, notes, _ = _tap([("Settleme", 371, 757), ("BUILD COSTS", 399, 699)], "settlement", monkeypatch)
    assert clicks == ["tap spot for settlement at (1, -5)"]
    assert any("no option for settlement" in n for n in notes)


def test_promote_regex_present_and_inactive_knight_activated(monkeypatch):
    """Solo 2026-10-03: the promote word list was lost in a comment; the popup on our inactive
    knight only offers 'Activate Knight'."""
    from catanbot import runner as R
    assert "promote" in R.Runner.POPUP_WORDS
    clicks, notes, _ = _tap(["Promote knight"], "promote", monkeypatch)
    assert len(clicks) >= 2 and "promote" in clicks[1].lower()
    clicks, notes, _ = _tap(["Activate Knight"], "promote", monkeypatch, grain=1)
    assert len(clicks) >= 2 and "activate" in clicks[1].lower()
    clicks, notes, _ = _tap(["Activate Knight"], "promote", monkeypatch, grain=0)
    assert not clicks[1:] and any("no grain" in n for n in notes)


def test_failed_tap_skipped_rest_of_turn(monkeypatch):
    import types
    from catanbot import runner as R
    from catanbot.brain import Action
    r = R.Runner.__new__(R.Runner)
    r._failed_taps = {("promote", (0, -4))}
    tapped, notes = [], []
    r.tap_build = tapped.append
    r.note = notes.append
    r.click = lambda *a: None
    r._run_builds(None, [Action("promote", (0, -4)), Action("road", (1, 2))])
    assert [a.kind for a in tapped] == ["road"] and any("already failed" in n for n in notes)


def test_improve_dialog_clicks_ok(monkeypatch):
    """Solo 2026-10-03: 'City improvements' title was clicked instead of the OK button."""
    import types
    import numpy as np
    from catanbot import runner as R
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    mk = lambda t, x, y: {"text": t, "x": x, "y": y, "w": 40, "h": 16, "conf": 1.0}
    ocr = [mk("City improvements", 600, 370), mk("Do you want to improve your city?", 400, 420),
           mk("CANCEL", 470, 470), mk("OK", 690, 470), mk("BUILD COSTS", 397, 699)]
    clicks = []
    r = R.Runner.__new__(R.Runner)
    r.io = types.SimpleNamespace(snapshot=lambda: np.zeros((10, 10, 3), np.uint8), ocr=lambda img: ocr)
    r.click = lambda xy, why, *a: clicks.append((xy, why))
    r.note = lambda m: None
    assert r._tap_improve_chart("politics", (796, 722)) is True
    assert clicks[-1] == ((710, 478), "improve politics (OK)")


def test_banner_colours_read_every_slot_even_when_name_ocr_misses():
    """Live 2026-10-03 game 6: Jean was dropped (name not OCR'd), Lin's grey banner read brown,
    and the colours changed after a relaunch. Read by banner slot; keep slot names."""
    import cv2
    from catanbot import runner as R
    r = R.Runner.__new__(R.Runner)
    r.cfg = {"bot_name": "marlow.systems"}
    a = cv2.imread("data/calib/live_g6_colors_a.png")
    ocr = [{"text": t, "x": x, "y": 90, "w": w, "h": 16, "conf": 1.0}
           for t, x, w in (("marlow.systems", 97, 120), ("Lin", 428, 26), ("Marianne", 763, 70), ("Jean", 1091, 36))]
    assert r._banner_colors(a, ocr) == [("marlow.systems", "magenta"), ("Lin", "grey"),
                                        ("Marianne", "green"), ("Jean", "red")]
    b = cv2.imread("data/calib/live_g6_colors_b.png")
    only_two = [o for o in ocr if o["text"] in ("marlow.systems", "Marianne")]
    assert r._banner_colors(b, only_two) == [("marlow.systems", "magenta"), ("Lin", "brown"),
                                             ("Marianne", "red"), ("Jean", "blue")]


def test_third_setup_settlement_is_not_a_city_loss():
    # game 14 03:24: a phantom 3rd building turned the real 3rd setup prompt into "lose weakest city"
    r = runner(["gc_setup1.jpg", "gc_setup1.jpg"])
    r.read_rules(RULES)
    assert r.step() == "setup_settlement"
    me = r.gs.me()
    free = [v for v in r.gs.legal_settlement_spots(me, setup=True) if v not in me.settlements]
    me.settlements = list(me.settlements) + free[:3]
    assert len(me.buildings()) >= 3
    r._in_play, r._saw_game_start = False, True
    r.step()
    whys = [w for *_, w in r.io.clicks]
    assert not any("lose" in w for w in whys), whys


def test_setup_road_takes_the_ringed_edge():
    # game 14 03:24: "Road, Ship" at the coast -- 16 land edges refused while the game ringed a sea edge
    r = runner(["gc_setup1.jpg", "gc_setup3_road.jpg"])
    r.read_rules(RULES)
    assert r.step() == "setup_settlement"
    gs, lat = r.gs, r.br.lattice
    v = r._last_setup_v
    target = next(e for e in sorted(gs.board.vertex_edges(v)) if not gs.edge_owner(e))
    others = [e for e in gs.board.vertex_edges(v) if e != target]
    r._road_tried = set(others)            # the brain would pick another edge first
    r._rings = lambda img: [lat.edge_px(target)]
    assert r.step() == "setup_road"
    roads = [w for *_, w in r.io.clicks if w.startswith("setup road")]
    assert roads == [f"setup road {target}"], r.io.clicks


class _ScreensIO:
    """Fake live IO: each snapshot shows the next list of OCR texts; clicks are recorded."""
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


def test_leave_flag_walks_out_with_game_buttons(tmp_path, monkeypatch):
    # owner 2026-10-05: leave by the game's buttons -- never kill the app (a killed game rejoins on relaunch)
    import subprocess, catanbot.runner as R
    io = _ScreensIO([["marlow.systems", "Hildegard"], ["Do you really want to quit this game?"],
                     ["Leaderboards", "Almanac", "Guilds"]])
    cfg = config.load(); cfg["dry_run"] = False
    r = Runner(ReplayIO([]), cfg)
    r.io = io
    flag, stop = tmp_path / "leave.flag", tmp_path / "STOP"
    flag.touch()
    monkeypatch.setattr(Runner, "LEAVE_FLAG", flag)
    monkeypatch.setattr(R, "STOP", stop)
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail(f"no process calls: {a}"))
    assert r.step() == "left"
    assert [w for *_, w in io.clicks] == ["leave game: quit gear", "leave game: green check"]
    assert not flag.exists() and not stop.exists()      # waits at home for the next invite


def test_leave_mid_game_presses_yes(tmp_path, monkeypatch):
    # live 2026-10-08 19:22: mid-game "Quit game / Do you really want to quit the game? / NO / YES"
    import subprocess, catanbot.runner as R
    io = _ScreensIO([["marlow.systems", "Hildegard"],
                     ["Quit game", "Do you really want to quit the game?", "NO", "YES"],
                     ["Leaderboards", "Almanac", "Guilds"]])
    cfg = config.load(); cfg["dry_run"] = False
    r = Runner(ReplayIO([]), cfg)
    r.io = io
    flag, stop = tmp_path / "leave.flag", tmp_path / "STOP"
    flag.touch()
    monkeypatch.setattr(Runner, "LEAVE_FLAG", flag)
    monkeypatch.setattr(R, "STOP", stop)
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    monkeypatch.setattr(Runner, "_save_leave_frame", lambda *a: None)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail(f"no process calls: {a}"))
    assert r.step() == "left"
    assert [w for *_, w in io.clicks] == ["leave game: quit gear", "leave game: YES"]


def test_hand_digits_are_one_digit_only(tmp_path, monkeypatch):
    """Stage 2026-10-10: OCR read a lone 2 next to a card edge as "21"/"24" and the reader learned
    those glyphs, so a slot showing 2 could come back as 21."""
    import numpy as np
    from catanbot import runner as R
    g = np.zeros((28, 22), np.float32)
    path = tmp_path / "d.npz"
    np.savez(path, **{"2_0": g, "21_0": g, "24_0": g})
    monkeypatch.setattr(R.HandReader, "PATH", path)
    h = R.HandReader()
    assert sorted(h.t) == [2]
    h.add(14, g)
    h.add(5, g)
    assert sorted(h.t) == [2, 5]
    shipped = np.load("data/calib/hand_digits.npz")
    assert all(int(k.split("_")[0]) <= 9 for k in shipped.files)
    img = np.full((1080, 1920, 3), 255, np.uint8)
    for text, want in (("VP 21", None), ("VP 2", 2), ("VP7", 7)):
        h.ocr_fn = lambda _img, t=text: [{"text": t}]
        assert h.ocr_digit(img, 900) == want
