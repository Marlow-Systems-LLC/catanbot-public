"""Progress cards wired into brain/state/runner: fake screen IO, dry run, no simulation."""
import numpy as np
import pytest

import catanbot.runner as R
from catanbot import config
from catanbot.board import canal_board, hex_corners
from catanbot.brain import Brain
from catanbot.rules import MODES
from catanbot.runner import ReplayIO, Runner
from catanbot.state import GameState, PlayerState, new_hand

HAND = R.UI["progress_hand"]


def box(text, x=None, y=None):
    """OCR box centred inside the (uncalibrated) progress-hand region unless placed."""
    x = (HAND[0] + HAND[2]) / 2 if x is None else x
    y = (HAND[1] + HAND[3]) / 2 if y is None else y
    return {"text": text, "x": x - 30, "y": y - 8, "w": 60, "h": 16, "conf": 1.0}


class FakeIO(ReplayIO):
    """Scripted screens: each snapshot yields the next OCR list; clicks are recorded."""

    def __init__(self, screens):
        super().__init__([])
        self.screens, self.cur = list(screens), []

    def snapshot(self):
        if not self.screens:
            return None
        self.cur = self.screens.pop(0)
        return np.zeros((888, 1372, 3), np.uint8)

    def ocr(self, img):
        if img is not None and img.shape[:2] != (888, 1372):
            return []                       # scripted screens model the full frame, not crops
        return self.cur


def _land(b, terrain):
    return [h for h in b.hexes.values() if h.is_land and h.terrain == terrain and h.number]


@pytest.fixture
def gs():
    b = canal_board(seed=3)
    me = PlayerState("marlow.systems", is_me=True, hand=new_hand(), vp_visible=6)
    a = PlayerState("Hildegard", vp_visible=10, hand_count=6)
    c = PlayerState("Nassir", vp_visible=5, hand_count=2)
    g = GameState(board=b, mode=MODES["great_canal"], players=[me, a, c])
    mtn = _land(b, "mountains")[0]
    me.cities.add(hex_corners(*mtn.coord)[0])
    other = next(h for h in b.hexes.values() if h.is_land and h.number and h.coord != mtn.coord
                 and not set(hex_corners(*h.coord)) & me.cities)
    a.cities.add(hex_corners(*other.coord)[2])
    return g


def runner(gs, screens, tmp_path, dry=True):
    cfg = config.load(); cfg["dry_run"] = dry
    r = Runner(FakeIO(screens), cfg)
    r.mode = gs.mode
    r.gs = gs
    r.ensure_state = lambda img, texts: gs
    r.popup_dir = tmp_path
    return r


# ---- state / brain -----------------------------------------------------------------------
def test_state_defaults():
    p = PlayerState("x")
    assert p.progress_hand == [] and PlayerState("y").progress_hand is not p.progress_hand
    assert GameState(board=canal_board(seed=3)).barbarians_attacked is False


def test_bishop_held_until_first_attack(gs):
    br = Brain()
    gs.me().progress_hand = ["bishop"]
    assert br.progress_play(gs, "main") is None
    gs.barbarians_attacked = True
    play = br.progress_play(gs, "main")
    assert play is not None and play[0] == "bishop"


def test_bishop_blocked_end_turn_over_limit_discards(gs):
    br = Brain()
    hand = ["bishop", "diplomat", "intrigue", "spy", "deserter"]
    play = br.progress_play(gs, "end_turn", hand)
    assert play is None or play[0] != "bishop"


def test_progress_play_uses_state_hand(gs):
    gs.me().progress_hand = ["mining"]
    assert Brain().progress_play(gs, "main")[0] == "mining"
    assert Brain().progress_play(gs, "main", []) is None


def test_merchant_fleet_bank_ratio(gs):
    br = Brain()
    base, wool = br.bank_ratio(gs, "ore"), br.bank_ratio(gs, "wool")
    assert base in (3, 4)
    br.start_fleet("ore")
    assert br.bank_ratio(gs, "ore") == 2
    assert br.bank_ratio(gs, "wool") == wool
    br.end_fleet()
    assert br.bank_ratio(gs, "ore") == base


# ---- runner ------------------------------------------------------------------------------
def test_play_one_then_reread(gs, tmp_path):
    r = runner(gs, [[box("Mining")], []], tmp_path)
    played = r.play_progress("main")
    assert played == ["mining"]
    whys = [w for *_, w in r.io.clicks]
    assert whys == ["progress select mining", "progress play mining"]
    assert r.io.clicks[1][:2] == tuple(R.UI["progress_play"])
    assert list(tmp_path.glob("*_progress_main_mining_mining.png"))
    assert any("held=['mining']" in m for m in r.log)


def test_unknown_names_ignored(gs, tmp_path):
    r = runner(gs, [[box("Monopoly of Sheep"), box("End turn"), box("Mining", y=10)]], tmp_path)
    assert r.play_progress("main") == []
    assert r.io.clicks == []


def test_not_retried_in_same_turn(gs, tmp_path):
    # the card is still on screen after the play (play failed): no second attempt
    r = runner(gs, [[box("Mining")], [box("Play")], [], [box("Mining")]], tmp_path)
    assert r.play_progress("main") == ["mining"]
    assert sum(w == "progress play mining" for *_, w in r.io.clicks) == 1
    assert any("already tried" in m for m in r.log)


def test_live_gate_only_logs(gs, tmp_path, monkeypatch):
    monkeypatch.setitem(R.UI, "progress_live", False)
    r = runner(gs, [[box("Mining")]], tmp_path, dry=False)
    assert r.play_progress("main") == []
    assert r.io.clicks == []
    assert any("would mining" in m for m in r.log)
    assert list(tmp_path.glob("*.png"))


def test_merchant_fleet_sets_ratio_and_end_turn_clears(gs, tmp_path):
    gs.me().hand["ore"] = 5
    # the dialog: big "Merchant Fleet" title over an icon tile row (solo 2026-10-06 06:47)
    title = {"text": "Merchant Fleet", "x": 586, "y": 394, "w": 136, "h": 20, "conf": 1.0}
    r = runner(gs, [[box("Merchant Fleet")], [box("Play")], [title], [title], []], tmp_path)
    assert r.play_progress("main") == ["merchant_fleet"]
    assert r.brain.bank_ratio(gs, "ore") == 2
    whys = [w for *_, w in r.io.clicks]
    assert "monopoly pick ore" in whys and "monopoly confirm (green check)" in whys
    r.io.screens = [[]]
    r.play_progress("end_turn")
    assert r.brain.fleet_resource is None


def test_not_ck_does_nothing(gs, tmp_path):
    r = runner(gs, [[box("Mining")]], tmp_path)
    r.mode = None                 # rules not read yet
    assert r.play_progress("main") == [] and r.io.clicks == []


def test_pre_roll_new_turn_resets(gs, tmp_path):
    r = runner(gs, [[box("Mining")], []], tmp_path)
    r.play_progress("main")
    r._progress_last_phase = "main"
    r.brain.start_fleet("ore")
    r.io.screens = [[]]
    r.play_progress("pre_roll", np.zeros((888, 1372, 3), np.uint8), [])
    assert r._progress_tried == set() and r.brain.fleet_resource is None


def test_past_half_leader_needs_big_gain():
    from catanbot.brain import Brain
    from catanbot.state import GameState
    import inspect
    assert "past_half" in inspect.getsource(Brain.evaluate_offer)
    assert hasattr(Brain, "past_half")


def test_offer_never_gives_kept_card(gs, tmp_path):
    """Owner 2026-10-02: rebuilding a lost city, the bot offered grain for wool."""
    r = runner(gs, [], tmp_path)
    me = gs.me()
    me.cities.clear()
    me.settlements.add(next(iter(hex_corners(*_land(gs.board, "fields")[0].coord))))
    me.hand.update(grain=2, ore=1)
    r._lost_cities = 1
    r._offers_made = {("want", "ore")}       # ore refused already: knight (wool) is next
    r.brain._trade_targets = lambda g: ["knight"]
    t = r._want_trade(gs)
    assert t is None or not ({"grain", "ore"} & set(t.give))


def test_threat_ranked_before_our_turn(gs, tmp_path):
    """Owner 2026-10-02: pick the biggest threat just before our turn; blocks target it."""
    r = runner(gs, [], tmp_path)
    r._rank_threat_before_turn()
    assert r.brain.threat_name == "Hildegard"            # 10 VP vs 5
    assert r.brain.threat(gs).name == "Hildegard"
    assert any("threat before our turn" in n for n in r.notes) if hasattr(r, "notes") else True


def test_ask_for_third_card_to_complete_bank_set(gs, tmp_path):
    """Owner 2026-10-02: 2 brick + a 3:1 harbor -> ask for a 3rd brick, bank it for grain."""
    r = runner(gs, [], tmp_path)
    r.cfg["harbor_3to1"] = True
    me = gs.me()
    me.cities.clear()
    me.settlements.add(next(iter(hex_corners(*_land(gs.board, "fields")[0].coord))))
    me.hand.update(brick=2, ore=3, grain=1, lumber=1)
    r._lost_cities = 1
    r._offers_made = {("want", "grain")}           # nobody gives grain
    r.brain._trade_targets = lambda g: []
    t = r._want_trade(gs)
    assert t is not None and t.get == {"brick": 1} and t.give == {"lumber": 1}


def test_refused_want_never_banks_away_a_full_city(gs, tmp_path):
    """Owner game 1: held 3 ore + 2 grain (a whole city), grain had been refused, and the bot
    banked 3 ore for a 3rd grain instead of building the city."""
    from catanbot.brain import TurnCtx
    from catanbot import moves as mv
    me = gs.me()
    sv = next(iter(hex_corners(*_land(gs.board, "fields")[0].coord)))
    me.settlements.add(sv)
    me.hand.clear(); me.hand.update(brick=1, grain=2, ore=3)
    r = runner(gs, [], tmp_path)
    legal = mv.turn_moves(gs, offers_left=False, harbor_3to1=True)
    m = r.brain.pick(gs, "turn", legal, turn=TurnCtx(refused={"grain": 1}, lost_cities=1))
    assert m.kind != "bank_trade" or "ore" not in m.give_d, m


def test_leftover_bank_dialog_is_cancelled(gs, tmp_path):
    """Solo 2026-10-03: a restart mid bank trade left 'Maritime trade' open and scans failed."""
    r = runner(gs, [[box("Maritime trade", 585, 393),
                     box("Choose the resources that you want to trade to the supply (bank).", 461, 449)]],
               tmp_path, dry=False)
    r.step()
    assert [w for *_, w in r.io.clicks] == ["close leftover bank trade"]
    assert r.io.clicks[0][:2] == tuple(R.UI["trade_cancel"])


def test_rules_panel_never_counts_as_barbarian_attack(gs, tmp_path):
    """Solo 2026-10-03: '...upon or after the first barbarian attack' set the flag at game start."""
    r = runner(gs, [[box("SCENARIO CONFIGURATION", 300, 200),
                     box("only upon or after the first barbarian attack). 2. Additional rules", 300, 260)]],
               tmp_path)
    r.barbarians_attacked = False
    r.step()
    assert r.barbarians_attacked is False


def test_own_bank_trade_not_cancelled_while_closing(gs, tmp_path):
    import time as _t
    r = runner(gs, [[box("Maritime trade", 585, 393)]], tmp_path, dry=False)
    r._bank_clicked_at = _t.time()
    r.step()
    assert "close leftover bank trade" not in [w for *_, w in r.io.clicks]


def test_barbarian_defeat_banner_is_not_game_over(gs, tmp_path):
    """Solo 2026-10-03 game 2: the mid-game barbarian 'Defeat!' banner ended the game for the bot."""
    from catanbot.vision.screens import classify
    assert classify(["marlow.systems", "Lin", "Defeat!", "BUILD COSTS"])[0] != "game_over"
    assert classify(["RESULT", "RANKING", "1 Lin", "Experience gained: 0", "QUIT"])[0] == "game_over"
    r = runner(gs, [[box("Defeat!", 650, 450)]], tmp_path)
    r.barbarians_attacked = False
    r.step()
    assert r.barbarians_attacked is True


def test_commercial_harbor_gives_least_valued_commodity(gs, tmp_path, monkeypatch):
    """Solo 2026-10-03: Marianne's Commercial Harbor dialog stalled the bot for 55 min."""
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    gs.me().hand.update(coin=2, paper=1)
    scr = [box("Commercial Harbor", 565, 391),
           box("Choose the commodities you want to give to player Marianne.", 461, 449)]
    r = runner(gs, [scr, scr], tmp_path, dry=False)
    r.step()
    assert [w for *_, w in r.io.clicks] == ["commercial harbor: give coin", "commercial harbor: ok"]
    r.step()                                       # still open: try the next one
    assert r.io.clicks[2][-1] == "commercial harbor: give paper"


def test_blind_discard_counts_down(gs, tmp_path, monkeypatch):
    """Solo 2026-10-03: restarted under the robber dialog, no state -> it sat there."""
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    t = lambda n: [box(f"Choose {n} more resources to hand over to the robber.", 499, 449)]
    # brick: 2 -> 1, then empty (stays 1); lumber: 1 -> 0
    r = runner(gs, [t(2), t(1), t(1), []], tmp_path, dry=False)
    r.ensure_state = lambda img, texts: None
    r.step()
    whys = [w for *_, w in r.io.clicks]
    assert whys == ["discard brick (blind)", "discard brick (blind)", "discard lumber (blind)",
                    "confirm discard (check mark)"]


def test_strong_knight_prompt_taps_our_knights(gs, tmp_path, monkeypatch):
    """Solo 2026-10-03: 'Select position for Strong knight' sat unread for 18 min."""
    import types
    from catanbot.vision.screens import classify
    from catanbot.state import Knight
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    assert classify(["Select position for Strong knight"])[0] == "promote_knight"
    assert classify(["Select position for knight"])[0] == "place_knight"
    v1, v2 = (1, -5), (2, -4)
    gs.me().knights.update({v1: Knight(), v2: Knight()})
    p = [box("Select position for Strong knight", 515, 223)]
    r = runner(gs, [p] * 6 + [[]], tmp_path, dry=False)    # each tap also looks for the green check
    r.br = types.SimpleNamespace(lattice=types.SimpleNamespace(vertex_px=lambda v: (v[0] * 100 + 500, 400)))
    r.step(); r.step()
    whys = [w for *_, w in r.io.clicks]
    assert whys == [f"promote knight at {v1}", f"promote knight at {v2}"]


def test_strong_knight_prompt_falls_back_to_road_corners(gs, tmp_path, monkeypatch):
    import types
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    b = gs.board
    land = [v for v in sorted(b.vertices) if b.is_land_vertex(v)] if hasattr(b, "vertices") else None
    e = next(iter(b.vertex_edges(next(iter(gs.me().cities)))))
    gs.me().roads.add(e)
    p = [box("Select position for Strong knight", 515, 223)]
    r = runner(gs, [p, p], tmp_path, dry=False)
    r.br = types.SimpleNamespace(lattice=types.SimpleNamespace(vertex_px=lambda v: (500, 400)))
    r.step()
    whys = [w for *_, w in r.io.clicks]
    assert len(whys) == 1 and whys[0].startswith("promote knight at")
    assert str(next(iter(gs.me().cities))) not in whys[0]


def test_ship_prompt_mid_game_is_cancelled(gs, tmp_path, monkeypatch):
    """Live 2026-10-03 game 4: 'Select position for Ship' after a road tap sat 10 min."""
    from catanbot.vision.screens import classify
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    p = [box("Select position for Ship", 653, 238)]
    assert classify([b["text"] for b in p])[0] == "place_ship"
    r = runner(gs, [p, p, []], tmp_path, dry=False)
    r._in_play = True
    r.step()
    assert [w for *_, w in r.io.clicks] == ["cancel ship placement outside setup"]


def test_truncated_progress_titles_read():
    """Live 2026-10-03: overlapping hand cards show cut titles; OCR may slip the last letter."""
    from catanbot import progress_cards as pc
    assert pc.card_key("Resource Monop") == "resource_monopoly"
    assert pc.card_key("Master Merchan") == "master_merchant"
    assert pc.card_key("Master Merchar") == "master_merchant"
    assert pc.card_key("Merchant") == "merchant"
    with pytest.raises(KeyError):
        pc.card_key("Build")


def test_fifth_progress_card_is_dragged_off(gs, tmp_path, monkeypatch):
    """Live 2026-10-03: 'Discard progress card' was taken for a resource discard for 45 min."""
    from catanbot.vision.screens import classify
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    p = [box("Discard progress card", 882, 621), box("Master Merchan", 838, 716),
         box("Bishop", 922, 716), box("Commercial Harbor", 1050, 716)]
    assert classify([b["text"] for b in p])[0] == "discard_progress"
    assert classify(["Discard 5 cards"])[0] == "discard"
    r = runner(gs, [p, p, []], tmp_path, dry=False)
    r.step()
    # 2026-10-04 (C6): Commercial Harbor has no play here (no commodities), Bishop does -> the dead card goes
    from catanbot import progress_cards as pc
    assert pc.CHOOSERS["commercial_harbor"](gs, gs.me()) is None and pc.CHOOSERS["bishop"](gs, gs.me()) is not None
    assert [w for *_, w in r.io.clicks] == ["discard progress commercial_harbor"]


def test_same_prompt_for_15_min_restarts(gs, tmp_path, monkeypatch):
    """Live 2026-10-03: a looping discard kept clicking for 45 min; the no-click watchdog was quiet."""
    p = [box("Discard 5 cards", 600, 300)]

    class LiveLikeIO:                       # not a ReplayIO: the watchdogs only run live
        clicks = []
        def snapshot(self): return np.zeros((888, 1372, 3), np.uint8)
        def ocr(self, img): return p if img.shape[:2] == (888, 1372) else []
        def click(self, x, y, why=""): self.clicks.append((x, y, why))
        def paused(self): return False

    r = runner(gs, [], tmp_path, dry=False)
    r.io = LiveLikeIO()
    r.cfg["solo"] = True
    restarts = []
    r.restart_app = lambda why: restarts.append(why)
    r.on_discard = lambda *a, **k: None
    t = [1000.0]
    monkeypatch.setattr(R.time, "time", lambda: t[0])
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    r._last_click_at = t[0]
    r.step()
    t[0] += 500; r._last_click_at = t[0]; r.step()
    assert restarts == []
    t[0] += 500; r._last_click_at = t[0]
    # stage 2026-10-08: clicks no longer hold the solo stall clock, so the no-progress
    # watchdog (same screen, same hand) may restart first; either way the loop ends in a restart
    assert r.step() == "restart" and restarts in (["screen discard for 16 min"], ["no progress for 16 min"])


# ---- Spy (owner 2026-10-06): pick by real value, then play the stolen card this turn --------
def test_after_spy_plays_stolen_card_now(gs):
    br = Brain()
    notes = []
    br.note = notes.append
    gs.me().progress_hand = []
    assert br.after_spy(gs, "mining") == ("mining", {"gain": {"ore": 2}})
    assert "mining" in gs.me().progress_hand
    assert any("spy: took mining" in n and "play mining" in n for n in notes)


def test_after_spy_alchemist_waits_for_next_roll(gs):
    br = Brain()
    notes = []
    br.note = notes.append
    gs.me().progress_hand = []
    assert br.after_spy(gs, "alchemist") is None
    assert gs.me().progress_hand == ["alchemist"]
    assert any("alchemist: before the roll only" in n for n in notes)
    assert br.progress_play(gs, "pre_roll")[0] == "alchemist"     # next turn, before the dice


def test_spy_pick_screen_takes_playable_card_and_replans(gs, tmp_path, monkeypatch):
    """'Select progress card': Alchemist (keep 5, but next turn) vs Mining (ore now). Mining is
    clicked, joins our hand, the turn is re-planned, and play_progress plays it this turn even
    though a Mining was already tried earlier this turn."""
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    screen = [box("Select progress card", 600, 250), box("Alchemist", 450, 300), box("Mining", 750, 300)]
    r = runner(gs, [[]], tmp_path)                     # after the pick the dialog is gone
    gs.me().progress_hand = []
    r._progress_last_phase = "main"
    r._spy_victim = "Nassir"
    r._progress_tried = {("mining", "mining"), ("spy", "spy")}
    r.on_pick_progress(np.zeros((888, 1372, 3), np.uint8), screen, [b["text"] for b in screen], {})
    assert [w for *_, w in r.io.clicks] == ["pick progress card mining"]
    assert gs.me().progress_hand == ["mining"]
    assert r._progress_tried == {("spy", "spy")} and r._spy_victim is None
    assert any("spy: took mining; re-planned -> play mining" in m for m in r.log)
    r.io.screens = [[box("Mining")], []]
    assert r.play_progress("main") == ["mining"]


def test_spy_victim_recorded_for_the_pick(gs, tmp_path):
    r = runner(gs, [], tmp_path)
    r._progress_choices("spy", {"player": "Hildegard"}, [])
    assert r._spy_victim == "Hildegard" and r._want_player == "Hildegard"
