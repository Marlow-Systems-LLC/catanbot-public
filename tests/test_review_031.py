"""v0.3.1 code-review fixes: one regression test per fix (FIXES.md lists them)."""
import copy

from catanbot.board import ekey
from catanbot.moves import build_moves
from catanbot.state import Knight
from tests.test_canal_playbook import _gs


def _settle_spot(gs):
    me = gs.me()
    for v in gs.board.vertices:
        if gs.can_settle(v):
            return v
    raise AssertionError("no buildable corner on the fixture")


# ---- 1. a corner with a knight is never a settlement spot (live game 20261009_014102) ----------
def test_own_knight_corner_is_not_a_settlement_spot():
    for canal in (True, False):
        gs = _gs()
        gs.mode = copy.copy(gs.mode)
        gs.mode.canal = canal
        v = _settle_spot(gs)
        me = gs.me()
        n = sorted(gs.board.neighbors(v))[0]
        me.roads.add(ekey(v, n))
        me.hand = {"brick": 1, "lumber": 1, "wool": 1, "grain": 1}
        assert ("settlement", v) in {(m.kind, m.target) for m in build_moves(gs)}
        me.knights[v] = Knight(1, True)
        assert not gs.can_settle(v)
        assert v not in gs.legal_settlement_spots(me)
        assert ("settlement", v) not in {(m.kind, m.target) for m in build_moves(gs)}


def test_rival_knight_corner_is_not_a_settlement_spot():
    gs = _gs()
    v = _settle_spot(gs)
    gs.players[1].knights[v] = Knight(1, False)
    assert not gs.can_settle(v)


# ---- 2. game over: the handler sees the game, then the runner forgets it --------------------
def test_game_over_handler_sees_the_game_before_the_reset(monkeypatch):
    from tests.test_lobby_guard import _runner
    r = _runner([["marlow.systems has won", "Player statistics"]], monkeypatch)
    r.gs, r._game_dir = _gs(), "data/game_logs/x"
    r._vp, r._progress_once = {"red": 9}, {"engineer"}
    seen = {}
    monkeypatch.setattr(r, "on_game_over",
                        lambda *a: seen.update(gs=r.gs, dir=r._game_dir, vp=dict(r._vp)))
    assert r.step() == "game_over"
    assert seen["gs"] is not None and seen["dir"] == "data/game_logs/x" and seen["vp"] == {"red": 9}
    assert r.gs is None and r._game_dir is None
    assert "_vp" not in r.__dict__ and "_progress_once" not in r.__dict__


# ---- 3. every lazily-set per-game field is forgotten at the lobby ---------------------------
def test_lobby_forgets_lazy_per_game_fields(monkeypatch):
    from tests.test_lobby_guard import _runner
    r = _runner([["Waiting for players"]], monkeypatch)
    r._in_play = False
    for n in r._PER_GAME_LAZY:
        setattr(r, n, {"stale"})
    r._last_hand = {"ore": 3}
    r.chat.seen = {("x", "marlow hi")}
    r.step()
    assert not [n for n in r._PER_GAME_LAZY if r.__dict__.get(n) is not None]
    assert r.chat.seen == set()


# ---- 4. a setup prompt without text follows settlement, city, settlement (Great Canal) -------
def test_textless_setup_follows_the_rules_order(monkeypatch):
    from tests.test_lobby_guard import _runner
    r = _runner([[]], monkeypatch)
    gs = _gs()
    spots = [v for v in gs.board.home_spots()][:3]
    monkeypatch.setattr(r, "ensure_state", lambda *a: gs)
    monkeypatch.setattr(r, "_vertex_rings", lambda img: [(v, 1.0) for v in spots] * 3)
    kinds = []
    monkeypatch.setattr(r, "_setup", lambda img, texts, kind: kinds.append(kind))
    me = gs.me()
    r._setup_textless(None, [])
    me.settlements.add(spots[0])
    r._setup_textless(None, [])
    me.cities.add(spots[1])
    r._setup_textless(None, [])
    assert kinds == ["settlement", "city", "settlement"]


# ---- 5. Commercial Harbor never hands grain to a player --------------------------------------
def test_commercial_harbor_never_gives_grain(monkeypatch):
    from catanbot import progress_cards as pc
    from tests.test_lobby_guard import _runner
    gs = _gs()
    me = gs.me()
    for p in gs.players[1:]:
        p.hand_count = 3
    me.hand = {"grain": 4, "wool": 1}
    assert pc.choose_commercial_harbor(gs, me) is None          # 2 opponents, 1 non-grain card
    assert not pc.can_play(gs, me, "commercial_harbor", "main")[0]
    me.hand["ore"] = 1
    assert sorted(pc.choose_commercial_harbor(gs, me)["give"]) == ["ore", "wool"]
    r = _runner([[]], monkeypatch)
    me.hand = {"grain": 4}
    assert r._harbor_resource(gs) is None
    me.hand = {"grain": 4, "brick": 1}
    assert r._harbor_resource(gs) == "brick"


# ---- 6. a yes to our own offer still passes the partner rules (14 VP, commodity) ------------
def _accept_run(monkeypatch, give, get, accepter):
    import time as _t
    from tests.test_lobby_guard import _runner
    r = _runner([[]], monkeypatch)
    gs = _gs()
    gs.me().vp_visible = 10
    gs.players[1].vp_visible = 14            # leader: inside 1.5x, at the 14-VP lock
    gs.me().hand = {"wool": 3, "paper": 2, "ore": 1}
    r.gs = gs
    r.offer_sent, r._pending_offer = _t.time() - 100, (give, get)
    monkeypatch.setattr(r, "offer_responses",
                        lambda img: [(100, "accept" if i == accepter else "decline") for i in range(2)])
    monkeypatch.setattr(r, "_confirm_accept", lambda snap: None)
    monkeypatch.setattr(r, "_expect_trade", lambda *a: None)
    whys = []
    monkeypatch.setattr(r, "click", lambda xy, why, ocr=None: whys.append(why))
    r._handle_responses(None)
    return whys


def test_own_offer_yes_from_a_14_vp_player_is_not_taken(monkeypatch):
    clicks = _accept_run(monkeypatch, {"wool": 1}, {"ore": 1}, accepter=0)
    assert not any("accept trade" in c for c in clicks)


def test_own_offer_commodity_for_one_card_is_not_taken(monkeypatch):
    clicks = _accept_run(monkeypatch, {"paper": 1}, {"ore": 1}, accepter=1)
    assert not any("accept trade" in c for c in clicks)
    clicks = _accept_run(monkeypatch, {"wool": 1}, {"ore": 1}, accepter=1)
    assert any("accept trade" in c for c in clicks)


# ---- 7. our own offers never put grain on the table ------------------------------------------
def test_propose_trade_never_offers_grain():
    from tests.test_canal_playbook import brain
    gs = _gs()
    gs.players[1].vp_visible = 6
    me = gs.me()
    me.settlements.add(sorted(gs.board.home_spots())[0])
    me.hand = {"grain": 4, "ore": 2}                 # a city one ore short; the only spare is grain
    t = brain().propose_trade(gs)
    assert t is None or not t[0].get("grain")
    me.hand = {"grain": 4, "ore": 2, "brick": 1}
    t = brain().propose_trade(gs)
    assert t is not None and t[0] == {"brick": 1} and t[1] == {"ore": 1}


# ---- 8. the outlook never plans a metropolis a rival locked at level 5 -----------------------
def test_outlook_skips_a_metropolis_locked_at_level_5():
    from catanbot.outlook import player_moves
    gs = _gs()
    me, rival = gs.me(), gs.players[1]
    me.cities.add(sorted(gs.board.home_spots())[0])
    me.improvements = {"trade": 3, "politics": 3, "science": 0}
    rival.improvements = {"trade": 5, "politics": 0, "science": 0}
    tracks = {mv.target for mv in player_moves(gs, me) if mv.kind == "metropolis"}
    assert "politics" in tracks and "trade" not in tracks


# ---- 9. Smith: two picks share the free pieces of the next level ----------------------------
def test_smith_picks_only_as_many_knights_as_free_pieces():
    from catanbot import progress_cards as pc
    gs = _gs()
    me = gs.me()
    c = sorted(gs.board.home_spots())
    me.knights = {c[0]: Knight(1, True), c[1]: Knight(1, True), c[2]: Knight(2, True)}
    assert len(pc.choose_smith(gs, me)["knights"]) == 1       # one strong piece left
    me.knights[c[2]] = Knight(1, False)
    assert len(pc.choose_smith(gs, me)["knights"]) == 2


# ---- 10. the pause corner reads the screen size from NSScreen --------------------------------
def test_pause_corner_uses_nsscreen(monkeypatch):
    import sys
    import types
    from catanbot.runner import LiveIO

    class _P:
        def __init__(self, x, y):
            self.x, self.y = x, y

    class _Frame:
        class size:
            height = 1080

    class _Screen:
        def frame(self):
            return _Frame()

    at = {"p": _P(1, 1079)}
    fake = types.ModuleType("AppKit")
    fake.NSEvent = type("NSEvent", (), {"mouseLocation": staticmethod(lambda: at["p"])})
    fake.NSScreen = type("NSScreen", (), {"screens": staticmethod(lambda: [_Screen()])})
    monkeypatch.setitem(sys.modules, "AppKit", fake)
    io = LiveIO.__new__(LiveIO)
    assert io.paused()
    at["p"] = _P(500, 500)
    assert not io.paused()


# ---- 11. a refused click does not count as the handler acting --------------------------------
def test_refused_click_is_not_stamped(monkeypatch):
    from tests.test_lobby_guard import _runner
    r = _runner([[]], monkeypatch)
    r._last_click_at = 0
    ocr = [{"text": "Play online", "x": 90, "y": 90, "w": 40, "h": 20}]
    from catanbot.runner import FORBIDDEN_BUTTONS
    if not FORBIDDEN_BUTTONS.search("Play online"):
        ocr[0]["text"] = "Matchmaking"
    assert FORBIDDEN_BUTTONS.search(ocr[0]["text"])
    r.click((110, 100), "test", ocr=ocr)
    assert r._last_click_at == 0
    monkeypatch.setattr(r, "_click_loop", lambda xy, why: True)
    r.click((500, 500), "blocked")
    assert r._last_click_at == 0
    monkeypatch.setattr(r, "_click_loop", lambda xy, why: False)
    r.click((500, 500), "real")
    assert r._last_click_at > 0


# ---- 12. a city loss is recorded once, when the game takes it --------------------------------
def test_lose_city_counts_one_loss_over_several_prompt_frames(monkeypatch):
    import types
    import numpy as np
    from tests.test_lobby_guard import _runner
    prompt = ["Select position for Settlement"]
    r = _runner([prompt] * 4 + [[]] * 4, monkeypatch)
    gs = _gs()
    me = gs.me()
    c = sorted(gs.board.home_spots())
    me.cities |= {c[0], c[5]}
    r._lost_cities = 0
    monkeypatch.setattr(r, "ensure_state", lambda img, texts: gs)
    monkeypatch.setattr(r, "_rings", lambda img: [])
    monkeypatch.setattr(r, "_green_check", lambda img: False)
    monkeypatch.setattr(r, "_barb_attack", lambda why: None)
    r.br = types.SimpleNamespace(lattice=types.SimpleNamespace(vertex_px=lambda v: (100, 100)))
    img = np.zeros((888, 1372, 3), np.uint8)
    r.lose_city(img, prompt)                  # tap + confirm, the prompt stays: nothing recorded
    assert r._lost_cities == 0 and me.cities == {c[0], c[5]}
    r.lose_city(img, prompt)
    r.lose_city(img, prompt)                  # the prompt is gone after this tap
    assert r._lost_cities == 1
    assert len(me.cities) == 1 and len(me.settlements & {c[0], c[5]}) == 1


# ---- 13. a barbarian attack is saved at once (a restart keeps it) ----------------------------
def test_barbarian_attack_is_saved_to_the_game_flags(monkeypatch):
    import json
    from tests.test_lobby_guard import _runner
    r = _runner([[]], monkeypatch)
    r.barbarians_attacked = False
    r._my_knights = {(1, 1): True}
    r._barb_attack("test")
    d = json.loads(r.FLAGS.read_text())
    assert d["barbarians_attacked"] is True and d["my_knights"][0][1] is False


# ---- 14. Engineer without rings: the wall we placed is recorded (limit 7 -> 9) -------------
def test_place_wall_without_rings_records_the_wall(monkeypatch):
    import types
    import numpy as np
    from tests.test_lobby_guard import _runner
    r = _runner([[]], monkeypatch)
    gs = _gs()
    me = gs.me()
    v = sorted(gs.board.home_spots())[0]
    me.cities.add(v)
    r.gs, r.mode = gs, gs.mode
    monkeypatch.setattr(r, "ensure_state", lambda img, texts: gs)
    monkeypatch.setattr(r, "_green_check", lambda img: False)
    monkeypatch.setattr(r, "_vertex_rings", lambda img: [])
    monkeypatch.setattr(r, "_confirm_if_check", lambda why: True)
    r.br = types.SimpleNamespace(lattice=types.SimpleNamespace(vertex_px=lambda v: (100, 100)))
    r.on_place_wall(np.zeros((888, 1372, 3), np.uint8), [], [], {})
    assert v in me.walls and v in r._my_walls


def test_place_wall_check_only_frame_records_the_pending_city(monkeypatch):
    import numpy as np
    from tests.test_lobby_guard import _runner
    r = _runner([[]], monkeypatch)
    gs = _gs()
    me = gs.me()
    v = sorted(gs.board.home_spots())[0]
    me.cities.add(v)
    r.gs, r.mode, r._wall_pending = gs, gs.mode, v
    monkeypatch.setattr(r, "ensure_state", lambda img, texts: gs)
    monkeypatch.setattr(r, "_green_check", lambda img: True)
    monkeypatch.setattr(r, "_ringed_pick", lambda *a: True)
    r.on_place_wall(np.zeros((888, 1372, 3), np.uint8), [], [], {})
    assert v in me.walls and r._wall_pending is None


# ---- 15. a landed improvement or knight move keeps the turn building -----------------------
def test_landed_improve_and_move_reset_the_build_rounds(monkeypatch):
    from catanbot.moves import Move
    from tests.test_lobby_guard import _runner
    r = _runner([[]], monkeypatch)
    gs = _gs()
    r._build_rounds = 2
    monkeypatch.setattr(r, "click", lambda *a, **k: None)
    monkeypatch.setattr(r, "tap_improve",
                        lambda gs, t: r.__dict__.setdefault("_my_improve", {}).update({t: 1}))
    r._run_builds(gs, [Move("improve", "science")])
    assert r._build_rounds == 0
    r._build_rounds = 2
    monkeypatch.setattr(r, "tap_move_knight", lambda a: True)
    r._run_builds(gs, [Move("move_knight", (1, 1))])
    assert r._build_rounds == 0


# ---- 16. with no city left a commodity is still never a 1-for-1 give ----------------------
def test_no_city_commodity_is_not_a_one_for_one_give(monkeypatch):
    from tests.test_lobby_guard import _runner
    r = _runner([[]], monkeypatch)
    gs = _gs()
    me = gs.me()
    me.cities = set()
    me.settlements.add(sorted(gs.board.home_spots())[0])
    me.hand = {"paper": 2, "coin": 1, "grain": 2, "wool": 1}     # a city wants 3 ore
    assert r._improve_commodities(gs) == {"paper", "coin", "cloth"}
    monkeypatch.setattr(r.brain, "_trade_targets", lambda gs: ["city", "knight"])
    monkeypatch.setattr(r.brain, "table_has", lambda gs, res: True)
    monkeypatch.setattr(r.brain, "grain_rush", lambda gs: False)
    for _ in range(4):
        a = r._want_trade(gs)
        if a is None:
            break
        assert not set(a.give) & {"paper", "coin", "cloth"}, a
        r.__dict__.setdefault("_offers_made", set()).update(("want", x) for x in a.get)
    monkeypatch.setattr(r.brain, "grain_rush", lambda gs: True)
    r._offers_made = set()
    me.hand = {"paper": 2, "coin": 1, "wool": 1}
    a = r._want_trade(gs)
    assert a is None or not set(a.give) & {"paper", "coin", "cloth"}, a


# ---- 17. a counter offer whose sender was not read must pass with every opponent ----------
def test_counter_offer_unknown_partner_uses_the_strictest_check(monkeypatch):
    from tests.test_lobby_guard import _runner
    r = _runner([[]], monkeypatch)
    gs = _gs()
    gs.me().hand = {"wool": 3, "ore": 0, "lumber": 2}
    gs.players[1].vp_visible = 15                              # leader past the 14-VP lock
    give, get = {"wool": 1}, {"ore": 1}
    assert not r.brain.evaluate_offer(gs, "leader", give, get)
    weak_ok = r.brain.evaluate_offer(gs, "weak", give, get)
    assert r._counter_ok(gs, "weak", give, get) == weak_ok
    assert not r._counter_ok(gs, "", give, get)


# ---- 18. an accepted offer answered again is booked in the hand ledger once ---------------
def test_reanswered_offer_is_booked_once(monkeypatch):
    import catanbot.runner as R
    from tests.test_runner import RULES, runner
    r = runner(["gc_trade_offer_in.jpg"] * 12)
    r.read_rules(RULES)
    clock = {"t": 1000.0}
    monkeypatch.setattr(R.time, "time", lambda: clock["t"])
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    monkeypatch.setattr(r.brain, "evaluate_offer", lambda *a: True)
    booked = []
    led = r._ledger()
    monkeypatch.setattr(led, "trade", lambda *a, **k: booked.append(a))
    monkeypatch.setattr(r, "_ledger", lambda: led)
    for _ in range(12):
        r.step()
        clock["t"] += 25                           # past the 20 s re-answer timer each frame
    assert r.__dict__.get("_answer_retries", 0) >= 1    # it did answer again
    assert len(booked) == 1


# ---- 19. new cards from anywhere (not only a trade) unblock a build the popup showed red ------
def test_hand_change_unblocks_a_red_activate(monkeypatch):
    from tests.test_lobby_guard import _runner
    r = _runner([[]], monkeypatch)
    gs = _gs()
    t = (-3, -1)
    gs.me().hand = {"grain": 1, "ore": 1}
    monkeypatch.setattr(r, "ensure_state", lambda img, texts: gs)
    for n in ("_handle_responses", "_confirm_accept", "_counter_offer", "bar_hidden", "_chat_poll_turn"):
        monkeypatch.setattr(r, n, lambda *a: False)
    monkeypatch.setattr(r, "play_progress", lambda *a, **k: False)
    monkeypatch.setattr(r, "offer_wait_s", lambda: 0)
    import time
    r.offer_sent, r.turn_started = 0, time.time()    # 0.4.3: a turn 1e9 s old hits the timer's End turn
    r._build_turn = r.turn_started
    r._plan_hand = (("ore", 1),)                       # the hand before Wedding brought the grain
    r._unaffordable = {"activate"}
    r._failed_taps = {("activate", t), ("road", (1, 1))}
    seen = {}

    class _Stop(Exception):
        pass

    def plan(g):
        seen["blocked"] = set(g.blocked)
        raise _Stop
    monkeypatch.setattr(r.brain, "plan_turn", plan)
    try:
        r.on_main(None, [], [], {})
    except _Stop:
        pass
    assert ("activate", t) not in seen["blocked"] and ("road", (1, 1)) in seen["blocked"]


# ---- 20. a trade panel with no offer open does not re-arm a used-up want ------------------
def test_trade_panel_fallback_rearms_only_a_live_offer(monkeypatch):
    import catanbot.runner as R
    from tests.test_lobby_guard import _runner
    r = _runner([[]], monkeypatch)
    monkeypatch.setattr(r, "_confirm_accept", lambda *a: False)
    monkeypatch.setattr(r, "_handle_responses", lambda *a: False)
    monkeypatch.setattr(r, "offer_wait_s", lambda: 15)
    monkeypatch.setattr(r, "click", lambda *a, **k: None)
    r.offer_sent, r.last_want, r.trade_refused = 0, "ore", None
    r.on_trade_panel(None, [], [], {})
    assert r.trade_refused is None and r.last_want is None
    r.offer_sent, r.last_want = R.time.time() - 30, "ore"
    r.on_trade_panel(None, [], [], {})
    assert r.trade_refused == "ore" and r.offer_sent == 0


# ---- 21. the chart's generic label path counts a level only when the hand paid --------------
def test_improve_chart_label_needs_the_commodity_to_leave(monkeypatch):
    import numpy as np
    from tests.test_lobby_guard import _runner
    r = _runner([[]], monkeypatch)
    lbl = [{"text": "Upgrade", "x": 400, "y": 300, "w": 60, "h": 20, "conf": 1.0}]
    monkeypatch.setattr(r.io, "snapshot", lambda: np.zeros((888, 1372, 3), np.uint8))
    monkeypatch.setattr(r.io, "ocr", lambda img: lbl)
    monkeypatch.setattr(r, "click", lambda *a, **k: None)
    monkeypatch.setattr(r, "_improve_levels", lambda: {"science": 0, "trade": 0, "politics": 0})
    hands = iter([{"paper": 3}, {"paper": 3}])
    monkeypatch.setattr(r, "_hand_now", lambda img=None: next(hands))
    assert r._tap_improve_chart("science", (10, 10)) is False
    hands = iter([{"paper": 3}, {"paper": 2}])
    monkeypatch.setattr(r, "_hand_now", lambda img=None: next(hands))
    assert r._tap_improve_chart("science", (10, 10)) is True


# ---- 22. a discard is confirmed even when the after-click frame is missing -----------------
def test_discard_confirms_without_an_after_frame(monkeypatch, tmp_path):
    import numpy as np
    import catanbot.runner as R
    from tests.test_lobby_guard import _runner
    monkeypatch.setattr(R, "ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    r = _runner([[]], monkeypatch)
    gs = _gs()
    gs.me().hand = {"wool": 4, "ore": 4}
    monkeypatch.setattr(r, "ensure_state", lambda img, texts: gs)
    monkeypatch.setattr(r.io, "snapshot", lambda: None)
    whys = []
    monkeypatch.setattr(r, "click", lambda xy, why, ocr=None: whys.append(why))
    r.on_discard(np.zeros((888, 1372, 3), np.uint8), [], ["Choose 4 more resources"], {})
    assert whys and whys[-1] == "confirm discard (check mark)"


# ---- 23. a new card's player pick and a new gain pick start fresh -------------------------
def test_new_card_and_new_gain_pick_forget_old_tries(monkeypatch, tmp_path):
    import numpy as np
    import catanbot.runner as R
    from tests.test_lobby_guard import _runner
    r = _runner([[]], monkeypatch)
    r._player_tried = {"Jean"}
    r._progress_choices("master_merchant", {"player": "Jean"}, [])
    assert r._player_tried == set() and r._want_player == "Jean"
    r.popup_dir = tmp_path
    monkeypatch.setattr(r.brain, "choose_resource", lambda gs, n: ["grain"])
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    whys = []
    monkeypatch.setattr(r, "click", lambda xy, why, ocr=None: whys.append(why))
    r._gain_round, r._gain_at = 1, R.time.time() - 300          # left over from an old pick
    r.on_choose_resource(np.zeros((888, 1372, 3), np.uint8), [], ["Science bonus"], {})
    assert whys[0] == "gain grain"


# ---- 24. phantom setup pieces are dropped before the 3rd-pick rules count our buildings ------
def test_setup_reconciles_phantoms_before_the_third_pick_rules(monkeypatch):
    import types
    import catanbot.vision.board_reader as BR
    from tests.test_lobby_guard import _runner
    r = _runner([[]], monkeypatch)
    gs = _gs()
    spots = sorted(gs.board.home_spots())
    me = gs.me()
    me.settlements, me.cities = {spots[0], spots[7]}, set()      # spots[7] is a phantom
    r._setup_picks = [{"kind": "settlement", "v": spots[0]}]
    monkeypatch.setattr(r, "ensure_state", lambda img, texts: gs)
    r.br = types.SimpleNamespace(highlights=[])
    seen = {}

    class _Stop(Exception):
        pass

    def opts(g, br):
        seen["n"] = len(g.me().settlements) + len(g.me().cities)
        raise _Stop
    monkeypatch.setattr(BR, "setup_options", opts)
    try:
        r._setup(None, [], "city")
    except _Stop:
        pass
    assert seen["n"] == 1                       # the city pick is pick 2, not the coastal 3rd


# ---- 25. after a handler click the fallback looks at a fresh frame, not the pre-click one -------
def test_fallback_after_a_handler_click_uses_a_fresh_frame(monkeypatch):
    import numpy as np
    from tests.test_lobby_guard import _runner
    r = _runner([[]] * 4, monkeypatch)
    seen = []
    monkeypatch.setattr(r, "_forced_fallback", lambda img, state, *a, **k: seen.append(img))
    old = np.full((888, 1372, 3), 7, np.uint8)
    r._same_n = ["place_robber", 5]
    r._after_handler("place_robber", True, old)
    assert len(seen) == 1 and seen[0] is not old and int(seen[0].max()) == 0
    seen.clear()
    r._same_n = ["place_robber", 5]
    r._after_handler("place_robber", False, old)      # no click: the frame is still current
    assert seen == [old]


# ---- 26. textless ringed prompt: the after-click count includes per-corner rings --------------
def test_ringed_fallback_counts_corner_rings_after_the_confirm(monkeypatch):
    import numpy as np
    from tests.test_lobby_guard import _runner
    r = _runner([[]] * 6, monkeypatch)
    frames = iter([np.full((888, 1372, 3), k, np.uint8) for k in (2, 3)])
    r.io.snapshot = lambda: next(frames)
    monkeypatch.setattr(r, "click", lambda *a, **k: None)
    before = np.full((888, 1372, 3), 1, np.uint8)
    monkeypatch.setattr(r, "_rings", lambda im: [(400, 300), (520, 300)] if int(im.max()) == 1 else [])
    monkeypatch.setattr(r, "_vertex_rings",
                        lambda im: [] if int(im.max()) == 1 else [(None, (400, 300)), (None, (520, 300))])
    res = []
    monkeypatch.setattr(r.probe, "result", lambda kind, ok, why="": res.append(ok))
    r._forced_fallback(before, "ringed_prompt")
    assert res == [False]                 # both rings still up: the prompt did not leave


# ---- 27. sim_learn: model writes are atomic, a torn model file is retried, typos fail --------
def test_model_save_is_atomic_and_leaves_no_temp(tmp_path):
    from catanbot.learning import Model
    p = tmp_path / "model.json"
    m = Model(path=None)
    m.path, m.games = p, 3
    m.save()
    assert Model(p).games == 3 and [f.name for f in tmp_path.iterdir()] == ["model.json"]


def test_corrupt_model_keeps_the_old_brain_and_retries(monkeypatch, tmp_path):
    import os
    from catanbot import learning
    from catanbot import runner as R
    from catanbot.learning import Model
    from tests.test_lobby_guard import _runner
    p = tmp_path / "model.json"
    m = Model(path=None); m.path, m.games = p, 1; m.save()
    monkeypatch.setattr(learning, "MODEL", p)
    monkeypatch.setattr(R, "Model", lambda: Model(p))
    r = _runner([[]], monkeypatch)
    r._model_mtime = None
    r._reload_model_if_promoted()                     # first poll only records the mtime
    old = r.brain
    p.write_text('{"weights": {')                     # a half-written file
    os.utime(p, (1, 1))
    r._reload_model_if_promoted()
    assert r.brain is old and r._model_mtime != 1
    m.games = 5; m.save(); os.utime(p, (2, 2))
    r._reload_model_if_promoted()
    assert r.brain is not old and r.model.games == 5 and r._model_mtime == 2


def test_unknown_brain_flag_is_an_error():
    import pytest
    from catanbot.brain import Brain, set_flag
    from catanbot.learning import Model
    b = Brain(Model(path=None))
    set_flag(b, "road_cuts", False)
    assert b.road_cuts is False
    with pytest.raises(AttributeError):
        set_flag(b, "road_cutz", True)


def test_live_learn_read_reports_a_corrupt_file(tmp_path, capsys):
    from catanbot import live_learn as L
    bad = tmp_path / "pending.json"
    bad.write_text("{not json")
    assert L._read(bad) is None
    assert "unreadable" in capsys.readouterr().err
    assert L._read(tmp_path / "missing.json") is None


# ---- 28. one source for the 7-discard limit -------------------------------------------------
def test_discard_limit_has_one_source():
    import re
    from pathlib import Path
    from catanbot import progress_cards as pc
    from catanbot.rules import discard_limit
    root = Path(__file__).resolve().parent.parent / "catanbot"
    hits = [f"{f.name}:{i}" for f in root.glob("*.py") if f.name != "rules.py" for i, line in enumerate(f.read_text().splitlines(), 1)
            if re.search(r"2 \* len\(.*walls|discard_limit *\+", line)]
    assert hits == [], hits
    gs = _gs()
    me = gs.me()
    base = gs.mode.discard_limit
    me.walls = set(list(me.cities)[:1]) or {(0, 0)}
    assert discard_limit(gs.mode, me) == pc.discard_limit(gs, me) == base + 2
