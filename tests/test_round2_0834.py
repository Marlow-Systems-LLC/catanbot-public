"""Round 2 (stage game 0834, 0.3.4): knight prompt buttons found by colour, city popup icons by
position, one action per knight per turn, a skipped promote leaves the plan, declined offers
capped, hand read once more just after a screen change."""
import types

import cv2
import numpy as np
import pytest

import catanbot.runner as R
from board_fixture import frame
from catanbot import moves as mv
from catanbot.brain import Action, Brain
from catanbot.state import Knight
from tests.test_progress_wiring import gs, runner  # noqa: F401 (fixture)


@pytest.fixture
def live_layout():
    """The game host's 1920x1080 full screen: UI setup_confirm at (654,645) in the frame."""
    from catanbot.vision.layout import Layout
    saved = {k: (dict(v) if isinstance(v, dict) else v) for k, v in R.UI.items()}
    glob = {k: getattr(R, k) for k in ("LAY", "HAND_XS", "HAND_Y", "BOARD_ROI", "BAR_BOX")}
    resp = R.Runner.RESP_XS
    R.set_layout(Layout(1920, 1080))
    yield
    R.UI.clear(); R.UI.update(saved)
    for k, v in glob.items():
        setattr(R, k, v)
    R.Runner.RESP_XS = resp


GREEN, RED = (60, 170, 40), (40, 40, 210)      # BGR


def _bar(green=None, red=None):
    img = np.zeros((888, 1372, 3), np.uint8)
    if green:
        cv2.circle(img, green, 20, GREEN, -1)
    if red:
        cv2.circle(img, red, 20, RED, -1)
    return img


class _IO(R.ReplayIO):
    def __init__(self, img, texts=()):
        super().__init__([])
        self.img, self.texts = img, list(texts)

    def snapshot(self):
        return self.img

    def ocr(self, img):
        return [{"text": t, "x": 600, "y": 300, "w": 60, "h": 16, "conf": 1.0} for t in self.texts]


# ---- 1. confirm / cancel by colour ------------------------------------------------------------
def test_check_found_beside_the_x(live_layout):
    img = _bar(green=(621, 645), red=(683, 645))
    x, y = R.Runner._check_xy(img)
    assert abs(x - 621) <= 3 and abs(y - 645) <= 3
    assert R.Runner._green_check(img) and not R.Runner._red_x(img)
    alone = _bar(red=(654, 645))
    assert R.Runner._check_xy(alone) is None and R.Runner._red_x(alone)
    assert not R.Runner._green_check(alone)


@pytest.mark.parametrize("name,check,x_alone", [
    ("knightbar_085334_540_610.png", (621, 645), False),     # two buttons: check + X
    ("knightbar_090715_540_610.png", None, True),            # the red X alone
])
def test_real_knight_bars(live_layout, name, check, x_alone):
    img = frame(name)
    got = R.Runner._check_xy(img)
    if check is None:
        assert got is None
    else:
        assert abs(got[0] - check[0]) <= 3 and abs(got[1] - check[1]) <= 3
    assert R.Runner._red_x(img) is x_alone
    assert R.Runner._green_check(img) is (check is not None)


def test_fixed_point_missed_both_buttons(live_layout):
    """The old read at (654,645): neither button -- the ring loop of 08:53."""
    img = frame("knightbar_085334_540_610.png")
    assert not R.Runner._green_check_fixed(img) and not R.Runner._red_x_fixed(img)


def test_ringed_pick_clicks_the_check_where_it_is(gs, tmp_path, live_layout):
    r = runner(gs, [], tmp_path, dry=False)
    img = _bar(green=(621, 645), red=(683, 645))
    r.io = _IO(img)
    assert r._ringed_pick(img, [], "place_knight", lambda g, o: o[0], "knight")
    x, y, why = r.io.clicks[-1]
    assert "green check" in why and abs(x - 621) <= 3


def test_cfg_off_uses_the_fixed_point(gs, tmp_path, live_layout):
    r = runner(gs, [], tmp_path, dry=False)
    r.cfg["confirm_by_colour"] = False
    img = _bar(green=(654, 645))
    r.io = _IO(img)
    assert r._green_check(img)
    r._ringed_pick(img, [], "place_knight", lambda g, o: o[0], "knight")
    assert r.io.clicks[-1][:2] == (654, 645)
    assert not r._green_check(_bar(green=(621, 645), red=(683, 645)))


def test_red_x_alone_skips_promote_this_turn_only(gs, tmp_path, live_layout, monkeypatch):
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    r = runner(gs, [], tmp_path, dry=False)
    r.io = _IO(_bar(red=(654, 645)))
    r._turn_n, r._promote_fb_n = 5, 2
    r._after_handler("promote_knight", True, r.io.img)
    assert "cancel strong-knight" in r.io.clicks[-1][2]
    assert getattr(r, "_promote_disabled", False) is False
    assert r._promote_skipped()
    r._turn_n = 6                                  # next own turn: promotions planned again
    assert not r._promote_skipped()


def test_fallback_rings_skip_both_buttons(gs, tmp_path, live_layout):
    r = runner(gs, [], tmp_path)
    img = _bar(green=(621, 645), red=(683, 645))
    r._rings = lambda im: [(621, 645), (683, 646), (400, 300)]
    r._vertex_rings = lambda im: []
    assert r._fallback_rings(img) == [(400, 300)]


# ---- 2. city popup by position ----------------------------------------------------------------
XS = (560, 685, 746, 875)


def _popup_runner(monkeypatch, kinds):
    import catanbot.vision.screens as S
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    monkeypatch.setattr(R.cv2, "imwrite", lambda *a, **k: True)
    monkeypatch.setattr(S, "option_rings", lambda img, near=None:
                        [{"x": x, "y": 255, "colour": "green"} for x in XS] if near is not None else [])
    monkeypatch.setattr(S, "icon_kind", lambda img, x, y: (dict(zip(XS, kinds))[x], None))
    r = R.Runner.__new__(R.Runner)
    r.br = types.SimpleNamespace(lattice=types.SimpleNamespace(vertex_px=lambda t: (720, 300)), board=None)
    r.io = types.SimpleNamespace(snapshot=lambda: np.zeros((888, 1372, 3), np.uint8), ocr=lambda img: [])
    r.clicks, r.notes = [], []
    r.click = lambda p, why, *a: r.clicks.append((why, p))
    r.note = r.notes.append
    r._save_game_flags = lambda: None
    me = types.SimpleNamespace(cities={(0, 0, 0)}, hand={})
    return r, types.SimpleNamespace(me=lambda: me)


@pytest.mark.parametrize("track,x", [("politics", 746), ("science", 875), ("trade", 685)])
def test_repeated_icons_relabelled_by_order(monkeypatch, track, x):
    r, g = _popup_runner(monkeypatch, ["wall", "trade", "science", "science"])
    r._improve_via_city(g, track)
    ring = [p for why, p in r.clicks if why.endswith("(city ring)")]
    assert ring == [(x, 255)]
    assert any("relabelled by order" in n for n in r.notes)


def test_distinct_icons_kept_and_cfg_off(monkeypatch):
    r, g = _popup_runner(monkeypatch, ["wall", "trade", "politics", "science"])
    rings = [{"x": x, "kind": k} for x, k in zip(XS, ["wall", "trade", "science", "politics"])]
    r._improve_by_order(rings)                      # all different: the read stands
    assert [q["kind"] for q in rings] == ["wall", "trade", "science", "politics"]
    r.cfg = {"improve_by_order": False}
    rings = [{"x": x, "kind": k} for x, k in zip(XS, ["wall", "trade", "science", "science"])]
    r._improve_by_order(rings)
    assert [q["kind"] for q in rings] == ["wall", "trade", "science", "science"]
    rings = [{"x": x, "kind": k} for x, k in zip(XS, ["road", "ship", None, None])]
    r.cfg = {}
    r._improve_by_order(rings)                      # not the city popup
    assert [q["kind"] for q in rings] == ["road", "ship", None, None]


# ---- 3. one action per knight per turn --------------------------------------------------------
def _knight_net(g):
    """Two roads off our city and an active knight at their far end; a free corner beyond."""
    me = g.me()
    c = next(iter(me.cities))
    e1 = next(e for e in g.board.vertex_edges(c) if all(g.board.is_land_vertex(v) for v in e))
    me.roads.add(e1)
    v1 = e1[0] if e1[1] == c else e1[1]
    e2 = next(e for e in g.board.vertex_edges(v1) if e != e1 and all(g.board.is_land_vertex(v) for v in e))
    me.roads.add(e2)
    v2 = e2[0] if e2[1] == v1 else e2[1]
    return v1, v2


def test_knight_activated_this_turn_does_not_move(gs, tmp_path):
    v, free = _knight_net(gs)
    r = runner(gs, [], tmp_path)
    r.turn_started = 1000.0
    me = gs.me()
    me.knights[v] = Knight(active=True)
    r._activated_at = {v: 1001.0}
    r._mark_acted(me)
    assert not any(m.target[0] == v for m in mv.knight_moves(gs))
    me.knights[v] = Knight(active=True)
    r._activated_at = {v: 900.0}                   # activated on an earlier turn
    r._mark_acted(me)
    assert any(m.target[0] == v for m in mv.knight_moves(gs))
    r._moved_knights = {v}                          # moved here this turn
    r._mark_acted(me)
    assert not any(m.target[0] == v for m in mv.knight_moves(gs))
    me.knights[v] = Knight(active=True)
    r.cfg["knight_acted_memory"] = False
    r._mark_acted(me)
    assert any(m.target[0] == v for m in mv.knight_moves(gs))


# ---- 4. a skipped promote blocks what needs it ------------------------------------------------
def _two_basic(g):
    v1, v2 = _knight_net(g)
    me = g.me()
    me.knights[v1] = Knight(active=True)
    c = next(iter(me.cities))
    me.knights[c] = Knight(active=True)             # (a corner of ours; only the count matters)
    me.improvements["politics"] = 3
    me.hand.update(lumber=9, brick=9, wool=9, grain=9, ore=9, cloth=9, coin=9, paper=9)
    return v1, c, v2


def test_blocked_promote_frees_no_knight_piece(gs):
    v1, c, v2 = _two_basic(gs)
    kinds = {m.kind for m in mv.build_moves(gs)}
    assert "promote" in kinds and "knight" not in kinds        # both basic pieces are out
    gs.blocked = {"promote"}
    kinds = {m.kind for m in mv.build_moves(gs)}
    assert "promote" not in kinds and "knight" not in kinds
    plan = Brain().plan_turn(gs)
    assert not any(a.kind in ("promote", "knight") for a in plan)


def test_knight_behind_a_skipped_promote_is_dropped(gs):
    v1, c, v2 = _two_basic(gs)
    b = Brain()
    pro, kn = Action("promote", v1), Action("knight", v2)
    assert b.still_legal(gs, [], [], kn) is True                # nothing skipped: no check
    assert b.still_legal(gs, [pro], [], kn) is True             # the promote stands: knight legal after it
    assert b.still_legal(gs, [pro], [pro], kn) is False         # skipped: no basic piece for it
    b.replan_after_skip = False
    assert b.still_legal(gs, [pro], [pro], kn) is True


def test_run_builds_skips_knight_after_refused_promote(gs, tmp_path):
    v1, c, v2 = _two_basic(gs)
    r = runner(gs, [], tmp_path)
    r.br = types.SimpleNamespace(lattice=types.SimpleNamespace(vertex_px=lambda t: (500, 400)))
    r._turn_n, r._promote_skip_turn = 3, 3
    tapped, notes = [], []
    r.note = notes.append
    r.tap_build = lambda a: tapped.append(a.kind) or False
    r._run_builds(gs, [Action("promote", v1), Action("knight", v2), Action("wall", c)])
    assert tapped == ["wall"]
    assert any("no longer legal after skipping" in n and "promote" in n for n in notes)


def test_ab_flag_is_a_brain_flag():
    from catanbot.brain import set_flag
    b = Brain()
    assert b.replan_after_skip is True
    set_flag(b, "replan_after_skip", False)


# ---- 5. declined offers -----------------------------------------------------------------------
def _decline_setup(gs, tmp_path):
    from catanbot.hands import HandLedger
    me = gs.me()
    me.hand.clear()
    me.hand.update(lumber=1, brick=1, wool=1, ore=2)
    r = runner(gs, [], tmp_path)
    r.brain._trade_targets = lambda g: ["settlement"]
    led = HandLedger(); led.me = me.name
    r._hands = gs.hands = led
    r._turn_n = 1
    t = r._want_trade(gs)
    assert t is not None
    return r, t


def test_two_declines_block_the_pair_for_the_game(gs, tmp_path):
    r, t = _decline_setup(gs, tmp_path)
    r._remember_decline(t.give, t.get)
    r._remember_decline(t.give, t.get)
    r._turn_n += 7
    r._offers_made = set()
    t2 = r._want_trade(gs)
    assert t2 is None or (t2.give, t2.get) != (t.give, t.get)
    r.cfg["offer_max_declines"] = 0                  # cap off: the old expiry again
    assert not r._was_declined(t.give, t.get)


def test_real_gain_reopens_estimate_does_not(gs, tmp_path):
    r, t = _decline_setup(gs, tmp_path)
    r._remember_decline(t.give, t.get)
    held = r._declined[r._offer_key(t.give, t.get)][1]
    r._wanted_held = lambda get: held + 0.6          # income estimate only
    assert r._was_declined(t.give, t.get)
    del r._wanted_held
    want = next(iter(t.get))
    r._hands.p(gs.opponents()[0].name).add({want: 1})
    r._offers_made = set(); r._turn_n = 2
    t2 = r._want_trade(gs)
    assert t2 is not None and (t2.give, t2.get) == (t.give, t.get)


# ---- 6. hand read after a screen change -------------------------------------------------------
class _Hand:
    def __init__(self, *reads):
        self.reads = list(reads)

    def read(self, img):
        return self.reads.pop(0)


def test_hand_read_again_just_after_a_screen_change(gs, tmp_path, monkeypatch):
    slept = []
    monkeypatch.setattr(R.time, "sleep", slept.append)
    r = runner(gs, [], tmp_path)
    r.io = _IO(np.zeros((888, 1372, 3), np.uint8))
    full = {"lumber": 2, "brick": 1, "wool": 0, "grain": 1, "ore": 3}
    r.hand = _Hand({k: None for k in full}, dict(full))
    r._screen_since = R.time.time()
    assert r._hand_settled(r.io.img) == full and slept == [0.3]


def test_hand_old_screen_reads_once(gs, tmp_path, monkeypatch):
    slept = []
    monkeypatch.setattr(R.time, "sleep", slept.append)
    r = runner(gs, [], tmp_path)
    r.io = _IO(np.zeros((888, 1372, 3), np.uint8))
    none = {"lumber": None, "brick": 1}
    r.hand = _Hand(dict(none), {"lumber": 4, "brick": 1})
    r._screen_since = R.time.time() - 5
    assert r._hand_settled(r.io.img) == none and slept == []
    r.cfg["hand_settle_s"] = 0
    r._screen_since = R.time.time()
    assert r._hand_settled(r.io.img) == {"lumber": 4, "brick": 1} and slept == []
