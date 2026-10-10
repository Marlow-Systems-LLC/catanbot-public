"""Round 4 (stage games 1548/1632, 0.3.6): game A sat 10 min on a textless Diplomat prompt and the
stall rescue quit a match we led 14/14/13/7. Road rings reach the fallback and the Diplomat looks
again, a textless prompt of ours falls under the 90 s cap, the rescue tries in-game exits first
and holds a led match one more window, the Inventor's chip snap is tight and logged."""
import dataclasses
import math

import numpy as np

import catanbot.runner as R
from board_fixture import canvas, frame
from catanbot import config
from catanbot.runner import ReplayIO, Runner
from catanbot.state import PlayerState
from catanbot.vision import board_reader as brd
from catanbot.vision.layout import Layout
from tests.test_progress_wiring import box, gs, runner  # noqa: F401 (fixture)
from tests.test_round2_0834 import live_layout  # noqa: F401 (fixture)

BOARD = "diplomat_162049_355_220.png"           # 162049_094_progress_after_diplomat, the board only
NATIVE = "diplomat_162049_1920x1080_900_280.png"  # its native capture, the ringed roads only
WANT = ((-2, -4), (-1, -5))                      # the brain's edge at 16:20:43, ringed at (639,294)
RING = (639, 294)


def _near(xy, to=RING, d=6):
    return math.hypot(xy[0] - to[0], xy[1] - to[1]) < d


_BR = {}


def _board():
    if "br" not in _BR:
        _BR["br"] = brd.read_board(frame(BOARD), R.BOARD_ROI, canal=True)
    return _BR["br"]


class _Shifted(brd.Lattice):
    """The live lattice off by d px (hypothesis (a) of the r4 findings)."""
    d = 10

    def to_px(self, ux, uy):
        x, y = super().to_px(ux, uy)
        return x + self.d, y + self.d


def _dip_runner(monkeypatch, img, native=None, cfg=None):
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    r = Runner.__new__(Runner)
    r.cfg = dict(cfg or {})
    r.br = _board()
    r.gs = None
    r.clicks, r.notes = [], []
    r.note = r.notes.append
    r.click = lambda xy, why="", ocr=None: r.clicks.append((tuple(map(int, xy)), why))

    class IO:
        def snapshot(self):
            if native is not None:
                self.native, self.lay, self._frame = native, Layout(1920, 1080), img
            return img
    r.io = IO()
    r._confirm_if_check = lambda why: True
    r._red_x = lambda im: False
    return r
RESCUE = "rescue_162521_610_250.png"             # 162521_102_rescue_240: the ringed roads, End turn grey
RESCUE_NATIVE = "rescue_162521_1920x1080_900_280.png"


# ---- 1. the stall rescue tries in-game exits first, a led match gets one more window ----------
class _StallIO:
    """Solo, live, the same textless frame every poll (game 1548 16:21-16:31)."""
    mac = True

    def __init__(self, img=None, native=None):
        self.img = img if img is not None else np.zeros((888, 1372, 3), np.uint8)
        self.nat = native

    def snapshot(self):
        if self.nat is not None:
            self.native, self.lay, self._frame = self.nat, Layout(1920, 1080), self.img
        return self.img

    def ocr(self, img):
        return [{"text": "nothing here", "x": 600, "y": 400, "w": 80, "h": 20}]

    def click(self, x, y, why=""):
        pass

    def move(self, x, y, why=""):
        pass

    def paused(self):
        return False


class _G:
    def __init__(self, ours, rival):
        self.players = [PlayerState("marlow.systems", is_me=True, vp_visible=ours),
                        PlayerState("Ann", vp_visible=rival), PlayerState("Bo", vp_visible=7)]

    def me(self):
        return self.players[0]


def _stall(monkeypatch, img=None, native=None, cfg=None, vp=None):
    c = config.load(); c["dry_run"] = True; c["solo"] = True
    c.update(cfg or {})
    r = Runner(ReplayIO([]), c)
    r.io = _StallIO(img, native)
    r.br = _board() if img is not None else None
    r.gs = _G(*vp) if vp else None
    now = [10_000.0]
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    monkeypatch.setattr(R.time, "time", lambda: now[0])
    r._probe = lambda *a, **k: None
    r.restarts, r.clicked, r.log = [], [], []
    r.note = r.log.append
    r.restart_app = lambda reason: r.restarts.append(reason)
    r.click = lambda xy, why="", ocr=None: (r.clicked.append((tuple(map(int, xy)), why)),
                                            setattr(r, "_last_click_at", now[0]))
    r._confirm_if_check = lambda why: True
    return r, now


def test_rescue_taps_the_road_ring_before_the_table(monkeypatch):
    r, now = _stall(monkeypatch, frame(RESCUE), canvas(RESCUE_NATIVE))
    r.step()
    now[0] = 10_250
    r.step()
    # no brain plan without a game state: the best-scoring ring is tapped (the 16:20 ring among them)
    ringed = [xy for _, xy in r._edge_rings(r.io.snapshot())]
    assert any(_near(xy, d=3) for xy in ringed)
    assert r.clicked and any(_near(r.clicked[0][0], xy) for xy in ringed)
    assert "ringed road" in r.clicked[0][1]
    assert all(xy != R.Runner.PARK_SPOT for xy, _ in r.clicked)
    assert r.probe.card["rescue_240_road"]["tries"] == 1


def test_rescue_without_rings_clicks_the_table_then_end_turn_once(monkeypatch):
    r, now = _stall(monkeypatch, frame(RESCUE), canvas(RESCUE_NATIVE))
    r._edge_rings = lambda img, *a, **k: []
    monkeypatch.setattr(R, "turn_phase", lambda img: "idle")
    r.step()
    now[0] = 10_250
    r.step()
    assert r.clicked == [(R.Runner.PARK_SPOT, "rescue: click empty table")]
    now[0] = 10_430
    r.step()
    assert [w for _, w in r.clicked].count("rescue: End turn grey, pressing once "
                                           "(UNCONFIRMED whether the game takes it)") == 1
    assert r.clicked[-1][0] == tuple(R.UI["dice_or_end_turn"])


def test_rescue_old_ladder_when_switched_off(monkeypatch):
    r, now = _stall(monkeypatch, frame(RESCUE), canvas(RESCUE_NATIVE), cfg={"rescue_in_game_exits": False})
    r.step()
    now[0] = 10_250
    r.step()
    assert r.clicked == [(R.Runner.PARK_SPOT, "rescue: click empty table")]   # blob rings miss road rings


def test_stall_while_level_gets_one_more_window(monkeypatch):
    r, now = _stall(monkeypatch, vp=(14, 14))
    r.step()
    now[0] = 10_610
    assert r.step() != "restart" and not r.restarts
    assert any("no progress for 10 min but we lead 14 vs 14: one more window" in m for m in r.log)
    now[0] = 10_850
    assert r.step() != "restart"
    now[0] = 10_910
    assert r.step() == "restart" and r.restarts


def test_stall_while_behind_or_switched_off_quits_at_ten_minutes(monkeypatch):
    r, now = _stall(monkeypatch, vp=(10, 14))
    r.step()
    now[0] = 10_610
    assert r.step() == "restart"
    r, now = _stall(monkeypatch, vp=(14, 14), cfg={"stall_lead_extra_s": 0})
    r.step()
    now[0] = 10_610
    assert r.step() == "restart"


# ---- 3. road rings in the fallback; the Diplomat re-reads, logs its looks, blocks more cards ----
def test_diplomat_finds_the_stall_ring_on_the_frame(monkeypatch):
    img = frame(BOARD)
    r = _dip_runner(monkeypatch, img)
    assert r._diplomat_pick(WANT, img=img)
    assert _near(r.clicks[0][0])
    assert any(n.startswith("edge rings: ") and "src frame" in n for n in r.notes)


def test_diplomat_finds_the_stall_ring_on_the_native(monkeypatch):
    img = frame(BOARD)
    r = _dip_runner(monkeypatch, img, native=canvas(NATIVE))
    assert r._diplomat_pick(WANT, img=r.io.snapshot())
    assert _near(r.clicks[0][0])
    assert any("src native" in n for n in r.notes)


def _shift(r):
    lat = brd.Lattice(r.br.lattice.A, r.br.lattice.W)
    lat.__class__ = _Shifted
    r.br = dataclasses.replace(r.br, lattice=lat)
    return r


def test_shifted_lattice_reread_still_finds_the_ring(monkeypatch):
    img = frame(BOARD)
    r = _shift(_dip_runner(monkeypatch, img))
    assert r._edge_rings(img) == []                    # the live miss: an offset lattice sees nothing
    assert r._diplomat_pick(WANT, img=img)
    assert _near(r.clicks[0][0])
    assert any("board re-read" in n for n in r.notes)
    assert not isinstance(r.br.lattice, _Shifted)      # same hexes: the fresh read is kept


def test_shifted_lattice_without_reread_gives_up_and_blocks_cards(monkeypatch):
    img = frame(BOARD)
    r = _shift(_dip_runner(monkeypatch, img, cfg={"diplomat_reread": False}))
    assert not r._diplomat_pick(WANT, img=img)
    assert r.clicks == []
    assert any("no road ring on screen" in n for n in r.notes)
    assert r._progress_blocked[0] == "diplomat"


def test_fallback_rings_include_the_road_rings(monkeypatch):
    img = frame(BOARD)
    r = _dip_runner(monkeypatch, img)
    r._rings = lambda im: []
    r._vertex_rings = lambda im: []
    rings = r._fallback_rings(img)
    assert len(rings) >= 6 and any(_near(xy, d=3) for xy in rings)
    r.cfg["edge_rings_in_fallback"] = False
    assert r._fallback_rings(img) == []


def test_blocked_progress_plays_no_more_cards(gs, tmp_path, monkeypatch):   # noqa: F811
    r = runner(gs, [[box("Inventor")]] * 4, tmp_path)
    monkeypatch.setattr(r.brain, "progress_play", lambda g, ph, hand_cards=None: ("inventor", {}))
    r._progress_blocked = ("diplomat", R.time.time())
    assert r.play_progress("main") == []
    assert r.io.clicks == []
    assert any("diplomat prompt still open -- no more cards this turn" in m for m in r.log)
    r._stall_progress("my_turn")                       # the prompt is gone: the block clears
    assert "_progress_blocked" not in r.__dict__


# ---- 2. a textless prompt of ours falls under the 90 s cap ----------------------------------
class _LiveIO:
    """Not a ReplayIO, no text: the step loop reads 'unknown' and runs its live-only branches."""

    def __init__(self, img):
        self.img, self.clicks = img, []

    def snapshot(self):
        return self.img

    def ocr(self, img):
        return []

    def click(self, x, y, why=""):
        self.clicks.append((int(x), int(y), why))

    def paused(self):
        return False


def _unknown(gs, tmp_path, monkeypatch, cfg=None, edge=False):   # noqa: F811
    r = runner(gs, [], tmp_path, dry=False)
    r.cfg.update(cfg or {})
    r.io = _LiveIO(frame(BOARD))
    r.br = _board()
    r._in_play = True
    t = [1000.0]
    monkeypatch.setattr(R.time, "time", lambda: t[0])
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    monkeypatch.setattr(R, "turn_phase", lambda img: "idle")
    r._last_click_at = t[0] - 50
    r._progress_click_at = t[0] - 60               # the Diplomat drag a minute ago
    r._rings = lambda img: []
    r._vertex_rings = lambda img: []
    if not edge:
        r._edge_rings = lambda img, *a, **k: []
    r._recenter_board = lambda img: False
    r._green_check = lambda img: False
    r._marker_seat = lambda img: 0
    r.setup_watch = lambda *a, **k: None
    r.restarts = []
    r.restart_app = lambda *a, **k: r.restarts.append(1)
    return r, t


def test_textless_prompt_of_ours_starts_the_clock_and_exits(gs, tmp_path, monkeypatch, live_layout):  # noqa: F811
    r, t = _unknown(gs, tmp_path, monkeypatch)
    assert r.step() == "unknown"
    assert r._prompt_since["unknown"] == 1000.0
    assert r.io.clicks == []
    t[0] += 91
    r.step()
    assert r.probe.card["prompt_cap_unknown"]["tries"] == 1
    assert r.restarts == []
    # no rings, no X, no button, idle under our marker: End turn once, marked unconfirmed
    assert [why for *_, why in r.io.clicks] == ["prompt cap: End turn grey, pressing once "
                                               "(UNCONFIRMED whether the game takes it)"]
    t[0] += 31
    r.step()
    assert len(r.io.clicks) == 1                   # once per prompt
    assert any("no way out" in m for m in r.log)


def test_unknown_cap_taps_the_road_ring_first(gs, tmp_path, monkeypatch, live_layout):  # noqa: F811
    r, t = _unknown(gs, tmp_path, monkeypatch, edge=True)
    r._diplomat_played_t = 0.0                     # the Diplomat gate itself is spent
    r._last_click_at = t[0]                        # and the ringed gate waits its 45 s
    r._ring_wait_s = lambda texts: 1e9
    r.step()
    t[0] += 91
    r.step()
    assert r.io.clicks and _near(r.io.clicks[0][:2])
    assert any("prompt cap unknown: road rings shown -> pick one" in m for m in r.log)


def test_unknown_clock_clears_and_switch_off(gs, tmp_path, monkeypatch, live_layout):  # noqa: F811
    r, t = _unknown(gs, tmp_path, monkeypatch)
    r.step()
    assert "unknown" in r._prompt_since
    r._stall_progress("my_turn")
    assert "unknown" not in r.__dict__.get("_prompt_since", {})
    r2, t2 = _unknown(gs, tmp_path, monkeypatch, cfg={"prompt_cap_unknown": False})
    r2.step()
    t2[0] += 91
    r2.step()
    assert "unknown" not in r2.__dict__.get("_prompt_since", {})
    assert r2.io.clicks == []


# ---- 4. Inventor: tight chip snap, the planned pair logged against the rings -----------------
class _Lat:
    """A flat hex lattice of the live board's size: hex (q, r) centre at (650 + 58 (q + r/2), 420 + 50 r)."""
    size = 26.6

    def hex_px(self, q, r):
        return 650 + 58 * (q + r / 2), 420 + 50 * r

    def nearest_hex(self, x, y):
        r = round((y - 420) / 50)
        return round((x - 650) / 58 - r / 2), r


def _inv(cfg=None):
    import types
    r = Runner.__new__(Runner)
    r.cfg = dict(cfg or {})
    r.br = types.SimpleNamespace(lattice=_Lat())
    r.notes = []
    r.note = r.notes.append
    s = _Lat.size
    r._rings = lambda img: [(650, 420), (708 + 0.3 * s, 420), (766 + 0.5 * s, 420)]
    return r


def test_token_rings_snap_tightly_and_log_the_drop():
    img = np.zeros((888, 1372, 3), np.uint8)
    r = _inv()
    assert sorted(r._token_rings(img)) == [(0, 0), (1, 0)]
    assert any("dropped (snap 0.35): (2, 0) at 0.50" in n for n in r.notes)
    assert sorted(_inv({"token_ring_snap": 0.6})._token_rings(img)) == [(0, 0), (1, 0), (2, 0)]


def test_inventor_logs_the_plan_against_the_rings(gs, tmp_path, monkeypatch):   # noqa: F811
    from catanbot import progress_cards as pc
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    r = runner(gs, [], tmp_path, dry=True)
    r._token_rings = lambda img: {(1, -2): (600, 300), (3, -1): (700, 300)}
    monkeypatch.setattr(pc, "choose_inventor", lambda *a, **k: {"swap": ((1, -2), (3, -1))})
    probes = []
    r._probe = lambda kind, *a, **k: probes.append(kind)
    r._check_xy = lambda img: None
    assert r._inventor_swap("inventor", planned=[(-1, -2), (3, -1)], img=np.zeros((888, 1372, 3), np.uint8))
    assert any("inventor: planned (-1, -2)<->(3, -1); ringed [(1, -2), (3, -1)]; planned ringed: a=no b=yes" in m
               for m in r.log)
    assert any("inventor: re-planned (1, -2)<->(3, -1) (planned hex (-1, -2) had no ring)" in m for m in r.log)
    assert probes == ["inventor_pairs"]

# ---- 5. marlow says what version it runs ----------------------------------------------------
def test_api_reports_the_version(tmp_path, monkeypatch):
    import catanbot
    from catanbot import api, game_log
    monkeypatch.setattr(catanbot, "COMMIT", "abc1234")
    monkeypatch.setattr(game_log, "DIR", tmp_path)
    monkeypatch.setattr(api, "DATA", tmp_path)
    assert api.route("/version", {}) == (200, {"version": catanbot.__version__, "commit": "abc1234"})
    for path in ("/health", "/game", "/live"):
        code, body = api.route(path, {})
        assert code == 200 and body["version"] == catanbot.__version__ and body["commit"] == "abc1234", path
    assert api.route("/health", {})[1]["ok"] is True


def test_commit_is_a_short_sha_or_empty():
    import re

    import catanbot
    assert catanbot.COMMIT == "" or re.fullmatch(r"[0-9a-f]{4,40}", catanbot.COMMIT)


def test_chat_answers_the_version_and_keeps_the_scenario_line(monkeypatch):
    import catanbot
    from catanbot import chat
    from catanbot.rules import MODES

    class G:
        mode = MODES["great_canal"]
    monkeypatch.setattr(catanbot, "COMMIT", "abc1234")
    want = f"I'm running marlow {catanbot.__version__} (abc1234)."
    for msg in ("marlow what version are you?", "marlow which version do you run", "marlow version?"):
        assert chat.Chat(clock=lambda: 1e9).respond("Lin", msg, G()) == want, msg
    for msg in ("marlow which version of catan is this?", "marlow what game version is this"):
        r = chat.Chat(clock=lambda: 1e9).respond("Lin", msg, G())
        assert "The Great Canal" in r and "running marlow" not in r, msg
    monkeypatch.setattr(catanbot, "COMMIT", "")
    assert chat.version_line() == f"I'm running marlow {catanbot.__version__}."
