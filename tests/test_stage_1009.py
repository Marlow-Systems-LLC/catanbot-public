"""Stage solo game 2026-10-09 00:31-01:07: the open issues it showed."""
import math
from pathlib import Path

import cv2
import pytest

from catanbot import runner as R
from catanbot.board import canal_board, ekey, hex_corners, snap_to_canal_template
from catanbot.brain import Brain
from catanbot.rules import MODES
from catanbot.runner import Runner
from catanbot.state import GameState, PlayerState, new_hand
from catanbot.vision.screens import active_seat, classify, dialog_close_x

FIX = Path(__file__).parent / "fixtures"
ME = "marlow.systems"


def _img(name):
    img = cv2.imread(str(FIX / name))
    assert img is not None, name
    return img


# ---- 1: rivals' turns read as other_turn (the gold marker over the active portrait)
@pytest.mark.parametrize("name, seat", [
    ("other_turn_lin_003643.png", 1),
    ("other_turn_jean_003720.png", 3),
    ("remove_city_prompt_224638.png", 3),
])
def test_active_seat_on_rival_turns(name, seat):
    assert active_seat(_img(name)) == seat


@pytest.mark.parametrize("name", ["diplomat_rings_222130.png", "commercial_harbor_player_004049.png"])
def test_active_seat_on_our_turn(name):
    assert active_seat(_img(name)) in (0, None)


def _marker_runner(rings=(), check=False):
    r = Runner.__new__(Runner)
    r._in_play = True
    r._slot_names = {1: "Lin", 3: "Jean"}
    r._rings = lambda img: list(rings)
    r._green_check = lambda img: check
    return r


def test_other_turn_by_marker():
    r, info = _marker_runner(), {}
    assert r._other_turn_by_marker(object(), info, 3) == "other_turn"
    assert info == {"player": "Jean", "seat": 3}
    assert r._other_turn_by_marker(object(), {}, 0) == "unknown"
    assert r._other_turn_by_marker(object(), {}, None) == "unknown"


def test_rings_or_check_keep_unknown():
    # a ringed prompt / green check of ours behind the marker stays with the unknown fallbacks
    assert _marker_runner(rings=[(500, 400)])._other_turn_by_marker(object(), {}, 1) == "unknown"
    assert _marker_runner(check=True)._other_turn_by_marker(object(), {}, 1) == "unknown"


def test_marker_seat_only_in_play():
    r = _marker_runner()
    r._in_play = False
    assert r._marker_seat(_img("other_turn_jean_003720.png")) is None
    r._in_play = True
    assert r._marker_seat(_img("other_turn_jean_003720.png")) == 3


# ---- 2: our Commercial Harbor dialog
HARBOR_TEXTS = ["Commercial Harbor", "Choose a player with whom you want to trade.", "Lin", "Marianne"]


def test_harbor_player_classified():
    assert classify(HARBOR_TEXTS, ME)[0] == "harbor_player"
    # Master Merchant's quoted 'trade' stays a plain player pick
    assert classify(['Choose a player with whom you want to "trade".'], ME)[0] != "harbor_player"


def test_harbor_fixture_classified():
    pytest.importorskip("Vision")
    from catanbot import mac
    texts = [o["text"] for o in mac.ocr(_img("commercial_harbor_player_004049.png"))]
    assert classify(texts, ME)[0] == "harbor_player"


def test_harbor_dialog_close_x():
    xy = dialog_close_x(_img("commercial_harbor_player_004049.png"), {"x": 556, "y": 264, "w": 196, "h": 24})
    assert xy is not None and math.hypot(xy[0] - 819, xy[1] - 275) < 6


def _b(text, x, y, w=60, h=22):
    return {"text": text, "x": x, "y": y, "w": w, "h": h}


HARBOR_OCR = [_b("Commercial Harbor", 556, 264, 196, 24),
              _b("Choose a player with whom you want to trade.", 470, 330, 380),
              _b("Lin", 568, 390, 24), _b("Marianne", 650, 386, 72)]


class _Probe:
    def __init__(self):
        self.kinds = []

    def tried(self, k):
        self.kinds.append(k)


def _harbor_runner(monkeypatch, after_texts):
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    b = canal_board(seed=3)
    me = PlayerState(ME, is_me=True, hand=new_hand())
    me.hand.update({"brick": 2, "grain": 1})
    gs = GameState(board=b, mode=MODES["great_canal"],
                   players=[me, PlayerState("Lin"), PlayerState("Marianne"), PlayerState("Jean")])
    r = Runner.__new__(Runner)
    r.cfg = {"bot_name": ME}
    r.gs, r.brain = gs, Brain()
    r.game_colors = [(ME, "magenta"), ("Lin", "red"), ("Marianne", "orange"), ("Jean", "green")]
    r.probe = _Probe()
    r.probes = []
    r._probe = lambda kind, *a, **k: r.probes.append(kind)
    r.clicks, r.notes = [], []
    r.note = r.notes.append
    r.click = lambda xy, why="", ocr=None: r.clicks.append((tuple(map(int, xy)), why))

    class IO:
        def snapshot(self):
            return object()

        def ocr(self, img):
            return [_b(t, 500, 400) for t in after_texts]
    r.io = IO()
    return r, gs


def test_harbor_picks_a_player_then_gives_least_valued(monkeypatch):
    r, gs = _harbor_runner(monkeypatch, ["Give one resource"])
    r._harbor_give = ["brick"]
    r.on_harbor_player(None, HARBOR_OCR, HARBOR_TEXTS, {})
    (pxy, pwhy), (gxy, gwhy), (okxy, okwhy) = r.clicks
    assert "choose" in pwhy and pxy[1] > 400            # the check under the name
    assert "give brick" in gwhy and gxy[1] == R.UI["trade_give_row"]["y"]
    assert okxy == tuple(R.UI["trade_ok"])
    assert {"harbor_after_pick", "harbor_after_give"} <= set(r.probes)


def test_harbor_never_offers_grain_while_holding_else(monkeypatch):
    r, gs = _harbor_runner(monkeypatch, ["Give one resource"])
    r._harbor_give = []
    assert r._harbor_resource(gs) == "brick"
    gs.me().hand["brick"] = 0
    # review 0.3.1 (owner: grain is never traded to a player): only grain left -> give nothing,
    # the dialog closes with its X after all players / the rounds / 120 s
    assert r._harbor_resource(gs) is None


def test_harbor_closes_after_all_players_tried(monkeypatch):
    r, gs = _harbor_runner(monkeypatch, HARBOR_TEXTS)   # the dialog stays after each pick
    for _ in range(2):
        r.on_harbor_player(None, HARBOR_OCR, HARBOR_TEXTS, {})
    assert len(r.clicks) == 2 and all("choose" in w for _, w in r.clicks)
    r.on_harbor_player(None, HARBOR_OCR, HARBOR_TEXTS, {})
    assert "close the dialog" in r.clicks[-1][1]
    assert "harbor_close" in r.probes


def test_harbor_closes_after_max_rounds(monkeypatch):
    r, gs = _harbor_runner(monkeypatch, HARBOR_TEXTS)
    r._harbor = {"t0": R.time.time(), "tried": set(), "rounds": R.Runner.HARBOR_MAX_ROUNDS}
    r.on_harbor_player(None, HARBOR_OCR, HARBOR_TEXTS, {})
    assert "close the dialog" in r.clicks[-1][1]


def test_harbor_player_is_a_stall_screen():
    assert "harbor_player" in R.STALL_SCREENS


# ---- 3: the Great Canal 3:1 harbours of sea hexes (3,0) and (3,-3)
def test_template_harbour_edges():
    b = snap_to_canal_template(canal_board(seed=1)) or None
    import json
    d = json.load(open(Path(R.ROOT) / "data/boards/canal_template.json"))
    edges = {ekey(tuple(e[0]), tuple(e[1])): k for e, k in d["harbors"]}
    assert edges[ekey((5, -1), (6, -2))] == "3:1"
    assert edges[ekey((2, -8), (3, -7))] == "3:1"
    assert ekey((5, -1), (5, 1)) not in edges and ekey((3, -7), (4, -8)) not in edges


@pytest.mark.parametrize("name", ["other_turn_lin_003643.png", "canal_board_025119.png", "diplomat_rings_222130.png"])
def test_harbour_icon_points_at_template_edge(name):
    """The 3:1 badge in sea hex (3,0) sits off-centre toward its dock; that edge is the template's."""
    import numpy as np
    from catanbot.vision import board_reader as brd
    img = _img(name)
    br = brd.read_board(img, R.BOARD_ROI, canal=True)
    b = snap_to_canal_template(br.board)
    assert b is not None and b.template_offset == (0, 0)
    lat = br.lattice
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    n, _, st, cen = cv2.connectedComponentsWithStats(cv2.inRange(hsv, (0, 0, 190), (180, 70, 255)))
    cx, cy = lat.hex_px(3, 0)
    icons = [cen[i] for i in range(1, n) if 60 < st[i][4] < 600 and math.hypot(cen[i][0] - cx, cen[i][1] - cy) < 22]
    assert icons
    bx, by = icons[0]
    cs = hex_corners(3, 0)

    def cos(e):
        mx, my = np.mean([lat.vertex_px(v) for v in e], axis=0)
        return ((bx - cx) * (mx - cx) + (by - cy) * (my - cy)) / (math.hypot(mx - cx, my - cy) * math.hypot(bx - cx, by - cy))
    best = max((ekey(cs[i], cs[(i + 1) % 6]) for i in range(6)), key=cos)
    assert best == ekey((5, -1), (6, -2))
    assert b.harbor_at((6, -2)) == "3:1" and b.harbor_at((5, 1)) is None


# ---- 4: Diplomat moving our own road -- the new spot is a shown ring
def _dip_runner(monkeypatch, rings, want_xy=(700, 300)):
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    r = Runner.__new__(Runner)

    class Lat:
        def edge_px(self, e):
            return want_xy

    class Br:
        lattice = Lat()
    r.br = Br()
    r.clicks, r.notes = [], []
    r.note = r.notes.append
    r.click = lambda xy, why="", ocr=None: r.clicks.append((tuple(map(int, xy)), why))

    class IO:
        def snapshot(self):
            return object()
    r.io = IO()
    r._edge_rings = lambda img: list(rings)
    r._confirm_if_check = lambda why: True
    return r


def test_diplomat_move_taps_the_wanted_ring(monkeypatch):
    want = ((1, 1), (2, 0))
    r = _dip_runner(monkeypatch, [(((0, 0), (1, 1)), (650, 300)), (want, (705, 304))])
    assert r._diplomat_place(want)
    assert r.clicks[0][0] == (705, 304)


def test_diplomat_move_unringed_want_taps_nearest_ring(monkeypatch):
    r = _dip_runner(monkeypatch, [(((0, 0), (1, 1)), (900, 500)), (((3, 3), (4, 2)), (720, 310))])
    assert r._diplomat_place(((1, 1), (2, 0)))
    assert r.clicks[0][0] == (720, 310)


def test_diplomat_move_no_ring_falls_back_to_lattice(monkeypatch):
    r = _dip_runner(monkeypatch, [])
    r._diplomat_place(((1, 1), (2, 0)))
    assert r.clicks == [((700, 300), r.clicks[0][1])] and "no ring" in r.clicks[0][1]


# ---- 5: a robber prompt with no ring on a rival's turn is not ours to tap
def test_robber_no_ring_on_rival_turn_does_not_tap(monkeypatch):
    b = canal_board(seed=3)
    me = PlayerState(ME, is_me=True, hand=new_hand())
    gs = GameState(board=b, mode=MODES["great_canal"], players=[me, PlayerState("Jean")])
    r = Runner.__new__(Runner)
    r.gs, r.brain = gs, Brain()
    r.ensure_state = lambda img, texts: gs
    r._rings = lambda img: []
    r.br = type("Br", (), {"lattice": None})()
    r._in_play = True
    r._slot_names = {3: "Jean"}
    r.clicks, r.notes = [], []
    r.note = r.notes.append
    r.click = lambda xy, why="", ocr=None: r.clicks.append(why)
    img = _img("other_turn_jean_003720.png")
    r.on_place_robber(img, [], ["Select position for Robber"], {})
    assert r.clicks == [] and "Jean" in r.notes[-1]


# ---- 6: the city-loss discard kept the ore for the rebuild (unchanged, pinned)
def test_city_loss_discard_keeps_ore():
    b = canal_board(seed=3)
    me = PlayerState(ME, is_me=True, hand=new_hand())
    me.hand.update({"brick": 1, "ore": 3, "paper": 3, "cloth": 2})
    land = [v for v in sorted(b.vertices) if any(h.is_land and h.number for h in b.vertex_hexes(v))]
    me.settlements |= {land[0], land[len(land) // 2]}
    gs = GameState(board=b, mode=MODES["great_canal"], players=[me, PlayerState("Jean")])
    out = Brain().choose_discard(gs, 4)
    assert sum(out.values()) == 4 and out.get("ore", 0) == 0
