"""Owner 2026-10-04, live game 13: free knights (Deserter) placed by the brain on a ring a road
split, the Deserter victim gives up its least useful knight, every useful progress card is played
before End turn (one stuck card no longer blocks the rest), a held Warlord replaces grain."""
from pathlib import Path

import cv2
import numpy as np
import pytest

import catanbot.runner as R
from catanbot import config
from catanbot import progress_cards as pc
from catanbot import sim_cards
from catanbot.board import canal_board, hex_corners
from catanbot.brain import Action, Brain
from catanbot.rules import MODES
from catanbot.runner import ReplayIO, Runner
from catanbot.sim import Sim
from catanbot.state import GameState, Knight, PlayerState, new_hand
from catanbot.vision.screens import classify

FIX = Path(__file__).parent / "fixtures"
HAND = R.UI["progress_hand"]


def box(text, x=None, y=None):
    x = (HAND[0] + HAND[2]) / 2 if x is None else x
    y = (HAND[1] + HAND[3]) / 2 if y is None else y
    return {"text": text, "x": x - 30, "y": y - 8, "w": 60, "h": 16, "conf": 1.0}


class FakeIO(ReplayIO):
    def __init__(self, screens):
        super().__init__([])
        self.screens, self.cur = list(screens), []

    def snapshot(self):
        if self.screens:
            self.cur = self.screens.pop(0)
        return np.zeros((888, 1372, 3), np.uint8)

    def ocr(self, img):
        return self.cur if img is not None and img.shape[:2] == (888, 1372) else []


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(R.time, "sleep", lambda s: None)


@pytest.fixture
def gs():
    b = canal_board(seed=3)
    me = PlayerState("marlow.systems", is_me=True, hand=new_hand(), vp_visible=6)
    a = PlayerState("Hildegard", vp_visible=10, hand_count=6)
    c = PlayerState("Nassir", vp_visible=5, hand_count=2)
    g = GameState(board=b, mode=MODES["great_canal"], players=[me, a, c])
    mtn = next(h for h in b.hexes.values() if h.is_land and h.terrain == "mountains" and h.number)
    me.cities.add(hex_corners(*mtn.coord)[0])
    return g


def runner(gs, screens):
    cfg = config.load(); cfg["dry_run"] = False
    r = Runner(FakeIO(screens), cfg)
    r.mode, r.gs = gs.mode, gs
    r.ensure_state = lambda img, texts: gs
    return r


# ---- A: free knight placement -------------------------------------------------------------
def test_basic_knight_and_metropolis_prompts_are_read():
    assert classify(["Select position for Basic knight"])[0] == "place_knight"
    assert classify(["Select position for Metropofis"])[0] == "place_metropolis"
    assert classify(["Select position for Strong knight"])[0] == "promote_knight"
    assert classify(["Choose a player to remove a knight. | hide overlay"])[0] == "choose_player"


def test_ring_split_by_roads_is_found_per_corner():
    """Game 13 16:51: the only legal corner's ring was cut by two roads; ring_spots found none."""
    from catanbot.vision.board_reader import read_board
    from catanbot.vision.rings import vertex_rings
    img = cv2.imread(str(FIX / "free_knight_165155.png"))
    br = read_board(img)
    found = vertex_rings(img, br.lattice, br.board)
    assert [v for _, v, _ in found] == [(-1, -5)]
    x, y = found[0][2]
    assert abs(x - 711) < 6 and abs(y - 289) < 6


def test_free_knight_goes_where_the_brain_says(gs):
    r = runner(gs, [[], [], []])
    corners = sorted(hex_corners(*next(iter(gs.board.hexes))))
    r._vertex_rings = lambda img: [(corners[0], (300, 300)), (corners[1], (400, 400))]
    r.brain.choose_free_knight_spot = lambda g, opts: corners[1]
    r.on_place_knight(np.zeros((888, 1372, 3), np.uint8), [], ["Select position for Basic knight"], {})
    assert any(w.startswith(f"place free knight at {corners[1]}") for *_, w in r.io.clicks)
    assert (400, 400) in [(x, y) for x, y, _ in r.io.clicks]


def test_free_strong_knight_on_empty_corners_is_a_placement(gs):
    r = runner(gs, [[], [], []])
    v = sorted(hex_corners(*next(iter(gs.board.hexes))))[2]
    r._vertex_rings = lambda img: [(v, (350, 350))]
    seen = []
    r.on_place_knight = lambda *a: seen.append("place")
    r.on_promote_knight(np.zeros((888, 1372, 3), np.uint8), [], ["Select position for Strong knight"], {})
    assert seen == ["place"]


def test_free_knight_spot_picks_a_legal_option(gs):
    land = [v for v in gs.board.vertices if gs.board.is_land_vertex(v) and not gs.occupied(v)][:6]
    assert Brain().choose_free_knight_spot(gs, land) in land


# ---- B: Deserter victim --------------------------------------------------------------------
def _canal_and_plain(gs):
    b = gs.board
    canal = next(h for h in b.hexes.values() if b.is_canal(h))
    on = next(v for v in hex_corners(*canal.coord) if b.is_land_vertex(v) and not gs.occupied(v))
    off = next(v for v in sorted(b.vertices) if b.is_land_vertex(v) and not gs.occupied(v)
               and not any(b.is_canal(h) for h in b.vertex_hexes(v))
               and Brain()._prospector_value(gs, gs.me(), v) == 0)
    return on, off


def test_victim_keeps_the_knight_still_digging_the_canal(gs):
    on, off = _canal_and_plain(gs)
    me = gs.me()
    me.knights[on] = Knight(level=1, active=False)
    me.knights[off] = Knight(level=1, active=False)
    br = Brain()
    assert br.knight_keep_value(gs, on) > br.knight_keep_value(gs, off)
    assert br.choose_knight_to_lose(gs) == off


def test_victim_keeps_the_defender_of_its_city(gs):
    on, off = _canal_and_plain(gs)
    me = gs.me()
    me.knights[on] = Knight(level=1, active=False)        # idle on the canal
    me.knights[off] = Knight(level=2, active=True)        # the only active knight: defends the city
    gs.barbarian_pos = gs.barbarian_len - 1
    assert Brain().choose_knight_to_lose(gs) == on


def test_sim_deserter_asks_the_victims_brain(monkeypatch):
    sim = Sim("great_canal", 3, seed=7)
    gs = sim.gs
    for p in gs.players:
        p.knights.clear()
    me, victim = gs.players[0], gs.players[1]
    vs = [v for v in sorted(gs.board.vertices) if gs.board.is_land_vertex(v) and not gs.occupied(v)][:2]
    victim.knights[vs[0]] = Knight(level=1, active=False)
    victim.knights[vs[1]] = Knight(level=1, active=False)
    asked = []

    def pick(self, g, options=None):
        asked.append(g.me().name)
        return vs[1]
    monkeypatch.setattr(Brain, "choose_knight_to_lose", pick)
    monkeypatch.setattr(pc, "choose_deserter", lambda g, m: {"player": victim.name})
    sim.as_me(0)
    assert sim_cards._deserter(sim, 0, gs, me)
    assert asked == [victim.name] and vs[1] not in victim.knights and vs[0] in victim.knights
    assert gs.players[0].is_me and not victim.is_me        # seat restored


# ---- C: Warlord replaces activation grain --------------------------------------------------
def _idle_knight(gs):
    me = gs.me()
    v = next(v for v in sorted(gs.board.vertices) if gs.board.is_land_vertex(v) and not gs.occupied(v))
    me.knights[v] = Knight(level=1, active=False)
    me.hand.update(brick=3, ore=1, wool=1)
    return me


def test_held_warlord_means_no_grain_for_activation(gs):
    me = _idle_knight(gs)
    br = Brain()
    me.progress_hand = ["warlord"]
    assert br.warlord_covers(gs)
    assert "activate" not in br._trade_targets(gs)
    assert br._grain_first_trade(gs) is None
    assert not br._needs_grain(gs)
    assert all(a.kind != "activate" for a in br.candidate_actions(gs))
    assert pc.choose_warlord(gs, me) == {}                # and the card itself wants to go now


def test_without_warlord_activation_still_wants_grain(gs):
    _idle_knight(gs)
    br = Brain()
    assert not br.warlord_covers(gs)
    assert br._needs_grain(gs) and "activate" in br._trade_targets(gs)


def test_sim_plays_warlord_for_idle_knights_before_the_barbarians_near():
    sim = Sim("great_canal", 3, seed=7)
    gs = sim.gs
    p = gs.players[0]
    p.knights.clear()
    vs = [v for v in sorted(gs.board.vertices) if gs.board.is_land_vertex(v) and not gs.occupied(v)][:2]
    for v in vs:
        p.knights[v] = Knight(level=1, active=False)
    p.progress_hand = ["warlord"]
    gs.barbarian_pos = 0
    sim.as_me(0)
    sim.play_held(0)
    assert "warlord" not in p.progress_hand and all(k.active for k in p.knights.values())


# ---- C: play every useful card; once-per-game tries; pending prompt ------------------------
def _progress_runner(gs, monkeypatch, hand, screens):
    monkeypatch.setitem(R.UI, "progress_live", True)
    r = runner(gs, screens)
    gs.me().knights.setdefault((9, 9), Knight(active=False))   # the Warlord is playable (pc.can_play)
    r._read_progress_hand = lambda ocr, img=None: [(k, box(k)) for k in hand]
    drags = []

    def drag(p0, p1, why=""):
        card = why.split()[2]
        drags.append(card)
        if r.__dict__.get("_drag_lands", True):
            hand.remove(card)
    r.drag = drag
    r._park_pointer = lambda why: None
    r.PLAY_HELD = {"alchemist", "inventor"}
    return r, drags


def test_skipped_card_no_longer_blocks_the_rest(gs, monkeypatch):
    """Game 13: Bishop, Alchemist and Inventor in hand, nothing played after the held one."""
    # 2026-10-06: the Alchemist can't go after the roll (pc.can_play), so Mining stands in as the
    # once-only (unconfirmed UI) card here
    hand = ["inventor", "warlord", "mining"]
    r, drags = _progress_runner(gs, monkeypatch, hand, [[]] * 40)
    r.PLAY_HELD = {"inventor", "mining"}
    r._progress_once = {"inventor"}                      # already tried once this game
    order = ["inventor", "warlord", "mining"]
    r.brain.progress_play = lambda g, ph, hand_cards=None: next(
        ((c, {}) for c in order if c in (hand_cards or [])), None)
    r.play_progress("main")
    assert drags == ["warlord", "mining"]
    # the once-only try left the hand: the flow works, the next copy plays normally
    assert "mining" not in r._progress_once and "inventor" in r._progress_once and hand == ["inventor"]


def test_once_only_try_is_spent_only_by_a_drag(gs, monkeypatch):
    """Review 0.3.1: a once-only card skipped as refused (or not on screen) keeps its try."""
    hand = ["mining"]
    r, drags = _progress_runner(gs, monkeypatch, hand, [[]] * 40)
    r.PLAY_HELD = {"mining"}
    r._progress_refused = {"mining"}
    r.brain.progress_play = lambda g, ph, hand_cards=None: next(((c, {}) for c in (hand_cards or [])), None)
    r.play_progress("main")
    assert drags == [] and "mining" not in r.__dict__.get("_progress_once", set())
    r._progress_refused = set()
    r.play_progress("main")
    assert drags == ["mining"]                           # the try was still there


def test_pending_prompt_is_answered_first_and_not_a_refusal(gs, monkeypatch):
    """Game 13 16:56: Warlord dragged while the Bishop's robber prompt was up -> marked refused."""
    hand = ["warlord"]
    robber = [{"text": "Select position for Robber", "x": 500, "y": 100, "w": 200, "h": 20, "conf": 1.0}]
    r, drags = _progress_runner(gs, monkeypatch, hand, [[], []] + [robber] * 20)
    r._drag_lands = False
    r.brain.progress_play = lambda g, ph, hand_cards=None: ("warlord", {}) if hand_cards else None
    r.play_progress("main")
    assert drags == ["warlord"]
    assert "warlord" not in r.__dict__.get("_progress_refused", set())


def test_refused_card_is_not_counted_as_a_resource(gs, monkeypatch):
    hand = ["warlord"]
    r, _ = _progress_runner(gs, monkeypatch, hand, [[]] * 10)
    r._progress_refused = {"warlord"}
    r.brain.progress_play = lambda g, ph, hand_cards=None: None
    r.play_progress("main")
    assert gs.me().progress_hand == []


def _two_dug_canal_corners(gs):
    b = gs.board
    canal = [h for h in b.hexes.values() if b.is_canal(h)]
    out = []
    for h in canal:
        for v in hex_corners(*h.coord):
            if b.is_land_vertex(v) and not gs.occupied(v) and v not in out \
                    and Brain()._prospector_value(gs, gs.me(), v) == 0 \
                    and all(not b.is_canal(x) or x.coord == h.coord for x in b.vertex_hexes(v)):
                out.append(v)
                gs.canal_dug.add(h.coord)
                break
        if len(out) == 2:
            return out
    pytest.skip("no two single-canal corners")


def test_victim_gives_up_the_lowest_level_on_a_dug_canal(gs):
    # owner 2026-10-06: "give up the lowest level knight that is already on a dug canal"
    lo, hi = _two_dug_canal_corners(gs)
    me = gs.me()
    me.knights[lo] = Knight(level=1, active=True)
    me.knights[hi] = Knight(level=2, active=False)
    assert Brain().choose_knight_to_lose(gs) == lo


def test_victim_gives_up_the_inactive_of_two_spent_knights(gs):
    a, b = _two_dug_canal_corners(gs)
    me = gs.me()
    me.knights[a] = Knight(level=1, active=True)
    me.knights[b] = Knight(level=1, active=False)
    assert Brain().choose_knight_to_lose(gs) == b


def test_victim_gives_up_a_spent_active_knight_to_keep_one_still_digging(gs):
    # owner: give up the active one when that is the better future (the other still digs)
    on, off = _canal_and_plain(gs)
    me = gs.me()
    me.knights[on] = Knight(level=1, active=False)      # undug canal: a point later
    me.knights[off] = Knight(level=1, active=True)      # nothing left to do
    gs.barbarian_pos = 0
    assert Brain().choose_knight_to_lose(gs) == off


def test_textless_rings_on_our_knights_only_read_as_deserter(gs):
    # solo 2026-10-06 03:51: no prompt text, rings on our two knights -> the generic fallback took
    # the first ring; now it is the deserter choice
    a, b = _two_dug_canal_corners(gs)
    gs.me().knights[a] = Knight(level=1, active=True)
    gs.me().knights[b] = Knight(level=2, active=False)
    r = Runner.__new__(Runner)
    r.mode = "great_canal"
    r.ensure_state = lambda img, texts: gs
    r._vertex_rings = lambda img: [(a, (1, 1)), (b, (2, 2))]
    assert r._rings_only_on_our_knights(None, []) is True
    r._vertex_rings = lambda img: [(a, (1, 1)), ((99, 99), (3, 3))]
    assert r._rings_only_on_our_knights(None, []) is False


def test_merchant_fleet_failed_pick_sets_no_two_to_one(gs, monkeypatch):
    """Review 0.3.1: the 2:1 was set after the card even when the pick dialog stayed up."""
    hand = ["merchant_fleet"]
    r, drags = _progress_runner(gs, monkeypatch, hand, [[]] * 40)
    r.PLAY_HELD = set()
    r._monopoly_pick = lambda choice, ocr=None: False
    r.brain.progress_play = lambda g, ph, hand_cards=None: (
        ("merchant_fleet", {"resource": "ore"}) if hand_cards else None)
    r.brain.fleet_resource = None
    r.play_progress("main")
    assert drags == ["merchant_fleet"]
    assert r.brain.fleet_resource is None
