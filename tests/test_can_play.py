"""Owner 2026-10-06: the bot knows when a progress card can't be played and never tries it.
One failing and one passing precondition per card (pc.can_play), plus the gates in choose_play,
brain.progress_play and runner.play_progress, and live wall recording."""
import numpy as np
import pytest

import catanbot.runner as R
from catanbot import progress_cards as pc
from catanbot import config
from catanbot.board import canal_board, hex_corners
from catanbot.brain import Brain
from catanbot.rules import MODES
from catanbot.runner import Runner
from catanbot.state import GameState, Knight, PlayerState, new_hand

from test_progress_wiring import FakeIO, box


def _land(b, terrain=None):
    return [h for h in b.hexes.values() if h.is_land and h.terrain != "canal" and h.number
            and (terrain is None or h.terrain == terrain)]


@pytest.fixture
def gs():
    b = canal_board(seed=3)
    me = PlayerState("marlow", is_me=True, hand=new_hand(), vp_visible=6)
    a = PlayerState("Hildegard", vp_visible=10, hand_count=6)
    c = PlayerState("Nassir", vp_visible=5, hand_count=2)
    g = GameState(board=b, mode=MODES["great_canal"], players=[me, a, c])
    mtn, fld = _land(b, "mountains")[0], _land(b, "fields")[0]
    me.cities.add(hex_corners(*mtn.coord)[0])
    me.settlements.add(hex_corners(*fld.coord)[3])
    other = next(h for h in _land(b) if h.coord not in (mtn.coord, fld.coord)
                 and not set(hex_corners(*h.coord)) & (me.cities | me.settlements))
    a.cities.add(hex_corners(*other.coord)[2])
    return g


def ok(gs, card, phase="main"):
    return pc.can_play(gs, gs.me(), card, phase)[0]


def test_every_card_has_a_limit_line():
    assert set(pc.CANT_PLAY) == set(pc.CARDS)


def test_vp_cards_never_played(gs):
    for c in pc.VP_CARDS:
        assert not ok(gs, c)


def test_phase(gs):
    assert ok(gs, "alchemist", "pre_roll") and not ok(gs, "alchemist", "main")
    assert not ok(gs, "mining", "pre_roll") and ok(gs, "mining", "main")
    assert not ok(gs, "mining", "opp_turn")


# ---- science ---------------------------------------------------------------------------
def test_crane(gs):
    me = gs.me()
    me.improvements = {"science": 1, "politics": 1, "trade": 1}
    assert not ok(gs, "crane")                      # level 2 costs 2, Crane 1: no commodity
    me.hand["paper"] = 1
    assert ok(gs, "crane")
    me.improvements = {"science": 0, "politics": 0, "trade": 0}
    me.hand["paper"] = 0
    assert ok(gs, "crane")                          # owner: level 1 is free with the Crane
    me.cities.clear()
    assert not ok(gs, "crane")


def test_engineer(gs):
    me = gs.me()
    assert ok(gs, "engineer")
    me.walls |= set(me.cities)
    assert not ok(gs, "engineer")


def test_inventor(gs):
    assert ok(gs, "inventor")
    for h in gs.board.hexes.values():
        if pc.inventor_movable(gs, h):
            h.number = 6
    assert not ok(gs, "inventor")


def test_irrigation_and_mining(gs):
    assert ok(gs, "irrigation") and ok(gs, "mining")
    gs.me().settlements.clear()
    assert not ok(gs, "irrigation") and ok(gs, "mining")
    gs.me().cities.clear()
    assert not ok(gs, "mining")


def test_medicine(gs):
    me = gs.me()
    assert not ok(gs, "medicine")                   # can't pay
    me.hand.update(ore=2, grain=1)
    assert ok(gs, "medicine")
    me.settlements.clear()
    assert not ok(gs, "medicine")


def test_smith(gs):
    me = gs.me()
    assert not ok(gs, "smith")                      # no knights
    me.knights[(9, 9)] = Knight(level=1)
    assert ok(gs, "smith")
    me.knights = {(9, 9): Knight(level=2), (9, 8): Knight(level=2)}
    assert not ok(gs, "smith")                      # strong needs politics 3 to go mighty
    me.improvements["politics"] = 3
    assert ok(gs, "smith")


def test_road_building(gs):
    assert ok(gs, "road_building")
    gs.me().cities.clear(); gs.me().settlements.clear()
    assert not ok(gs, "road_building")


# ---- trade -----------------------------------------------------------------------------
def test_commercial_harbor(gs):
    assert not ok(gs, "commercial_harbor")          # no resource to give
    gs.me().hand["wool"] = 1
    assert not ok(gs, "commercial_harbor")          # two opponents hold cards: one card is short
    gs.me().hand["grain"] = 3
    assert not ok(gs, "commercial_harbor")          # grain is never given to a player
    gs.me().hand["ore"] = 1
    assert ok(gs, "commercial_harbor")
    for p in gs.players[1:]:
        p.hand_count = 0
    assert not ok(gs, "commercial_harbor")


def test_master_merchant_and_wedding_need_someone_ahead(gs):
    for c in ("master_merchant", "wedding"):
        assert ok(gs, c)                            # Hildegard 10 VP > 6
    gs.me().vp_visible = 11                          # owner's example: we lead everyone
    for c in ("master_merchant", "wedding"):
        assert not ok(gs, c)
        assert "more VP" in pc.can_play(gs, gs.me(), c)[1]


def test_merchant(gs):
    assert ok(gs, "merchant")
    gs.me().cities.clear(); gs.me().settlements.clear()
    assert not ok(gs, "merchant")


def test_merchant_fleet(gs):
    assert ok(gs, "merchant_fleet")          # owner-confirmed: no pair needed to play it


def test_merchant_never_on_gold(gs):
    me = gs.me()
    for h in pc._land_hexes_near(gs, me):
        h.terrain = "gold"
    assert not ok(gs, "merchant")


def test_monopolies(gs):
    assert ok(gs, "resource_monopoly") and ok(gs, "trade_monopoly")
    for p in gs.players[1:]:
        p.hand_count = 0
    assert not ok(gs, "resource_monopoly") and not ok(gs, "trade_monopoly")


# ---- politics --------------------------------------------------------------------------
def test_bishop(gs):
    assert not ok(gs, "bishop")
    gs.barbarians_attacked = True
    assert ok(gs, "bishop")


def test_deserter_and_warlord(gs):
    assert not ok(gs, "deserter") and not ok(gs, "warlord")
    gs.players[1].knights[(9, 9)] = Knight()
    gs.me().knights[(8, 8)] = Knight(active=False)
    assert ok(gs, "deserter") and ok(gs, "warlord")
    gs.me().knights[(8, 8)].active = True
    assert not ok(gs, "warlord")


def test_deserter_needs_a_free_knight_piece(gs):
    """owner-confirmed 2026-10-06: refused with no free piece; a taken strong knight is set as
    basic when both our strong knights are out."""
    gs.players[1].knights[(9, 9)] = Knight(level=2)
    me = gs.me()
    for i, lv in enumerate((1, 1, 2, 2, 3, 3)):
        me.knights[(20 + i, 0)] = Knight(level=lv)
    assert not ok(gs, "deserter")
    del me.knights[(20, 0)]                  # one basic piece free
    assert ok(gs, "deserter")


def test_diplomat(gs):
    assert not ok(gs, "diplomat")
    a = gs.players[1]
    v = next(iter(a.cities))
    e = next(e for e in gs.board.vertex_edges(v) if all(gs.owner_at(x) in (None, a) for x in e))
    a.roads.add(e)
    assert ok(gs, "diplomat")
    a.cities.add(next(x for x in e if x != v))       # both ends built: no longer open
    assert not ok(gs, "diplomat")


def test_intrigue_needs_enemy_knight_on_our_road(gs):
    me, a = gs.me(), gs.players[1]
    v = next(iter(me.cities))
    e = next(iter(gs.board.vertex_edges(v)))
    me.roads.add(e)
    w = next(x for x in e if x != v)
    a.knights[(99, 99)] = Knight()
    assert not ok(gs, "intrigue")                   # knight nowhere near our road
    a.knights[w] = Knight()
    assert ok(gs, "intrigue")


def test_saboteur(gs):
    assert ok(gs, "saboteur")                       # Hildegard ahead with 6 cards
    gs.me().vp_visible = 11
    assert not ok(gs, "saboteur")
    gs.me().vp_visible = 10                          # owner-confirmed: ties are hit
    assert ok(gs, "saboteur")


def test_spy(gs):
    assert not ok(gs, "spy")
    gs.players[2].progress_cards = 1
    assert ok(gs, "spy")


# ---- gates -----------------------------------------------------------------------------
def test_choose_play_logs_cant_play_reason(gs):
    gs.me().vp_visible = 11
    why = {}
    plays = pc.choose_play(gs, gs.me(), ["wedding", "master_merchant", "spy"], "main", why)
    assert plays == []
    assert why["wedding"].startswith("can't play — ")
    assert why["spy"].startswith("can't play — ")


def test_choose_play_never_returns_unplayable(gs):
    """No card choose_play returns may fail can_play, in any phase, whatever the hand."""
    hand = [c for c in pc.CARDS]
    for attacked in (False, True):
        gs.barbarians_attacked = attacked
        for phase in ("pre_roll", "main", "end_turn", "opp_turn"):
            for c, _ in pc.choose_play(gs, gs.me(), hand, phase):
                if c != "discard":
                    assert pc.can_play(gs, gs.me(), c, phase)[0], (phase, c)


def test_brain_holds_with_reason(gs):
    br = Brain()
    notes = []
    br.note = notes.append
    gs.me().vp_visible = 11
    assert br.progress_play(gs, "main", ["wedding"]) is None
    assert any("holding wedding: can't play — " in n for n in notes), notes


def _runner(gs, screens, tmp_path):
    cfg = config.load(); cfg["dry_run"] = True
    r = Runner(FakeIO(screens), cfg)
    r.mode, r.gs = gs.mode, gs
    r.ensure_state = lambda img, texts: gs
    r.popup_dir = tmp_path
    return r


def test_runner_never_drags_unplayable_card(gs, tmp_path, monkeypatch):
    gs.me().vp_visible = 11                          # Wedding can't be played: we lead
    r = _runner(gs, [[box("Wedding")]] * 4, tmp_path)
    # even if the brain asks for it, the runner's own gate stops the drag
    monkeypatch.setattr(r.brain, "progress_play", lambda g, ph, hand_cards=None:
                        ("wedding", {"from": []}) if "wedding" in (hand_cards or []) else None)
    assert r.play_progress("main") == []
    assert r.io.clicks == []
    assert any("holding wedding: can't play" in m for m in r.log)


# ---- walls -----------------------------------------------------------------------------
def test_wall_build_is_recorded_and_raises_the_limit(gs, tmp_path):
    r = _runner(gs, [], tmp_path)
    r.FLAGS = tmp_path / "flags.json"
    v = next(iter(gs.me().cities))
    base = pc.discard_limit(gs, gs.me())
    r._record_build("wall", v)
    assert v in r._my_walls and v in gs.me().walls
    assert pc.discard_limit(gs, gs.me()) == base + 2
    r2 = _runner(gs, [], tmp_path)
    r2.FLAGS = r.FLAGS
    r2._load_game_flags()
    assert v in r2._my_walls                         # survives a runner restart


def test_engineer_wall_prompt_records_wall(gs, tmp_path, monkeypatch):
    import types
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    v = next(iter(gs.me().cities))
    r = _runner(gs, [[]] * 4, tmp_path)
    r.FLAGS = tmp_path / "flags.json"
    r.br = types.SimpleNamespace(lattice=types.SimpleNamespace(vertex_px=lambda v: (500, 400)))
    r._green_check = lambda img: False
    r._vertex_rings = lambda img: [(v, (500, 400))]
    r.on_place_wall(np.zeros((888, 1372, 3), np.uint8), [], [], {})
    assert v in gs.me().walls and v in r._my_walls


def test_merchant_fleet_waits_for_a_pair_and_the_plan_asks_for_it(gs):
    """owner 2026-10-06: "with merchant fleet, bot needs to trade first and get two of a kind
    before using the card, otherwise it may be wasted"."""
    me = gs.me()
    me.progress_hand = ["merchant_fleet"]
    me.hand.update(grain=1, ore=3, wool=1)            # 2 wool at 2:1 -> the 2nd grain -> a city
    me.improvements = {"trade": 3, "politics": 3, "science": 3}   # no cheap improvement in reach
    assert pc.choose_play(gs, me, ["merchant_fleet"], "main") == []
    assert pc.fleet_pair_target(gs, me) == "wool"
    assert Brain().plan_missing(gs) == {"wool": 1}
    me.hand["wool"] = 2                              # the trade landed: now play it
    assert pc.choose_play(gs, me, ["merchant_fleet"], "main") == [("merchant_fleet", {"resource": "wool"})]
    assert pc.fleet_pair_target(gs, me) is None
