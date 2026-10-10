"""Great Canal setup as a set (owner 2026-10-05, live game 14; Brain.setup_set).

Board = game 14 scan (tests/fixtures/scan_game14_2026-10-05.json) snapped to the canal template.
Opponents as they sat at our first pick (03:21:23): Rhea (5,5), Lin (-1,-1), Lars (-2,2); we
were last in the first round, so settlement and city went down back to back.

Owner: "it could have picked 6 sheep 9 wood ... city, and the settlement here best would have
been at 5 9 10 where stone and clay were going to give it all it needed, then just a harbor
would have been the target for the third settlement".
"""
import copy
import json

import pytest

from catanbot.board import Board, Hex, snap_to_canal_template
from catanbot.brain import Brain
from catanbot.learning import Model
from catanbot.rules import MODES
from catanbot.state import GameState, PlayerState

FX = "tests/fixtures/scan_game14_2026-10-05.json"
OPP = {"Rhea": [(5, 5)], "Lin": [(-1, -1)], "Lars": [(-2, 2)]}
OWNER_SETTLEMENT = (1, 1)     # hills 9 + mountains 10 + mountains 5: brick + ore
OWNER_CITY = (-4, 2)          # pasture 6 + forest 9 (+ sea): sheep + wood; (-3,1) is next to Lars
LIVE_SETTLEMENT, LIVE_CITY, PHANTOM = (4, -4), (5, -1), (-6, -8)


@pytest.fixture(scope="module")
def board():
    with open(FX) as f:
        d = json.load(f)
    b = Board()
    for h in d["hexes"]:
        b.add(Hex(h["q"], h["r"], h["terrain"], h["number"]))
    s = snap_to_canal_template(b)
    assert s is not None and s.template_offset == (0, 0)
    return s


def _gs(board, mine=(), kinds=None, opp=None):
    me = PlayerState("marlow.systems", is_me=True)
    ps = [me] + [PlayerState(n) for n in ("Rhea", "Lin", "Lars")]
    for p in ps[1:]:
        p.settlements.update((opp or OPP).get(p.name, []))
    for i, v in enumerate(mine):
        (me.cities if kinds and kinds[i] == "city" else me.settlements).add(v)
    return GameState(board=copy.deepcopy(board), mode=copy.copy(MODES["great_canal"]), players=ps)


def _brain(on=True):
    b = Brain(Model(path=None), setup_weights={})
    b.setup_set = on
    return b


def _terr(board, v):
    return sorted((h.terrain, h.number) for h in board.vertex_hexes(v) if h.number)


def test_owner_corners_are_what_the_owner_named(board):
    assert _terr(board, OWNER_SETTLEMENT) == [("hills", 9), ("mountains", 5), ("mountains", 10)]
    assert ("pasture", 6) in _terr(board, OWNER_CITY) and ("forest", 9) in _terr(board, OWNER_CITY)
    gs = _gs(board)
    legal = gs.legal_settlement_spots(gs.me(), setup=True)
    assert OWNER_SETTLEMENT in legal and OWNER_CITY in legal and (-3, 1) not in legal


def test_live_picks_reproduced_with_old_scorer(board):
    """Old scorer + the phantom (-6,-8) read: the settlement was scored as a city (one piece
    down) and the city as the coastal 3rd settlement (two down) -> (4,-4), then (5,-1)."""
    br = _brain(False)
    gs = _gs(board, [PHANTOM])
    assert br.choose_setup_settlement(gs) == LIVE_SETTLEMENT
    gs = _gs(board, [PHANTOM, LIVE_SETTLEMENT])
    assert br.choose_setup_settlement(gs) == LIVE_CITY


def test_set_scorer_picks_owner_settlement_then_city(board):
    br = _brain()
    gs = _gs(board)
    assert br.choose_setup_settlement(gs) == OWNER_SETTLEMENT
    gs = _gs(board, [OWNER_SETTLEMENT])
    assert br._setup_kind(gs, gs.me()) == "city"
    assert br.choose_setup_settlement(gs) == OWNER_CITY


def test_set_scorer_third_settlement_takes_a_3to1(board):
    """Set covered (wood, sheep, brick, ore) -> the last coastal settlement goes on a 3:1,
    never on the lone barren grain 6 at (-6,-8) where the live game put it by mistake."""
    br = _brain()
    opp = {"Rhea": [(5, 5)], "Lin": [(-1, -1), (3, 5)], "Lars": [(-2, 2), (0, -4), (3, -7)]}
    gs = _gs(board, [OWNER_SETTLEMENT, OWNER_CITY], ["settlement", "city"], opp)
    v = br.choose_setup_settlement(gs)
    assert gs.board.harbor_at(v) == "3:1" and gs.board.is_coastal_vertex(v)
    assert br.setup_vertex_score(gs, v, gs.me()) > br.setup_vertex_score(gs, PHANTOM, gs.me()) + 10


def test_whole_game_yield_prices_barren_grain_low(board):
    br = _brain()
    gs = _gs(board)
    p = br._life_pips(gs, PHANTOM, "settlement")
    assert p.get("grain", 0) <= 0.15 * 5 + 1e-9      # grain 6 (5 pips) under the canal: ~nothing


def test_flag_off_keeps_old_scores(board):
    gs = _gs(board, [OWNER_SETTLEMENT])
    old = _brain(False)
    assert not old._set_mode(gs)
    assert old.choose_setup_settlement(gs) == LIVE_SETTLEMENT   # old: ore 11 + more pips wins the city
