"""Outlook for The Great Canal (docs/audit/rules_diff.md rows 1, 14, 41, 42)."""
import copy

from catanbot import chat, outlook
from catanbot.board import hex_corners, real_canal_board
from catanbot.rules import COSTS, MODES
from catanbot.state import GameState, Knight, PlayerState


def _gs(n=3, vp_target=None):
    b = real_canal_board(7)
    ps = [PlayerState("a", is_me=True)] + [PlayerState(x) for x in "bcd"[: n - 1]]
    return GameState(board=b, mode=copy.copy(MODES["great_canal"]), players=ps, vp_target=vp_target)


# row 1: threat ranking and chat use gs.target (screen, else 21 with 3 / 18 with 4)
def test_rank_threats_uses_three_player_target():
    gs = _gs(3)
    gs.players[1].vp_visible = 7
    o = outlook.Outlook([], {}, {}, set())
    (_, _, why), *_ = outlook.rank_threats(gs, o, {})
    assert "vp 7/21" in why


def test_rank_threats_uses_four_player_and_screen_target():
    gs = _gs(4)
    o = outlook.Outlook([], {}, {}, set())
    assert all("/18," in w for _, _, w in outlook.rank_threats(gs, o, {}))
    gs = _gs(3, vp_target=15)                      # host-set value on screen wins
    assert all("/15," in w for _, _, w in outlook.rank_threats(gs, o, {}))


def test_chat_scenario_line_uses_target():
    assert "21 VP" in chat.answer("which game is this?", _gs(3))
    assert "18 VP" in chat.answer("which game is this?", _gs(4))
    assert "16 VP" in chat.answer("which game is this?", _gs(3, vp_target=16))


# row 42 + C&K: no island / largest army / dev VP sources
def test_vp_sources_canal_only():
    names = {n for n, _, _ in outlook.vp_sources(_gs())}
    assert "island" not in names and "largest_army" not in names and "dev_vp" not in names
    assert {"settlement", "city", "longest_road", "metropolis", "canal_chip"} <= names
    assert dict((n, v) for n, v, _ in outlook.VP_TABLE)["canal_chip"] == 2


# row 14: one owner's dig = 2 VP, shared = 1 VP
def _canal_move(gs, p):
    return next(m for m in outlook.player_moves(gs, p) if m.kind == "canal_chip")


def _quiet_hex(gs):
    """A canal hex whose corners touch no other canal hex, so the test reads one dig."""
    canal = gs.board.canal_cells()
    for c in sorted(canal):
        cs = set(hex_corners(*c))
        if not any(cs & set(hex_corners(*o)) for o in canal if o != c):
            return c
    return sorted(canal)[0]


def test_canal_chip_two_vp_when_both_knights_ours():
    gs = _gs()
    a = gs.me()
    cs = hex_corners(*_quiet_hex(gs))
    a.knights[cs[0]] = Knight(active=True)
    a.knights[cs[3]] = Knight(active=False)
    mv = _canal_move(gs, a)
    assert mv.vp == 2 and mv.target == cs[3] and mv.cost == dict(COSTS["activate"])


def test_canal_chip_one_vp_when_shared():
    gs = _gs()
    a, b = gs.players[0], gs.players[1]
    cs = hex_corners(*_quiet_hex(gs))
    a.knights[cs[0]] = Knight(active=False)
    b.knights[cs[3]] = Knight(active=True)
    mv = _canal_move(gs, a)
    assert mv.vp == 1 and mv.target == cs[0]


def test_canal_chip_two_idle_knights_cost_two_activations():
    gs = _gs()
    a = gs.me()
    cs = hex_corners(*_quiet_hex(gs))
    a.knights[cs[0]] = Knight(active=False)
    a.knights[cs[3]] = Knight(active=False)
    mv = _canal_move(gs, a)
    assert mv.vp == 2 and mv.cost == {k: 2 * v for k, v in COSTS["activate"].items()}


# row 41: rule notes
def test_great_canal_notes_confirmed():
    notes = " | ".join(MODES["great_canal"].notes)
    for s in ("21 with 3", "2 VP one owner, 1+1 two owners", "9 canal sections", "no VP",
              "only on an 8", "max 1 per player", "no activation", "no settlements on the small islands",
              "no pirate", "first barbarian attack"):
        assert s in notes, s
