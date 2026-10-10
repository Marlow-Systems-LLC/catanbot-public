"""Live 2026-10-09 19:44: the discard threw away 3 paper + 3 wood (paper 3 -> 4 and the metropolis),
and the robber went on an 8 with nobody to rob."""
from tests.test_canal_playbook import _gs, brain, hex_corners


def _me_city(gs):
    me = gs.me()
    me.cities.add(hex_corners(1, 0)[0])
    me.improvements["science"] = 3       # paper track
    return me


def test_discard_keeps_paper_and_wood_for_the_metropolis():
    gs = _gs()
    me = _me_city(gs)
    me.hand = {"brick": 1, "lumber": 4, "wool": 2, "ore": 1, "paper": 3, "coin": 1}
    b = brain()
    b.bank_ratio = lambda g, r: 3
    out = b.choose_discard(gs, 6)
    assert sum(out.values()) == 6
    assert me.hand["paper"] - out.get("paper", 0) == 3
    assert me.hand["lumber"] - out.get("lumber", 0) >= 3


def test_discard_without_a_reachable_level_is_unchanged():
    gs = _gs()
    me = _me_city(gs)
    me.hand = {"brick": 1, "lumber": 4, "wool": 2, "ore": 1, "paper": 3, "coin": 1}
    b = brain()
    b.bank_ratio = lambda g, r: 4                # 4:1 -- 3 paper + 4 wood would be 7 cards, only 6 stay
    assert "paper" not in b.improve_keep(gs, me.hand, 6)   # (politics 0 -> 1 with the 1 coin still fits)


def test_metropolis_level_beats_a_lower_one():
    gs = _gs()
    me = _me_city(gs)
    me.improvements["trade"] = 0                 # cloth 0 -> 1 costs 1 cloth
    hand = {"cloth": 1, "paper": 3, "lumber": 4, "wool": 2}
    b = brain()
    b.bank_ratio = lambda g, r: 3
    assert b.improve_keep(gs, hand, 3) == {"paper": 3, "lumber": 3}


def test_robber_goes_where_someone_can_be_robbed():
    gs = _gs()
    gs.mode.friendly_robber = False
    me, leader, weak = gs.players
    leader.settlements.add(hex_corners(1, -1)[1])   # hills 5
    leader.hand_count = 4
    # the gold 8 has nobody on it; the 5 has the leader with cards
    hx, victim = brain().choose_robber(gs, [(0, 1), (1, -1)])
    assert hx == (1, -1) and victim is leader
