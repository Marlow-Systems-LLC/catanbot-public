"""Live 2026-10-08 21:55 (solo Great Canal): 12 cards over an 11 limit, 2 cities, 1 knight,
no grain and no bank set -- plan=[] and end turn. Over the limit the cards are spent: a 2nd
knight first (2 cities, 1 knight), then the best other build."""
from test_canal_playbook import _gs, brain

from catanbot.board import ekey, hex_corners
from catanbot.state import Knight

HAND = {"brick": 3, "lumber": 3, "wool": 1, "ore": 2, "paper": 2, "cloth": 1}


def _live(active: bool, far: bool = True):
    gs = _gs()
    me = gs.me()
    c = hex_corners(0, -1)
    me.cities |= {c[0], c[3]}
    me.walls |= {c[0], c[3]}                 # limit 7 + 2*2 = 11
    for i in range(5):
        me.roads.add(ekey(c[i], c[i + 1]))
    me.knights[c[1]] = Knight(1, active)
    me.improvements.update({"science": 2, "trade": 1})
    me.hand.update(HAND)
    gs.barbarian_pos = 0 if far else gs.barbarian_len - 2
    return gs


def _kinds(plan):
    return [a.kind for a in plan]


def test_idle_knight_over_the_limit_still_spends():
    # the bot's live reading: targets started with 'activate' (the knight read idle)
    for far in (True, False):
        k = _kinds(brain()._plan_turn(_live(active=False, far=far)))
        assert k, f"far={far}: plan=[] with 12 cards over an 11 limit"
        assert k[0] == "knight", k               # 2 cities, 1 knight: the 2nd knight first


def test_active_knight_over_the_limit_builds_the_second():
    for far in (True, False):
        k = _kinds(brain()._plan_turn(_live(active=True, far=far)))
        assert k and k[0] == "knight", k


def test_spending_leaves_the_hand_at_the_limit():
    from catanbot.rules import COSTS
    plan = brain()._plan_turn(_live(active=False))
    spent = sum(sum(COSTS.get(a.kind, {}).values()) for a in plan if a.kind != "bank_trade")
    assert sum(HAND.values()) - spent <= 11, plan
