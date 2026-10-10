"""Stage 2026-10-09 06:35: a late turn dropped the bank trade and kept the road it paid for."""
from types import SimpleNamespace

from catanbot.brain import Action
from catanbot.runner import Runner
from catanbot.state import new_hand


def _gs(**hand):
    h = new_hand(); h.update(hand)
    me = SimpleNamespace(hand=h, improvements={"trade": 0, "politics": 0, "science": 0})
    return SimpleNamespace(me=lambda: me)


def test_bank_trade_paying_a_build_stays():
    gs = _gs(lumber=2, ore=5)
    plan = [Action("bank_trade", give={"ore": 4}, get={"brick": 1}), Action("road", ((2, -4), (2, -2)))]
    assert [a.kind for a in Runner._late_plan(gs, plan)] == ["bank_trade", "road"]


def test_build_without_its_cards_goes():
    gs = _gs(lumber=2, ore=1)
    plan = [Action("bank_trade", give={"ore": 4}, get={"brick": 1}), Action("road", ((2, -4), (2, -2)))]
    assert Runner._late_plan(gs, plan) == []


def test_player_offers_and_trailing_trades_go():
    gs = _gs(brick=1, lumber=1, wool=1, ore=8, cloth=2)
    plan = [Action("player_trade", give={"ore": 2}, get={"grain": 1}),
            Action("bank_trade", give={"cloth": 2}, get={"grain": 1}),
            Action("settlement", (3, -1)),
            Action("bank_trade", give={"ore": 4}, get={"lumber": 1})]
    assert [a.kind for a in Runner._late_plan(gs, plan)] == ["bank_trade", "settlement"]


def test_plain_knight_spot_is_not_our_settlement_spot():
    # stage 2026-10-09 06:11: knight on (4,2), the corner the roads had just reached
    import sys
    sys.path.insert(0, "tests")
    from test_settle_when_far import _canal_idle_knight
    g, b = _canal_idle_knight()
    me = g.me()
    me.knights.clear()
    me.hand.update(wool=1, ore=1, grain=1)
    spots = set(g.legal_settlement_spots(me))
    assert spots
    reach = {v for e in me.roads for v in e}
    if not (reach - spots - me.cities - me.settlements):
        return
    knights = [a.target for a in b.candidate_actions(g) if a.kind == "knight"]
    plain = [v for v in knights if v not in spots]
    assert plain and knights[0] not in spots
