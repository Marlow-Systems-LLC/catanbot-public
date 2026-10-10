from catanbot.sim import Sim
from catanbot import progress_cards as pc


def _sim():
    s = Sim("great_canal", seed=7)
    s.setup()
    return s


def test_record_roll_counts_production():
    s = _sim()
    gs = s.gs
    for roll in (2, 3, 4, 5, 6, 8, 9, 10, 11, 12):
        gs.record_roll(roll)
    assert any(sum(g.values()) > 0 for g in gs.seen_gain.values())


def test_est_hand_splits_unknown_count_like_seen_gain():
    s = _sim()
    gs = s.gs
    p = gs.players[1]
    p.hand = {}
    p.hand_count = 6
    gs.seen_gain[p.name] = {"ore": 2, "wool": 1}
    est = gs.est_hand(p)
    assert abs(sum(est.values()) - p.total_cards()) < 1e-6
    assert est["ore"] > est["wool"] > 0


def test_monopoly_take_caps_at_two_per_player():
    s = _sim()
    gs = s.gs
    me = gs.players[0]
    for p in gs.players[1:]:
        p.hand = {"ore": 5}
    assert pc.monopoly_take(gs, me, "ore", 2.0) == 2.0 * (len(gs.players) - 1)


def test_monopoly_names_the_missing_resource():
    s = _sim()
    gs = s.gs
    me = gs.players[0]
    for p in gs.players[1:]:
        p.hand = {"ore": 1, "wool": 3}
    gs.plan_missing = {"ore": 2}
    assert pc.choose_resource_monopoly(gs, me) == {"resource": "ore"}
    gs.plan_missing = {}
    assert pc.choose_resource_monopoly(gs, me) == {"resource": "wool"}


def test_spend_turn_banks_spare_cards_toward_near_target():
    from catanbot.brain import Brain
    s = _sim()
    gs = s.as_me(0)
    me = gs.me()
    b = Brain()
    me.hand = {"brick": 4, "lumber": 0, "wool": 0, "grain": 0, "ore": 0}
    trades = b._spend_toward_target(gs)
    assert all(t.kind == "bank_trade" and "grain" not in t.give for t in trades)
    b.spend_need_cap = 0
    assert b._spend_toward_target(gs) == []
