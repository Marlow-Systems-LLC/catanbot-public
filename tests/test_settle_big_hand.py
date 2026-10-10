"""Live 2026-10-09 21:08 (Great Canal, target 19), owner: "marlow should build settlements now,
still not doing that even with 25 resources in hand". 28 cards, an open spot on our road, and
the plans were only city / activate / promote / improve / bank_trade -- the improvements and
their bank trades ate the settlement's wool and grain every step."""
from catanbot.sim import Sim
from catanbot.state import Knight, new_hand

LIVE_HAND = {"brick": 1, "lumber": 5, "wool": 4, "grain": 3, "ore": 5, "paper": 4, "cloth": 4, "coin": 2}


def _live(seed, spot=True):
    """2 cities, 1 settlement, one active and one idle knight, ship at 4/7, the live hand.
    spot=True: roads out until an open spot sits on our road; False: one road (spot 1 road away)."""
    sim = Sim("great_canal", seed=seed)
    sim.setup()
    g, me, b = sim.gs, sim.gs.players[0], sim.brains[0]
    for p in g.players:
        p.hand, p.progress_hand = new_hand(), []
    for _ in range(4 if spot else 1):
        me.hand.update(brick=2, lumber=2, wool=1, grain=1)
        if spot and any(a.kind == "settlement" for a in b.candidate_actions(g)):
            break
        road = max((a for a in b.candidate_actions(g) if a.kind == "road"), key=lambda a: b.action_value(g, a))
        sim.execute(0, road)
    s = next(iter(me.settlements))
    me.settlements.discard(s)
    me.cities.add(s)
    spots = set(g.legal_settlement_spots(me))
    for e in sorted(me.roads):
        for v in e:
            if len(me.knights) < 2 and not g.occupied(v) and v not in spots:
                me.knights[v] = Knight(1, not me.knights)
    g.barbarian_pos = 4
    me.hand = new_hand()
    me.hand.update(LIVE_HAND)
    return g, b


def test_live_spot_ready_old_plan_never_settles():
    g, b = _live(5)
    assert g.legal_settlement_spots(g.me())
    b.settle_big_hand = False
    assert "settlement" not in [a.kind for a in b.plan_turn(g)]


def test_live_spot_ready_big_hand_settles():
    g, b = _live(5)
    kinds = [a.kind for a in b.plan_turn(g)]
    assert "settlement" in kinds
    assert kinds.index("city") < kinds.index("settlement")        # owner: cities first


def test_spot_one_road_away_road_then_settlement():
    g, b = _live(1, spot=False)
    me = g.me()
    assert not g.legal_settlement_spots(me) and b._spot_distance(g, me) == 1
    kinds = [a.kind for a in b.plan_turn(g)]
    assert "road" in kinds and "settlement" in kinds
    assert kinds.index("road") < kinds.index("settlement")


def test_no_road_toward_a_far_spot():
    g, b = _live(1, spot=False)
    b.settle_big_hand_reach = 0
    assert b._big_hand_settle(g, False, True, -1.0) is None


def test_small_hand_is_the_old_order():
    g, b = _live(5)
    g.me().hand = new_hand()
    g.me().hand.update(brick=1, lumber=1, wool=1, grain=1, ore=1)
    assert not b._big_hand(g)
    on = [a.kind for a in b.plan_turn(g)]
    b.settle_big_hand = False
    assert on == [a.kind for a in b.plan_turn(g)]


# owner 2026-10-09: "even if the canal is done, marlow should continue to build roads and
# settlements and then cities"
def _canal_done(seed=6):
    sim = Sim("great_canal", seed=seed)
    sim.setup()
    g, me, b = sim.gs, sim.gs.players[0], sim.brains[0]
    for p in g.players:
        p.hand, p.progress_hand = new_hand(), []
    me.hand.update(brick=2, lumber=2, wool=1, grain=1)
    road = max((a for a in b.candidate_actions(g) if a.kind == "road"), key=lambda a: b.action_value(g, a))
    sim.execute(0, road)
    for e in sorted(me.roads):
        for v in e:
            if not me.knights and not g.occupied(v):
                me.knights[v] = Knight(1, True)
    g.barbarian_pos = 1
    g.canal_progress = g.canal_total
    me.hand = new_hand()
    me.hand.update(brick=2, lumber=2, wool=1, grain=1)
    return g, b


def test_canal_done_old_greedy_built_a_wall():
    g, b = _canal_done()
    assert not g.legal_settlement_spots(g.me()) and b._spot_distance(g, g.me()) == 1
    b.road_settle_lookahead = False
    assert "settlement" not in [a.kind for a in b.plan_turn(g)]


def test_canal_done_road_then_settlement():
    g, b = _canal_done()
    plan = b.plan_turn(g)
    kinds = [a.kind for a in plan]
    assert kinds[:2] == ["road", "settlement"]
    assert plan[1].target in plan[0].target                       # the road reaches the spot
    assert b._road_target(g, g.me(), plan[1].target)             # 7+ pips, no rival road


def test_canal_done_then_a_city():
    g, b = _canal_done()
    for a in b.plan_turn(g):
        g = b._apply(g, a)
    g.me().hand = new_hand()
    g.me().hand.update(grain=2, ore=3)
    cities = [a.target for a in b.plan_turn(g) if a.kind == "city"]
    assert cities and cities[0] in g.me().settlements    # owner: on the best producing settlement
