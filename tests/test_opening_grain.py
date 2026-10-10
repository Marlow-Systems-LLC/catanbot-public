"""Owner 2026-10-03 opening: bank a set for grain early and keep it for the knight."""
from catanbot.brain import Brain
from catanbot.learning import Model
from catanbot.sim import Sim
from catanbot.state import Knight


def _setup(hand, knight=True):
    sim = Sim("great_canal", seed=5)
    sim.setup()
    i = 0
    p = sim.gs.players[i]
    p.knights.clear()
    if knight:
        v = next(v for v in sim.gs.board.vertices if sim.gs.board.is_land_vertex(v)
                 and not sim.gs.board.is_small_island_vertex(v) and all(v not in q.knights for q in sim.gs.players)
                 and all(v not in dict(q.buildings()) for q in sim.gs.players))
        p.knights[v] = Knight(1, False)
    p.hand.clear(); p.hand.update(hand)
    b = Brain(Model(path=None))
    b.grain_first = True
    return sim, b, sim.as_me(i)


def test_banks_set_for_grain_then_activates():
    sim, b, gs = _setup({"lumber": 4})
    plan = b.plan_turn(gs)
    assert plan[0].kind == "bank_trade" and plan[0].get == {"grain": 1}
    assert any(a.kind == "activate" for a in plan)


def test_keeps_wool_and_ore_when_no_knight_yet():
    sim, b, gs = _setup({"wool": 4, "ore": 1}, knight=False)
    t = b._grain_first_trade(gs)
    assert t is None or t.give.get("wool", 0) <= 3


def test_off_when_flag_off():
    sim, b, gs = _setup({"lumber": 4})
    b.grain_first = False
    assert b._grain_first_trade(gs) is None
