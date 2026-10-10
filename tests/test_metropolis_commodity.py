"""Stage 2026-10-09 07:32:59: the trade metropolis was lost at level 4 after the cloth went to the
bank (07:22:12, 07:31:52) and to an opponent's Commercial Harbor (07:25:15)."""
from catanbot.sim import Sim
from catanbot.state import new_hand


def _metro_at_4(**hand):
    sim = Sim("great_canal", seed=1)
    sim.setup()
    g, b = sim.gs, sim.brains[0]
    me = g.me()
    for p in g.players:
        p.hand, p.progress_hand = new_hand(), []
    city = next(iter(me.cities))
    me.improvements.update(trade=4)
    me.metropolises.add(city)
    g.metropolis_owner["trade"] = (me.name, city)
    me.hand.update(hand)
    return g, b


def test_cloth_stays_brick_pays():
    g, b = _metro_at_4(cloth=2, brick=4)
    trades = b.bank_trades_for(g, {"grain": 1})
    assert trades and not any(t.give.get("cloth") for t in trades)
    assert any(t.give.get("brick") for t in trades)


def test_flag_off_sells_the_cloth():
    g, b = _metro_at_4(cloth=2, brick=4)
    b.lock_metropolis_commodity = False
    assert any(t.give.get("cloth") for t in b.bank_trades_for(g, {"grain": 1}))


def test_lifts_at_level_5_or_without_the_metropolis():
    g, b = _metro_at_4(cloth=2)
    assert b.reserved_commodities(g) == {"cloth"}
    g.me().improvements["trade"] = 5
    assert b.reserved_commodities(g) == set()
    g.me().improvements["trade"] = 4
    g.metropolis_owner["trade"] = (g.players[1].name, next(iter(g.me().cities)))
    assert b.reserved_commodities(g) == set()


def test_no_player_gets_the_cloth():
    g, b = _metro_at_4(cloth=3, brick=1)
    assert not b.evaluate_offer(g, g.players[1].name, {"cloth": 1}, {"grain": 1})


def test_harbour_and_runner_bank_skip_the_cloth():
    from catanbot import config
    from catanbot.runner import ReplayIO, Runner
    g, b = _metro_at_4(cloth=2, coin=1, brick=1)
    cfg = config.load(); cfg["dry_run"] = True
    r = Runner(ReplayIO([]), cfg)
    r.brain, r.gs = b, g
    clicks = []
    r.click = lambda xy, why="": clicks.append(why)
    import catanbot.runner as R
    R.time.sleep, sleep = (lambda s: None), R.time.sleep
    try:
        r._commercial_harbor()
    finally:
        R.time.sleep = sleep
    assert clicks[0] == "commercial harbor: give coin"
    t = r._bank_trade_for(g, {"grain": 1})
    assert t is None or not t.give.get("cloth")
