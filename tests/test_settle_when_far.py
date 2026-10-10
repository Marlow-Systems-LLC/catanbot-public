"""Stage game 2026-10-09 050606: 0 settlements in 89 turns -- the knight-first hold took every grain."""
from catanbot.sim import Sim
from catanbot.state import Knight, new_hand


def _canal_idle_knight():
    sim = Sim("great_canal", seed=1)
    sim.setup()
    g, me, b = sim.gs, sim.gs.players[0], sim.brains[0]
    for p in g.players:
        p.hand, p.progress_hand = new_hand(), []
    for _ in range(4):                               # roads out until a settlement spot opens
        me.hand.update(brick=2, lumber=2, wool=1, grain=1)
        if any(a.kind == "settlement" for a in b.candidate_actions(g)):
            break
        road = max((a for a in b.candidate_actions(g) if a.kind == "road"), key=lambda a: b.action_value(g, a))
        sim.execute(0, road)
    me.hand = new_hand()
    me.hand.update(brick=1, lumber=1, wool=1, grain=1)
    spot = next(v for e in me.roads for v in e if v not in me.cities and v not in me.settlements
                and all(v not in p.knights for p in g.players))
    me.knights[spot] = Knight(1, False)              # after an attack: the knight sits idle
    assert b.defence_gap(g, me) > 0
    assert any(a.kind == "settlement" for a in b.candidate_actions(g))
    return g, b


def test_ship_far_settlement_takes_the_grain():
    g, b = _canal_idle_knight()
    g.barbarian_pos = 0
    assert "settlement" in [a.kind for a in b.plan_turn(g)]


def test_ship_near_knight_still_first():
    g, b = _canal_idle_knight()
    g.barbarian_pos = g.barbarian_len - 2
    kinds = [a.kind for a in b.plan_turn(g)]
    assert kinds[:1] == ["activate"] and "settlement" not in kinds


def test_flag_off_is_the_old_hold():
    g, b = _canal_idle_knight()
    g.barbarian_pos = 0
    b.settle_when_far = False
    assert "settlement" not in [a.kind for a in b.plan_turn(g)]


def test_a_build_spends_the_post_trade_hand(monkeypatch, tmp_path):
    # 04:59:40: "grain read 0 just after the trade, using 1" after the activation spent that grain
    from catanbot import config
    from catanbot import runner as R
    from catanbot.runner import ReplayIO, Runner
    cfg = config.load(); cfg["dry_run"] = True
    r = Runner(ReplayIO([]), cfg)
    monkeypatch.setattr(Runner, "FLAGS", tmp_path / "game_flags.json")
    r._trade_expect = (R.time.time(), {"grain": 1, "brick": 2}, {"grain"})
    r._record_build("activate", (1, -7))
    assert r._trade_expect[1]["grain"] == 0 and r._trade_expect[1]["brick"] == 2


def test_held_settlement_never_banks_knight_cards():
    # review 8825311: bank_trades_for may pick ore 3 -> wool for the settlement while the knight waits
    g, b = _canal_idle_knight()
    g.barbarian_pos = 0
    me = g.me()
    me.hand.update(wool=0, ore=4, cloth=4)
    plan = b.plan_turn(g)
    held = plan[:next((i for i, a in enumerate(plan) if a.kind == "activate"), len(plan))]
    bad = ("wool", "ore", "grain", "cloth", "coin", "paper")
    assert not any(a.kind == "bank_trade" and any(a.give.get(c) for c in bad) for a in held)


def test_an_improvement_keeps_the_grain_correction(monkeypatch, tmp_path):
    from catanbot import config
    from catanbot import runner as R
    from catanbot.runner import ReplayIO, Runner
    cfg = config.load(); cfg["dry_run"] = True
    r = Runner(ReplayIO([]), cfg)
    monkeypatch.setattr(Runner, "FLAGS", tmp_path / "game_flags.json")
    r._trade_expect = (R.time.time(), {"grain": 1, "cloth": 2}, {"grain", "cloth"})
    r._record_build("improve", (1, -7))
    assert r._trade_expect[2] == {"grain"}
