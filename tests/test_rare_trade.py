"""Owner 2026-10-06: always accept a trade that brings a resource we (almost) never roll."""
from catanbot.brain import Brain
from catanbot.learning import Model
from catanbot.sim import Sim


def _setup():
    sim = Sim("great_canal", 4, seed=7)
    sim.setup()
    gs = sim.gs
    me = gs.me()
    inc = gs.income(me)
    spare = max(inc, key=inc.get)                     # what we roll most
    me.hand = {spare: 2}
    me.hand_count = 2
    return gs, me, spare, Brain(Model(path=None))


def test_grain_for_a_spare_card_is_taken():
    gs, me, spare, b = _setup()
    assert gs.income(me).get("grain", 0) == 0          # Great Canal: no grain before the canal
    partner = next(p for p in gs.opponents()).name
    assert b.rare_gain(gs, {spare: 1}, {"grain": 1})
    assert b.evaluate_offer(gs, partner, {spare: 1}, {"grain": 1})


def test_rare_rule_limits():
    gs, me, spare, b = _setup()
    assert not b.rare_gain(gs, {spare: 2}, {spare: 1})            # not rare for us
    me.hand = {spare: 3}
    assert not b.rare_gain(gs, {spare: 3}, {"grain": 1})          # 3 for 1 is too much
    b.keep = {spare}
    assert not b.rare_gain(gs, {spare: 1}, {"grain": 1})          # never pay with kept cards


def test_never_feeds_a_player_about_to_win():
    gs, me, spare, b = _setup()
    p = next(iter(gs.opponents()))
    p.vp_visible = gs.target - 1
    assert not b.evaluate_offer(gs, p.name, {spare: 1}, {"grain": 1})
