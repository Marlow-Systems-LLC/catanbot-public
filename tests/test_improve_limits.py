"""C&K p.7-8 limits: a city is needed, max level 5, level 4+ needs a city that can hold the metropolis."""
from catanbot.moves import improve_allowed
from catanbot.sim import Sim


def _me():
    sim = Sim("great_canal", 4, seed=5)
    sim.setup()
    gs = sim.gs
    me = gs.players[0]
    return gs, me


def test_needs_a_city_and_caps_at_5():
    gs, me = _me()
    me.cities = set()
    assert not improve_allowed(gs, me, "science")
    me.cities = {next(iter(me.settlements))}
    assert improve_allowed(gs, me, "science")
    me.improvements["science"] = 5
    assert not improve_allowed(gs, me, "science")


def test_level_4_needs_a_city_without_a_metropolis():
    gs, me = _me()
    c = next(iter(me.settlements))
    me.cities, me.metropolises = {c}, {c}
    gs.metropolis_owner = {"trade": (me.name, c)}
    me.improvements.update(science=3, trade=4, politics=2)
    assert not improve_allowed(gs, me, "science")      # its only city already holds a metropolis
    assert improve_allowed(gs, me, "trade")            # 4 -> 5 keeps our own trade metropolis
    assert improve_allowed(gs, me, "politics")         # 2 -> 3 needs no metropolis city
