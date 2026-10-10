"""fleet opus 2026-10-04: grain/knight economy flags (grain_gate, knight_guard), default off."""
from catanbot.board import ekey, hex_corners
from catanbot.brain import Action, Brain
from catanbot.state import Knight

from test_canal_playbook import C, _gs, brain


# knights off the canal hex (no dig) and off gold: corners of the forest hex away from the canal
K = [v for v in hex_corners(1, 0) if v not in C]


def _defended():
    """One city, one active knight (no gap), a 2nd idle knight; the rival is at strength 2."""
    gs = _gs(barbarian_pos=3)
    me, a = gs.players[0], gs.players[1]
    me.cities.add(C[0])
    me.knights[K[0]] = Knight(1, True)
    me.knights[K[1]] = Knight(1, False)
    a.cities.add(C[3])
    a.knights[C[4]] = Knight(2, True)
    return gs


def test_defaults_are_off():
    br = Brain()
    assert br.grain_gate is False and br.knight_guard == 0.0


def test_activation_useful_cases():
    br = brain()
    gs = _defended()
    assert not br._activation_useful(gs)            # 1 -> 2 only ties the rival's 2: no VP, no gap
    gs.players[1].knights[C[4]].level = 1           # rival at 1: our 2nd knight makes us sole top
    assert br._activation_useful(gs)
    gs = _defended()
    gs.me().cities.add(C[5])                        # 2 cities, 1 active: a city at risk
    assert br._activation_useful(gs)
    gs = _defended()
    gs.me().knights[K[1]].active = True
    assert not br._activation_useful(gs)            # nothing idle


def test_grain_gate_stops_the_set_for_grain_bank_trade_when_activation_is_useless():
    br = brain()
    br.bank_ratio = lambda g, r: 3
    gs = _defended()
    gs.me().hand = {"lumber": 3}
    assert br._grain_first_trade(gs) is not None    # owner rule: an idle knight -> bank the set
    br.grain_gate = True
    assert br._grain_first_trade(gs) is None
    gs.players[1].knights[C[4]].level = 1           # now it would make us sole top
    assert br._grain_first_trade(gs) is not None


def test_grain_gate_drops_spend_grain_on_a_useless_activation_only():
    gs = _defended()
    gs.me().hand = {"grain": 1}
    off, on = brain(), brain()
    on.grain_gate = True
    act = Action("activate", K[1])
    assert off.action_value(gs, act) - on.action_value(gs, act) > 0.02
    gs.players[1].knights[C[4]].level = 1
    assert abs(off.action_value(gs, act) - on.action_value(gs, act)) < 1e-9


def test_grain_gate_drops_the_activate_trade_target_and_third_of_set():
    br = brain()
    gs = _defended()
    gs.me().hand = {"ore": 2, "lumber": 1}
    br.bank_ratio = lambda g, r: 3 if r == "ore" else 4
    assert "activate" in br._trade_targets(gs)
    assert br._third_of_set(gs, br.resource_value(gs, gs.me()), None) is not None
    br.grain_gate = True
    # owner 2026-10-06: an idle knight always puts activate first (the gate no longer drops it)
    assert br._trade_targets(gs)[0] == "activate"
    assert br._third_of_set(gs, br.resource_value(gs, gs.me()), None) is None


def test_knight_guard_penalises_a_knight_that_cannot_be_activated():
    gs = _gs()
    me = gs.me()
    me.settlements.add(C[0]); me.roads.add(ekey(C[0], C[1]))
    me.knights[C[3]] = Knight(1, False)               # already one idle knight
    me.hand = {"wool": 1, "ore": 1, "grain": 1}
    off, on = brain(), brain()
    on.knight_guard = 0.02
    k = Action("knight", C[1])
    assert not on._on_gold(gs, C[1])
    assert off.action_value(gs, k) - on.action_value(gs, k) > 0.025   # 1 grain, 1 idle: not covered...
    assert not on._knight_activatable(gs, C[1])
    me.hand["grain"] = 2                              # ...2 grain covers the idle one and the new one
    assert on._knight_activatable(gs, C[1])
    assert abs(off.action_value(gs, k) - on.action_value(gs, k)) < 1e-9


def test_knight_guard_value_drop():
    gs = _gs()
    me = gs.me()
    me.settlements.add(C[0]); me.roads.add(ekey(C[0], C[1]))
    me.hand = {"wool": 1, "ore": 1}
    off, on = brain(), brain()
    on.knight_guard = 0.02
    k = Action("knight", C[1])
    assert off.action_value(gs, k) - on.action_value(gs, k) > 0.025


def test_grain_engine_values_science_to_aqueduct_and_first_gold_knight():
    gs = _gs()
    me = gs.me()
    br = brain()
    assert br.grain_engine == 0.0
    br.grain_engine = 0.02
    me.improvements["science"] = 2
    assert br._grain_engine_value(gs, Action("improve", "science")) == 0.04
    me.improvements["science"] = 3
    assert br._grain_engine_value(gs, Action("improve", "science")) == 0.0
    gold = next(v for v in hex_corners(0, 1) if br._on_gold(gs, v) and v not in C)
    assert br._grain_engine_value(gs, Action("knight", gold)) == 0.04
    me.knights[gold] = Knight(1, False)
    other = next(v for v in hex_corners(0, 1) if v not in C and v != gold)
    assert br._grain_engine_value(gs, Action("knight", other)) == 0.0
    gs.canal_progress = gs.canal_total                 # canal done: no gold, no extra
    off = brain(); on = brain(); on.grain_engine = 0.02
    me.improvements["science"] = 1
    a = Action("improve", "science")
    assert abs(off.action_value(gs, a) - on.action_value(gs, a)) < 1e-9
