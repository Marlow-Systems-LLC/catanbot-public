"""owner 2026-10-08: "should not give out development cards for just one resource, ask at least for
2" and "avoid giving them to players that have more points or can get the 4th level and level up"."""
from catanbot import moves as mv

from test_canal_playbook import _gs, brain


def _state():
    gs = _gs()
    me = gs.me()
    me.vp_visible = 6
    me.hand = {"paper": 2, "wool": 2, "lumber": 1}
    me.hand_count = 5
    gs.players[2].vp_visible = 4                 # "weak": behind us
    return gs


def test_one_resource_for_a_commodity_is_refused():
    br, gs = brain(), _state()
    assert not br.commodity_ok(gs, "weak", {"paper": 1}, {"ore": 1})
    assert br.commodity_ok(gs, "weak", {"paper": 1}, {"ore": 2})
    assert br.commodity_ok(gs, "weak", {"paper": 1}, {"coin": 1})     # commodity for commodity
    assert not br.evaluate_offer(gs, "weak", {"paper": 1}, {"ore": 1})


def test_no_commodity_to_a_player_ahead():
    br, gs = brain(), _state()
    gs.players[2].vp_visible = 7
    assert not br.commodity_ok(gs, "weak", {"paper": 1}, {"ore": 2})


def test_no_commodity_that_can_reach_level_4():
    br, gs = brain(), _state()
    gs.players[2].improvements["science"] = 3
    assert not br.commodity_ok(gs, "weak", {"paper": 1}, {"ore": 2})
    assert br.commodity_ok(gs, "weak", {"wool": 1}, {"ore": 1})         # plain resources unaffected


def test_wildcard_for_a_commodity_counters_with_two_cards():
    br, gs = brain(), _state()
    legal = mv.answer_moves(gs, {"paper": 1}, None, wildcard=True)
    assert all(m.kind == "decline" or sum(m.get_d.values()) == 2 for m in legal)
    ans = br.choose_answer(gs, legal, "weak")
    assert ans.kind == "decline" or sum(ans.get_d.values()) == 2
    gs.players[2].vp_visible = 9
    assert br.choose_answer(gs, legal, "weak").kind == "decline"
