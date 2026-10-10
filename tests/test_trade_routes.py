"""Owner 2026-10-08: "if it doesn't have a harbor it needs 4:1, a gold mine, science bonus or the
merchant (fleet) to get help. it needs to save the city and know all trade routes it can do at a
specific time". Live 21:54: "opening: bank {'lumber': 3} -> 1 grain for the knight", the bank
dialog showed lumber at 4, the trade was cancelled and the knight never switched on."""
from test_canal_playbook import _gs, brain

from catanbot import moves as mv
from catanbot.board import Harbor, ekey, hex_corners
from catanbot.state import Knight

C = hex_corners(0, -1)


def _idle_knight(hand, near=True):
    gs = _gs()
    me = gs.me()
    me.cities.add(C[0])
    me.roads.add(ekey(C[0], C[1]))
    me.knights[C[1]] = Knight(1, False)
    me.hand.update(hand)
    gs.barbarian_pos = gs.barbarian_len - 2 if near else 0
    return gs


def _trades(plan):
    return [(a.give, a.get) for a in plan if a.kind == "bank_trade"]


def test_no_harbour_never_plans_3_to_1():
    gs = _idle_knight({"lumber": 3, "brick": 1})
    assert mv.bank_ratio(gs, "lumber") == 4
    plan = brain()._plan_turn(gs)
    assert not any(g.get("lumber") == 3 for g, _ in _trades(plan)), plan
    assert "activate" not in [a.kind for a in plan]          # holds: no route to grain this turn
    gs4 = _idle_knight({"lumber": 4, "brick": 1})
    plan4 = brain()._plan_turn(gs4)
    assert ({"lumber": 4}, {"grain": 1}) in _trades(plan4), plan4
    assert "activate" in [a.kind for a in plan4]


def test_seen_bank_ratios_override_a_wrong_harbour_guess():
    gs = _idle_knight({"lumber": 3, "brick": 3})
    gs.board.harbors.append(Harbor(ekey(C[0], C[5]), "3:1"))   # the board read puts a 3:1 on our city
    br = brain()
    assert br.bank_ratio(gs, "lumber") == 3                      # the guess, before the dialog is seen
    br.ratio_seen.update({"lumber": 4})                          # the dialog: lumber 4
    assert br.bank_ratio(gs, "lumber") == 4
    assert br.bank_ratio(gs, "brick") == 4                       # one 4:1 seen: no 3:1 harbour at all
    src = br.ratio_source(gs, "lumber")
    assert src[1] == "bank dialog" and src[2]
    plan = br._plan_turn(gs)
    assert not any(n == 3 for g, _ in _trades(plan) for n in g.values()), plan
    # the progress cards read the same truth through the state
    assert mv.bank_ratio(gs, "brick") == 4


def test_merchant_fleet_saves_the_city():
    # the improvements are paid for already: before the save-the-city rule the Fleet found no use
    gs = _idle_knight({"lumber": 2, "brick": 1, "paper": 1, "coin": 1, "cloth": 1})
    gs.me().progress_hand = ["merchant_fleet"]
    br = brain()
    routes = br.trade_routes(gs, "grain")
    assert routes[0]["how"] == "play merchant fleet" and routes[0]["give"] == {"lumber": 2}, routes
    play = br.progress_play(gs, "main")
    assert play == ("merchant_fleet", {"resource": "lumber"}), play
    br.start_fleet("lumber")
    gs.me().progress_hand = []
    plan = br._plan_turn(gs)
    assert ({"lumber": 2}, {"grain": 1}) in _trades(plan), plan
    assert "activate" in [a.kind for a in plan], plan


def test_routes_list_future_sources_and_players_last():
    gs = _idle_knight({"lumber": 1, "brick": 1})
    gs.me().improvements["science"] = 3
    hows = [r["how"] for r in mv.trade_routes(gs, "grain")]
    assert hows[0] == "player trade"                              # the only route usable now
    assert any(h.startswith("aqueduct") for h in hows)
    gs.me().improvements["trade"] = 3
    gs.me().hand.update({"cloth": 2})
    r = mv.trade_routes(gs, "grain")[0]
    assert r["how"] == "trading house" and r["give"] == {"cloth": 2}
