"""Solo 2026-10-06 (01:17, 6 VP, last): the only city lost to the first attack with the knight
inactive and no grain asked for; a strong knight "promoted" again 36 times and traded for."""
from catanbot import moves as mv
from catanbot import runner as R
from catanbot.brain import Brain
from catanbot.learning import Model
from catanbot.sim import Sim
from catanbot.state import Knight


def _gs():
    sim = Sim("great_canal", 4, seed=7)
    sim.setup()
    gs = sim.gs
    me = gs.me()
    me.knights = {}
    return gs, me, Brain(Model(path=None))


def test_failed_build_leaves_the_move_list():
    gs, me, _ = _gs()
    v = next(iter(me.settlements | me.cities))
    me.knights = {v: Knight(level=1, active=True)}
    me.hand = {"wool": 2, "ore": 2}
    assert mv.Move("promote", v) in mv.build_moves(gs)
    gs.blocked = {("promote", v)}
    assert mv.Move("promote", v) not in mv.build_moves(gs)


def test_strong_knight_is_not_promoted_without_politics_3():
    gs, me, _ = _gs()
    v = next(iter(me.settlements | me.cities))
    me.knights = {v: Knight(level=2, active=True)}
    me.hand = {"wool": 2, "ore": 2}
    me.improvements["politics"] = 0
    assert mv.Move("promote", v) not in mv.build_moves(gs)


def test_landed_promotion_is_remembered():
    r = R.Runner.__new__(R.Runner)
    r.brain = Brain(Model(path=None))
    r._record_build("knight", (1, -1))
    r._record_build("promote", (1, -1))
    assert r._knight_levels[(1, -1)] == 2
    r._record_build("promote", (1, -1)); r._record_build("promote", (1, -1))
    assert r._knight_levels[(1, -1)] == 3


def test_grain_rush_when_barbarians_close():
    gs, me, b = _gs()
    if not me.cities:
        me.cities = {next(iter(me.settlements))}
    v = next(iter(me.cities))
    me.knights = {v: Knight(level=1, active=False)}
    me.hand = {"brick": 2, "wool": 1}
    gs.barbarian_pos = 5
    assert b.grain_rush(gs)
    gs.barbarian_pos = 2
    assert not b.grain_rush(gs)                       # far away: normal trading
    gs.barbarian_pos = 5
    me.hand["grain"] = 1
    assert not b.grain_rush(gs)                       # grain in hand: activate, no rush


def test_live_offer_asks_grain_two_for_one():
    gs, me, b = _gs()
    if not me.cities:
        me.cities = {next(iter(me.settlements))}
    me.knights = {next(iter(me.cities)): Knight(level=1, active=False)}
    me.hand = {"brick": 2, "wool": 1}
    gs.barbarian_pos = 5
    r = R.Runner.__new__(R.Runner)
    r.brain, r.note = b, (lambda *a, **k: None)
    a = r._want_trade(gs)
    assert a.get == {"grain": 1} and sum(a.give.values()) == 2


def test_grain_refused_completes_a_bank_set_two_for_two():
    gs, me, b = _gs()
    if not me.cities:
        me.cities = {next(iter(me.settlements))}
    me.knights = {next(iter(me.cities)): Knight(level=1, active=False)}
    me.hand = {"brick": 1, "wool": 1, "ore": 1}
    gs.barbarian_pos = 5
    r = R.Runner.__new__(R.Runner)
    r.brain, r.note, r.cfg = b, (lambda *a, **k: None), {"harbor_3to1": True}
    r._cfg_3to1 = lambda: True
    r._offers_made = {("want", "grain")}               # the table refused grain
    a = r._want_trade(gs)
    (res, n), = a.get.items()
    assert res != "grain" and n == 2 and sum(a.give.values()) == 2 and res not in a.give


def test_second_knight_waits_while_the_first_is_idle():
    """owner 03:13: knight at (1,-1) built, then a 2nd with the last wool+ore -- keep them to trade."""
    gs, me, b = _gs()
    if not me.cities:
        me.cities = {next(iter(me.settlements))}
    v = next(iter(me.cities))
    me.knights = {v: Knight(level=1, active=False)}
    me.hand = {"brick": 2, "wool": 1, "ore": 1}
    gs.barbarian_pos = 1
    assert all(a.kind != "knight" for a in b.plan_turn(gs))
    me.knights = {}                                   # no knight yet: the first one is built
    assert any(a.kind == "knight" for a in b.plan_turn(gs))


def test_three_paper_take_science_3_while_there_is_time():
    """owner 03:18: sold 3 paper for grain with the ship at 3/7 instead of science level 3."""
    gs, me, b = _gs()
    if not me.cities:
        me.cities = {next(iter(me.settlements))}
    vs = list(me.settlements | me.cities)
    me.knights = {vs[0]: Knight(level=1, active=False), vs[1]: Knight(level=1, active=False)}
    me.improvements["science"] = 2
    me.hand = {"brick": 1, "lumber": 2, "ore": 1, "paper": 3}
    gs.barbarian_pos = 3
    plan = b.plan_turn(gs)
    assert plan[0].kind == "improve" and not any(a.kind == "bank_trade" for a in plan)
    gs.barbarian_pos = 5                              # ship next to the coast: the city first
    assert b.plan_turn(gs)[0].kind == "bank_trade"


def test_dug_canal_survives_a_restart(tmp_path):
    """owner 03:22: after a runner restart the knight that had already dug was activated again."""
    r = R.Runner.__new__(R.Runner)
    r.FLAGS = tmp_path / "game_flags.json"
    r.mode = None
    r._improve_levels = lambda: {}
    r._canal_seen = {(1, -1), (2, -2)}
    r._my_cities, r._my_blds = {(5, -1)}, {(2, -2), (5, -1)}
    r._save_game_flags()
    r2 = R.Runner.__new__(R.Runner)
    r2.FLAGS, r2.mode = r.FLAGS, None
    r2._load_game_flags()
    assert r2._canal_seen == {(1, -1), (2, -2)}
    assert r2._my_cities == {(5, -1)} and r2._my_blds == {(2, -2), (5, -1)}   # not the (2,-2) guess


def test_three_paper_ask_for_the_wood_that_banks_the_4th():
    """owner 03:23: 3 paper at science 3 -- ask for the 3rd wood, bank 3 wood -> paper, science 4."""
    gs, me, b = _gs()
    if not me.cities:
        me.cities = {next(iter(me.settlements))}
    me.improvements["science"] = 3
    me.hand = {"lumber": 2, "paper": 3, "brick": 1, "wool": 1}
    r = R.Runner.__new__(R.Runner)
    r.brain, r.note, r.cfg = b, (lambda *a, **k: None), {"harbor_3to1": True}
    r._cfg_3to1 = lambda: True
    b.table_has = lambda gs, res: True
    a = r._complete_bank_set(gs, [])
    assert a is not None and a.get == {"lumber": 1} and "paper" not in a.give
    me.hand["lumber"] = 3                             # the set is complete: the plan banks it
    me.knights = {next(iter(me.cities)): Knight(level=1, active=True)}
    plan = b.plan_turn(gs)
    assert [x.kind for x in plan[:2]] == ["bank_trade", "improve"] and plan[0].get == {"paper": 1}


def test_no_city_keeps_ore_and_skips_knights():
    """owner 03:3x: city lost -> banked 3 ore for grain, activated a knight it no longer needed,
    and asked for sheep every round for more knights."""
    gs, me, b = _gs()
    me.cities = set()
    v = next(iter(me.settlements))
    me.knights = {v: Knight(level=1, active=False)}
    gs.canal_dug = set(gs.board.canal_hexes)          # no dig left for this knight
    gs.canal_progress = len(gs.canal_dug) - 1
    o = gs.opponents()[0]
    o.knights = {next(iter(o.settlements | o.cities)): Knight(level=2, active=True)}   # no defender point either
    me.hand = {"ore": 3, "brick": 1, "wool": 1}
    gs.barbarian_pos = 2
    t = b._trade_targets(gs)
    assert t[0] == "city" and "knight" not in t and "activate" not in t
    assert b._grain_first_trade(gs) is None or "ore" not in b._grain_first_trade(gs).give
    me.hand = {"grain": 1, "ore": 2}
    assert all(a.kind not in ("activate", "knight") for a in b.plan_turn(gs))


def test_road_to_a_rival_canal_knight_then_push_it_off():
    """owner 03:3x: build a road toward the other bot's knight on the canal and push it off."""
    import copy
    from catanbot.board import ekey, hex_corners
    from catanbot.brain import Action
    gs, me, b = _gs()
    bd = gs.board
    o = gs.opponents()[0]
    for p in gs.players:
        p.roads, p.knights = set(), {}
    occupied = {v for p in gs.players for v, _ in p.buildings()}
    spot = None
    for h in bd.canal_hexes:
        for c in hex_corners(*h):
            if c not in bd.vertices or c in occupied or not bd.is_land_vertex(c):
                continue
            for e in bd.vertex_edges(c):
                n = e[0] if e[1] == c else e[1]
                far = [f for f in bd.vertex_edges(n) if c not in f]
                if n not in occupied and bd.is_land_vertex(n) and bd.is_land_edge(e) and far:
                    spot = (c, n, e, far[0])
                    break
            if spot:
                break
        if spot:
            break
    c, n, e, back = spot
    gs.canal_dug = set()
    o.knights = {c: Knight(level=1, active=True)}
    me.knights = {n: Knight(level=2, active=True)}
    me.roads = {back}
    g = copy.deepcopy(gs)
    g.me().roads.add(e)
    assert b._contest_reach_value(gs, g) > 0              # the road reaches the rival's canal knight
    me.roads.add(e)
    acts = b.candidate_actions(gs)
    assert Action("move_knight", (n, c)) in acts         # and the stronger knight may push it off


def test_move_knight_clicks_knight_ring_then_destination(tmp_path, monkeypatch):
    import types
    import numpy as np
    from catanbot import brain as B
    r = R.Runner.__new__(R.Runner)
    clicks = []
    px = {(1, 1): (100, 100), (2, 2): (200, 200)}
    r.br = types.SimpleNamespace(lattice=types.SimpleNamespace(vertex_px=lambda v: px[v]))
    r.io = types.SimpleNamespace(snapshot=lambda: np.zeros((40, 40, 3), np.uint8), ocr=lambda img: [])
    r.click = lambda xy, why="", *a: clicks.append((tuple(xy), why))
    r.note = lambda *a, **k: None
    r._close_leftover = lambda *a: None
    r._popup_options = lambda img, near=None: [{"x": 150, "y": 90, "kind": "move", "colour": "green"}]
    r.probe = types.SimpleNamespace(tried=lambda *a: None, result=lambda *a: None)
    r._save_game_flags = lambda: None
    r._my_knights, r._knight_levels = {(1, 1): True}, {(1, 1): 2}
    monkeypatch.setattr(R, "ROOT", tmp_path)
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    assert r.tap_move_knight(B.Action("move_knight", ((1, 1), (2, 2))))
    assert [c[0] for c in clicks] == [(100, 100), (150, 90), (200, 200)]
    assert r._my_knights == {(2, 2): False} and r._knight_levels == {(2, 2): 2}


def test_city_before_settlement_when_both_are_possible():
    """owner 03:5x: strive for more cities first, then settlements."""
    from catanbot.board import ekey
    gs, me, b = _gs()
    bd = gs.board
    found = None
    for s in sorted(me.settlements):
        for e1 in bd.vertex_edges(s):
            a = e1[0] if e1[1] == s else e1[1]
            for e2 in bd.vertex_edges(a):
                c = e2[0] if e2[1] == a else e2[1]
                if c == s or not (bd.is_land_edge(e1) and bd.is_land_edge(e2)):
                    continue
                me.roads |= {e1, e2}
                if c in gs.legal_settlement_spots(me):
                    found = (s, c)
                    break
                me.roads -= {e1, e2}
            if found:
                break
        if found:
            break
    assert found
    me.knights = {v: Knight(level=1, active=True) for v in me.cities}
    me.hand = {"grain": 3, "ore": 3, "brick": 1, "lumber": 1, "wool": 1}
    assert b._trade_targets(gs)[:2] == ["city", "settlement"]
    kinds = [a.kind for a in b.plan_turn(gs) if a.kind in ("city", "settlement")]
    assert kinds and kinds[0] == "city"


def test_trade_goes_before_the_build_when_it_spares_the_builds_cards():
    # owner 2026-10-06: "it built a ship and then offered trade ... it should first do all trades
    # and then build" -- hand lumber 1, wool 2, grain 1: ship (lumber+wool) and offer wool -> ore
    from catanbot.brain import Action
    gs, me, b = _gs()
    me.hand = {"lumber": 1, "wool": 2, "grain": 1}
    r = R.Runner.__new__(R.Runner)
    ship = [Action("ship", None)]
    assert r._trade_spares_builds(gs, Action("player_trade", give={"wool": 1}, get={"ore": 1}), ship)
    # the offer would give the ship's only lumber: build first
    assert not r._trade_spares_builds(gs, Action("player_trade", give={"lumber": 1}, get={"ore": 1}), ship)
    # a cost the runner can't count (improve): build first
    assert not r._trade_spares_builds(gs, Action("player_trade", give={"wool": 1}, get={"ore": 1}),
                                      [Action("improve", "science")])


def test_ask_a_player_for_the_commodity_that_buys_the_next_improvement():
    # owner 2026-10-06: the dice didn't bring the paper -- trade for it, improvements are worth it
    gs, me, b = _gs()
    if not me.cities:
        me.cities = {next(iter(me.settlements))}
    me.improvements = {"trade": 2, "politics": 2, "science": 1}
    me.hand = {"paper": 1, "brick": 2}               # science 2 costs 2 paper: 1 short (others 3 short)
    r = R.Runner.__new__(R.Runner)
    r.brain, r.note = b, (lambda *a, **k: None)
    r.cfg = {}
    a = r._improve_want(gs)
    assert a is not None and a.get == {"paper": 1} and a.give == {"brick": 1}
    me.hand = {"brick": 2}
    me.improvements["science"] = 2                   # science 3 = 3 paper, none held: too far to ask
    assert r._improve_want(gs) is None
    me.improvements = {"trade": 0, "politics": 2, "science": 1}
    me.hand = {"paper": 1, "brick": 2}               # ladder: trade 0 -> 1 (1 cloth) comes first
    assert r._improve_want(gs).get == {"cloth": 1}


def test_wool_refused_for_ore_then_offer_the_wood():
    # owner 2026-10-06 live: hand lumber 1, wool 2, 0 knights for 1 city. Asked wool -> ore (all
    # declined), then built a ship. "it could trade out the wood it has for the stone it needs"
    from catanbot.brain import Action
    gs, me, b = _gs()
    if not me.cities:
        me.cities = {next(iter(me.settlements))}
    me.knights = {}
    me.hand = {"lumber": 1, "wool": 2}
    r = R.Runner.__new__(R.Runner)
    r.brain, r.note = b, (lambda *a, **k: None)
    r.cfg = {}
    b.table_has = lambda g, res: True
    kinds = ["knight", "city", "settlement", "road", "ship"]
    made = {("want", "ore"), ("ask", "ore", "wool")}
    a = r._retry_top_ask(gs, kinds, made)
    assert a is not None and a.give == {"lumber": 1} and a.get == {"ore": 1} and a.note == "for:knight"
    # the knight outranks the ship: the offer goes before building the ship with that lumber
    b._trade_targets = lambda g: kinds
    assert r._trade_beats_builds(gs, a, [Action("ship", None)])
    assert not r._trade_spares_builds(gs, a, [Action("ship", None)])
    # wood refused too: nothing left to offer
    made.add(("ask", "ore", "lumber"))
    assert r._retry_top_ask(gs, kinds, made) is None
    # first ask not made yet: the normal ask path goes first
    assert r._retry_top_ask(gs, kinds, set()) is None


def test_progress_deck_follows_the_stage_of_the_game():
    # owner 2026-10-06: defender tie -> pick the deck by what we need and the stage, and never
    # an empty deck
    gs, me, b = _gs()
    for p in gs.players:
        p.vp_visible = 5
    me.improvements = {"trade": 1, "politics": 0, "science": 1}
    me.hand = {"grain": 1, "lumber": 2}               # nothing 1 short, grain held
    assert b.choose_progress_deck(gs) == "science"     # building up
    me.hand = {"lumber": 2}                           # canal, no grain: trade cards get it
    assert b.choose_progress_deck(gs) == "trade"
    gs.players[1].vp_visible = gs.target - 2          # a rival about to win: politics
    assert b.choose_progress_deck(gs) == "politics"
    assert b.choose_progress_deck(gs, {"trade": 4, "politics": 0, "science": 5}) == "trade"
    gs.players[1].vp_visible = 5
    me.hand = {"grain": 1, "lumber": 2}
    me.improvements = {"trade": 4, "politics": 3, "science": 3}
    assert b.choose_progress_deck(gs) == "trade"       # metropolis race on trade: its own deck


def test_merchant_fleet_played_when_2_to_1_finishes_a_build():
    # owner 2026-10-06 "bot doesn't know how to use the merchant fleet" (owner played it on sheep)
    from catanbot import progress_cards as pc
    gs, me, b = _gs()
    me.hand = {"wool": 5, "grain": 1, "ore": 2}       # city = 2 grain + 3 ore: 2 short
    fb = pc.fleet_build(gs, me)
    assert fb is not None and fb[0] == "wool"
    assert pc.choose_merchant_fleet(gs, me)["resource"] == "wool"
    me.hand = {"wool": 2, "grain": 1}                 # 2 wool buy 1 card: nothing completes
    assert pc.fleet_build(gs, me) is None or pc.fleet_build(gs, me)[0] == "wool"


def test_merchant_goes_where_we_get_most_and_brain_trades_2_to_1():
    from catanbot import progress_cards as pc
    from catanbot.board import TERRAIN_RES
    gs, me, b = _gs()
    hexes = [h for h in pc._my_hexes(gs, me) if h.terrain in TERRAIN_RES and h.terrain != "gold"]
    vals = {h.coord: pc.merchant_hex_value(gs, me, h) for h in hexes}
    pick = pc.choose_merchant(gs, me)["hex"]
    assert vals[pick] == max(vals.values())
    gs.board.robber = pick                            # owner: the robber's hex is still fine
    assert pc.choose_merchant(gs, me)["hex"] == pick
    # brain: the Merchant's resource is 2:1 even after the old dialog said 4:1
    res = TERRAIN_RES[gs.board.hexes[pick].terrain]
    b.ratio_seen = {res: 4}
    b.ratio_seen.pop(res)
    b.merchant_res = res
    assert b.bank_ratio(gs, res) == 2
    b.ratio_seen = {res: 4}                           # a later dialog read (Merchant taken) wins
    assert b.bank_ratio(gs, res) == 4


def test_resource_monopoly_takes_what_every_rival_holds_two_of_first():
    # owner 2026-10-06: "request whatever it knows it can get two of from each player, then the
    # most of what it can get, focusing on grain or resources it can trade"
    from catanbot import progress_cards as pc
    gs, me, b = _gs()
    opps = [p for p in gs.players if p is not me]
    hands = {opps[0].name: {"ore": 2, "lumber": 5}, opps[1].name: {"ore": 2}, opps[2].name: {"ore": 3}}
    gs.est_hand = lambda p: {k: float(v) for k, v in hands.get(p.name, {}).items()}
    gs.plan_missing = {}
    assert pc.choose_resource_monopoly(gs, me) == {"resource": "ore"}     # 2 from each, for sure
    hands[opps[2].name] = {"lumber": 3}                # ore no longer with everyone: the most cards
    hands[opps[1].name] = {"lumber": 2}
    assert pc.choose_resource_monopoly(gs, me) == {"resource": "lumber"}
    # grain in the canal leans the tie: 2 grain + 2 grain vs 2 wool + 2 wool
    hands = {opps[0].name: {"grain": 2, "wool": 2}, opps[1].name: {"grain": 2, "wool": 2}}
    assert pc.choose_resource_monopoly(gs, me) == {"resource": "grain"}
    # a card we trade 2:1 (Merchant on its hex) leans the same way
    hands = {opps[0].name: {"brick": 2, "wool": 2}, opps[1].name: {"brick": 2, "wool": 2}}
    gs.fleet_resource = "wool"
    assert pc.choose_resource_monopoly(gs, me) == {"resource": "wool"}


def test_trade_monopoly_takes_the_commodity_everyone_holds():
    from catanbot import progress_cards as pc
    gs, me, b = _gs()
    opps = [p for p in gs.players if p is not me]
    hands = {opps[0].name: {"coin": 1, "paper": 3}, opps[1].name: {"coin": 1}, opps[2].name: {"coin": 1}}
    gs.est_hand = lambda p: {k: float(v) for k, v in hands.get(p.name, {}).items()}
    assert pc.choose_trade_monopoly(gs, me) == {"commodity": "coin"}


def test_medicine_in_hand_trades_toward_the_cheap_city():
    # cards report 2026-10-06: Medicine sat in hand; the planner never traded toward 2 ore + 1 grain
    gs, me, b = _gs()
    me.hand = {"ore": 1, "grain": 1, "lumber": 3}
    me.knights = {v: Knight(level=1, active=True) for v in list(me.settlements)[:1]}   # knight covered
    me.progress_hand = []
    assert "medicine" not in b._trade_targets(gs)
    me.progress_hand = ["medicine"]
    kinds = b._trade_targets(gs)
    assert "medicine" in kinds and ("city" not in kinds or kinds.index("medicine") < kinds.index("city"))
    assert b.plan_missing(gs) == {"ore": 1}
    me.hand = {"ore": 3, "grain": 2}                    # the full city is paid: no detour
    assert "medicine" not in b._trade_targets(gs)


def test_merchant_2_to_1_is_spent_toward_a_target():
    # owner 2026-10-06: Merchant on wood with 5 wood + 1 ore and no city -- it kept all the wood.
    # 2:1 lumber buys brick + wool toward the settlement (grain left short)
    gs, me, b = _gs()
    for v in list(me.cities):
        me.cities.discard(v)
        me.settlements.add(v)
    me.hand = {"lumber": 5, "ore": 1}
    b.merchant_res = "lumber"
    out = b._spend_toward_target(gs)
    assert out and all(t.give == {"lumber": 2} for t in out)
    assert {r for t in out for r in t.get} <= {"brick", "wool", "grain", "ore"}
    b.merchant_res = None                       # 4:1 lumber: still nothing worth banking
    assert not b._spend_toward_target(gs) or all(t.give.get("lumber", 0) >= 3 for t in b._spend_toward_target(gs))
