"""Great Canal playbook behaviours (docs/canal_playbook.md)."""
import pytest
import copy

from catanbot.board import Board, Harbor, Hex, hex_corners, ekey
from catanbot.brain import Action, Brain
from catanbot.learning import Model
from catanbot.rules import MODES
from catanbot.state import GameState, Knight, PlayerState


def _board():
    b = Board()
    b.add(Hex(0, 0, "canal"))
    terr = ["forest", "hills", "pasture", "mountains", "fields", "gold"]
    nums = [6, 5, 9, 10, 4, 8]
    for (dq, dr), t, n in zip([(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)], terr, nums):
        b.add(Hex(dq, dr, t, n, barren=(t == "fields")))
    return b


def _gs(**kw):
    b = _board()
    me = PlayerState("bot", is_me=True, vp_visible=5)
    a = PlayerState("leader", vp_visible=12)
    c = PlayerState("weak", vp_visible=4)
    gs = GameState(board=b, mode=copy.copy(MODES["great_canal"]), players=[me, a, c], vp_target=19)
    for k, v in kw.items():
        setattr(gs, k, v)
    return gs


def brain():
    return Brain(Model(path=None))


C = hex_corners(0, 0)


def test_second_own_knight_on_canal_hex_beats_shared():
    gs = _gs()
    me, leader = gs.players[0], gs.players[1]
    me.knights[C[0]] = Knight(1, True)
    br = brain()
    own = br._canal_pair_value(gs, Action("knight", C[3]))
    # a different hex state: only the leader's active knight is waiting
    gs2 = _gs()
    gs2.me().cities.add(hex_corners(2, 0)[0])     # with a city: deny the leader
    gs2.players[1].knights[C[0]] = Knight(1, True)
    shared = br._canal_pair_value(gs2, Action("activate", C[3]))
    assert own > 0.02 > shared
    assert shared < 0           # never hand the leader a chip


def test_own_pair_activation_digs_for_two_vp():
    gs = _gs()
    me = gs.me()
    me.knights[C[0]] = Knight(1, True)
    me.knights[C[3]] = Knight(1, False)
    g = brain()._apply(gs, Action("activate", C[3]))
    assert g.me().canal_chips == 2 and g.me().vp_visible == 7


def _islet(gs):
    """A one-hex small island off the test board (audit row 22: merchant harbours are there)."""
    gs.board.add(Hex(3, 0, "desert"))
    gs.board._main = None
    S = hex_corners(3, 0)
    assert gs.board.is_small_island_vertex(S[0])
    return S


def test_trader_knight_gives_harbour_ratio():
    """Almanac: only a knight on a SMALL-island harbour corner trades there (row 22)."""
    gs = _gs()
    gs.board.harbors.append(Harbor(ekey(C[1], C[2]), "3:1"))
    br = brain()
    assert br.bank_ratio(gs, "ore") == 4
    gs.me().knights[C[1]] = Knight(1, False)     # home-island harbour corner: nothing
    assert br.bank_ratio(gs, "ore") == 4
    S = _islet(gs)
    gs.board.harbors.append(Harbor(ekey(S[0], S[1]), "3:1"))
    gs.me().knights[S[0]] = Knight(1, False)     # needn't be active
    assert br.bank_ratio(gs, "ore") == 3


def test_three_to_one_harbour_knight_valued_until_owned():
    gs = _gs()
    gs.board.harbors.append(Harbor(ekey(C[1], C[2]), "3:1"))
    br = brain()
    assert br._prospector_value(gs, gs.me(), C[1]) == 0.0      # home island: no merchant job
    S = _islet(gs)
    gs.board.harbors.append(Harbor(ekey(S[0], S[1]), "3:1"))
    assert br._prospector_value(gs, gs.me(), S[0]) >= 0.04
    gs.me().knights[S[1]] = Knight(1, False)
    assert br._prospector_value(gs, gs.me(), S[0]) < 0.04


def test_gold_prospector_counts_once():
    gs = _gs()
    gold = gs.board.hexes[(0, 1)]
    gc = [v for v in hex_corners(0, 1) if v not in C]
    br = brain()
    assert br._prospector_value(gs, gs.me(), gc[0]) > 0
    gs.me().knights[gc[1]] = Knight(1, False)
    assert br._prospector_value(gs, gs.me(), gc[0]) == 0     # 2 knights still give 1
    assert gold.number == 8


def test_robber_skips_barren_farmland_and_hits_leader():
    gs = _gs()
    gs.mode.friendly_robber = False     # "weak" (4 VP, lowest) would be protected otherwise
    me, leader, weak = gs.players
    fields_c = hex_corners(-1, 1)
    hills_c = hex_corners(1, -1)
    leader.settlements.add(fields_c[4])
    leader.cities.add(hills_c[1])
    weak.settlements.add(hills_c[4])
    leader.hand_count = 3
    hx, victim = brain().choose_robber(gs)
    assert hx == (1, -1) and victim is leader
    # barren farmland (grain locked) is worth less than an empty hex
    b = brain()
    b.robber_steal_first = False    # live 2026-10-09 19:44: with steal-first on, the leader's hex wins (a card)
    hx, _ = b.choose_robber(gs, [(-1, 1), (0, -1)])
    assert hx == (0, -1)


def test_robber_avoids_a_hex_a_rival_knight_guards():
    # owner 2026-10-06: an active knight next to the robber chases it off at once
    gs = _gs()
    gs.mode.friendly_robber = False
    me, leader, _ = gs.players
    hills_c = hex_corners(1, -1)
    leader.cities.add(hills_c[1])
    leader.settlements.add(next(v for v in hex_corners(0, -1) if v not in hills_c))   # a weaker 2nd hex
    leader.hand_count = 3
    hx, _ = brain().choose_robber(gs)
    assert hx == (1, -1)
    leader.knights[hills_c[3]] = Knight(1, True)
    hx, _ = brain().choose_robber(gs)
    assert hx != (1, -1)
    leader.knights[hills_c[3]].active = False     # an inactive knight can't chase
    hx, _ = brain().choose_robber(gs)
    assert hx == (1, -1)


def test_wool_wanted_when_knight_owed():
    gs = _gs()
    me = gs.me()
    me.cities.add(hex_corners(1, 0)[1])
    br = brain()
    v_gap = br.resource_value(gs, me)["wool"]
    me.knights[hex_corners(1, 0)[2]] = Knight(1, True)
    assert v_gap > br.resource_value(gs, me)["wool"]


def test_propose_asks_for_knight_card_first():
    gs = _gs()
    me = gs.me()
    me.cities.add(hex_corners(1, 0)[1])
    # one card from a city (ore) and one from a knight (wool)
    me.hand = {"ore": 3, "grain": 2, "wool": 0, "brick": 2, "lumber": 0}
    me.hand["ore"] = 2
    give, get, _ = brain().propose_trade(gs)
    assert get == {"wool": 1}


def test_refuse_trade_with_player_near_win():
    gs = _gs()
    me = gs.me()
    me.cities.add(hex_corners(1, 0)[1])
    c1 = hex_corners(1, 0)
    me.roads.add(ekey(c1[1], c1[2]))     # knight needs a spot on our road
    me.hand = {"brick": 3, "ore": 1, "grain": 1}
    br = brain()
    assert br.evaluate_offer(gs, "weak", {"brick": 1}, {"wool": 1}) is True   # knight + activate
    gs.players[2].vp_visible = 17
    assert br.evaluate_offer(gs, "weak", {"brick": 1}, {"wool": 1}) is False


def test_city_without_knight_penalised_more_when_barbarians_close():
    br = brain()
    br.spend_grain = br.improve_ladder = br.city_first = 0   # isolate the knight penalty
    gs = _gs()
    me = gs.me()
    s = hex_corners(1, 0)[1]
    me.settlements.add(s)
    me.hand = {"grain": 2, "ore": 3}
    spot = 0.004 * br.city_keep_value(gs, s)        # the production bonus, same both times
    gs.barbarian_pos = 0
    far = br.action_value(gs, Action("city", s)) - br.my_prob(br._apply(gs, Action("city", s))) - spot
    gs.barbarian_pos = 6
    near = br.action_value(gs, Action("city", s)) - br.my_prob(br._apply(gs, Action("city", s))) - spot
    assert near < far < 0


def test_spend_commodities_near_discard_limit():
    gs = _gs()
    me = gs.me()
    me.cities.add(hex_corners(1, 0)[1])
    me.hand = {"paper": 3, "coin": 2, "cloth": 1, "brick": 1}
    plan = brain().plan_turn(gs)
    assert any(a.kind == "improve" and a.target == "science" for a in plan)


def test_city_goes_near_gold_on_game4_board():
    """Owner 2026-10-02 game 4: the city at (3,5) "could have been lower to give access to the
    gold mine" -- (3,7) has the same mountains 8 + forest 11 and is 2 steps from the gold 8."""
    import cv2
    from catanbot.runner import BOARD_ROI
    from catanbot.vision.board_reader import read_board
    img = cv2.imread("data/calib/gc_game4_board.png")
    if img is None:
        pytest.skip("live frame not in this checkout")
    b = read_board(img, BOARD_ROI, canal=True).board
    me = PlayerState(name="marlow", color="magenta", is_me=True)
    others = [PlayerState(name=n, color=c) for n, c in (("A", "red"), ("B", "green"), ("C", "purple"))]
    gs = GameState(board=b, mode=copy.copy(MODES["great_canal"]), players=[me] + others, vp_target=19)
    me.settlements.add((-3, 1)) if isinstance(me.settlements, set) else me.settlements.append((-3, 1))
    assert brain().choose_setup_settlement(gs) == (3, 7)


def test_idle_knight_asks_grain_two_for_one_near_attack():
    gs = _gs(barbarian_pos=5)
    me = gs.me()
    me.cities.add(C[0]); me.knights[C[2]] = Knight()
    me.hand.update({"wool": 2, "ore": 1, "brick": 1})
    give, get, partners = brain().propose_trade(gs)
    assert get == {"grain": 1} and sum(give.values()) == 2 and "leader" not in partners
    gs.barbarian_pos = 1                       # far away: 1-for-1 is enough
    give, get, _ = brain().propose_trade(gs)
    assert get == {"grain": 1} and sum(give.values()) == 1


def test_first_settlement_takes_good_harbour_when_rule_on():
    """Owner 2026-10-03: first settlement on a harbour with good numbers, then get the rest."""
    import cv2
    from catanbot.runner import BOARD_ROI
    from catanbot.vision.board_reader import read_board
    img = cv2.imread("data/calib/gc_game4_board.png")
    if img is None:
        pytest.skip("live frame not in this checkout")
    b = read_board(img, BOARD_ROI, canal=True).board
    me = PlayerState(name="marlow", is_me=True)
    gs = GameState(board=b, mode=copy.copy(MODES["great_canal"]),
                   players=[me] + [PlayerState(name=n) for n in "ABC"], vp_target=19)
    br = brain(); br.first_harbour = 10
    br.setup_set = False                       # the per-spot scorer this rule lives in
    v = br.choose_setup_settlement(gs)
    assert b.harbor_at(v) == "3:1"
    # holding that 3:1, another 3:1 adds nothing for the later picks
    me.settlements.add(v)
    other = next(w for h in b.harbors if h.kind == "3:1" for w in h.edge if w not in b.neighbors(v) and w != v)
    assert br._spot_extras(gs, other, {}) < 8


def test_gold_corner_is_not_a_spot_and_its_knight_is_no_defender():
    """Owner 2026-10-03: nothing can be built on a gold hex corner; a knight there collects the
    gold but never counts against the barbarians (it only chases the robber off that gold)."""
    gs = _gs()
    me = gs.me()
    gold = next(h for h in gs.board.hexes.values() if h.terrain == "gold")
    corners = set(hex_corners(*gold.coord))
    assert not corners & set(gs.legal_settlement_spots(me, setup=True))
    me.cities.add(C[0])
    on_gold = next(iter(corners - set(C)))
    me.knights[on_gold] = Knight(active=True)
    assert gs.defence_strength(me) == 0 and Brain.defence_gap(gs, me) == 1
    main = next(v for v in C if v not in corners and v != C[0])
    me.knights[main] = Knight(active=True)
    assert gs.defence_strength(me) == 1 and Brain.defence_gap(gs, me) == 0


def test_city_goes_on_the_producing_settlement_not_the_gold_coast():
    """Owner 2026-10-03: "worst choice for the city, the one that had no resources or
    developments" -- a gold/sea corner pays a building nothing in Great Canal."""
    gs = _gs()
    b, me = gs.board, gs.me()
    gold = min(hex_corners(0, 1), key=lambda v: len(b.vertex_hexes(v)))
    good = max(hex_corners(1, 0), key=lambda v: sum(h.terrain != "canal" for h in b.vertex_hexes(v)))
    me.settlements |= {gold, good}
    me.hand = {"grain": 2, "ore": 3}
    br = brain()
    assert br.action_value(gs, Action("city", good)) > br.action_value(gs, Action("city", gold))
    assert br.city_keep_value(gs, gold) < br.city_keep_value(gs, good)


def test_two_of_a_kind_asks_the_table_for_the_third():
    """Owner 2026-10-04: no grain, 2 of a kind at a 3:1 -> trade 1 spare card for the 3rd."""
    gs = _gs()
    me = gs.me()
    me.hand = {"ore": 2, "lumber": 1}
    br = brain()
    br.bank_ratio = lambda g, r: 3 if r == "ore" else 4
    give, get, partners = br._third_of_set(gs, br.resource_value(gs, me), None)
    assert get == {"ore": 1} and give == {"lumber": 1} and partners
    me.hand["grain"] = 1
    assert br._third_of_set(gs, br.resource_value(gs, me), None) is None
    br.set_for_grain = False
    me.hand["grain"] = 0
    assert br._third_of_set(gs, br.resource_value(gs, me), None) is None


def test_activation_that_makes_us_sole_top_defender_gets_a_bonus():
    from catanbot.state import Knight
    gs = _gs()
    me, a = gs.players[0], gs.players[1]
    me.cities.add(C[0]); a.cities.add(C[3])
    me.knights[C[1]] = Knight(1, True)
    me.knights[C[5]] = Knight(1, False)
    a.knights[C[2]] = Knight(1, True)                 # tie at 1: nobody gets the VP
    gs.barbarian_pos = 6
    br = brain()
    g = copy.deepcopy(gs)
    g.me().knights[C[5]].active = True
    assert br._defender_bonus(gs, g) > 0
    assert br._defender_bonus(g, g) == 0              # no change, no bonus
    far = copy.deepcopy(gs); far.barbarian_pos = 0
    assert br._defender_bonus(gs, g) > br._defender_bonus(far, g)


def test_knight_build_always_gets_a_nudge():
    gs = _gs()
    br = brain()
    me = gs.me()
    me.settlements.add(C[0]); me.roads.add(ekey(C[0], C[1]))
    me.hand = {"wool": 1, "ore": 1}
    v1 = br.action_value(gs, Action("knight", C[1]))
    br.knight_always = 0
    assert v1 - br.action_value(gs, Action("knight", C[1])) == pytest.approx(0.01)


def test_city_opens_a_wall_slot_when_every_city_is_walled():
    """Owner 2026-10-04: walls stand on cities -- more cities, more cards we can hold."""
    gs = _gs()
    me = gs.me()
    s = hex_corners(1, 0)[1]
    me.settlements.add(s)
    me.cities.add(C[0]); me.walls.add(C[0])
    me.hand = {"grain": 2, "ore": 3}
    br = brain()
    walled = br.action_value(gs, Action("city", s))
    br.city_wall_slot = 0
    assert walled - br.action_value(gs, Action("city", s)) == pytest.approx(0.01)
    me.walls.clear()                                   # a free wall slot already: no extra
    br.city_wall_slot = 0.01
    a = br.action_value(gs, Action("city", s))
    br.city_wall_slot = 0
    assert a == pytest.approx(br.action_value(gs, Action("city", s)))


def test_development_ladder_lowest_track_then_lock_or_take_the_metropolis():
    """Owner 2026-10-04: every track to 3 (lowest first), then 4 and 5; 5 locks the metropolis."""
    gs = _gs()
    me, a = gs.players[0], gs.players[1]
    br = brain()
    w = br.improve_ladder
    me.improvements.update({"science": 2, "trade": 0, "politics": 1})
    assert br._ladder_value(gs, "trade", 1) == pytest.approx(0.5 * w)       # lowest track
    assert br._ladder_value(gs, "science", 3) == 0
    me.improvements.update({"science": 4, "trade": 3, "politics": 3})
    gs.metropolis_owner = {"science": (me.name, C[0])}
    a.improvements["science"] = 2
    assert br._ladder_value(gs, "science", 5) == 0                         # nobody close
    a.improvements["science"] = 4
    assert br._ladder_value(gs, "science", 5) == pytest.approx(2 * w)      # lock it now
    gs.metropolis_owner = {"science": (a.name, C[3])}
    assert br._ladder_value(gs, "science", 5) == pytest.approx(w)          # take theirs
    a.improvements["science"] = 5
    assert br._ladder_value(gs, "science", 5) == 0                         # theirs is locked


def test_share_canal_with_anyone_when_city_out_of_reach():
    """Owner 2026-10-05: no city -> share digs with anyone (leader too), else the others take them."""
    br = brain()
    for who in (1, 2):
        gs = _gs()
        gs.players[who].knights[C[0]] = Knight(1, True)
        gs.me().knights[C[3]] = Knight(1, False)
        assert br._canal_pair_value(gs, Action("activate", C[3])) >= 0.03
        gs.me().cities.add(hex_corners(2, 0)[0])
        v = br._canal_pair_value(gs, Action("activate", C[3]))
        assert (v < 0) if who == 1 else (0 < v < 0.03)


def test_city_hosts_the_next_metropolis_when_every_city_has_one():
    """Owner 2026-10-09 live 21:12: "for the third metropolis, marlow needs a third city"."""
    gs = _gs()
    me = gs.me()
    s = hex_corners(1, 0)[1]
    me.settlements.add(s)
    me.cities.update({C[0], C[3]}); me.metropolises.update({C[0], C[3]})
    gs.metropolis_owner = {"politics": (me.name, C[0]), "science": (me.name, C[3])}
    me.improvements.update({"politics": 4, "science": 4, "trade": 3})
    me.hand = {"grain": 2, "ore": 3}
    br = brain()
    full = br.action_value(gs, Action("city", s))
    br.city_metro_slot = 0
    assert full - br.action_value(gs, Action("city", s)) == pytest.approx(0.04)
    br.city_metro_slot = 0.04
    me.metropolises.discard(C[3])                      # a free city already: no extra
    gs.metropolis_owner = {"politics": (me.name, C[0])}
    a = br.action_value(gs, Action("city", s))
    br.city_metro_slot = 0
    assert a == pytest.approx(br.action_value(gs, Action("city", s)))
