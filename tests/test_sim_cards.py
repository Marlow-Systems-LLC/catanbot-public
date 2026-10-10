"""Real progress-card effects in the simulator (catanbot/sim_cards.py), one test per card."""
import pytest

from catanbot import progress_cards as pc
from catanbot import sim_cards
from catanbot.board import hex_corners
from catanbot.sim import Sim
from catanbot.state import Knight


@pytest.fixture
def sim():
    s = Sim(seed=1, n=4)
    s.setup()
    for p in s.gs.players:
        for r in p.hand:
            p.hand[r] = 0
    s.as_me(0)
    return s


def P(s, i):
    return s.gs.players[i]


def play(s, j, card):
    """Take the card into the hand and play it through Sim._play_progress (the real dispatch)."""
    s._play_progress(j, card)
    return getattr(s, "progress_played", {}).get(card, 0)


def adj_hexes(s, p, terrain):
    return {h.coord for v, _ in p.buildings() for h in s.gs.board.vertex_hexes(v) if h.terrain == terrain}


def free_corner(s, h):
    return next(v for v in hex_corners(*h.coord) if s.gs.can_settle(v))


# ---- science -------------------------------------------------------------------------------

def test_alchemist_only_before_roll(sim):
    me = P(sim, 0)
    me.hand.update({"brick": 4, "lumber": 4})        # over the hand limit: worth fixing the dice
    assert play(sim, 0, "alchemist") == 0
    assert me.progress_hand == ["alchemist"]          # after the roll it stays in hand
    dice = sim_cards.pre_roll(sim, 0)
    assert dice is not None and sum(dice) != 7 and all(1 <= d <= 6 for d in dice)
    assert me.progress_hand == [] and sim.progress_played["alchemist"] == 1


def test_crane_builds_improvement_one_cheaper(sim):
    me = P(sim, 0)
    me.improvements["science"] = 1
    me.hand["paper"] = 1                              # level 2 normally costs 2 paper
    assert play(sim, 0, "crane") == 1
    assert me.improvements["science"] == 2 and me.hand["paper"] == 0


def test_engineer_free_wall(sim):
    me = P(sim, 0)
    assert play(sim, 0, "engineer") == 1
    assert me.walls == set(me.cities)


def test_inventor_swaps_movable_tokens(sim):
    b = sim.gs.board
    before = {c: h.number for c, h in b.hexes.items()}
    assert play(sim, 0, "inventor") == 1
    changed = [c for c, h in b.hexes.items() if h.number != before[c]]
    assert len(changed) == 2
    for c in changed:
        h = b.hexes[c]
        assert before[c] in (3, 4, 5, 9, 10, 11) and h.terrain != "gold"   # barren ok (owner 2026-10-06)


def test_inventor_skips_gold_but_may_move_barren(sim):
    """Owner-confirmed 2026-10-06: barren farmland chips may be swapped while the canal is dug."""
    gs = sim.gs
    barren = next(h for h in gs.board.hexes.values() if h.barren and h.number in (3, 4, 5, 9, 10, 11))
    gold = next(h for h in gs.board.hexes.values() if h.terrain == "gold")
    assert pc.inventor_movable(gs, barren) and not pc.inventor_movable(gs, gold)


def test_irrigation_pays_barren_fields(sim):
    me = P(sim, 0)
    if not any(sim.gs.board.hexes[c].barren for c in adj_hexes(sim, me, "fields")):
        fld = next(h for h in sim.gs.board.hexes.values() if h.terrain == "fields" and h.barren
                   and any(sim.gs.can_settle(v) for v in hex_corners(*h.coord)))
        me.settlements.add(free_corner(sim, fld))
    fields = adj_hexes(sim, me, "fields")
    assert fields and any(sim.gs.board.hexes[c].barren for c in fields)
    assert play(sim, 0, "irrigation") == 1
    assert me.hand["grain"] == 2 * len(fields)        # per hex; cities don't double


def test_medicine_city_for_2_ore_1_grain(sim):
    me = P(sim, 0)
    me.hand.update({"ore": 2, "grain": 1})
    n_set, n_city = len(me.settlements), len(me.cities)
    assert play(sim, 0, "medicine") == 1
    assert (len(me.settlements), len(me.cities)) == (n_set - 1, n_city + 1)
    assert me.hand["ore"] == 0 and me.hand["grain"] == 0


def test_mining_two_ore_per_mountain(sim):
    me = P(sim, 0)
    if not adj_hexes(sim, me, "mountains"):
        mtn = next(h for h in sim.gs.board.hexes.values() if h.terrain == "mountains"
                   and any(sim.gs.can_settle(v) for v in hex_corners(*h.coord)))
        me.settlements.add(free_corner(sim, mtn))
    n = len(adj_hexes(sim, me, "mountains"))
    assert play(sim, 0, "mining") == 1
    assert me.hand["ore"] == 2 * n


def test_road_building_two_free_pieces(sim):
    me = P(sim, 0)
    before = len(me.roads) + len(me.ships)
    assert play(sim, 0, "road_building") == 1
    assert len(me.roads) + len(me.ships) == before + 2


def test_road_building_offers_ships(sim):
    me = P(sim, 0)
    me.roads.clear()
    me.roads.update({(("x", i), ("y", i)) for i in range(15)})   # road supply used up: ships only
    roads = set(me.roads)
    args = pc.choose_road_building(sim.as_me(0), me)
    assert args and len(args["ships"]) == 2 and set(args["ships"]) == set(args["edges"])
    assert me.roads == roads                          # the chooser leaves the pieces untouched
    assert play(sim, 0, "road_building") == 1
    assert len(me.ships) >= 2


def test_smith_promotes_two_knights(sim):
    me = P(sim, 0)
    vs = [v for e in me.roads for v in e if v not in me.settlements and v not in me.cities][:2]
    for v in vs:
        me.knights[v] = Knight(1, active=False)
    assert play(sim, 0, "smith") == 1
    assert sorted(k.level for k in me.knights.values()) == [2] * len(vs)


# ---- trade ---------------------------------------------------------------------------------

def test_commercial_harbor_resource_for_commodity(sim):
    me = P(sim, 0)
    me.hand["brick"] = 3
    P(sim, 1).hand["cloth"] = 1
    P(sim, 2).hand["coin"] = 2
    assert play(sim, 0, "commercial_harbor") == 1
    assert me.hand["cloth"] == 1 and me.hand["coin"] == 1 and me.hand["brick"] == 1
    assert P(sim, 1).hand["brick"] == 1 and P(sim, 2).hand["brick"] == 1
    assert sum(P(sim, 3).hand.values()) == 0          # no commodity: no trade


def test_master_merchant_takes_two_from_leader(sim):
    me, lead = P(sim, 0), P(sim, 1)
    lead.special_vp = 4
    lead.hand.update({"ore": 3, "wool": 1})
    assert play(sim, 0, "master_merchant") == 1
    assert sum(me.hand.values()) == 2 and sum(lead.hand.values()) == 2


def test_merchant_moves_to_last_player(sim):
    a, b = P(sim, 0), P(sim, 1)
    vp_a = sim.vp(a)
    assert play(sim, 0, "merchant") == 1
    h = a.merchant_hex
    assert h is not None
    assert h in {x.coord for v, _ in a.buildings() for x in sim.gs.board.vertex_hexes(v)}
    assert sim.vp(a) == vp_a + 1                      # Sim.vp counts the merchant
    play(sim, 1, "merchant")
    assert a.merchant_hex is None and b.merchant_hex is not None
    assert sim.vp(a) == vp_a


def test_merchant_fleet_trades_two_for_one(sim):
    me = P(sim, 0)
    me.hand["wool"] = 5
    assert play(sim, 0, "merchant_fleet") == 1
    assert me.hand["wool"] == 1 and sum(me.hand.values()) == 3


def test_resource_monopoly_two_each(sim):
    me = P(sim, 0)
    for i in (1, 2, 3):
        P(sim, i).hand["ore"] = 3
    assert play(sim, 0, "resource_monopoly") == 1
    assert me.hand["ore"] == 6 and all(P(sim, i).hand["ore"] == 1 for i in (1, 2, 3))


def test_trade_monopoly_one_each(sim):
    me = P(sim, 0)
    for i in (1, 2, 3):
        P(sim, i).hand["paper"] = 2
    assert play(sim, 0, "trade_monopoly") == 1
    assert me.hand["paper"] == 3 and all(P(sim, i).hand["paper"] == 1 for i in (1, 2, 3))


# ---- politics ------------------------------------------------------------------------------

def test_bishop_locked_then_friendly(sim):
    me = P(sim, 0)
    for i in (1, 3):
        P(sim, i).special_vp = 4                      # 8 VP: fair game
        P(sim, i).hand["ore"] = 3
    P(sim, 2).hand["ore"] = 3                         # 4 VP: protected by the friendly robber
    assert play(sim, 0, "bishop") == 0                # robber locked before the first attack
    assert me.progress_hand == ["bishop"] and sim.gs.board.robber is None
    me.progress_hand.clear()
    sim.attacks = 1
    assert play(sim, 0, "bishop") == 1
    b = sim.gs.board
    assert b.robber is not None
    assert P(sim, 2).hand["ore"] == 3
    hit = [i for i in (1, 3) if any(h.coord == b.robber for v, _ in P(sim, i).buildings()
                                    for h in b.vertex_hexes(v))]
    assert me.hand["ore"] == len(hit)


def test_deserter_swaps_a_knight(sim):
    me, opp = P(sim, 0), P(sim, 1)
    v = next(v for e in opp.roads for v in e if not sim.gs.occupied(v))
    opp.knights[v] = Knight(2, active=True)
    assert play(sim, 0, "deserter") == 1
    assert not opp.knights
    assert [(k.level, k.active) for k in me.knights.values()] == [(2, True)]


def test_diplomat_removes_leaders_open_road(sim):
    lead = P(sim, 1)
    lead.special_vp = 4
    lead.has_longest_road = True
    n = len(lead.roads)
    assert play(sim, 0, "diplomat") == 1
    assert len(lead.roads) == n - 1


def test_intrigue_displaces_knight_on_our_road(sim):
    me, opp = P(sim, 0), P(sim, 1)
    v = next(v for e in me.roads for v in e if not sim.gs.occupied(v))
    opp.knights[v] = Knight(1, active=True)
    assert play(sim, 0, "intrigue") == 1
    assert v not in opp.knights


def test_saboteur_halves_richer_hands(sim):
    me, opp = P(sim, 0), P(sim, 1)
    opp.special_vp = 2
    opp.hand.update({"ore": 5, "wool": 4})
    me.special_vp = -1                                # we trail everyone
    P(sim, 2).hand["brick"] = 3
    assert play(sim, 0, "saboteur") == 1
    assert sum(opp.hand.values()) == 5 and P(sim, 2).hand["brick"] == 2


def test_spy_steals_a_progress_card(sim):
    me, opp = P(sim, 0), P(sim, 1)
    opp.progress_hand[:] = ["alchemist", "smith"]
    assert play(sim, 0, "spy") == 1
    # owner 2026-10-06: nothing playable now (no knights for the Smith) -> the card worth most
    # to hold; the Alchemist waits in hand for our next roll
    assert me.progress_hand == ["alchemist"] and opp.progress_hand == ["smith"]


def test_spy_plays_the_stolen_card_this_turn(sim):
    """owner 2026-10-06: a card we can use now beats one for later, and it is played at once."""
    me, opp = P(sim, 0), P(sim, 1)
    opp.progress_hand[:] = ["alchemist", "mining"]
    assert adj_hexes(sim, me, "mountains") or pytest.skip("seat 0 has no mountains on this seed")
    ore = me.hand.get("ore", 0)
    assert play(sim, 0, "spy") == 1
    assert opp.progress_hand == ["alchemist"] and me.progress_hand == []
    assert sim.progress_played.get("mining") == 1 and me.hand["ore"] > ore


def test_wedding_two_from_each_leader(sim):
    me = P(sim, 0)
    P(sim, 1).special_vp = 2
    P(sim, 1).hand.update({"ore": 4})
    P(sim, 2).special_vp = 1
    P(sim, 2).hand.update({"wool": 1})                # only 1 card: gives that one
    assert play(sim, 0, "wedding") == 1
    assert me.hand["ore"] == 2 and me.hand["wool"] == 1


def test_every_non_vp_card_has_an_effect():
    real = {"warlord"} | pc.VP_CARDS
    assert set(sim_cards.EFFECTS) == set(pc.CARDS) - real
