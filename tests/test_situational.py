"""Owner 2026-10-04: per-turn selector -- BLOCK the advancing rival, RACE when we lead."""
from catanbot import situational as st
from catanbot.board import hex_corners
from catanbot.sim import Sim
from catanbot.state import Knight


def _land(b):
    return [h for h in b.hexes.values() if h.is_land and h.terrain not in ("desert", "gold") and h.number
            and not h.barren]


def _sim(vps=(9, 9, 9, 9)):
    sim = Sim("great_canal", 4, seed=7)
    gs = sim.gs
    for p, vp in zip(gs.players, vps):
        p.settlements.clear(); p.cities.clear(); p.metropolises.clear(); p.knights.clear()
        p.roads.clear(); p.ships.clear(); p.hand = {}
        p.special_vp = vp
    sim.attacks = 1
    gs.barbarian_pos = 0
    sim.refresh()
    for br in sim.brains:
        br.situational = True
        br.plan_missing = lambda gs: {}          # no plan shortfall: only the owner order counts
    return sim


def _begin(sim, i=0):
    gs = sim.as_me(i)
    return sim.brains[i].situational_begin(gs), gs


def test_block_when_rival_three_ahead():
    sim = _sim((9, 12, 9, 8))
    o, _ = _begin(sim)
    assert o["stance"] == "block" and o["focus"] == 1
    assert o["road_cuts"] and o["knight_bounce"]


def test_race_when_we_lead():
    sim = _sim((12, 9, 8, 7))
    o, _ = _begin(sim)
    assert o["stance"] == "race" and o["focus"] is None and o["spare"] == 3


def test_focus_picks_advancing_rival():
    sim = _sim((9, 8, 10, 13))
    o, gs = _begin(sim)
    assert o["focus"] == 3
    br = sim.brains[0]
    assert br.threat(gs) is gs.players[3]
    assert not br.evaluate_offer(gs, "P3", {"wool": 1}, {"ore": 1})   # no trades to the focus


def test_rival_near_win_is_focus():
    sim = _sim((9, 16, 10, 10))
    o, _ = _begin(sim)
    assert o["stance"] == "block" and o["focus"] == 1


def test_flag_off_identical():
    sim = _sim((9, 12, 9, 8))
    br = sim.brains[0]
    br.situational = False
    keys = st.FLAG_KEYS + st.TARGET_KEYS
    before = {k: getattr(br, k) for k in keys}
    gs = sim.as_me(0)
    assert br.situational_begin(gs) is None
    br.plan_turn(gs)
    assert {k: getattr(br, k) for k in keys} == before
    assert st.prefer(br, gs, gs.opponents()) is None


def test_flags_restored_after_plan_turn():
    sim = _sim((9, 12, 9, 8))
    br = sim.brains[0]
    before = {k: getattr(br, k) for k in st.FLAG_KEYS}
    gs = sim.as_me(0)
    br.plan_turn(gs)
    assert {k: getattr(br, k) for k in st.FLAG_KEYS} == before
    assert br.focus == 1                         # the target stays for the round


def test_friendly_robber_still_respected():
    sim = _sim((1, 4, 2, 2))                    # rival 3 ahead but at 4 VP: protected
    o, gs = _begin(sim)
    assert o["focus"] == 1
    b = gs.board
    hexes = _land(b)
    h1, h2 = hexes[0], next(h for h in hexes[1:] if not set(hex_corners(*h.coord)) & set(hex_corners(*hexes[0].coord)))
    rival, other = gs.players[1], gs.players[2]
    rival.settlements.add(sorted(hex_corners(*h1.coord))[0]); rival.hand = {"grain": 3}
    other.settlements.add(sorted(hex_corners(*h2.coord))[0]); other.hand = {"wool": 1}
    other.special_vp = 5; rival.special_vp = 3; sim.refresh()   # rival shows 4 VP with its settlement
    gs = sim.as_me(0)
    opts = gs.friendly_robber_hexes([h1.coord, h2.coord], gs.me())
    hexc, victim = sim.brains[0].choose_robber(gs, opts)
    assert hexc != h1.coord and victim is not rival
    assert sim.brains[0].choose_victim(gs, ["P1", "P2"]) == "P2"


def test_race_robs_second_not_rich_last():
    sim = _sim((12, 9, 6, 5))
    _begin(sim)
    gs = sim.as_me(0)
    b = gs.board
    hexes = _land(b)
    h1 = hexes[0]
    h2 = next(h for h in hexes[1:] if not set(hex_corners(*h.coord)) & set(hex_corners(*h1.coord)))
    last, second = gs.players[3], gs.players[1]
    last.settlements.add(sorted(hex_corners(*h1.coord))[0]); last.hand = {"grain": 4, "ore": 4}
    second.settlements.add(sorted(hex_corners(*h2.coord))[0]); second.hand = {"wool": 2}
    hexc, victim = sim.brains[0].choose_robber(gs, [h1.coord, h2.coord])
    assert victim is second and hexc == h2.coord


def test_race_last_placed_only_target_is_robbed():
    sim = _sim((12, 9, 6, 5))
    _begin(sim)
    gs = sim.as_me(0)
    hexes = _land(gs.board)
    last = gs.players[3]
    last.settlements.add(sorted(hex_corners(*hexes[0].coord))[0]); last.hand = {"wool": 2}
    hexc, victim = sim.brains[0].choose_robber(gs, [hexes[0].coord])
    assert victim is last
    assert sim.brains[0].choose_victim(gs, ["P3"]) == "P3"


def test_grain_holder_robbed_before_big_hand():
    sim = _sim((12, 9, 8, 5))
    _begin(sim)
    gs = sim.as_me(0)
    gs.players[1].hand = {"grain": 1, "brick": 2}       # 1 grain in 3 cards
    gs.players[2].hand = {"wool": 6}                     # 6 cards, no grain
    assert sim.brains[0].choose_victim(gs, ["P1", "P2"]) == "P1"


def test_needed_coin_beats_wool():
    sim = _sim((12, 9, 9, 5))
    _begin(sim)
    gs = sim.as_me(0)
    me = gs.me()
    me.cities.add(sorted(hex_corners(*_land(gs.board)[0].coord))[0])
    me.improvements = {"trade": 3, "politics": 0, "science": 3}   # next improvement: politics = coin
    gs.players[1].hand = {"wool": 2}
    gs.players[2].hand = {"coin": 2}
    assert st.next_improve_commodity(sim.brains[0], gs) == "coin"
    assert sim.brains[0].choose_victim(gs, ["P1", "P2"]) == "P2"


def test_bounce_still_max_steals():
    sim = _sim((12, 9, 8, 7))
    gs = sim.gs
    b = gs.board
    hexes = _land(b)
    h1 = hexes[0]
    a = next(iter(hex_corners(*h1.coord)))
    h2 = next(h for h in hexes if h is not h1 and a not in hex_corners(*h.coord)
              and not set(hex_corners(*h.coord)) & set(hex_corners(*h1.coord)))
    c2 = sorted(hex_corners(*h2.coord))
    me, rival = gs.players[0], gs.players[1]
    me.knights[a] = Knight(level=1, active=True)
    me.knights[c2[0]] = Knight(level=1, active=True)
    rival.settlements.add(c2[3]); rival.hand = {"wool": 2}
    b.robber = h1.coord
    sim.refresh()
    _begin(sim)
    assert sim.brains[0].knight_bounce            # RACE keeps robbing the most per round
    sim.chase_robber(0)
    assert sim.chases == 2
    assert sum(rival.hand.values()) < 2


def test_last_city_costs_more_than_second():
    sim = _sim()
    gs = sim.gs
    hexes = _land(gs.board)
    p = gs.players[1]
    c1 = sorted(hex_corners(*hexes[0].coord))[0]
    far = next(h for h in hexes[1:] if not set(hex_corners(*h.coord)) & set(hex_corners(*hexes[0].coord)))
    c2 = sorted(hex_corners(*far.coord))[0]
    p.cities |= {c1, c2}
    two = st.city_loss_cost(gs, 1)
    p.cities.discard(c2)
    one = st.city_loss_cost(gs, 1)
    assert one > two > 0
    p.cities.clear()
    assert st.city_loss_cost(gs, 1) == 0.0


def test_defend_overrides_block():
    sit = {"rival_to_win": 8, "rival_lead": 3, "rival_prob": 0.4, "my_prob": 0.2, "turns_to_attack": 2.0,
           "my_city_at_risk": True, "my_loss_cost": 4.0, "rival": 1, "last": 2,
           "rivals": [{"index": 1, "at_bottom": True, "loss_cost": 4.0}, {"index": 2, "at_bottom": False, "loss_cost": 0}]}
    o = st.choose(sit, {"pay_two_danger": 0.5})
    assert o["stance"] == "block" and not o["knight_bounce"] and not o["barb_sabotage"]
    sit["my_city_at_risk"] = False
    o = st.choose(sit, {"pay_two_danger": 0.5})
    assert o["barb_sabotage"]                     # the advancing rival's city is worth a stand-down
