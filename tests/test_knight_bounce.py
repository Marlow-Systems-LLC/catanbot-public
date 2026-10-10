"""Owner 2026-10-04: knights steal and bounce the robber round the table; stand a knight
down near an attack when only a rival loses a city."""
from catanbot.board import hex_corners
from catanbot.sim import Sim
from catanbot.state import Knight


def _land(b):
    return [h for h in b.hexes.values() if h.is_land and h.terrain not in ("desert", "gold") and h.number
            and not h.barren]


def _sim(bounce=True, sabotage=False):
    sim = Sim("great_canal", 4, seed=7)
    gs = sim.gs
    for p in gs.players:
        p.settlements.clear(); p.cities.clear(); p.metropolises.clear(); p.knights.clear()
        p.roads.clear(); p.ships.clear(); p.hand = {"wool": 2}; p.vp_visible = 9
    for br in sim.brains:
        br.knight_bounce, br.barb_sabotage = bounce, sabotage
    sim.attacks = 1
    gs.barbarian_pos = 0
    return sim


def _setup_two_hops(sim):
    """Our knight A next to the robber hex H1; our knight B next to H2, where a rival has cards."""
    b = sim.gs.board
    hexes = _land(b)
    h1 = hexes[0]
    a = next(iter(hex_corners(*h1.coord)))
    h2 = next(h for h in hexes if h is not h1 and a not in hex_corners(*h.coord)
              and not set(hex_corners(*h.coord)) & set(hex_corners(*h1.coord)))
    c2 = sorted(hex_corners(*h2.coord))
    bv, rv = c2[0], c2[3]                                  # opposite corners of H2
    me, rival = sim.gs.players[0], sim.gs.players[1]
    me.knights[a] = Knight(level=1, active=True)
    me.knights[bv] = Knight(level=1, active=True)
    rival.settlements.add(rv)
    rival.special_vp = 5                                   # above the friendly-robber 4 VP
    b.robber = h1.coord
    return me, rival, h2


def test_bounce_chases_twice_and_steals():
    sim = _sim()
    me, rival, h2 = _setup_two_hops(sim)
    sim.chase_robber(0)
    assert sim.chases == 2
    assert all(not k.active and k.acted for k in me.knights.values())
    assert sum(rival.hand.values()) < 2 and sum(me.hand.values()) > 2


def test_old_play_never_chases_just_to_steal():
    sim = _sim(bounce=False)
    _setup_two_hops(sim)
    sim.chase_robber(0)
    assert sim.chases == 0


def test_bounce_holds_back_near_attack_without_grain_or_warlord():
    sim = _sim()
    me, rival, _ = _setup_two_hops(sim)
    city = next(v for v in sim.gs.board.vertices if v not in me.knights and sim.gs.owner_at(v) is None)
    me.cities.add(city)                                    # a city at risk: the knights must defend
    sim.gs.barbarian_pos = sim.gs.barbarian_len - 1
    sim.chase_robber(0)
    assert sim.chases == 0


def test_sabotage_drops_a_knight_when_only_a_rival_loses():
    sim = _sim(bounce=False, sabotage=True)
    gs = sim.gs
    me, rival = gs.players[0], gs.players[1]
    free = [v for v in gs.board.vertices]
    me.knights[free[0]] = Knight(level=2, active=True)
    me.cities |= {free[10], free[12]}                     # 2 cities: a knight short either way
    rival.cities.add(free[20])
    gs.barbarian_pos = gs.barbarian_len - 1
    me.knights[free[0]].active = False                    # the candidate stands down
    # we keep strength 0 vs rival 0 -> both at the bottom: not allowed
    assert not sim._can_drop(0, True, True)
    me.knights[free[30]] = Knight(level=1, active=True)    # 1 strength vs 3 cities: lost, but we 1 > rival 0
    assert sim._can_drop(0, True, True)
    assert not sim._can_drop(0, True, False)               # old play: defence first
