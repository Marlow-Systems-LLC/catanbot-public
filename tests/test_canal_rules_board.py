"""Great Canal board rules (docs/audit/rules_diff.md rows 7, 13, 18/19, 22/23, 24, 25, 26, 27, 35)
and the Merchant hook (PlayerState.merchant_hex)."""
import copy

from catanbot import moves as mv
from catanbot.board import hex_corners, real_canal_board
from catanbot.brain import Action, Brain
from catanbot.rules import MODES
from catanbot.sim import Sim
from catanbot.state import GameState, Knight, PlayerState, new_hand


def _gs(seed=7):
    b = real_canal_board(seed)
    ps = [PlayerState("a", is_me=True), PlayerState("b"), PlayerState("c")]
    for p in ps:
        p.hand = new_hand()
    return GameState(board=b, mode=copy.copy(MODES["great_canal"]), players=ps, vp_target=18)


def _small_harbour(b):
    hb = next(x for x in b.harbors if x.kind != "3:1")
    return hb.kind, hb.edge[0]


def _main_coast_with_sea_edges(gs, n=2):
    b = gs.board
    for v in sorted(b.vertices):
        if gs.can_settle(v) and b.is_coastal_vertex(v):
            sea = [e for e in b.vertex_edges(v) if b.is_sea_edge(e)]
            if len(sea) >= n:
                return v, sea
    raise AssertionError("no coastal corner")


# ---- row 25: no settlements on small islands ------------------------------------------
def test_no_settlement_on_any_small_island_corner():
    gs = _gs()
    b = gs.board
    small = [v for v in b.vertices if b.is_land_vertex(v) and b.is_small_island_vertex(v)]
    assert small
    assert not any(gs.can_settle(v) for v in small)
    assert not any(b.is_small_island_vertex(v) for v in gs.legal_settlement_spots(gs.me(), setup=True))
    # home-island corners stay legal
    assert any(gs.can_settle(v) for v in b.vertices if b.is_land_vertex(v) and not b.is_small_island_vertex(v))


# ---- row 35: piece limits -------------------------------------------------------------
def _rich(p):
    for r in p.hand:
        p.hand[r] = 20


def test_settlement_and_city_caps_counted_apart():
    gs = _gs()
    me = gs.me()
    _rich(me)
    spots = [v for v in gs.board.vertices if gs.can_settle(v)]
    v0 = spots[0]
    me.settlements = {v0}
    me.roads = set(gs.board.vertex_edges(v0)[:1])
    assert any(m.kind == "city" for m in mv.build_moves(gs))
    me.cities = set(spots[1:5])                          # 4 cities: no 5th
    assert not any(m.kind == "city" for m in mv.build_moves(gs))
    me.cities = set()
    me.settlements = set(spots[:5])                      # 5 settlements: no 6th, cities still fine
    kinds = {m.kind for m in mv.build_moves(gs)}
    assert "settlement" not in kinds and "city" in kinds
    # a city only on our own settlement
    assert all(m.target in me.settlements for m in mv.build_moves(gs) if m.kind == "city")


def test_road_and_ship_caps():
    gs = _gs()
    me = gs.me()
    v, sea = _main_coast_with_sea_edges(gs, 1)
    me.settlements = {v}
    assert gs.legal_road_edges(me) and gs.legal_road_edges(me, ship=True)
    far = [e for e in gs.board.edges if v not in e]
    me.roads = set(far[:15])
    me.ships = set(far[15:30])
    assert gs.legal_road_edges(me) == []
    assert gs.legal_road_edges(me, ship=True) == []


# ---- row 26: no branching ship lines --------------------------------------------------
def test_ship_line_cannot_branch():
    gs = _gs()
    b = gs.board
    me = gs.me()
    v, _ = _main_coast_with_sea_edges(gs, 1)
    me.settlements = {v}
    # build a 2-ship line out to sea, then check every corner on it
    e1 = next(e for e in gs.legal_road_edges(me, ship=True) if v in e)
    me.ships = {e1}
    w = e1[0] if e1[1] == v else e1[1]
    e2 = next(e for e in gs.legal_road_edges(me, ship=True) if w in e and v not in e)
    me.ships.add(e2)
    x = e2[0] if e2[1] == w else e2[1]
    legal = gs.legal_road_edges(me, ship=True)
    # nothing more at w (it already has 2 ships: a third would branch)
    assert not any(w in e for e in legal)
    # the open end x extends
    assert any(x in e for e in legal)
    # every legal new ship keeps every corner at <= 2 own ships
    for e in legal:
        net = me.ships | {e}
        for c in e:
            assert c == v or sum(1 for y in b.vertex_edges(c) if y in net) <= 2


def test_building_starts_a_ship_line_on_every_sea_edge():
    """Seafarers: a coastal building may start a line on each of its sea edges; only an open
    corner (no own building) is limited to 2 own ships."""
    gs = _gs()
    b, me = gs.board, gs.me()
    v = max((v for v in b.vertices if gs.can_settle(v)),
            key=lambda v: sum(1 for e in b.vertex_edges(v) if b.is_sea_edge(e)))
    sea = [e for e in b.vertex_edges(v) if b.is_sea_edge(e)]
    assert len(sea) >= 2
    me.settlements = {v}
    me.ships = set(sea[:-1])
    assert sea[-1] in gs.legal_road_edges(me, ship=True)


# ---- row 27: ship move ----------------------------------------------------------------
def _line(gs):
    me = gs.me()
    v, _ = _main_coast_with_sea_edges(gs, 1)
    me.settlements = {v}
    e1 = next(e for e in gs.legal_road_edges(me, ship=True) if v in e)
    me.ships = {e1}
    w = e1[0] if e1[1] == v else e1[1]
    e2 = next(e for e in gs.legal_road_edges(me, ship=True) if w in e and v not in e)
    me.ships.add(e2)
    return v, e1, e2


def test_ship_move_only_open_end_once_not_new():
    gs = _gs()
    me = gs.me()
    v, e1, e2 = _line(gs)
    ms = mv.ship_moves(gs)
    assert ms and {m.target[0] for m in ms} == {e2}       # only the last ship of the line
    for m in ms:
        assert m.target[1] != e2 and m.target[1] not in me.ships
    # built this turn: can't move
    me.ships_new = {e2}
    assert mv.ship_moves(gs) == []
    me.ships_new = set()
    # a knight at the open end holds the ship
    x = e2[1] if e2[0] in e1 else e2[0]
    me.knights[x] = Knight(1)
    assert all(m.target[0] != e2 for m in mv.ship_moves(gs))
    me.knights.clear()
    # one move a turn
    brain = Brain(setup_weights={})
    g2 = brain._apply(gs, Action("move_ship", ms[0].target))
    assert ms[0].target[1] in g2.me().ships and e2 not in g2.me().ships
    assert mv.ship_moves(g2) == []


def test_ship_move_in_sim_is_legal_and_resets():
    sim = Sim("great_canal", 4, seed=11)
    sim.setup()
    for t in range(60):
        sim.gs.turn = t
        sim.turn(t % 4)
        for p in sim.gs.players:
            assert len(p.ships) <= sim.mode.max_ships
    assert all(isinstance(p.ship_moved, bool) for p in sim.gs.players)


# ---- row 7: setup ships ---------------------------------------------------------------
def test_setup_coastal_piece_gets_ship():
    gs = _gs()
    v, sea = _main_coast_with_sea_edges(gs, 1)
    gs.me().settlements = {v}
    e = Brain(setup_weights={}).choose_setup_road(gs, v)
    assert gs.board.is_sea_edge(e)


def test_sim_setup_places_ships_on_coast():
    total = 0
    for seed in range(4):
        sim = Sim("great_canal", 4, seed=seed)
        sim.setup()
        b = sim.gs.board
        for i, moves in sim.setup_moves.items():
            for kind, v, e, t in moves:
                if b.is_coastal_vertex(v) and b.is_sea_edge(e):
                    assert t == "ship"
                    total += 1
            assert any(b.is_coastal_vertex(v) for _, v, _, _ in moves)
    assert total > 0


# ---- rows 22/23: merchant knight harbour ----------------------------------------------
def test_harbour_knight_only_on_small_island_and_survives_canal():
    gs = _gs()
    b = gs.board
    me = gs.me()
    kind, hv = _small_harbour(b)
    assert b.is_small_island_vertex(hv)
    me.knights[hv] = Knight(1, active=False)            # no activation needed
    assert mv.bank_ratio(gs, kind) == 2
    gs.canal_progress = gs.canal_total                  # canal done: merchant job kept
    assert mv.bank_ratio(gs, kind) == 2
    me.knights.clear()
    main3 = next(x.edge[0] for x in b.harbors if x.kind == "3:1")
    me.knights[main3] = Knight(1, active=True)          # home-island harbour corner: nothing
    assert mv.bank_ratio(gs, "wool") == 4
    brain = Brain(setup_weights={})
    me.knights.clear()
    assert brain._prospector_value(gs, me, main3) == 0.0
    assert brain._prospector_value(gs, me, hv) > 0


# ---- rows 18/19: gold on 8, max 1 per player ------------------------------------------
def test_gold_only_on_eight_one_card_per_player():
    sim = Sim("great_canal", 3, seed=5)
    gs = sim.gs
    b = gs.board
    golds = [h for h in b.hexes.values() if h.terrain == "gold"]
    assert len(golds) == 2
    p = gs.players[0]
    gs.board.robber = None
    p.knights = {hex_corners(*golds[0].coord)[0]: Knight(1), hex_corners(*golds[1].coord)[0]: Knight(1)}
    for h in golds:
        h.number = 8 if h is golds[0] else 6        # a roll of 8 pays, whatever chip
    n0 = sum(p.hand.values())
    sim.produce(6)
    assert sum(p.hand.values()) == n0
    sim.produce(8)
    assert sum(p.hand.values()) == n0 + 1               # 2 gold knights: still 1 card
    gs.canal_progress = gs.canal_total
    sim.produce(8)
    assert sum(p.hand.values()) == n0 + 1               # canal done: no gold
    brain = Brain(setup_weights={})
    gs.canal_progress = 0
    gs.players[0].is_me = True
    other_gold = hex_corners(*golds[1].coord)[3]
    p.knights = {hex_corners(*golds[0].coord)[0]: Knight(1)}
    assert brain._prospector_value(gs, p, other_gold) == 0.0   # a second gold knight adds no gold


# ---- row 24: small-island knights don't defend ----------------------------------------
def test_small_island_knights_never_defend():
    gs = _gs()
    b = gs.board
    me = gs.me()
    small = next(v for v in b.vertices if b.is_land_vertex(v) and b.is_small_island_vertex(v))
    home = next(v for v in b.vertices if gs.can_settle(v))
    me.knights = {small: Knight(3, active=True), home: Knight(1, active=True)}
    assert gs.defence_strength(me) == 1


# ---- row 13: dig after move_knight ----------------------------------------------------
def test_apply_digs_after_move_knight():
    gs = _gs()
    b = gs.board
    me = gs.me()
    c = sorted(b.canal_hexes)[0]
    cs = hex_corners(*c)
    o = gs.players[1]
    src = next(v for v in b.vertices if v not in cs and b.is_land_vertex(v))
    # with two active knights already on the hex, any move_knight apply triggers the dig check
    me.knights = {cs[0]: Knight(1, active=True), cs[1]: Knight(1, active=True), src: Knight(1, active=True)}
    o.knights = {}
    gs.canal_dug = set()
    g = Brain(setup_weights={})._apply(gs, Action("move_knight", (src, cs[2])))
    assert c in g.canal_dug and g.me().canal_chips >= 2


# ---- Merchant hook --------------------------------------------------------------------
def test_merchant_hex_rate_and_vp():
    sim = Sim("great_canal", 3, seed=2)
    gs = sim.gs
    p = gs.players[0]
    for q in gs.players:
        q.is_me = q is p
    c, h = next((c, h) for c, h in gs.board.hexes.items() if h.terrain == "pasture")
    assert p.merchant_hex is None
    vp0 = sim.vp(p)
    assert mv.bank_ratio(gs, "wool") == 4
    p.merchant_hex = c
    assert mv.bank_ratio(gs, "wool") == 2 and mv.bank_ratio(gs, "ore") == 4
    assert sim.vp(p) == vp0 + 1


# ---- Alchemist pre-roll hook (sim_cards.pre_roll) in Sim.turn --------------------------
def test_held_alchemist_sets_the_dice(monkeypatch):
    from catanbot import progress_cards as pc
    sim = Sim("great_canal", 3, seed=4)
    sim.setup()
    p = sim.gs.players[0]
    p.progress_hand = ["alchemist"]
    monkeypatch.setattr(pc, "choose_alchemist", lambda gs, me: {"dice": (4, 2), "sum": 6})
    rolls = []
    monkeypatch.setattr(sim, "produce", lambda roll: rolls.append(roll))
    sim.turn(0)
    assert rolls == [6] and "alchemist" not in p.progress_hand
