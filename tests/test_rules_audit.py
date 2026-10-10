"""Rules-audit fixes in the simulator (docs/great_canal_rules_research.md): robber lock,
defender tie, pillaged walls, per-track metropolises, aqueduct, small-island knights, gold
prospector stop, discarded progress cards, knights chasing the robber."""
from catanbot.board import Board, Hex
from catanbot.brain import Action
from catanbot.rules import MODES
from catanbot.sim import Sim
from catanbot.state import GameState, Knight, PlayerState


def _sim(seed=5):
    sim = Sim("great_canal", seed=seed)
    sim.setup()
    return sim


def _field_spots(sim, n):
    """n free main-island corners off gold and away from buildings."""
    gs, b = sim.gs, sim.gs.board
    out = []
    for v in sorted(b.vertices):
        if gs.can_settle(v) and not b.is_small_island_vertex(v) and not gs.occupied(v) \
                and not any(v in b.neighbors(w) for w in out) and v not in out:
            out.append(v)
        if len(out) == n:
            return out
    raise AssertionError("no free spots")


def test_robber_locked_until_first_attack():
    sim = _sim()
    gs = sim.gs
    land = next(c for c, h in gs.board.hexes.items() if h.number and not h.barren)
    gs.board.robber = land
    hands = [dict(p.hand) for p in gs.players]
    sim.seven(1)
    assert gs.board.robber == land                  # no move, no steal
    assert [p.hand for p in gs.players] == hands    # nobody over the limit: nothing changes
    sim.attacks = 1
    sim.seven(1)
    assert gs.board.robber != land


def test_seven_still_discards_before_first_attack():
    sim = _sim()
    p = sim.gs.players[0]
    for k in p.hand:
        p.hand[k] = 0
    p.hand.update(brick=6, lumber=6)
    sim.seven(1)
    assert sum(p.hand.values()) == 6


def test_defender_tie_draws_progress_card_no_vp():
    sim = _sim()
    gs = sim.gs
    a, b = gs.players[:2]
    va, vb = _field_spots(sim, 2)
    cities = sum(len(p.cities) for p in gs.players)
    a.knights[va] = Knight(cities // 2 + 1, True)     # strength level > 3 is fine for the sum
    b.knights[vb] = Knight(cities // 2 + 1, True)
    a.progress_hand = ["merchant", "spy", "crane", "smith"]
    sim.decks = {"science": ["mining"], "politics": ["warlord"], "trade": ["merchant"]}
    gs.current = 2                                  # someone else rolled the ship
    gs.barbarian_pos = gs.barbarian_len - 1
    sim.barbarians()
    assert all(p.defender_vp == 0 for p in gs.players)
    assert sim.progress_drawn == {a.name: 1, b.name: 1}
    assert len(a.progress_hand) == 4 and sim.progress_discarded == 1   # off-turn limit
    assert len(b.progress_hand) == 1


def test_defender_tie_roller_keeps_fifth_card():
    sim = _sim()
    gs = sim.gs
    a, b = gs.players[:2]
    va, vb = _field_spots(sim, 2)
    cities = sum(len(p.cities) for p in gs.players)
    a.knights[va] = Knight(cities // 2 + 1, True)
    b.knights[vb] = Knight(cities // 2 + 1, True)
    a.progress_hand = ["merchant", "spy", "crane", "smith"]
    sim.decks = {"science": ["mining"], "politics": ["warlord"], "trade": ["merchant"]}
    gs.current = 0                                  # a rolled the ship
    gs.barbarian_pos = gs.barbarian_len - 1
    sim.barbarians()
    assert len(a.progress_hand) == 5 and sim.progress_discarded == 0


def test_pillaged_city_loses_wall():
    sim = _sim()
    gs = sim.gs
    p = gs.players[0]
    v = next(iter(p.cities))
    p.walls.add(v)
    gs.barbarian_pos = gs.barbarian_len - 1
    sim.barbarians()                                # no knights: everyone is pillaged
    assert v in p.settlements and v not in p.cities and v not in p.walls


def test_metropolis_per_track_first_to_4_and_steal_at_5():
    sim = _sim()
    gs = sim.gs
    a, b = gs.players[:2]
    a.cities.add(a.settlements.pop())               # a: 2 cities
    for p in (a, b):
        p.hand.update(paper=20, coin=20)
    a.improvements["science"] = 3
    vp0 = sim.vp(a)
    sim.execute(0, Action("improve", "science"))
    assert len(a.metropolises) == 1 and gs.metropolis_owner["science"][0] == a.name
    assert sim.vp(a) == vp0 + 2
    b.improvements["science"] = 3
    sim.execute(1, Action("improve", "science"))    # level 4: taken, nothing
    assert not b.metropolises
    sim.execute(1, Action("improve", "science"))    # level 5 first: steals it
    assert len(b.metropolises) == 1 and not a.metropolises
    assert gs.metropolis_owner["science"][0] == b.name and sim.vp(a) == vp0
    a.improvements["politics"] = 3
    sim.execute(0, Action("improve", "politics"))   # a player may hold several (one per track)
    sim.execute(0, Action("improve", "science"))    # a level 5 too late: b holds 5
    assert len(a.metropolises) == 1 and gs.metropolis_owner["politics"][0] == a.name
    assert gs.metropolis_owner["science"][0] == b.name


def test_brain_apply_tracks_metropolis_vp():
    sim = _sim()
    gs = sim.as_me(1)
    a, b = gs.players[:2]
    a.improvements["science"] = 4
    gs.metropolis_owner["science"] = (a.name, next(iter(a.cities)))
    a.metropolises.add(next(iter(a.cities)))
    b.improvements["science"] = 4
    b.hand.update(paper=10)
    b.vp_visible, a.vp_visible = 5, 7
    g = sim.brains[1]._apply(gs, Action("improve", "science"))
    ga, gb = g.players[:2]
    assert gb.vp_visible == 7 and ga.vp_visible == 5 and not ga.metropolises
    assert a.metropolises and gs.metropolis_owner["science"][0] == a.name   # original untouched


def _nothing_roll(sim, p):
    b = sim.gs.board
    nums = {h.number for v, _ in p.buildings() for h in b.vertex_hexes(v)}
    return next(r for r in (2, 12, 3, 11, 4, 10, 5, 9, 6, 8) if r not in nums)


def test_aqueduct_pays_one_on_an_empty_roll():
    sim = _sim()
    p = sim.gs.players[0]
    roll = _nothing_roll(sim, p)
    n = sum(p.hand.values())
    sim.produce(roll)
    assert sum(p.hand.values()) == n
    p.improvements["science"] = 3
    sim.produce(roll)
    assert sum(p.hand.values()) == n + 1


def test_small_island_knights_never_defend():
    b = Board()
    for c in [(0, 0), (1, 0), (0, 1), (-1, 1)]:
        b.add(Hex(*c, "pasture", 5))
    b.add(Hex(5, 0, "forest", 6))                  # small island
    b.add(Hex(-5, 0, "canal"))                     # canal counts as main island
    for q in range(-7, 8):
        for r in range(-3, 4):
            if (q, r) not in b.hexes:
                b.add(Hex(q, r, "sea"))
    from catanbot.board import hex_corners
    small, main, canal = hex_corners(5, 0)[0], hex_corners(0, 0)[0], hex_corners(-5, 0)[0]
    assert b.is_small_island_vertex(small) and not b.is_small_island_vertex(main)
    assert not b.is_small_island_vertex(canal)
    me = PlayerState("bot", is_me=True)
    gs = GameState(board=b, mode=MODES["great_canal"], players=[me])
    me.knights[small] = Knight(3, True)
    assert gs.defence_strength(me) == 0
    me.knights[main] = Knight(1, True)
    assert gs.defence_strength(me) == 1


def test_gold_prospector_stops_when_canal_complete():
    sim = _sim()
    gs, b = sim.gs, sim.gs.board
    p = gs.players[0]
    p.settlements.clear()
    p.cities.clear()
    gold = next(h for h in b.hexes.values() if h.terrain == "gold" and h.number)
    from catanbot.board import hex_corners
    p.knights[hex_corners(*gold.coord)[0]] = Knight(1, False)
    n = sum(p.hand.values())
    sim.produce(gold.number)
    assert sum(p.hand.values()) == n + 1
    gs.canal_progress = gs.canal_total
    sim.produce(gold.number)
    assert sum(p.hand.values()) == n + 1


def test_discarded_progress_card_returns_to_its_deck():
    sim = _sim()
    p = sim.gs.players[0]
    p.progress_hand = ["warlord", "spy", "crane", "smith", "merchant"]
    before = {t: len(d) for t, d in sim.decks.items()}
    sim._progress_limit(p)
    assert len(p.progress_hand) == 4 and sim.progress_discarded == 1
    after = {t: len(d) for t, d in sim.decks.items()}
    assert sum(after.values()) == sum(before.values()) + 1


def test_knight_chases_robber_off_own_hex_unless_needed_for_defence():
    sim = _sim()
    gs, b = sim.gs, sim.gs.board
    p = gs.players[0]
    h = next(x for v, _ in p.buildings() for x in b.vertex_hexes(v) if x.number and not x.barren)
    from catanbot.board import hex_corners
    v = next(c for c in hex_corners(*h.coord)
             if not any(x.terrain == "gold" for x in b.vertex_hexes(c)) and not gs.occupied(c))
    p.knights[v] = Knight(1, True)
    b.robber = h.coord
    gs.barbarian_pos = 0
    sim.chase_robber(0)
    assert b.robber == h.coord and sim.chases == 0          # robber locked before the 1st attack
    sim.attacks = 1
    gs.barbarian_pos = gs.barbarian_len - 1                 # barbarians next to land: keep it
    sim.chase_robber(0)
    assert b.robber == h.coord and p.knights[v].active
    gs.barbarian_pos = 0
    sim.chase_robber(0)
    assert b.robber != h.coord and not p.knights[v].active and sim.chases == 1


def test_knights_built_on_home_island_only():
    from catanbot import moves as mv
    from catanbot.board import ekey
    sim = _sim()
    gs = sim.as_me(0)
    b, p = gs.board, gs.players[0]
    far = [v for v in b.vertices if b.is_land_vertex(v) and not gs.occupied(v)
           and (b.is_small_island_vertex(v) or any(h.terrain == "gold" for h in b.vertex_hexes(v)))]
    assert far
    for w in far:
        p.ships.add(ekey(w, next(iter(b.neighbors(w)))))
    p.hand.update(wool=5, ore=5)
    spots = {m.target for m in mv.build_moves(gs) if m.kind == "knight"}
    assert spots and not spots & set(far)


def test_knight_moves_to_gold_along_ships_once_per_turn():
    from catanbot import moves as mv
    from catanbot.board import ekey
    sim = _sim()
    gs = sim.as_me(0)
    b, p = gs.board, gs.players[0]
    p.settlements.clear()
    p.cities.clear()
    gold = next(h for h in b.hexes.values() if h.terrain == "gold" and h.number)
    from catanbot.board import hex_corners
    dst = next(v for v in hex_corners(*gold.coord) if not gs.occupied(v))
    src = next(v for v in b.neighbors(dst) if not gs.occupied(v)
               and gold not in b.vertex_hexes(v))
    p.knights[src] = Knight(1, True)
    p.ships.add(ekey(src, dst))
    move = mv.Move("move_knight", (src, dst))
    assert move in mv.knight_moves(gs)
    a = Action("move_knight", (src, dst))
    assert a in sim.brains[0].candidate_actions(gs)          # prospector corner: simulated
    assert sim.brains[0]._prospector_value(gs, p, dst) > 0
    sim.execute(0, a)
    k = p.knights[dst]
    assert src not in p.knights and not k.active and k.acted
    k.active = True                                          # re-activated this turn: no 2nd move
    assert not any(m.target[0] == dst for m in mv.knight_moves(gs))
    n = sum(p.hand.values())
    sim.produce(gold.number)                                 # the moved knight collects gold
    assert sum(p.hand.values()) == n + 1


def test_friendly_robber_spares_players_at_4_vp_or_fewer():
    from catanbot.state import FRIENDLY_ROBBER_VP
    sim = _sim()
    gs = sim.gs
    assert gs.mode.friendly_robber and FRIENDLY_ROBBER_VP == 4
    for p in gs.players[:3]:
        p.cities |= p.settlements                           # 6 VP each
        p.settlements.clear()
    low = gs.players[3]                                     # 4 VP: protected
    sim.refresh()
    assert gs.robber_protected() == {low.name}
    touching = {h.coord for v, _ in low.buildings() for h in gs.board.vertex_hexes(v)}
    low.hand.update(ore=3)
    for _ in range(5):
        sim.move_robber(0)
        assert gs.board.robber not in touching
    # only protected players' hexes left: any hex is allowed, but nobody 4 VP or less is robbed
    g = sim.as_me(0)
    hx, victim = sim.brains[0].choose_robber(g, sorted(touching))
    assert hx in touching and (victim is None or victim.name != low.name)


def test_warlord_pair_digs_canal():
    from catanbot.board import hex_corners
    sim = _sim()
    gs = sim.gs
    a = gs.players[0]
    c = sorted(gs.board.canal_hexes)[0]
    cs = hex_corners(*c)
    a.knights[cs[0]] = Knight(1, False)
    a.knights[cs[3]] = Knight(1, False)
    chips = a.canal_chips
    sim._play_progress(0, "warlord")
    assert c in gs.canal_dug and a.canal_chips == chips + 2


def test_robbed_text_counts_once_and_trades_do_not(tmp_path, monkeypatch):
    import numpy as np
    from catanbot.runner import Runner
    monkeypatch.chdir(tmp_path)
    r = Runner.__new__(Runner)
    r.note = lambda *a, **k: None
    img = np.zeros((4, 4, 3), np.uint8)
    r._watch_robbed(img, ["Rhea took your offer"])
    assert getattr(r, "_cards_lost", 0) == 0
    r._watch_robbed(img, ["Rhea stole a card from marlow.systems"])
    r._watch_robbed(img, ["Rhea stole a card from marlow.systems"])     # same popup, next frame
    assert r._cards_lost == 1


def test_strong_knight_displaces_weaker_one():
    """C&K: an active knight may push a weaker enemy knight at the end of its road; the pushed
    knight keeps level and status and moves to a free corner on its owner's roads."""
    from catanbot import moves as mv
    from catanbot.board import ekey
    sim = _sim()
    gs = sim.as_me(0)
    b, me, them = gs.board, gs.players[0], gs.players[1]
    for p in gs.players:
        p.settlements.clear(); p.cities.clear(); p.roads.clear(); p.knights.clear(); p.ships.clear()   # setup ships (row 7) are corners too
    land = [v for v in sorted(b.vertices) if b.is_land_vertex(v)]
    src = land[len(land) // 2]
    dst = next(v for v in b.neighbors(src) if b.is_land_vertex(v))
    away = next(v for v in b.neighbors(dst) if v != src and b.is_land_vertex(v))
    me.knights[src] = Knight(2, True)
    me.roads.add(ekey(src, dst))
    them.knights[dst] = Knight(1, True)
    them.roads.add(ekey(dst, away))
    assert mv.Move("move_knight", (src, dst)) in mv.knight_moves(gs)
    them.knights[dst] = Knight(2, True)                      # equal strength: no push
    assert mv.Move("move_knight", (src, dst)) not in mv.knight_moves(gs)
    them.knights[dst] = Knight(1, True)
    sim.execute(0, Action("move_knight", (src, dst)))
    assert dst in me.knights and src not in me.knights
    assert dst not in them.knights and them.knights[away].level == 1 and them.knights[away].active
    assert sim.displaced == 1
