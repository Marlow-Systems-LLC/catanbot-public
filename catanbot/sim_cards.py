"""Real progress-card effects for the self-play simulator (Sim._play_progress calls play()).

Card texts: the in-game Almanac (data/ref/almanac_en.txt), docs/progress_cards.md and, for The
Great Canal, docs/great_canal_rules_research.md. Decisions (whom to hit, which hex) reuse the
progress_cards choosers, so the sim plays a card when, and how, the live bot would.

play(sim, j, card) -> True if the card was played. A card the chooser would not play now (no
target, not affordable, robber still locked ...) goes back into the hand; the 4-card limit in
Sim.play_held then discards the least-kept card if needed.

Alchemist is played BEFORE the roll: pre_roll(sim, i) -> (white, red) or None. It needs a hook
in Sim.turn (owned elsewhere); after the roll play() keeps the card in hand.
"""
from __future__ import annotations

from . import progress_cards as pc
from .board import COMMODITIES
from .brain import Action
from .moves import home_island
from .state import dig_canals, grant_metropolis


def _count(sim, card):
    d = sim.__dict__.setdefault("progress_played", {})
    d[card] = d.get(card, 0) + 1


def _give(p, r, n=1):
    p.hand[r] = p.hand.get(r, 0) + n


def _take(src, dst, r, n=1):
    n = min(n, src.hand.get(r, 0))
    if n > 0:
        src.hand[r] -= n
        _give(dst, r, n)
    return n


def _pool(p):
    return [r for r, n in p.hand.items() for _ in range(max(0, n))]


def _discard(sim, k, n):
    """Player k gives up n cards of its own choice (brain.choose_discard picks them)."""
    p = sim.gs.players[k]
    n = min(n, sum(p.hand.values()))
    if n <= 0:
        return {}
    pick = sim.brains[k].choose_discard(sim.as_me(k), n)
    out = {}
    for r, x in pick.items():
        x = min(x, p.hand.get(r, 0))
        if x > 0:
            p.hand[r] -= x
            out[r] = out.get(r, 0) + x
    short = n - sum(out.values())               # the brain under-picked: drop the most plentiful
    while short > 0 and sum(p.hand.values()):
        r = max(p.hand, key=lambda r: (p.hand[r] > 0, r != "grain", p.hand[r]))   # grain last (owner 2026-10-08)
        p.hand[r] -= 1
        out[r] = out.get(r, 0) + 1
        short -= 1
    return out


def _want_order(sim, j, cards):
    """Cards of an opponent's hand sorted most wanted first (plan shortfall, then card value)."""
    gs = sim.as_me(j)
    miss = sim.brains[j].plan_missing(gs)
    return sorted(cards, key=lambda r: (-(miss.get(r, 0) > 0), -pc.card_value(gs, r)))


def _knight_spot(sim, j):
    """Best free corner for a new knight: on our roads (and ships in the canal), home island."""
    gs = sim.as_me(j)
    p, b = gs.players[j], gs.board
    reach = {v for e in p.roads for v in e}
    if gs.mode.canal:
        reach |= {v for e in p.ships for v in e}
    spots = [v for v in sorted(reach) if b.is_land_vertex(v) and not gs.occupied(v) and home_island(gs, v)]
    if not spots:
        return None
    brain = sim.brains[j]
    if gs.mode.canal:
        return max(spots, key=lambda v: (brain._canal_pair_value(gs, Action("knight", v)), b.vertex_pips(v)))
    return max(spots, key=b.vertex_pips)


def _dig(sim, why):
    if sim.mode.canal:
        for c, x, y in dig_canals(sim.gs):
            sim.log.event(f"canal dug at {c} by {x}+{y} ({why})")


# ---- science ---------------------------------------------------------------------------

def _alchemist(sim, j, gs, me):
    return False                                # only before the roll: pre_roll()


def _gain_card(sim, j, gs, me, card):
    args = pc.CHOOSERS[card](gs, me)
    if not args:
        return False
    for r, n in args["gain"].items():           # 2 per adjacent hex; cities don't double
        _give(me, r, n)
    return True


def _crane(sim, j, gs, me):
    """Next improvement this turn costs 1 commodity less: the sim builds it at once."""
    args = pc.choose_crane(gs, me)
    if not args:
        return False
    track = args["track"]
    com = pc.IMPROVEMENT_TRACKS[track]
    lvl = me.improvements.get(track, 0)
    cost = lvl                                  # level n+1 costs n+1; Crane takes 1 off
    if me.hand.get(com, 0) < cost:
        return False
    me.hand[com] -= cost
    me.improvements[track] = lvl + 1
    grant_metropolis(gs, me, track)
    return True


def _engineer(sim, j, gs, me):
    args = pc.choose_engineer(gs, me)
    if not args:
        return False
    me.walls.add(args["city"])
    return True


def _inventor(sim, j, gs, me):
    args = pc.choose_inventor(gs, me)
    if not args:
        return False
    a, b = (gs.board.hexes[c] for c in args["swap"])
    a.number, b.number = b.number, a.number
    sim.log.event(f"P{j} inventor swaps {a.coord}<->{b.coord}")
    return True


def _medicine(sim, j, gs, me):
    args = pc.choose_medicine(gs, me)
    if not args:
        return False
    me.hand["ore"] -= 2
    me.hand["grain"] -= 1
    v = args["settlement"]
    me.settlements.discard(v)
    me.cities.add(v)
    return True


def _road_building(sim, j, gs, me):
    args = pc.choose_road_building(gs, me)
    if not args:
        return False
    ships = set(args.get("ships", []))
    for e in args["edges"]:
        (me.ships if e in ships else me.roads).add(e)
    sim.update_awards()
    return True


def _smith(sim, j, gs, me):
    args = pc.choose_smith(gs, me)
    if not args:
        return False
    done = 0
    for v in args["knights"]:
        k = me.knights[v]
        per = sum(1 for x in me.knights.values() if x.level == k.level + 1)
        if per < 2:                             # 2 knights per level
            k.level += 1
            done += 1
    return done > 0


# ---- trade -----------------------------------------------------------------------------

def _commercial_harbor(sim, j, gs, me):
    """Each opponent with a commodity takes one of our resources and gives a commodity of
    its choice (the one it holds most of, cheapest on ties)."""
    args = pc.choose_commercial_harbor(gs, me)
    opps = [q for q in gs.players if q is not me and any(q.hand.get(c, 0) for c in COMMODITIES)]
    if not args or not opps:
        return False
    give = list(args["give"])
    for q in opps:
        r = next((r for r in give if me.hand.get(r, 0) > 0), None)
        if r is None:
            break
        give.remove(r)
        c = max((c for c in COMMODITIES if q.hand.get(c, 0)), key=lambda c: (q.hand[c], -pc.card_value(gs, c)))
        _take(me, q, r)
        _take(q, me, c)
    return True


def _master_merchant(sim, j, gs, me):
    args = pc.choose_master_merchant(gs, me)
    if not args:
        return False
    q = next(p for p in gs.players if p.name == args["player"])
    if sim.vp(q) <= sim.vp(me):
        return False
    for r in _want_order(sim, j, _pool(q))[:2]:
        _take(q, me, r)
    return True


def _merchant(sim, j, gs, me):
    """Place the merchant next to our building: 2:1 for its resource and 1 VP while we hold
    it; whoever plays the next Merchant takes it (PlayerState.merchant_hex)."""
    args = pc.choose_merchant(gs, me)
    if not args:
        return False
    for q in gs.players:
        if getattr(q, "merchant_hex", None) is not None:
            q.merchant_hex = None
    me.merchant_hex = args["hex"]
    return True


def _merchant_fleet(sim, j, gs, me):
    """2:1 with the bank for one type this turn: trade pairs into what the plan wants,
    keeping 1 of the type unless over the hand limit."""
    over = sum(me.hand.values()) > pc.discard_limit(gs, me)
    args = pc.choose_merchant_fleet(gs, me, force=over)
    if not args:
        return False
    r = args["resource"]
    brain = sim.brains[j]
    traded = 0
    while me.hand.get(r, 0) >= 3 or (me.hand.get(r, 0) >= 2 and sum(me.hand.values()) > pc.discard_limit(gs, me)):
        got = brain.choose_resource(gs, 1)[0]
        if got == r:
            break
        me.hand[r] -= 2
        _give(me, got)
        traded += 1
    return traded > 0


def _resource_monopoly(sim, j, gs, me):
    old = getattr(gs, "plan_missing", None)
    gs.plan_missing = sim.brains[j].plan_missing(gs)
    try:
        args = pc.choose_resource_monopoly(gs, me)
    finally:
        gs.plan_missing = old
    if not args:
        return False
    for q in gs.players:
        if q is not me:
            _take(q, me, args["resource"], 2)
    return True


def _trade_monopoly(sim, j, gs, me):
    args = pc.choose_trade_monopoly(gs, me)
    if not args:
        return False
    for q in gs.players:
        if q is not me:
            _take(q, me, args["commodity"], 1)
    return True


# ---- politics --------------------------------------------------------------------------

def _bishop(sim, j, gs, me):
    """Move the robber (locked until the first attack; friendly robber hexes only) and take
    1 random card from each player with a building on the new hex (protected players: none)."""
    if gs.mode.ck and not sim.attacks:
        return False
    args = pc.choose_bishop(gs, me)
    if not args:
        return False
    b = gs.board
    b.robber = args["hex"]
    safe = gs.robber_protected() if gs.mode.friendly_robber else set()
    corners = {v for v in b.vertices if any(h.coord == b.robber for h in b.vertex_hexes(v))}
    for q in gs.players:
        if q is me or q.name in safe or not any(v in corners for v, _ in q.buildings()):
            continue
        pool = _pool(q)
        if pool:
            r = sim.rng.choice(pool)
            _take(q, me, r)
            sim.cards_lost[q.name] = sim.cards_lost.get(q.name, 0) + 1
            q.cards_lost += 1
    sim.log.event(f"P{j} bishop -> {b.robber}")
    return True


def _deserter(sim, j, gs, me):
    """The target removes the knight it values least; we place one of the same level and
    status free (owner-confirmed 2026-10-06: it keeps active/inactive and level; basic if that
    level is used up it drops to the next free level, owner-confirmed; none without a free corner)."""
    args = pc.choose_deserter(gs, me)
    if not args:
        return False
    q = next(p for p in gs.players if p.name == args["player"])
    qi = gs.players.index(q)
    try:                                         # owner 2026-10-04: the victim gives up its least useful knight
        v = sim.brains[qi].choose_knight_to_lose(sim.as_me(qi))
    finally:
        sim.as_me(j)
    if v is None:
        v = min(q.knights, key=lambda v: (q.knights[v].level, q.knights[v].active))
    k = q.knights.pop(v)
    w = _knight_spot(sim, j)
    if w is not None:
        per = {lv: sum(1 for x in me.knights.values() if x.level == lv) for lv in (1, 2, 3)}
        # owner-confirmed 2026-10-06: drops to the next free level (strong -> basic when both strong are out)
        lvl = next((lv for lv in range(k.level, 0, -1) if per[lv] < 2), None)
        if lvl is not None:
            me.knights[w] = type(k)(level=lvl, active=k.active, acted=True)
            _dig(sim, "deserter")
    sim.log.event(f"P{j} deserter takes {q.name}'s knight at {v}")
    return True


def _diplomat(sim, j, gs, me):
    args = pc.choose_diplomat(gs, me)
    if not args:
        return False
    if args.get("own"):                          # move our own open road or ship
        old, new = args["edges"]
        net = me.ships if args.get("ship") else me.roads
        net.discard(old)
        net.add(new)
        sim.update_awards()
        return True
    e = args["edge"]
    owner = gs.edge_owner(e)
    if owner is None or owner is me:
        return False
    owner.roads.discard(e)
    owner.ships.discard(e)                      # ships too (owner-confirmed 2026-10-06)
    sim.update_awards()
    return True


def _intrigue(sim, j, gs, me):
    args = pc.choose_intrigue(gs, me)
    if not args:
        return False
    sim._displace(j, args["knight_at"])         # moved by its owner to a free corner, else removed
    return True


def _saboteur(sim, j, gs, me):
    if pc.choose_saboteur(gs, me) is None:      # {} = play
        return False
    mine = sim.vp(me)
    for k, q in enumerate(gs.players):
        if q is not me and sim.vp(q) >= mine:
            _discard(sim, k, sum(q.hand.values()) // 2)
    sim.as_me(j)
    return True


def _spy(sim, j, gs, me):
    """Take the best progress card (by KEEP_VALUE) of the player holding most of them."""
    opps = [q for q in gs.players if q is not me and q.progress_hand]
    if not opps:
        return False
    q = max(opps, key=lambda q: (len(q.progress_hand), pc.score(gs, q)))
    from .situational import spy_target
    q = spy_target(sim.brains[j], gs, opps) or q   # selector: focus rival / not the last-placed
    # owner 2026-10-06: take by a real value (playable now first, the leader's best counts),
    # then play it this turn when it has a use; the Alchemist waits for our next roll
    c, _ = pc.spy_pick(gs, me, q.progress_hand, q.name, "main", bool(sim.attacks))
    if c is None:
        return False
    q.progress_hand.remove(c)
    if pc.playable_now(gs, me, c, "main", bool(sim.attacks)):
        sim._play_progress(j, c)                # not played after all -> back in our hand
    else:
        me.progress_hand.append(c)              # held like a draw (Alchemist: next pre-roll)
    return True


def _wedding(sim, j, gs, me):
    args = pc.choose_wedding(gs, me)
    if not args:
        return False
    mine = sim.vp(me)
    for k, q in enumerate(gs.players):
        if q is not me and sim.vp(q) > mine:
            for r, n in _discard(sim, k, 2).items():
                _give(me, r, n)
    sim.as_me(j)
    return True


EFFECTS = {
    "alchemist": _alchemist, "crane": _crane, "engineer": _engineer, "inventor": _inventor,
    "irrigation": lambda s, j, g, m: _gain_card(s, j, g, m, "irrigation"),
    "medicine": _medicine,
    "mining": lambda s, j, g, m: _gain_card(s, j, g, m, "mining"),
    "road_building": _road_building, "smith": _smith,
    "commercial_harbor": _commercial_harbor, "master_merchant": _master_merchant,
    "merchant": _merchant, "merchant_fleet": _merchant_fleet,
    "resource_monopoly": _resource_monopoly, "trade_monopoly": _trade_monopoly,
    "bishop": _bishop, "deserter": _deserter, "diplomat": _diplomat, "intrigue": _intrigue,
    "saboteur": _saboteur, "spy": _spy, "wedding": _wedding,
}


def play(sim, j: int, card: str) -> bool:
    """Apply `card` for seat j (already taken out of its hand). Not playable now -> back in hand."""
    sim.refresh()
    gs = sim.as_me(j)
    me = gs.players[j]
    played = EFFECTS[card](sim, j, gs, me)
    sim.as_me(j)
    if played:
        _count(sim, card)
    else:
        me.progress_hand.append(card)
    sim.refresh()
    return played


def pre_roll(sim, i: int):
    """Alchemist, own turn before the dice: (white, red) to roll, or None. Hook for Sim.turn."""
    p = sim.gs.players[i]
    if "alchemist" not in p.progress_hand:
        return None
    sim.refresh()
    args = pc.choose_alchemist(sim.as_me(i), p)
    if not args:
        return None
    p.progress_hand.remove("alchemist")
    _count(sim, "alchemist")
    return args["dice"]
