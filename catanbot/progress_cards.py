"""Cities & Knights progress cards: card data and when/how the bot plays them.

Data: docs/progress_cards.md (sources and unconfirmed points there).
Entry point: choose_play(gs, me, hand_cards, phase) -> [(card_key, args), ...] in play order.

Phases
- "pre_roll": own turn, before the dice. Only the Alchemist may be played (VP cards never are).
- "main":     own turn, after the roll and before building. Gain/steal cards come first so
              the brain can spend what they bring (owner rule: spend before the robber).
- "end_turn": own turn, after building. Free-value cards, barbarian defence, and the
              progress-hand limit (max 4 when the turn ends).
- "on_seven" / "opp_turn": not our turn. Progress cards are own-turn only, so only VP
              cards (Printer/Constitution, revealed when drawn) can be "played".
Args are dicts; what each key means is listed in CARDS[...]["args"].
"""
from __future__ import annotations

from .board import COMMODITIES, PIPS, RESOURCES, TERRAIN_RES
from .rules import COSTS, discard_limit as _rules_discard_limit, production
from . import knight_targets as kt
from .state import GameState, PlayerState

# ---------------------------------------------------------------------------------------
# Card data (English 5th-edition names; Catan Universe uses the same set: unconfirmed)
# ---------------------------------------------------------------------------------------


def _c(deck, count, effect, timing, use, args):
    return {"deck": deck, "count": count, "effect": effect, "timing": timing,
            "use": use, "args": args}


ANY = "own turn, after the roll"
CARDS: dict[str, dict] = {
    # ---- Science (green), 18 -----------------------------------------------------------
    "alchemist": _c("science", 2, "Choose the values of both production dice; roll the event die normally and resolve it first.",
                    "own turn, BEFORE rolling only", "Pick the sum that pays us most and the leader least; never 7; low red die for progress draws.",
                    "dice=(white, red)"),
    "crane": _c("science", 2, "Next city improvement this turn costs 1 commodity less (cannot stack two Cranes).",
                ANY, "Play when it makes an improvement affordable now; paper > coin > cloth.", "track"),
    "engineer": _c("science", 1, "Build one city wall free (max 3 walls; each wall +2 hand limit).",
                   ANY, "Always on an unwalled city; raises discard limit.", "city=vertex"),
    "inventor": _c("science", 2, "Swap two number tokens; 2, 12, 6 and 8 cannot be moved.",
                   ANY, "Move a 5/9 onto our hex and a 3/11 onto the leader's.", "swap=(hexA, hexB)"),
    "irrigation": _c("science", 2, "Take 2 grain per fields hex next to at least one of your buildings.",
                     ANY, "Play early in main so the grain is spent; works on barren canal farmland.", "none"),
    "medicine": _c("science", 2, "Upgrade a settlement to a city for 2 ore + 1 grain.",
                   ANY, "Play when we hold 2 ore+1 grain and a settlement; best-pip settlement.", "settlement=vertex"),
    "mining": _c("science", 2, "Take 2 ore per mountains hex next to at least one of your buildings.",
                 ANY, "Play early in main so the ore is spent.", "none"),
    "printer": _c("science", 1, "1 VP. Revealed immediately when drawn; cannot be stolen.",
                  "immediately when drawn", "Automatic.", "none"),
    "road_building": _c("science", 2, "Build 2 roads free (ships allowed with Seafarers).",
                        ANY, "Play when the roads reach a settlement spot, the canal or longest road.", "edges=[e1, e2]"),
    "smith": _c("science", 2, "Promote up to 2 knights one level free (level 3 needs politics 3).",
                ANY, "Promote active knights first; always worth it.", "knights=[v1, v2]"),
    # ---- Trade (yellow), 18 ------------------------------------------------------------
    "commercial_harbor": _c("trade", 2, "Each opponent must take 1 resource from you and give you 1 commodity if they have one.",
                            ANY, "Play when we have spare resources and opponents hold commodities.", "give=[resource per opponent, worst first]"),
    "master_merchant": _c("trade", 2, "Look at the hand of a player with MORE VP than you and take 2 cards.",
                          ANY, "Target the leader with the biggest hand.", "player=name"),
    "merchant": _c("trade", 6, "Place the merchant on a land hex next to your building: 2:1 trade for that resource and 1 VP while you hold it.",
                   ANY, "Always play (1 VP); hex of the resource we hold most of.", "hex=coord"),
    "merchant_fleet": _c("trade", 2, "This turn, trade one resource or commodity of your choice 2:1 with the bank, any number of times.",
                         ANY, "Play when we hold 4+ of one type, or the hand is over the discard limit.", "resource"),
    "resource_monopoly": _c("trade", 4, "Name a resource; each player gives you 2 of it if they can.",
                            ANY, "Play when expected take is 3+ cards of a resource we need.", "resource"),
    "trade_monopoly": _c("trade", 2, "Name a commodity; each player gives you 1 of it if they can.",
                         ANY, "Play when 2+ opponents likely hold it; paper > coin > cloth.", "commodity"),
    # ---- Politics (blue), 18 -----------------------------------------------------------
    "bishop": _c("politics", 2, "Move the robber; take 1 random card from each player with a building on the new hex.",
                 ANY, "Robber on the leader's best hex.", "hex=coord"),
    "constitution": _c("politics", 1, "1 VP. Revealed immediately when drawn; cannot be stolen.",
                       "immediately when drawn", "Automatic.", "none"),
    "deserter": _c("politics", 2, "An opponent removes one knight of their choice; you may place one of yours of the same strength free.",
                   ANY, "Target the strongest knight player (leader on ties); best when the barbarians are near.", "player=name"),
    "diplomat": _c("politics", 2, "Remove an open road (no piece at one end); if it is yours you may move it.",
                   ANY, "Break the leader's longest road / a road blocking our expansion.", "edge=e"),
    "intrigue": _c("politics", 2, "Displace an opponent knight standing on a junction touched by your road.",
                   ANY, "Remove a leader's knight that blocks us or sits on a canal corner.", "knight_at=vertex"),
    "saboteur": _c("politics", 2, "Each player with as many or more VP than you discards half their hand.",
                   ANY, "Play when we trail and targets hold big hands.", "none"),
    "spy": _c("politics", 3, "Look at a player's progress cards and take one (not VP cards).",
              ANY, "Target the player with most progress cards, the leader on ties.", "player=name"),
    "warlord": _c("politics", 2, "Activate all your knights free.",
                  ANY, "Play with 2+ inactive knights, or 1 when barbarians are near / canal digging.", "none"),
    "wedding": _c("politics", 2, "Each player with more VP than you gives you 2 cards of their choice.",
                  ANY, "Play when at least one (ideally two+) opponents lead us.", "none"),
}

VP_CARDS = {"printer", "constitution"}
PROGRESS_HAND_LIMIT = 4
IMPROVEMENT_TRACKS = {"science": "paper", "politics": "coin", "trade": "cloth"}
TRACK_ORDER = ["science", "politics", "trade"]          # owner rule: paper > coin > cloth

_ALIASES = {
    "printing press": "printer", "construction crane": "crane", "mining industry": "mining",
    "forge": "smith", "road construction": "road_building", "road building": "road_building",
    "trade fleet": "merchant_fleet", "trade master": "master_merchant", "trade port": "commercial_harbor",
    "trader": "merchant", "raw material monopoly": "resource_monopoly", "commodity monopoly": "trade_monopoly",
    "defectors": "deserter", "plot": "intrigue", "field marshall": "warlord", "field marshal": "warlord",
}


def card_key(name: str, min_prefix: int = 5) -> str:
    """Normalize an on-screen card name (any edition) to a CARDS key."""
    n = name.strip().lower().replace("-", " ")
    if n in _ALIASES:
        return _ALIASES[n]
    k = n.replace(" ", "_")
    if k in CARDS:
        return k
    # live 2026-10-03: overlapping hand cards cut titles short ("Resource Monop", "Master
    # Merchan"); accept a prefix when it names exactly one card
    if len(n) >= min_prefix and n.isalpha() or len(n) >= 5:
        names = {a: v for a, v in _ALIASES.items()}
        names.update({c.replace("_", " "): c for c in CARDS})
        hits = {v for a, v in names.items() if a.startswith(n)}
        if len(hits) == 1:
            return hits.pop()
        if not hits and len(n) >= 8:          # one OCR slip on the last letter ("Master Merchar")
            hits = {v for a, v in names.items() if a.startswith(n[:-1])}
            if len(hits) == 1:
                return hits.pop()
    raise KeyError(name)


# ---------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------

def card_value(gs: GameState, res: str) -> float:
    """Rough worth of one card. Owner rules: paper > coin > cloth; grain is gold while the canal is open."""
    v = {"paper": 1.5, "coin": 1.35, "cloth": 1.15, "ore": 1.15, "grain": 1.1}.get(res, 1.0)
    if res == "grain" and gs.mode.canal and gs.canal_progress < gs.canal_total:
        v = 1.4
    return v


def discard_limit(gs: GameState, p: PlayerState) -> int:
    return _rules_discard_limit(gs.mode, p)


def hand_size(p: PlayerState) -> int:
    return sum(p.hand.values()) if p.is_me else p.total_cards()


def score(gs: GameState, p: PlayerState) -> float:
    """Leader metric: visible VP, income as tie-break."""
    return p.vp_visible + gs.hidden_vp_estimate(p) + 0.01 * gs.income_total(p)


def leader(gs: GameState, me: PlayerState) -> PlayerState | None:
    opps = [p for p in gs.players if p is not me]
    return max(opps, key=lambda p: score(gs, p), default=None)


def barbarians_near(gs: GameState) -> bool:
    return gs.barbarian_len - gs.barbarian_pos <= 2


def defence_gap(gs: GameState, me: PlayerState) -> int:
    """Owner rule: one active knight-level per city. Metropolis cities are safe."""
    at_risk = len(me.cities - me.metropolises)
    return max(0, at_risk - gs.defence_strength(me))


def _my_hexes(gs: GameState, p: PlayerState):
    seen = {}
    for v, _ in p.buildings():
        for h in gs.board.vertex_hexes(v):
            seen[h.coord] = h
    return list(seen.values())


def _roll_gain(gs: GameState, p: PlayerState, total: int) -> dict[str, int]:
    out: dict[str, int] = {}
    for v, kind in p.buildings():
        for h in gs.board.vertex_hexes(v):
            if h.number != total or h.coord == gs.board.robber or h.fog:
                continue
            if gs.mode.canal and h.barren and gs.canal_progress < gs.canal_total:
                continue
            for r, n in production(h.terrain, kind, gs.mode).items():
                out[r] = out.get(r, 0) + n
    return out


def _value(gs, cards: dict[str, int]) -> float:
    return sum(card_value(gs, r) * n for r, n in cards.items())


def _can_afford(hand: dict[str, int], cost: dict[str, int]) -> bool:
    return all(hand.get(r, 0) >= n for r, n in cost.items())


# ---------------------------------------------------------------------------------------
# Per-card choosers: return args dict, or None = don't play now
# ---------------------------------------------------------------------------------------

def choose_alchemist(gs: GameState, me: PlayerState) -> dict | None:
    lead = leader(gs, me)
    best, best_s = None, -1e9
    for total in range(2, 13):
        if total == 7:
            continue
        s = _value(gs, _roll_gain(gs, me, total))
        if lead is not None:
            s -= 0.5 * _value(gs, _roll_gain(gs, lead, total))
        if s > best_s:
            best, best_s = total, s
    over = hand_size(me) > discard_limit(gs, me)
    # owner 2026-10-06: the Alchemist may choose a 7 -- worth it only to clear the robber off our
    # own hex (usually a producing roll pays more), and never with too many cards in hand
    s7 = _alchemist_seven(gs, me)
    if s7 is not None and not over and s7 > best_s:
        best, best_s = 7, s7
    if best is None:
        return None
    # owner 2026-10-06 "should never hold cards it can use": a chosen roll beats a random one
    # whenever it pays us more than the leader. Held only when no sum does (playbook:
    # docs/progress_cards_playbook.md).
    if best_s <= 0 and not over:
        return None
    red = max(1, best - 6)          # low red die: most progress-card draws for our improvements
    return {"dice": (best - red, red), "sum": best}


def _alchemist_seven(gs: GameState, me: PlayerState) -> float | None:
    """Value of choosing 7: the robber leaves the hex it blocks for us (about 6 rolls until the
    next 7) and we steal a card. None when the robber is locked (before the first barbarian
    attack) or blocks nothing of ours."""
    if gs.mode.ck and not gs.barbarians_attacked:
        return None
    h = gs.board.hexes.get(gs.board.robber) if gs.board.robber is not None else None
    if h is None or not h.number:
        return None
    blocked: dict[str, int] = {}
    for v, kind in me.buildings():
        if h in gs.board.vertex_hexes(v):
            for r, n in production(h.terrain, kind, gs.mode).items():
                blocked[r] = blocked.get(r, 0) + n
    if not blocked:
        return None
    return 6 * PIPS[h.number] / 36 * _value(gs, blocked) + 1.0


def inventor_movable(gs: GameState, h) -> bool:
    """Inventor may swap 3/4/5/9/10/11 (never 2, 12, 6, 8), never a gold hex. Great Canal
    (owner-confirmed 2026-10-06): the face-down chips of barren farmland MAY be swapped while the
    canal is dug. A barren chip pays nothing while barren (gs.income), so the swap only helps by
    moving a good number away from a rival or onto our producing hex."""
    if h.number not in (3, 4, 5, 9, 10, 11) or h.terrain == "gold":
        return False
    # owner-confirmed 2026-10-06: barren chips may move (INVENTOR_BARREN True, cfg inventor_barren);
    # game 0959's unringed pick is handled by choose_inventor(allowed=ringed chips)
    if not INVENTOR_BARREN and getattr(h, "barren", False) and gs.mode is not None and gs.mode.canal \
            and gs.canal_progress < gs.canal_total:
        return False
    return True


INVENTOR_BARREN = True


def choose_inventor(gs: GameState, me: PlayerState, allowed=None, force: bool = False) -> dict | None:
    """Owner 2026-10-06: take the leader's best number for our own hex -- every swap is scored as
    our gain plus the leader's loss (full weight) plus half the loss of any other rival at or above
    our score; a swap that only moves a good chip off the leader still counts.
    allowed: hex coords the game rings (game 0959), None = any movable chip. force: the card is
    already played and the prompt waits -- take the best pair even without a 1-pip gain."""
    hexes = [h for h in gs.board.hexes.values() if inventor_movable(gs, h)
             and (allowed is None or h.coord in allowed)]
    lead = leader(gs, me)
    mine = score(gs, me)
    rivals = [(p, 1.0 if p is lead else 0.5) for p in gs.players
              if p is not me and (p is lead or score(gs, p) >= mine)]
    base_me = sum(gs.income(me).values())
    base = {p.name: sum(gs.income(p).values()) for p, _ in rivals}
    best, best_s = None, (-1e9 if force else 1.0 / 36)   # any real gain: at least 1 pip net (owner: never hold)
    for i, a in enumerate(hexes):
        for b in hexes[i + 1:]:
            if PIPS[a.number] == PIPS[b.number]:
                continue
            a.number, b.number = b.number, a.number
            try:
                d = sum(gs.income(me).values()) - base_me
                for p, w in rivals:
                    d -= w * (sum(gs.income(p).values()) - base[p.name])
            finally:
                a.number, b.number = b.number, a.number
            if d > best_s + 1e-9:
                best, best_s = (a.coord, b.coord), d
    return {"swap": best} if best else None


def _hexes_of(gs, me, terrain):
    return [h for h in _my_hexes(gs, me) if h.terrain == terrain]


def choose_irrigation(gs, me):
    n = len(_hexes_of(gs, me, "fields"))      # barren fields still count (rules_research.md)
    return {"gain": {"grain": 2 * n}} if n else None


def choose_mining(gs, me):
    n = len(_hexes_of(gs, me, "mountains"))
    return {"gain": {"ore": 2 * n}} if n else None


def choose_crane(gs, me) -> dict | None:
    for track in TRACK_ORDER:
        lvl = me.improvements.get(track, 0)
        if lvl >= 5:
            continue
        com = IMPROVEMENT_TRACKS[track]
        # level n+1 costs n+1; Crane takes 1 off, so level 1 is FREE (owner-confirmed 2026-10-06:
        # a Crane on a level-0 track is a free level)
        if me.hand.get(com, 0) >= lvl and me.cities:
            return {"track": track}
    return None


def choose_engineer(gs, me) -> dict | None:
    if len(me.walls) >= 3:
        return None
    open_cities = [v for v in me.cities if v not in me.walls]
    if not open_cities:
        return None
    return {"city": max(open_cities, key=gs.board.vertex_pips)}


def choose_medicine(gs, me) -> dict | None:
    if not me.settlements or len(me.cities) >= gs.mode.max_cities:
        return None
    if not _can_afford(me.hand, {"ore": 2, "grain": 1}):
        return None
    return {"settlement": max(me.settlements, key=gs.board.vertex_pips)}


def choose_smith(gs, me) -> dict | None:
    cap = 3 if me.improvements.get("politics", 0) >= 3 else 2
    per = {lvl: sum(1 for k in me.knights.values() if k.level == lvl) for lvl in (2, 3)}
    # 2 pieces per level: a knight can only go up if a piece of the next level is free
    # (sim 2026-10-04: Smith sat in hand 30 turns with 2 strong knights already out)
    ks = [(v, k) for v, k in me.knights.items() if k.level < cap and per.get(k.level + 1, 0) < 2]
    if not ks:
        return None
    ks.sort(key=lambda vk: (not vk[1].active, vk[1].level))
    # review 0.3.1: the two picks share the free pieces -- two basic knights with one strong
    # piece left get one promotion, not two (the second tap failed)
    free = {lvl: 2 - per.get(lvl, 0) for lvl in (2, 3)}
    picks = []
    for v, k in ks:
        if free.get(k.level + 1, 0) > 0 and len(picks) < 2:
            free[k.level + 1] -= 1
            picks.append(v)
    return {"knights": picks}


def choose_warlord(gs, me) -> dict | None:
    idle = sum(1 for k in me.knights.values() if not k.active)
    canal_open = gs.mode.canal and gs.canal_progress < gs.canal_total
    if idle >= 2 or (idle >= 1 and (barbarians_near(gs) or defence_gap(gs, me) > 0 or canal_open)):
        return {}
    return None


def choose_road_building(gs, me) -> dict | None:
    """Two free roads OR ships (Seafarers rules, audit row 39). "edges" lists both in build
    order; "ships" says which of them are ships. Piece caps (15 roads / 15 ships) respected.

    Owner 2026-10-08: it built "ships to nowhere" instead of the road alongside the canal it
    had room for at both ends. Both pieces are chosen together (the 2nd may extend the 1st) and
    scored by the corners they newly reach: settleable pips (+3 for a 7+ spot), +2 per undug
    canal hex on a land corner (a knight digs there), +3 for a gold-island corner (a knight
    collects there); a piece that only reaches open sea scores -1.5, and a road beats a ship
    on a tie."""
    p_roads, p_ships = set(me.roads), set(me.ships)
    m = gs.mode
    b = gs.board
    net = {v for e in p_roads | p_ships for v in e} | {v for v, _ in me.buildings()}

    def vval(v):
        s = 0.0
        if gs.can_settle(v):
            pips = b.vertex_pips(v)
            s += pips + (3 if pips >= 7 else 0)
        if b.is_land_vertex(v):
            for h in b.vertex_hexes(v):
                if b.is_canal(h) and h.coord not in gs.canal_dug:
                    s += 2
            if m.canal and any(h.terrain == "gold" for h in b.vertex_hexes(v)):
                s += 3
        else:
            s -= 1.5                           # open sea: a ship to nowhere
        return s

    def cands(skip):
        out = []
        if len(me.roads) < m.max_roads:
            out += [(e, False) for e in gs.legal_road_edges(me) if e not in skip]
        if m.seafarers and len(me.ships) < m.max_ships:
            out += [(e, True) for e in gs.legal_road_edges(me, ship=True) if e not in skip]
        return out

    def place(e, ship, add=True):
        (me.ships if ship else me.roads).add(e) if add else (me.ships if ship else me.roads).discard(e)

    def score(pieces):
        new = {v for e, _ in pieces for v in e} - net
        return sum(vval(v) for v in new) + sum(0.0 if ship else 0.5 for _, ship in pieces)

    best, best_s = [], None
    try:
        for e1, s1 in cands(()):
            place(e1, s1)
            try:
                seconds = cands({e1})
            finally:
                place(e1, s1, add=False)
            for pair in ([(e1, s1), c] for c in seconds) if seconds else ([(e1, s1)],):
                sc = score(pair)
                if best_s is None or sc > best_s:
                    best, best_s = pair, sc
    finally:
        me.roads.clear()
        me.roads.update(p_roads)
        me.ships.clear()
        me.ships.update(p_ships)
    if not best:
        return None
    return {"edges": [e for e, _ in best], "ships": [e for e, ship in best if ship]}


def merchant_hex_value(gs, me, h) -> float:
    """owner 2026-10-06: the Merchant goes where it pays most -- 2:1 on the card we get most of
    (our pips on that hex, a city counts double) plus what we already hold of it; nothing for a
    card that is already 2:1 for us (own harbour) or a field the canal left barren. The robber's
    hex is fine (owner: "it can stay on the same hex"). The +1 VP is the same on any hex."""
    from .moves import bank_ratio
    r = TERRAIN_RES[h.terrain]
    if h.fog:
        return -1.0
    if gs.mode.canal and getattr(h, "barren", False) and gs.canal_progress < gs.canal_total:
        return -1.0
    per_roll = 0.0
    for v, kind in me.buildings():
        if h in gs.board.vertex_hexes(v):
            per_roll += PIPS.get(h.number or 0, 0) / 36 * (2 if kind == "city" else 1)
    gain = 0.0 if bank_ratio(gs, r) <= 2 else 1.0
    return gain * (per_roll * 10 + me.hand.get(r, 0)) + per_roll


def choose_merchant(gs, me) -> dict | None:
    hexes = [h for h in _my_hexes(gs, me) if h.terrain in TERRAIN_RES and h.terrain != "gold"]
    if not hexes:
        return None
    best = max(hexes, key=lambda h: (merchant_hex_value(gs, me, h), PIPS.get(h.number or 0, 0)))
    return {"hex": best.coord}


def fleet_build(gs, me) -> tuple[str, str] | None:
    """owner 2026-10-06 "bot doesn't know how to use the merchant fleet": play it when 2:1 on one
    card we hold turns the hand into a build we can't make now (city > improvement > knight >
    settlement). Returns (card to trade 2:1, what it buys) or None."""
    from .moves import bank_ratio
    from .rules import IMPROVEMENT_TRACKS, improvement_cost, missing
    hand = {r: n for r, n in me.hand.items() if n > 0}
    targets = []
    if gs.mode.ck and barbarians_near(gs) and defence_gap(gs, me) > 0:
        # owner 2026-10-08 "it needs to save the city": barbarians close and a city would fall --
        # the Fleet buys the knight's grain (or the knight) before anything else
        if any(not k.active for k in me.knights.values()):
            targets.append(("activate", COSTS.get("activate", {"grain": 1})))
        else:
            targets.append(("knight", COSTS.get("knight", {})))
    if me.settlements:
        targets.append(("city", COSTS["city"]))
    if gs.mode.ck and me.cities:
        for t in sorted(IMPROVEMENT_TRACKS, key=lambda t: (me.improvements or {}).get(t, 0)):
            lvl = (me.improvements or {}).get(t, 0)
            if lvl < 5:
                targets.append((f"improve {t}", improvement_cost(t, lvl)))
    if gs.mode.ck:
        targets.append(("knight", COSTS.get("knight", {})))
    targets.append(("settlement", COSTS["settlement"]))
    for name, cost in targets:
        need = missing(dict(hand), cost)
        if not need:
            continue                                  # buildable already: no fleet needed
        for r, n in sorted(hand.items(), key=lambda x: card_value(gs, x[0])):
            if r in need or bank_ratio(gs, r) <= 2:
                continue
            spare = n - cost.get(r, 0)
            if spare >= 2 * sum(need.values()):
                return r, name
    return None


def fleet_pair_target(gs, me) -> str | None:
    """owner 2026-10-06 "with merchant fleet, bot needs to trade first and get two of a kind before
    using the card, otherwise it may be wasted": holding the Fleet with no 2:1 that builds, the card
    whose second copy (one trade away) makes a fleet build. None when the Fleet already has a use."""
    if "merchant_fleet" not in (me.progress_hand or []) or choose_merchant_fleet(gs, me) is not None:
        return None
    from .moves import bank_ratio
    best = None
    for r in sorted(me.hand, key=lambda r: -card_value(gs, r)):
        if me.hand.get(r, 0) % 2 == 0 or bank_ratio(gs, r) <= 2:
            continue                                 # pairs come from odd counts; a 2:1 port needs no fleet
        me.hand[r] += 1
        try:
            fb = fleet_build(gs, me)
        finally:
            me.hand[r] -= 1
        if fb is not None and fb[0] == r:
            rank = 0 if fb[1] == "city" else 1 if fb[1].startswith("improve") else 2 if fb[1] == "knight" else 3
            if best is None or rank < best[0]:
                best = (rank, r)                     # the pair whose fleet build is best (city first)
    return best[1] if best else None


def choose_merchant_fleet(gs, me, force=False) -> dict | None:
    kinds = [(n, r) for r, n in me.hand.items() if n >= 2]
    if not kinds:
        return None
    fb = fleet_build(gs, me)
    if fb is not None:
        return {"resource": fb[0]}           # 2:1 on it buys fb[1] this turn
    n, r = max(kinds, key=lambda x: (x[0], -card_value(gs, x[1])))
    if n >= 4 or force:
        return {"resource": r}
    return None


def choose_commercial_harbor(gs, me) -> dict | None:
    opps = [p for p in gs.players if p is not me and p.total_cards() > 0]
    # review 0.3.1: grain is never given to a player -- each opponent takes one card, so play the
    # card only when every one of them can get a non-grain card
    spare = sorted((r for r in RESOURCES if r != "grain" for _ in range(me.hand.get(r, 0))),
                   key=lambda r: card_value(gs, r))
    if not opps or len(spare) < len(opps):
        return None
    return {"give": spare[: len(opps)]}


def choose_master_merchant(gs, me) -> dict | None:
    ahead = [p for p in gs.players if p is not me and p.vp_visible > me.vp_visible and p.total_cards() > 0]
    if not ahead:
        return None
    led = getattr(gs, "hands", None)
    if led is not None:             # owner 2026-10-06: take the 2 best cards the ledger sees
        return {"player": max(ahead, key=lambda p: (min(p.total_cards(), 2),
                                                    round(led.best_cards_value(p.name, 2, lambda r: card_value(gs, r), gs), 3),
                                                    score(gs, p))).name}
    return {"player": max(ahead, key=lambda p: (min(p.total_cards(), 2), score(gs, p))).name}


def _est(p: PlayerState, r: str, kinds: int) -> float:
    if p.hand.get(r):
        return p.hand[r]
    return p.total_cards() / kinds          # unknown: spread evenly


def monopoly_take(gs, me, r: str, per_player: float) -> float:
    """Expected cards of r taken from all opponents, at most per_player each."""
    led = getattr(gs, "hands", None)
    return sum(led.expected_take(p.name, r, per_player, gs) if led is not None and led.knows(p.name)
               and not sum(p.hand.values()) else min(per_player, gs.est_hand(p).get(r, 0.0))
               for p in gs.players if p is not me)


def _monopoly_pick(gs, me, cards, per_player: float, bonus) -> tuple[str, float, bool]:
    """owner 2026-10-06: first what we know every rival holds the full amount of (2 each for
    Resource Monopoly, 1 for Trade Monopoly), then the most cards; `bonus` favours cards that
    are worth more to us (grain in the canal, a card we trade 2:1)."""
    opps = [p for p in gs.players if p is not me]
    take = {r: monopoly_take(gs, me, r, per_player) for r in cards}
    full = {r: bool(opps) and all(gs.est_hand(p).get(r, 0.0) >= per_player for p in opps) for r in cards}
    best = max(cards, key=lambda r: (full[r], take[r] + bonus(r), card_value(gs, r)))
    return best, take[best], full[best]


def _two_to_one(gs, r) -> bool:
    from .moves import bank_ratio
    return bank_ratio(gs, r) <= 2        # own harbour, the Merchant's hex, Trade 3 commodities


def choose_resource_monopoly(gs, me) -> dict | None:
    """Owner 2026-10-03: up to 2 from each player (6 with 3 opponents). Owner 2026-10-06: ask for
    the card every rival is known to hold 2 of; else the most cards, leaning to grain (canal) and
    cards we trade 2:1 (Merchant, Merchant Fleet, own harbour) since those turn into anything.
    Re-asked after each play, so a second copy sees the new hand."""
    need = {r: n for r, n in (getattr(gs, "plan_missing", None) or {}).items() if n > 0}
    fleet = getattr(gs, "fleet_resource", None)

    def bonus(r):
        b = 0.0
        if r == "grain" and gs.mode.canal:
            b += 1.0
        if r == fleet or _two_to_one(gs, r):
            b += 1.0
        if need.get(r):
            b += 0.5
        return b

    # a haul that covers everything the plan is missing of a card finishes the build: that first
    covers = [r for r, n in need.items() if r in RESOURCES and monopoly_take(gs, me, r, 2.0) >= n]
    if covers:
        return {"resource": max(covers, key=lambda r: (need[r], monopoly_take(gs, me, r, 2.0)))}
    best, take, full = _monopoly_pick(gs, me, list(RESOURCES), 2.0, bonus)
    return {"resource": best} if full or take >= 2.0 or (need.get(best) and take >= 0.8) else None


def choose_trade_monopoly(gs, me) -> dict | None:
    """1 commodity from each player: the one every rival holds first, then the most, leaning to
    the commodity our next improvement needs (owner 2026-10-06, same rule as Resource Monopoly)."""
    from .rules import IMPROVEMENT_TRACKS
    lv = me.improvements or {}
    near = {c for t, c in IMPROVEMENT_TRACKS.items() if lv.get(t, 0) < 5} if me.cities else set()

    def bonus(c):
        return (0.5 if c in near else 0.0) + (0.5 if _two_to_one(gs, c) else 0.0)

    best, take, full = _monopoly_pick(gs, me, list(COMMODITIES), 1.0, bonus)
    return {"commodity": best} if full or take >= 1.5 else None


def choose_bishop(gs, me) -> dict | None:
    b = gs.board
    safe = gs.robber_protected() if gs.mode.friendly_robber else set()
    lead = leader(gs, me)

    def dmg(h):
        d = 0.0
        for v in b.vertices:
            if h.coord not in [x.coord for x in b.vertex_hexes(v)]:
                continue
            o = gs.owner_at(v)
            if o is None:
                continue
            mult = 2 if v in o.cities else 1
            w = -3.0 if o is me else (2.0 if o is lead else 1.0)
            steal = 1.0 if (o is not me and o.total_cards() and o.name not in safe) else 0.0
            d += w * mult * PIPS[h.number] / 5 + steal
        return d
    cands = [h for h in b.hexes.values() if h.is_land and h.number and h.coord != b.robber]
    if not cands:
        return None
    if gs.mode.friendly_robber:         # owner: no hex touching a player with 4 VP or fewer
        ok = set(gs.friendly_robber_hexes([h.coord for h in cands], me))
        cands = [h for h in cands if h.coord in ok]
    h = max(cands, key=dmg)
    return {"hex": h.coord} if dmg(h) > 0 else None


def choose_deserter(gs, me, by_spot: bool | None = None) -> dict | None:
    """The card names a player; that player removes a knight of THEIR choice (live: player
    dialog only, no knight pick). Owner 2026-10-09 "when deleting a knight of another player, go
    for the best position if needed or the largest one otherwise": the leader-first pick stays
    unless a rival knight on a spot that matters (knight_targets) belongs to another player.
    "knight" is the knight we are after (log only: the victim chooses)."""
    opps = [p for p in gs.players if p is not me and p.knights]
    if not opps:
        return None
    lead = leader(gs, me)
    if lead in opps and lead.vp_visible >= me.vp_visible + 3:
        t = lead                                 # owner 2026-10-04: slow the runaway leader first
    else:
        t = max(opps, key=lambda p: (gs.defence_strength(p), score(gs, p)))
    if not (kt.ENABLED if by_spot is None else by_spot):
        return {"player": t.name}
    from_t = kt.pick_knight_target(gs, me, [(t, v) for v in t.knights])
    if not kt.spot_matters(gs, me, from_t[1]):
        best = kt.pick_knight_target(gs, me, [(p, v) for p in opps for v in p.knights])
        if kt.spot_matters(gs, me, best[1]):
            return {"player": best[0].name, "knight": best[1]}
    return {"player": t.name, "knight": from_t[1]}


def _open_roads(gs, p: PlayerState):
    """Open roads AND ships (owner-confirmed 2026-10-06: the Diplomat can take ships too): one
    end touches none of the owner's pieces."""
    out = []
    for e in p.roads | p.ships:
        for v in e:
            if gs.owner_at(v) is not None or any(v in q.knights for q in gs.players):
                continue
            if not any(x in p.roads or x in p.ships for x in gs.board.vertex_edges(v) if x != e):
                out.append(e)
                break
    return out


def choose_diplomat(gs, me) -> dict | None:
    """Remove an opponent's open road. Owner 2026-10-06 "never hold cards it can use": any
    rival's open road is a target; best is the Longest Road holder (2 VP at stake), then the
    leader, then a road that touches our network (it blocks our way)."""
    lead = leader(gs, me)
    mine_v = {v for e in me.roads | me.ships for v in e} | {v for v, _ in me.buildings()}
    best, best_s = None, 0.0
    for p in gs.players:
        if p is me:
            continue
        for e in sorted(_open_roads(gs, p)):
            s = 1.0 + (3.0 if p.has_longest_road else 0.0) + (2.0 if p is lead else 0.0) \
                + (1.5 if any(v in mine_v for v in e) else 0.0)
            if s > best_s:
                best, best_s = e, s
    if best is not None:
        return {"edge": best}
    return _diplomat_own_move(gs, me)


def _diplomat_own_move(gs, me) -> dict | None:
    """No rival road to take: the Diplomat may move one of OUR open roads or ships (C&K rule;
    ships owner-confirmed 2026-10-06). Owner 2026-10-06 "never hold cards it can use": move it
    when the new spot opens a settlement corner or lengthens our longest road. A ship moves to a
    ship edge, a road to a road edge. UI UNCONFIRMED: the runner taps the old edge, then the new
    one (probe-captured)."""
    spots0 = len(gs.legal_settlement_spots(me))
    len0 = gs.longest_road_len(me)
    best, best_s = None, 0.0
    for old in sorted(_open_roads(gs, me)):
        ship = old in me.ships
        net = me.ships if ship else me.roads
        net.discard(old)
        try:
            for new in sorted(gs.legal_road_edges(me, ship=ship)):
                if new == old:
                    continue
                net.add(new)
                gain = (len(gs.legal_settlement_spots(me)) - spots0) * 2.0 \
                    + (gs.longest_road_len(me) - len0)
                net.discard(new)
                if gain > best_s:
                    best, best_s = (old, new, ship), gain
        finally:
            net.add(old)
    if best is None:
        return None
    out = {"edges": [best[0], best[1]], "own": True}
    if best[2]:
        out["ship"] = True
    return out


def choose_intrigue(gs, me) -> dict | None:
    # knights next to our roads OR ships (owner-confirmed 2026-10-06)
    mine_v = {v for e in me.roads | me.ships for v in e}
    lead = leader(gs, me)
    best, best_s = None, 0.0
    for p in gs.players:
        if p is me:
            continue
        for v, k in p.knights.items():
            if v not in mine_v:
                continue
            s = k.level + (2 if p is lead else 0) + (1 if k.active else 0)
            # owner 2026-10-04: a knight breaking our road costs us length (and maybe the 2 VP)
            before = gs.longest_road_len(me)
            saved, p.knights = p.knights, {x: y for x, y in p.knights.items() if x != v}
            gain = gs.longest_road_len(me) - before
            p.knights = saved
            s += min(4, gain) + (3 if gain and me.longest_road + gain >= 5 else 0)
            if any(gs.board.is_canal(h) and h.coord not in gs.canal_dug for h in gs.board.vertex_hexes(v)):
                s += 2
            if s > best_s:
                best, best_s = v, s
    return {"knight_at": best} if best is not None else None


def choose_saboteur(gs, me) -> dict | None:
    # owner-confirmed 2026-10-06: everyone with as many VP as us OR MORE halves (round down)
    hit = [p for p in gs.players if p is not me and p.vp_visible >= me.vp_visible]
    loss = sum(p.total_cards() // 2 for p in hit)
    return {} if hit and loss >= 2 else None      # we are never hit ourselves


def choose_spy(gs, me) -> dict | None:
    opps = [p for p in gs.players if p is not me and p.progress_cards > 0]
    if not opps:
        return None
    return {"player": max(opps, key=lambda p: (p.progress_cards, score(gs, p))).name}


def choose_wedding(gs, me) -> dict | None:
    ahead = [p for p in gs.players if p is not me and p.vp_visible > me.vp_visible and p.total_cards() > 0]
    return {"from": [p.name for p in ahead]} if ahead else None


CHOOSERS = {
    "alchemist": choose_alchemist, "inventor": choose_inventor, "irrigation": choose_irrigation,
    "mining": choose_mining, "crane": choose_crane, "engineer": choose_engineer,
    "medicine": choose_medicine, "smith": choose_smith, "road_building": choose_road_building,
    "merchant": choose_merchant, "merchant_fleet": choose_merchant_fleet,
    "commercial_harbor": choose_commercial_harbor, "master_merchant": choose_master_merchant,
    "resource_monopoly": choose_resource_monopoly, "trade_monopoly": choose_trade_monopoly,
    "bishop": choose_bishop, "deserter": choose_deserter, "diplomat": choose_diplomat,
    "intrigue": choose_intrigue, "saboteur": choose_saboteur, "spy": choose_spy,
    "warlord": choose_warlord, "wedding": choose_wedding,
    "printer": lambda gs, me: {}, "constitution": lambda gs, me: {},
}

# "main" order: VP, steal/gain first (so it gets spent), then build helpers, then conversion.
MAIN_ORDER = ["printer", "constitution", "merchant",
              "wedding", "master_merchant", "spy", "saboteur", "bishop",
              "resource_monopoly", "trade_monopoly", "commercial_harbor", "irrigation", "mining",
              "warlord", "deserter", "smith", "intrigue", "diplomat", "inventor",
              "medicine", "crane", "engineer", "road_building", "merchant_fleet"]
DEFENCE = ["warlord", "smith", "deserter"]
# how much a card is worth holding for a later turn (for the 5th-card discard)
KEEP_VALUE = {"alchemist": 5, "inventor": 4, "warlord": 4, "deserter": 4, "smith": 3, "medicine": 3,
              "crane": 3, "spy": 3, "wedding": 3, "master_merchant": 3, "resource_monopoly": 3,
              "merchant_fleet": 2, "diplomat": 1, "intrigue": 2}


TIMED = {"warlord", "alchemist", "inventor", "saboteur"}   # held on purpose for the right moment


def keep_value(gs: GameState | None, me: PlayerState | None, card: str, lead_dead: bool = False) -> float:
    """KEEP_VALUE, halved for a card we can't play now (sim 2026-10-04: Medicine, Smith and
    Irrigation sat unplayable for 30+ turns while a Warlord or Deserter got discarded).
    lead_dead (Brain.wedding_dead_when_leading), stage 2026-10-09 07:22: Wedding held all game
    while we led -- with nobody ahead it is worth nothing, the first card to go."""
    if lead_dead and card == "wedding" and gs is not None and me is not None \
            and all(p.vp_visible <= me.vp_visible for p in gs.players if p is not me):
        return 0.0
    v = KEEP_VALUE.get(card, 1)
    if gs is None or me is None or card in TIMED or card not in CHOOSERS:
        return v
    try:
        dead = not can_play(gs, me, card, "main", attacked=True)[0] or CHOOSERS[card](gs, me) is None
    except Exception:
        dead = False
    return v * 0.5 if dead else v


# Why a card with no play now stays in hand (owner 2026-10-06: "should never hold cards it can
# use" -- a card is held only for a stated reason, logged by brain.progress_play). One line per
# chooser's None; docs/progress_cards_playbook.md has the full hold conditions.
NO_USE = {
    "alchemist": "no sum pays us more than the leader",
    "crane": "no improvement affordable this turn even 1 commodity cheaper (or no city)",
    "engineer": "no unwalled city (or 3 walls already)",
    "inventor": "no legal swap gains us 1 pip net",
    "irrigation": "no fields next to our buildings",
    "mining": "no mountains next to our buildings",
    "medicine": "need 2 ore + 1 grain and a settlement (city cap not reached)",
    "smith": "no knight can go up a level (pieces/politics 3)",
    "road_building": "no legal road or ship edge",
    "merchant": "no land hex next to our buildings",
    "merchant_fleet": "no 2:1 trade finishes a build and no 4+ of one card: trade for a pair first",
    "commercial_harbor": "no non-grain card for each opponent with cards",
    "master_merchant": "no player with more VP holding cards",
    "resource_monopoly": "expected haul under 2 cards",
    "trade_monopoly": "expected haul under 1.5 commodities",
    "bishop": "no hex hurts rivals more than us (friendly robber)",
    "deserter": "no opponent has a knight",
    "diplomat": "no opponent has an open road and moving ours gains nothing",
    "intrigue": "no opponent knight on a junction our road touches",
    "saboteur": "targets with VP >= ours would lose under 2 cards",
    "spy": "no opponent holds progress cards",
    "warlord": "1 idle knight, barbarians far: saved for the turn before an attack",
    "wedding": "no player with more VP holding cards",
}
GAIN_CARDS = ("irrigation", "mining", "resource_monopoly", "trade_monopoly", "commercial_harbor",
              "wedding", "master_merchant")


# ---------------------------------------------------------------------------------------
# Play preconditions: when a card CAN'T be played (legality, not strategy)
# ---------------------------------------------------------------------------------------
# Owner 2026-10-06: "bot should know when a card can not be used ... so it does not try them if
# he can't use them". can_play() is the gate in front of every chooser (choose_play) and of the
# runner's drag (runner.play_progress): a card whose precondition fails is never tried. The
# choosers then decide whether a playable card is worth playing now (NO_USE).
# CANT_PLAY: card -> (when it can't be played, owner_confirmed). False = from the card text or
# the C&K rulebook but not checked in Catan Universe; the check is the permissive reading.
# Shown in docs/progress_cards_playbook.md ("Can't be played when").
CANT_PLAY: dict[str, tuple[str, bool]] = {
    "alchemist": ("after the dice are rolled (before the roll only)", True),
    "crane": ("we have no city, or no track below level 5 whose next level we can pay "
              "with 1 commodity less (level 1 is free)", True),
    "engineer": ("we have no city without a wall, or 3 walls already", True),
    "inventor": ("fewer than two number chips that may move (3/4/5/9/10/11, not on gold)", True),
    "irrigation": ("no fields hex next to our settlements or cities (barren canal fields count)", True),
    "mining": ("no mountains hex next to our settlements or cities", False),
    "medicine": ("no settlement, the city cap is reached, or we can't pay 2 ore + 1 grain", True),
    "smith": ("no knight can go up: level cap (strong without politics 3) or both pieces of the "
              "next level are on the board", True),
    "road_building": ("no legal road or ship edge, or all road and ship pieces are out", True),
    "commercial_harbor": ("not one non-grain resource for each opponent with cards", True),
    "master_merchant": ("no opponent has more VP than us and holds cards", True),
    "merchant": ("no land hex (not gold) next to our settlements or cities", True),
    "merchant_fleet": ("never refused: it applies for the turn, so we can trade into a pair after", True),
    "resource_monopoly": ("no opponent holds cards (the game allows it, but it gains nothing: hold it)", True),
    "trade_monopoly": ("no opponent holds cards (the game allows it, but it gains nothing: hold it)", True),
    "bishop": ("before the first barbarian attack (robber locked), or no hex to move it to", True),
    "deserter": ("no opponent has a knight on the board, or we have no free knight piece to set "
                 "(a taken knight drops a level when we have both of its level out)", True),
    "diplomat": ("no open road or ship on the board (one end touching none of its owner's pieces)", True),
    "intrigue": ("no opponent knight on a junction one of our roads or ships touches", True),
    "saboteur": ("no opponent with as many VP as us or more holds 2+ cards", True),
    "spy": ("no opponent holds progress cards", True),
    "warlord": ("we have no inactive knight (the game allows it, but it gains nothing: hold it)", True),
    "wedding": ("no opponent has more VP than us and holds cards", True),
    "printer": ("never played: 1 VP, revealed when drawn", True),
    "constitution": ("never played: 1 VP, revealed when drawn", True),
}


def _opps(gs, me):
    return [p for p in gs.players if p is not me]


def _land_hexes_near(gs, me, terrains=None):
    out = []
    for h in _my_hexes(gs, me):
        if not h.is_land or getattr(h, "fog", False):
            continue
        if terrains is None and h.terrain in TERRAIN_RES or terrains is not None and h.terrain in terrains:
            out.append(h)
    return out


def can_play(gs: GameState, me: PlayerState, card: str, phase: str = "main",
             attacked: bool | None = None) -> tuple[bool, str]:
    """(True, "") when the game lets us play `card` now, else (False, why). Legality only: a
    playable card may still be held for strategy (the choosers). `attacked` overrides
    gs.barbarians_attacked (the sim tracks attacks itself)."""
    try:
        c = card if card in CARDS else card_key(card)
    except KeyError:
        return False, f"unknown card {card!r}"
    if c in VP_CARDS:
        return False, CANT_PLAY[c][0]
    if phase not in ("pre_roll", "main", "end_turn"):
        return False, "progress cards are played on our own turn only"
    if c == "alchemist":
        return (True, "") if phase == "pre_roll" else (False, CANT_PLAY[c][0])
    if phase == "pre_roll":
        return False, "after the roll (only the Alchemist goes before the dice)"
    opps = _opps(gs, me)
    with_cards = [p for p in opps if p.total_cards() > 0]
    ok = True
    if c == "crane":
        lv = me.improvements or {}
        ok = bool(me.cities) and any(lv.get(t, 0) < 5 and me.hand.get(IMPROVEMENT_TRACKS[t], 0) >= lv.get(t, 0)
                                     for t in IMPROVEMENT_TRACKS)
    elif c == "engineer":
        ok = len(me.walls) < 3 and any(v not in me.walls for v in me.cities)
    elif c == "inventor":
        ok = sum(1 for h in gs.board.hexes.values() if h.is_land and inventor_movable(gs, h)) >= 2
    elif c == "irrigation":
        # canal: barren fields still count (owner-confirmed 2026-10-06)
        ok = bool(_land_hexes_near(gs, me, ("fields",)))
    elif c == "mining":
        ok = bool(_land_hexes_near(gs, me, ("mountains",)))
    elif c == "medicine":
        ok = bool(me.settlements) and len(me.cities) < gs.mode.max_cities \
            and _can_afford(me.hand, {"ore": 2, "grain": 1})
    elif c == "smith":
        cap = 3 if (me.improvements or {}).get("politics", 0) >= 3 else 2
        per = {lvl: sum(1 for k in me.knights.values() if k.level == lvl) for lvl in (2, 3)}
        ok = any(k.level < cap and per.get(k.level + 1, 0) < 2 for k in me.knights.values())
    elif c == "road_building":
        m = gs.mode
        ok = (len(me.roads) < m.max_roads and bool(gs.legal_road_edges(me))) or \
             (m.seafarers and len(me.ships) < m.max_ships and bool(gs.legal_road_edges(me, ship=True)))
    elif c == "commercial_harbor":
        # grain is never given: one non-grain card for each opponent that holds cards
        ok = bool(with_cards) and sum(me.hand.get(r, 0) for r in RESOURCES if r != "grain") >= len(with_cards)
    elif c in ("master_merchant", "wedding"):
        # the owner's example: "has more points than everyone and card asks it to have less
        # than at least one player"
        ok = any(p.vp_visible > me.vp_visible for p in with_cards)
    elif c == "merchant":
        # owner-confirmed 2026-10-06: the merchant never goes on gold
        ok = any(h.terrain != "gold" for h in _land_hexes_near(gs, me))
    elif c == "merchant_fleet":
        ok = True                                # owner-confirmed 2026-10-06: no pair needed to play it
    elif c in ("resource_monopoly", "trade_monopoly"):
        ok = bool(with_cards)
    elif c == "bishop":
        att = gs.barbarians_attacked if attacked is None else attacked
        if gs.mode.ck and not att:
            return False, "robber locked until the first barbarian attack"
        ok = any(h.is_land and h.number and h.coord != gs.board.robber for h in gs.board.hexes.values())
    elif c == "deserter":
        # owner-confirmed 2026-10-06: refused with no free piece to set; a taken knight drops to the
        # next free level (a strong one becomes basic when both our strong are out)
        per = {lv: sum(1 for k in me.knights.values() if k.level == lv) for lv in (1, 2, 3)}
        ok = any(p.knights for p in opps) and any(n < 2 for n in per.values())
    elif c == "diplomat":
        ok = any(_open_roads(gs, p) for p in gs.players)
    elif c == "intrigue":
        mine_v = {v for e in me.roads | me.ships for v in e}
        ok = any(v in mine_v for p in opps for v in p.knights)
    elif c == "saboteur":
        ok = any(p.vp_visible >= me.vp_visible and p.total_cards() >= 2 for p in opps)
    elif c == "spy":
        ok = any(p.progress_cards > 0 for p in opps)
    elif c == "warlord":
        ok = any(not k.active for k in me.knights.values())
    return (True, "") if ok else (False, CANT_PLAY[c][0])


def choose_play(gs: GameState, me: PlayerState, hand_cards: list[str], phase: str,
                why: dict | None = None, lead_dead: bool = False) -> list[tuple[str, dict]]:
    """Which progress cards to play now, in order, with the choices each needs. `why`, when
    given, is filled with {card: reason} for every card left in hand by this call."""
    hand = [card_key(c) for c in hand_cards]
    out: list[tuple[str, dict]] = []
    # owner 2026-10-06: VP cards (Printer, Constitution) are revealed when drawn and never played
    vp: list[tuple[str, dict]] = []
    why = {} if why is None else why
    for c in hand:
        if c in VP_CARDS:
            why[c] = "VP card: revealed when drawn, never played"

    if phase not in ("pre_roll", "main", "end_turn"):
        # not our turn: only VP cards reveal. Owner: a 5th card drawn on someone else's turn
        # must go at once (the roller alone may hold 5 until its turn ends).
        left = [c for c in hand if c not in VP_CARDS]
        for c in left:
            why[c] = "progress cards are played on our own turn only"
        if len(left) > PROGRESS_HAND_LIMIT:
            vp.append(("discard", {"card": min(left, key=lambda c: keep_value(gs, me, c, lead_dead))}))
        return vp
    if phase == "pre_roll":
        if "alchemist" in hand and not (ok := can_play(gs, me, "alchemist", phase))[0]:
            why["alchemist"] = f"can't play — {ok[1]}"
        elif "alchemist" in hand and (a := choose_alchemist(gs, me)) is not None:
            out.append(("alchemist", a))
        elif "alchemist" in hand:
            why["alchemist"] = NO_USE["alchemist"]
        for c in hand:
            if c not in VP_CARDS and c != "alchemist":
                why[c] = "after the roll (only the Alchemist goes before the dice)"
        return vp + out

    order = [c for c in MAIN_ORDER if c not in VP_CARDS]
    if barbarians_near(gs) and defence_gap(gs, me) > 0:
        # losing a city is near fatal: defence cards jump the queue
        order = [c for c in order if c in VP_CARDS] + DEFENCE + [c for c in order if c not in DEFENCE and c not in VP_CARDS]

    over = hand_size(me) > discard_limit(gs, me)
    used: set[int] = set()
    for card in order:
        for i, c in enumerate(hand):
            if c != card or i in used:
                continue
            if phase == "end_turn" and card in GAIN_CARDS and hand_size(me) + 2 > discard_limit(gs, me):
                why[card] = "cards gained now would sit through a 7: play after next roll"
                continue
            ok, reason = can_play(gs, me, card, phase)
            if not ok:                              # owner 2026-10-06: never try what can't be played
                why[card] = f"can't play — {reason}"
                break
            if card == "merchant_fleet":
                args = choose_merchant_fleet(gs, me, force=over)
            else:
                args = CHOOSERS[card](gs, me)
            if args is not None:
                out.append((card, args))
                used.add(i)
            else:
                why[card] = NO_USE.get(card, "no use now")
            break                                   # one copy per card per call
    if "alchemist" in hand:
        why["alchemist"] = "before the roll only: next turn"

    if phase == "end_turn":
        left = [c for i, c in enumerate(hand) if i not in used and c not in VP_CARDS and c != "alchemist"]
        keep_room = len(left) + (1 if "alchemist" in hand else 0)
        if keep_room > PROGRESS_HAND_LIMIT and left:
            # over the progress limit: play the weakest-held card rather than discard it
            for c in left:
                if can_play(gs, me, c, phase)[0] and (args := CHOOSERS[c](gs, me)) is not None:
                    out.append((c, args))
                    why.pop(c, None)
                    break
            else:
                # nothing playable: discard the card least worth keeping
                out.append(("discard", {"card": min(left, key=lambda c: keep_value(gs, me, c, lead_dead))}))
    return out


# ---------------------------------------------------------------------------------------
# Spy: which of the victim's cards to take
# ---------------------------------------------------------------------------------------
PLAY_NOW_BONUS = 8.0          # a card we can play this turn for a gain beats any card for later


def playable_now(gs, me, card: str, phase: str = "main", attacked: bool = True) -> bool:
    """`card` would be played this turn (after the roll) if it were in our hand."""
    if phase not in ("main", "end_turn") or card in VP_CARDS or card == "alchemist":
        return False
    if card == "bishop" and not attacked:
        return False              # owner rule: Bishop waits for the first barbarian attack
    try:
        if not can_play(gs, me, card, phase, attacked=attacked)[0]:
            return False
        if card == "merchant_fleet":
            return choose_merchant_fleet(gs, me, force=hand_size(me) > discard_limit(gs, me)) is not None
        return CHOOSERS[card](gs, me) is not None
    except Exception:                                # noqa: BLE001
        return False


def spy_pick(gs, me, cards: list[str], victim: str | None = None, phase: str = "main",
             attacked: bool = True) -> tuple[str | None, dict[str, float]]:
    """owner 2026-10-06: Spy takes the card worth most to us, by a real value -- a card we can
    play this turn for a gain first (PLAY_NOW_BONUS), then what it is worth to hold (keep_value;
    the Alchemist waits for our next roll). Taken from the leader, the card is one they lose
    too: half its keep value is added. VP cards cannot be taken. Returns (card, scores)."""
    lead = leader(gs, me) if gs is not None and me is not None else None
    hit_leader = victim is not None and lead is not None and str(victim) == lead.name
    scores: dict[str, float] = {}
    for x in cards:
        c = x if x in CARDS else card_key(x)
        if c in VP_CARDS or c in scores:
            continue
        v = keep_value(gs, me, c)
        if gs is not None and me is not None and playable_now(gs, me, c, phase, attacked):
            v += PLAY_NOW_BONUS
        if hit_leader:
            v += 0.5 * KEEP_VALUE.get(c, 1)
        scores[c] = v
    if not scores:
        return None, scores
    return max(scores, key=lambda c: scores[c]), scores
