"""Legal moves. Owner 2026-10-02: "deterministic code owns the legal moves and the model just
picks what to go for, trades included."

Every decision point has a generator here: pure functions of the game state (and what the
screen shows), no preferences, no clicks. The Brain picks one Move from the list it is given;
the runner checks the pick is in that list and only then turns it into clicks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .board import COMMODITIES, RESOURCES, TERRAIN_RES
from .rules import COSTS, IMPROVEMENT_TRACKS, can_afford, improvement_cost
from .state import GameState

BUILD_KINDS = ("settlement", "city", "road", "ship", "dev", "improve", "knight", "activate",
               "promote", "wall")


def _t(d: dict | None) -> tuple:
    return tuple(sorted((k, int(v)) for k, v in (d or {}).items() if v))


@dataclass(frozen=True)
class Move:
    """kind: a build kind, bank_trade, offer, end, accept, decline, counter, discard, steal,
    robber, resource, take, wait. give/get are sorted (card, n) tuples so moves hash."""
    kind: str
    target: Any = None
    give: tuple = ()
    get: tuple = ()

    @property
    def give_d(self) -> dict:
        return dict(self.give)

    @property
    def get_d(self) -> dict:
        return dict(self.get)

    @staticmethod
    def of(kind: str, target: Any = None, give: dict | None = None, get: dict | None = None) -> "Move":
        return Move(kind, target, _t(give), _t(get))

    def __str__(self):
        s = self.kind
        if self.target is not None:
            s += f" {self.target}"
        if self.give or self.get:
            s += f" {self.give_d}->{self.get_d}"
        return s


# ---- bank ratios -----------------------------------------------------------------------
def ratio_source(gs: GameState, res: str, harbor_3to1: bool = False,
                 fleet: str | None = None) -> tuple[int, str, bool]:
    """THE bank ratio for `res` right now, where it comes from, and whether it is certain.

    Owner 2026-10-08: "if it doesn't have a harbor it needs 4:1 ... know all trade routes it can
    do". Order: Merchant Fleet played this turn (2:1) > our Merchant's hex (2:1) > the Trading
    House (Trade 3: commodities 2:1) > the ratios read off the bank dialog (gs.ratio_seen, the
    truth for the rest of the game) > harbours guessed from the board read > config 3:1 > 4:1.
    A board guess is never certain, and one 4:1 seen in the dialog means we own no 3:1 harbour
    (live 21:54: 3 lumber planned at a 3:1, the dialog said 4)."""
    me = gs.me()
    fleet = fleet or getattr(gs, "fleet_resource", None)
    if fleet and res == fleet:
        return 2, "merchant fleet", True
    if gs.mode.ck and res in COMMODITIES and (me.improvements or {}).get("trade", 0) >= 3:
        return 2, "trading house", True
    seen = getattr(gs, "ratio_seen", None) or {}
    if res in seen:
        return int(seen[res]), "bank dialog", True      # placing our Merchant drops its card's entry
    mh = getattr(me, "merchant_hex", None)
    if mh is not None:
        h = gs.board.hexes.get(tuple(mh))
        if h is not None and TERRAIN_RES.get(h.terrain) == res:
            return 2, "merchant", True
    no_3to1 = any(n >= 4 for n in seen.values())
    ratio, src = 4, "bank"
    spots = [v for v, _ in me.buildings()]
    if gs.mode.canal:
        # Almanac "merchants": a knight (active or not) on a harbour corner of a SMALL island
        # gives that harbour's rate; a knight on a home-island harbour corner gives nothing.
        # Kept after the canal is complete.
        spots += [v for v in me.knights if gs.board.is_small_island_vertex(v)]
    for v in spots:
        hb = gs.board.harbor_at(v)
        if hb == res:
            return 2, "harbour 2:1 (board read)", False
        if hb == "3:1" and not no_3to1:
            ratio, src = 3, "harbour 3:1 (board read)"
    if harbor_3to1 and not no_3to1 and ratio > 3:     # owner-confirmed 3:1 (the board read may miss it)
        ratio, src = 3, "harbour 3:1 (config)"
    return ratio, src, ratio == 4


def bank_ratio(gs: GameState, res: str, harbor_3to1: bool = False, fleet: str | None = None) -> int:
    """Cards of `res` the bank wants for one card (ratio_source is the one source of truth)."""
    return ratio_source(gs, res, harbor_3to1, fleet)[0]


def trade_routes(gs: GameState, res: str, hand: dict | None = None, harbor_3to1: bool = False,
                 fleet: str | None = None) -> list[dict]:
    """Every way to get one `res`, best first. Each route: {"how", "give", "ratio", "now"
    (usable this turn with this hand), "certain"}. Owner 2026-10-08: bank 4:1, harbours we own,
    the ratios seen in the bank dialog, the Trading House, the Merchant, a held Merchant Fleet
    card (2:1 on one card of our choice for the turn); as future sources the Aqueduct (science 3:
    a card of our choice on a roll that gives us nothing) and, in the Great Canal, knights on
    gold; player trades last (uncertain)."""
    me = gs.me()
    hand = dict(me.hand if hand is None else hand)
    no_grain = gs.mode.canal                           # grain is never banked away in the canal
    out = []
    for give in list(RESOURCES) + list(COMMODITIES):
        if give == res or (no_grain and give == "grain"):
            continue
        ratio, how, sure = ratio_source(gs, give, harbor_3to1, fleet)
        if hand.get(give, 0) >= ratio:
            out.append({"how": how, "give": {give: ratio}, "ratio": ratio, "now": True, "certain": sure})
    if gs.mode.ck and not (fleet or getattr(gs, "fleet_resource", None)) \
            and "merchant_fleet" in (me.progress_hand or []):
        for give in list(RESOURCES) + list(COMMODITIES):
            if give == res or (no_grain and give == "grain"):
                continue
            if hand.get(give, 0) >= 2 and ratio_source(gs, give, harbor_3to1, fleet)[0] > 2:
                out.append({"how": "play merchant fleet", "give": {give: 2}, "ratio": 2, "now": True,
                            "certain": True, "card": "merchant_fleet", "fleet": give})
    if gs.mode.ck and (me.improvements or {}).get("science", 0) >= 3:
        out.append({"how": "aqueduct (a roll that gives us nothing)", "give": {}, "ratio": 0,
                    "now": False, "certain": False})
    if gs.mode.canal and res == "grain" and any(
            any(h.terrain == "gold" for h in gs.board.vertex_hexes(v)) for v in me.knights):
        out.append({"how": "knight on gold (its number rolled)", "give": {}, "ratio": 0,
                    "now": False, "certain": False})
    give = next((g for g in RESOURCES if g != res and hand.get(g, 0) >= 1
                 and not (no_grain and g == "grain")), None)
    if give:
        out.append({"how": "player trade", "give": {give: 1}, "ratio": 1, "now": True, "certain": False})
    out.sort(key=lambda r: (not r["now"], r["how"] == "player trade", not r["certain"], r["ratio"]))
    return out


# ---- builds ----------------------------------------------------------------------------
def build_cost(gs: GameState, kind: str, target: Any = None) -> dict:
    if kind == "improve":
        return improvement_cost(target, gs.me().improvements[target])
    return COSTS.get(kind, {})


def build_moves(gs: GameState) -> list[Move]:
    """Every affordable build the rules allow."""
    me = gs.me()
    m = gs.mode
    out: list[Move] = []
    # piece limits (Seafarers / C&K): 5 settlements, 4 cities on the board, each counted on its
    # own; a city replaces one of our settlements (the settlement piece goes back to the supply)
    if len(me.settlements) < m.max_settlements:
        out += [Move("settlement", v) for v in gs.legal_settlement_spots(me)]
    if len(me.cities) < m.max_cities:
        out += [Move("city", v) for v in me.settlements]
    out += [Move("road", e) for e in gs.legal_road_edges(me)]
    if m.seafarers:
        out += [Move("ship", e) for e in gs.legal_road_edges(me, ship=True)]
    if m.has_dev_cards:
        out.append(Move("dev"))
    if m.ck:
        if me.cities:                                   # any improvement needs a city
            out += [Move("improve", tr) for tr in IMPROVEMENT_TRACKS if improve_allowed(gs, me, tr)]
        reach = {v for e in me.roads for v in e}
        if m.canal:
            reach |= {v for e in me.ships for v in e}
        # C&K pieces: 2 basic, 2 strong and 2 mighty knights per player
        per = {lv: sum(1 for k in me.knights.values() if k.level == lv) for lv in (1, 2, 3)}
        if per[1] < 2:
            out += [Move("knight", v) for v in sorted(reach)
                    if gs.board.is_land_vertex(v) and not gs.occupied(v) and home_island(gs, v)]
        for v, k in me.knights.items():
            if not k.active:
                out.append(Move("activate", v))
            if (k.level < 2 or (k.level < 3 and me.improvements["politics"] >= 3)) and per[k.level + 1] < 2:
                out.append(Move("promote", v))
        if len(me.walls) < 3:
            out += [Move("wall", v) for v in me.cities if v not in me.walls]
    # solo 2026-10-06: a promote the game refused stayed in the plan and was traded for (3 brick
    # -> wool) 44 times; the live runner lists builds that failed so the plan drops them
    blocked = getattr(gs, "blocked", ())
    return [mv for mv in out if (mv.kind, mv.target) not in blocked and mv.kind not in blocked
            and can_afford(me.hand, build_cost(gs, mv.kind, mv.target))]


# ---- trades ----------------------------------------------------------------------------
def bank_moves(gs: GameState, harbor_3to1: bool = False, fleet: str | None = None) -> list[Move]:
    """One bank/harbour trade: `ratio` of a card we hold for 1 of any other resource."""
    hand = gs.me().hand
    out = []
    for give in list(RESOURCES) + list(COMMODITIES):
        r = bank_ratio(gs, give, harbor_3to1, fleet)
        if hand.get(give, 0) >= r:
            out += [Move.of("bank_trade", give={give: r}, get={res: 1}) for res in RESOURCES if res != give]
    return out


def offer_moves(gs: GameState) -> list[Move]:
    """1-for-1 offers to the table: any card we hold for any other resource."""
    hand = gs.me().hand
    return [Move.of("offer", give={g: 1}, get={w: 1})
            for g in RESOURCES if hand.get(g, 0) >= 1 for w in RESOURCES if w != g]


def turn_moves(gs: GameState, *, offers_left: bool, harbor_3to1: bool = False,
               fleet: str | None = None) -> list[Move]:
    out = build_moves(gs) + bank_moves(gs, harbor_3to1, fleet)
    if offers_left:
        out += offer_moves(gs)
    return out + [Move("end")]


def answer_moves(gs: GameState | None, give: dict | None, get: dict | None,
                 wildcard: bool = False) -> list[Move]:
    """Answers to an offer where we'd give `give` and get `get`. Declining is always legal;
    accepting needs a fully read offer and the cards in hand; a wildcard (no `get`) can be
    countered by naming the card we want."""
    out = [Move("decline")]
    if gs is None or not give:
        return out
    if not can_afford(gs.me().hand, give):
        return out
    if get:
        out.append(Move.of("accept", give=give, get=get))
    elif wildcard:
        # owner 2026-10-08: a commodity asked of us is only given for 2 cards
        n = 2 if any(k in COMMODITIES for k in give) else 1
        out += [Move.of("counter", give=give, get={r: n}) for r in RESOURCES]
    return out


def home_island(gs: GameState, v) -> bool:
    """Great Canal (owner 2026-10-03): knights are built on the home island only -- not on a
    small-island corner and not on a gold corner (gold lies on the small islands); they get
    there by moving."""
    if not gs.mode.canal:
        return True
    b = gs.board
    return not b.is_small_island_vertex(v) and not any(h.terrain == "gold" for h in b.vertex_hexes(v))


def knight_moves(gs: GameState, ships: bool = True) -> list[Move]:
    """C&K knight action: an ACTIVE knight that hasn't acted this turn moves along its owner's
    connected roads and ships (through its own pieces, not others') to an empty land corner,
    or onto a weaker enemy knight at the end of one of those roads (displace it). The knight
    is inactive afterwards. Owner 2026-10-09: "knights can move along the road or ships" --
    ships=False is the roads-only path (config knight_move_ships off).
    Target is (from, to). Live: runner.tap_move_knight (UNVERIFIED flow)."""
    me = gs.me()
    if not gs.mode.ck:
        return []
    net = me.roads | (me.ships if ships else set())
    adj: dict = {}
    for e in net:
        adj.setdefault(e[0], []).append(e[1])
        adj.setdefault(e[1], []).append(e[0])
    others = {v for p in gs.players if p is not me for v in list(p.knights) + [x for x, _ in p.buildings()]}
    out = []
    for src, k in me.knights.items():
        if not k.active or k.acted or src not in adj:
            continue
        seen, stack = {src}, [src]
        while stack:
            v = stack.pop()
            for w in adj.get(v, []):
                if w not in seen and w not in others:
                    seen.add(w)
                    stack.append(w)
        out += [Move("move_knight", (src, w)) for w in sorted(seen - {src})
                if gs.board.is_land_vertex(w) and not gs.occupied(w)]
        # displace: a weaker enemy knight one of our roads leads to (C&K knight action)
        weak = {v for p in gs.players if p is not me for v, o in p.knights.items() if o.level < k.level}
        out += [Move("move_knight", (src, w)) for w in sorted({w for v in seen for w in adj.get(v, [])} & weak)]
    return out


def chase_moves(gs: GameState) -> list[Move]:
    """C&K knight action (owner 2026-10-09: "knights can move the robber if active and get
    deactivated"): an ACTIVE knight that hasn't acted this turn, standing on a corner of the
    robber's hex -- in Seafarers also of the pirate's sea hex -- chases it away. The player then
    moves it as after a 7 (and steals); the knight becomes inactive. Never before the first
    barbarian attack. Target is (knight corner, hex chased off, "robber" | "pirate")."""
    if not gs.mode.ck or not gs.barbarians_attacked:
        return []
    b, me = gs.board, gs.me()
    out = []
    for piece, at in (("robber", b.robber), ("pirate", getattr(b, "pirate", None))):
        if at is None or (piece == "pirate" and not gs.mode.seafarers):
            continue
        h = b.hexes.get(at)
        if h is None:
            continue
        for v, k in sorted(me.knights.items()):
            if k.active and not k.acted and h in b.vertex_hexes(v):
                out.append(Move("chase", (v, at, piece)))
    return out


def ship_moves(gs: GameState) -> list[Move]:
    """Seafarers ship move (Almanac "Ship (move)"), free, once per turn: the last ship of an open
    line -- one end has no own building, knight or other ship of ours, the other end joins our
    line -- that was not built this turn, to any edge a new ship could go (the line rules of
    legal_road_edges, judged with the ship lifted). Target is (from_edge, to_edge). Live: the
    runner's tap_move_ship (config "ship_move")."""
    me = gs.me()
    if not gs.mode.seafarers or me.ship_moved:
        return []
    b = gs.board
    mine_v = {v for v, _ in me.buildings()}
    out = []
    for s in sorted(me.ships):
        if s in me.ships_new:
            continue
        ends = []
        for v in s:
            own = [x for x in b.vertex_edges(v) if x != s and x in me.ships]
            ends.append(bool(own) or v in mine_v or v in me.knights)
        if ends.count(True) != 1:
            continue                     # both ends in use (or a lone ship): not an open end
        me.ships.discard(s)
        try:
            dests = [e for e in gs.legal_road_edges(me, ship=True) if e != s]
        finally:
            me.ships.add(s)
        out += [Move("move_ship", (s, e)) for e in dests]
    return out


def discard_moves(hand: dict) -> list[Move]:
    """One card to hand over (the runner asks again for each card owed)."""
    return [Move("discard", r) for r in list(RESOURCES) + list(COMMODITIES) if hand.get(r, 0) > 0]


def steal_moves(names: list[str]) -> list[Move]:
    return [Move("steal", n) for n in names]


def robber_moves(hexes: list) -> list[Move]:
    return [Move("robber", c) for c in hexes]


def resource_moves() -> list[Move]:
    return [Move("resource", r) for r in RESOURCES]


def take_moves(accepted: list) -> list[Move]:
    """Our offer: take one of the players who said yes, or keep waiting."""
    return [Move("take", x) for x in accepted] + [Move("wait")]


__all__ = ["Move", "BUILD_KINDS", "bank_ratio", "build_cost", "build_moves", "bank_moves", "knight_moves", "chase_moves",
           "ship_moves",
           "offer_moves", "turn_moves", "answer_moves", "discard_moves", "steal_moves",
           "robber_moves", "resource_moves", "take_moves"]


def improve_allowed(gs, me, track: str) -> bool:
    """C&K p.7-8: improvements need a city; max level 5; level 4 and 5 need a city that can hold
    the metropolis (one that is not a metropolis yet, unless we already hold this track's)."""
    lv = me.improvements[track]
    if not me.cities or lv >= 5:
        return False
    if lv < 3:
        return True
    held = (getattr(gs, "metropolis_owner", {}) or {}).get(track)
    if held and held[0] == me.name:
        return True
    return any(v not in me.metropolises for v in me.cities)
