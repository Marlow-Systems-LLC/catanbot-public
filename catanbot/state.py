"""Game state as the bot understands it (from vision or from the simulator)."""
from __future__ import annotations

from dataclasses import dataclass, field

from .board import COMMODITIES, RESOURCES, Board, EKey, VKey
from .rules import MODES, Mode, production


FRIENDLY_ROBBER_VP = 4      # owner: the friendly robber spares players with 4 VP or fewer


@dataclass
class Knight:
    level: int = 1          # 1 basic, 2 strong, 3 mighty
    active: bool = False
    acted: bool = False     # activated, moved or chased this turn (one action per turn)


@dataclass(eq=False)
class PlayerState:
    name: str
    color: str = ""
    is_me: bool = False
    vp_visible: int = 0                     # points shown on screen
    hand: dict[str, int] = field(default_factory=dict)  # exact for me; estimate for others
    hand_count: int = 0                     # card count shown on screen
    settlements: set[VKey] = field(default_factory=set)
    cities: set[VKey] = field(default_factory=set)
    metropolises: set[VKey] = field(default_factory=set)
    walls: set[VKey] = field(default_factory=set)
    cards_lost: int = 0                     # cards lost to 7-discards and robbing so far
    roads: set[EKey] = field(default_factory=set)
    ships: set[EKey] = field(default_factory=set)
    knights: dict[VKey, Knight] = field(default_factory=dict)
    dev_cards: int = 0                      # unplayed dev cards (base/seafarers)
    progress_cards: int = 0                 # C&K progress cards in hand
    progress_hand: list[str] = field(default_factory=list)  # progress_cards.card_key names; mine only
    knights_played: int = 0                 # base largest army
    longest_road: int = 0
    has_longest_road: bool = False
    has_largest_army: bool = False
    improvements: dict[str, int] = field(default_factory=lambda: {"trade": 0, "politics": 0, "science": 0})
    defender_vp: int = 0                    # C&K "defender of Catan" points
    canal_chips: int = 0                    # Great Canal
    special_vp: int = 0                     # other scenario VP (islands, etc.)
    # C&K Merchant (progress card): hex id (q, r) the merchant stands on, else None. The owner
    # trades that hex's resource 2:1 (moves.bank_ratio) and scores +1 VP (Sim.vp).
    merchant_hex: tuple | None = None
    # Seafarers ship move (Almanac "Ship (move)"): ships built this turn can't move, one move a turn
    ships_new: set[EKey] = field(default_factory=set)
    ship_moved: bool = False

    def total_cards(self) -> int:
        return self.hand_count or sum(self.hand.values())

    @property
    def knight_strength(self) -> int:
        return sum(k.level for k in self.knights.values() if k.active)

    def buildings(self) -> list[tuple[VKey, str]]:
        return [(v, "settlement") for v in self.settlements] + [(v, "city") for v in self.cities]


@dataclass
class GameState:
    board: Board
    mode: Mode = field(default_factory=lambda: MODES["great_canal"])
    players: list[PlayerState] = field(default_factory=list)
    turn: int = 0
    current: int = 0
    barbarian_pos: int = 0          # C&K: 0..7, attack at 7
    barbarian_len: int = 7
    barbarians_attacked: bool = False   # C&K: Bishop (and robber) locked until the first attack
    canal_progress: int = 0         # Great Canal: canal tiles built
    canal_total: int = 9           # 9 tiles; the canal completes with the 8th dig (owner + TDE rulebook)
    canal_dug: set = field(default_factory=set)   # canal hexes already dug
    metropolis_owner: dict = field(default_factory=dict)  # C&K: track -> (player name, vertex)
    vp_target: int | None = None    # read from screen if known
    # owner 2026-10-03: what each player got from the dice this game (name -> card -> count);
    # with the banner card count it says what an opponent likely holds (steals, monopolies)
    seen_gain: dict = field(default_factory=dict)
    # live 2026-10-04: coast edges whose build popup offered only a ship -- no road goes there
    sea_only_edges: set = field(default_factory=set)
    # owner 2026-10-05: per-player hand ledger (catanbot.hands.HandLedger), kept by the runner
    # across frames; est_hand reads it when present
    hands: object | None = None

    @property
    def target(self) -> int:
        if self.vp_target:
            return self.vp_target          # read from the game screen: always wins
        if self.mode.key == "great_canal" and len(self.players) == 3:
            return 21                      # printed rules: 21 VP with 3, 18 with 4
        return self.mode.vp_target

    def robber_protected(self) -> set[str]:
        """Friendly robber (owner 2026-10-03): the robber can't attack a player with
        FRIENDLY_ROBBER_VP (4) or fewer -- no hex touching their buildings, never stolen from."""
        return {p.name for p in self.players if p.vp_visible <= FRIENDLY_ROBBER_VP}

    def friendly_robber_hexes(self, options: list, mover: PlayerState | None = None) -> list:
        """Hexes the robber may go to: none touching a protected player (the mover may still
        block itself); if that leaves nothing, any hex."""
        safe = self.robber_protected() - ({mover.name} if mover else set())
        if not safe:
            return list(options)
        b = self.board
        bad = {h.coord for p in self.players if p.name in safe
               for v, _ in p.buildings() for h in b.vertex_hexes(v)}
        return [c for c in options if c not in bad] or list(options)

    def me(self) -> PlayerState:
        return next(p for p in self.players if p.is_me)

    def opponents(self) -> list[PlayerState]:
        return [p for p in self.players if not p.is_me]

    def owner_at(self, v: VKey) -> PlayerState | None:
        for p in self.players:
            if v in p.settlements or v in p.cities:
                return p
        return None

    def occupied(self, v: VKey) -> bool:
        return self.owner_at(v) is not None or any(v in p.knights for p in self.players)

    def edge_owner(self, e: EKey) -> PlayerState | None:
        for p in self.players:
            if e in p.roads or e in p.ships:
                return p
        return None

    # ---- derived economics ----------------------------------------------
    def roll_yield(self, total: int) -> dict[str, dict[str, int]]:
        """What each player's buildings produce on this dice total (robber-aware, barren canal
        fields pay nothing, gold left out: a gold card is the player's pick)."""
        out: dict[str, dict[str, int]] = {}
        for p in self.players:
            got = out.setdefault(p.name, {})
            for v, kind in p.buildings():
                for h in self.board.vertex_hexes(v):
                    if h.number != total or h.coord == self.board.robber or h.fog:
                        continue
                    if self.mode.canal and h.barren and self.canal_progress < self.canal_total:
                        continue
                    for r, n in production(h.terrain, kind, self.mode).items():
                        if r != "gold":                  # gold: any card, can't say which
                            got[r] = got.get(r, 0) + n
        return out

    def record_roll(self, total: int) -> None:
        """Add what every player produces on this dice total to seen_gain (robber-aware)."""
        for name, cards in self.roll_yield(total).items():
            got = self.seen_gain.setdefault(name, {})
            for r, n in cards.items():
                got[r] = got.get(r, 0) + n

    def est_hand(self, p: PlayerState) -> dict[str, float]:
        """Likely hand of p: the exact hand when known, else its card count split like what it
        has been seen getting (or, before any roll is seen, like its expected income)."""
        if sum(p.hand.values()) > 0:
            return {r: float(n) for r, n in p.hand.items()}
        if self.hands is not None and not p.is_me and self.hands.knows(p.name):
            # owner 2026-10-05: the ledger follows gains, builds, trades, steals and counts
            return self.hands.est(p.name, self)      # an empty ledger hand is empty
        total = p.total_cards()
        if total <= 0:
            return {}
        base = {r: float(n) for r, n in self.seen_gain.get(p.name, {}).items() if n > 0}
        if not base:
            base = {r: n for r, n in self.income(p).items() if r != "gold" and n > 0}
        if not base:
            base = {r: 1.0 for r in RESOURCES}
        s = sum(base.values())
        return {r: total * n / s for r, n in base.items()}

    def income(self, p: PlayerState) -> dict[str, float]:
        """Expected cards per dice roll for player p (robber-aware)."""
        from .board import PIPS
        out: dict[str, float] = {}
        for v, kind in p.buildings():
            for h in self.board.vertex_hexes(v):
                if h.number is None or h.coord == self.board.robber or h.fog:
                    continue
                if self.mode.canal and h.barren and self.canal_progress < self.canal_total:
                    continue  # barren farmland: no grain until the canal is done
                for res, n in production(h.terrain, kind, self.mode).items():
                    if res == "gold":
                        n *= 1.0  # gold = any resource; spread as wildcard
                    out[res] = out.get(res, 0.0) + n * PIPS[h.number] / 36
        return out

    def income_total(self, p: PlayerState) -> float:
        return sum(self.income(p).values())

    def distance_rule_ok(self, v: VKey) -> bool:
        if self.owner_at(v):
            return False
        return not any(self.owner_at(n) for n in self.board.neighbors(v))

    def can_settle(self, v: VKey) -> bool:
        """Land corner, distance rule, and (Great Canal) home island only: Almanac "no settlement
        may be built on the small islands" (the desert islands and the gold islands), so gold
        corners are out too -- only a knight collects the gold. A corner holding a knight (ours
        or a rival's) takes no settlement: live game 20261009_014102 tapped our own knight's
        corner 6 times and the popup only offered knight actions."""
        if any(v in p.knights for p in self.players):
            return False
        if self.mode.canal:
            return v in self.board.home_spots() and self.distance_rule_ok(v)
        return self.board.is_land_vertex(v) and self.distance_rule_ok(v)

    def defence_strength(self, p: PlayerState) -> int:
        """Active knight strength against the barbarians. Great Canal (owner 2026-10-03): a knight
        on a gold corner is off the main island -- it never defends; active, it can only chase
        the robber off that gold hex. Printed rule: no knight on a small island defends."""
        if not self.mode.canal:
            return p.knight_strength
        b = self.board
        return sum(k.level for v, k in p.knights.items() if k.active
                   and not b.is_small_island_vertex(v)
                   and not any(h.terrain == "gold" for h in b.vertex_hexes(v)))

    def legal_settlement_spots(self, p: PlayerState, setup: bool = False) -> list[VKey]:
        out = []
        for v in self.board.vertices:
            if not self.can_settle(v):
                continue
            if setup or any(e in p.roads or e in p.ships for e in self.board.vertex_edges(v)):
                out.append(v)
        return out

    def legal_road_edges(self, p: PlayerState, ship: bool = False) -> list[EKey]:
        """Free edges p may build a road (land edge) or ship (sea edge) on, within the piece
        limit (mode.max_roads / max_ships). A road joins an own building or road. A ship starts
        at an own building or extends an own ship line; Great Canal (Almanac: "ship routes may not
        branch"): no corner without an own building may end up with 3 own ships, and a corner without an own building
        takes a new ship only at the open end of a line (exactly 1 own ship there). Ships never
        join a road directly (Almanac "Ship (build)")."""
        net = p.ships if ship else p.roads
        if len(net) >= (self.mode.max_ships if ship else self.mode.max_roads):
            return []
        out = []
        mine_v = {v for v, _ in p.buildings()}
        no_branch = ship and self.mode.canal
        for e in self.board.edges:
            if self.edge_owner(e):
                continue
            if ship and not self.board.is_sea_edge(e):
                continue
            if not ship and (not self.board.is_land_edge(e) or e in self.sea_only_edges):
                continue
            owns = [sum(1 for x in self.board.vertex_edges(v) if x != e and x in net) for v in e]
            if no_branch and any(own >= 2 and v not in mine_v for v, own in zip(e, owns)):
                continue                        # a third ship at an open corner = a branch;
                                                # a building may start a line on each sea edge
            for v, own in zip(e, owns):
                other = self.owner_at(v)
                if v in mine_v or (other in (None, p) and own and (not no_branch or own == 1)):
                    out.append(e)
                    break
        return out

    def longest_road_len(self, p: PlayerState) -> int:
        edges = p.roads | p.ships
        if not edges:
            return 0
        adj: dict[VKey, list[EKey]] = {}
        for e in edges:
            for v in e:
                adj.setdefault(v, []).append(e)
        # an opponent's settlement, city or knight (C&K, active or not) on a corner breaks the road
        theirs = {v for q in self.players if q is not p for v in q.knights}
        blocked = {v for v in adj if v in theirs or ((o := self.owner_at(v)) is not None and o is not p)}
        best = 0

        def dfs(v, used):
            nonlocal best
            best = max(best, len(used))
            if v in blocked and used:
                return
            for e in adj.get(v, []):
                if e not in used:
                    used.add(e)
                    dfs(e[0] if e[1] == v else e[1], used)
                    used.discard(e)

        for v in adj:
            dfs(v, set())
        return best

    def hidden_vp_estimate(self, p: PlayerState) -> float:
        """VP the screen doesn't show: dev-card VPs (about 5 of 25 base dev cards)."""
        if p.is_me or not self.mode.has_dev_cards:
            return 0.0
        return p.dev_cards * 0.2


def award_longest_road(gs: GameState) -> dict[str, int]:
    """Longest road (2 VP): 5+ and strictly longer than everyone. The holder keeps it while
    nobody is longer. When its road is broken: the strictly longest player with 5+ takes it;
    a tie among the others, or nobody at 5+, sets the card aside. Returns VP change per name."""
    for p in gs.players:
        p.longest_road = gs.longest_road_len(p)
    holder = next((p for p in gs.players if p.has_longest_road), None)
    top = max((p.longest_road for p in gs.players), default=0)
    leaders = [p for p in gs.players if p.longest_road == top]
    if top < 5:
        new = None
    elif holder is not None and holder.longest_road == top:
        new = holder
    else:
        new = leaders[0] if len(leaders) == 1 else None
    out: dict[str, int] = {}
    if new is not holder:
        if holder is not None:
            holder.has_longest_road = False
            out[holder.name] = -2
        if new is not None:
            new.has_longest_road = True
            out[new.name] = out.get(new.name, 0) + 2
    return out


def grant_metropolis(gs: GameState, p: PlayerState, track: str) -> int:
    """C&K: one metropolis per track. The first to level 4 takes it if it is free; the first to
    level 5 takes it from a holder still below 5. It goes on a city that is not a metropolis yet.
    Updates gs.metropolis_owner and the loser's pieces / vp_visible; returns p's VP change."""
    lvl = p.improvements.get(track, 0)
    cur = gs.metropolis_owner.get(track)
    free = [v for v in p.cities if v not in p.metropolises]
    if lvl < 4 or not free or (cur and cur[0] == p.name):
        return 0
    if cur:
        holder = next((q for q in gs.players if q.name == cur[0]), None)
        if lvl < 5 or (holder is not None and holder.improvements.get(track, 0) >= 5):
            return 0
        if holder is not None:
            holder.metropolises.discard(cur[1])
            holder.vp_visible -= 2
    v = free[0]
    p.metropolises.add(v)
    gs.metropolis_owner[track] = (p.name, v)
    return 2


def new_hand() -> dict[str, int]:
    return {k: 0 for k in RESOURCES + COMMODITIES}


from .board import hex_corners  # noqa: E402


def dig_canals(gs: "GameState") -> list[tuple]:
    """Great Canal (printed rules): a canal hex is dug as soon as two ACTIVATED knights
    stand on its corners -- no resource cost. Same owner takes both chips (+2 VP), two
    owners take one each. The canal is finished when the penultimate tile is dug."""
    if not gs.mode.canal or gs.canal_progress >= gs.canal_total:
        return []
    out = []
    for h in gs.board.hexes.values():
        if not gs.board.is_canal(h) or h.coord in gs.canal_dug:
            continue
        corners = set(hex_corners(*h.coord))
        knights = [p for p in gs.players for v, k in p.knights.items() if v in corners and k.active]
        if len(knights) < 2:
            continue
        a, b = knights[0], knights[1]
        a.canal_chips += 1
        b.canal_chips += 1
        gs.canal_dug.add(h.coord)
        gs.canal_progress += 1
        out.append((h.coord, a.name, b.name))
        if gs.canal_progress >= gs.canal_total - 1:
            gs.canal_progress = gs.canal_total     # last tile goes in with the 8th
            break
    return out

