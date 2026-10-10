"""Look ahead: every way to score, each player's next points, and our next 3 goals.

Owner 2026-10-02: "players can do actions that change the plans, bot should have a plan for
at least 3 moves ahead but should update that plan based on every move of the other players,
its goal is to stop them from making points and making as many points as it can itself."

The runner rebuilds the outlook whenever the board or anyone's VP changes. The brain uses it:
  - goals[0] gets a bonus in action_value, and its missing cards go in brain.keep (not traded away)
  - blocks (moves that deny an opponent's next points) get a bonus scaled by that player's threat
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .rules import COSTS, IMPROVEMENT_TRACKS, improvement_cost
from .board import ekey, hex_corners
from .state import GameState, PlayerState

# Every VP source the bot knows, by mode. (source, vp, how)
VP_TABLE = [
    ("settlement", 1, "build on an open spot (distance rule) connected to your road"),
    ("city", 1, "upgrade a settlement (+1, a city is worth 2); in C&K a lost city drops back to 1"),
    ("longest_road", 2, "5+ connected roads/ships, more than anyone; it moves when someone passes you; "
                        "a settlement on the road breaks it"),
    ("metropolis", 2, "C&K: first to improvement level 4 on a track (needs a city); level 5 steals it "
                      "from someone still on 4; can't be pillaged"),
    ("defender", 1, "C&K: barbarians beaten and you alone gave the most active knight strength"),
    ("progress_vp", 1, "C&K: Constitution / Printer progress cards"),
    ("merchant", 1, "C&K: whoever holds the merchant figure"),
    ("knights", 0, "C&K knights give no VP directly: they save cities (a pillaged city = -1 VP), "
                   "win Defender, chase off the robber, displace rival knights, and in the Great "
                   "Canal two ACTIVE knights on a canal hex's corners dig it"),
    ("canal_chip", 2, "Great Canal: a dug canal hex is 2 VP when both knights are yours, "
                      "1 VP each when two owners share it"),
]
# Almanac (Great Canal): no largest army / VP dev cards (C&K), no settlements on the small
# islands so no "new island" VP, no pirate.


def vp_sources(gs: GameState) -> list[tuple[str, int, str]]:
    m = gs.mode
    out = []
    for name, vp, how in VP_TABLE:
        if name in ("metropolis", "defender", "progress_vp", "merchant", "knights") and not m.ck:
            continue
        if name == "canal_chip" and not m.canal:
            continue
        out.append((name, vp, how))
    return out


@dataclass
class VPMove:
    player: str
    kind: str                # settlement/city/longest_road/metropolis/defender/canal_chip
    vp: float                # points gained (longest road also costs the holder 2)
    turns: float             # rough rounds until they can do it
    target: object = None
    cost: dict = field(default_factory=dict)

    def __str__(self):
        return f"{self.kind}+{self.vp:g}@{self.turns:.1f}"


@dataclass
class Outlook:
    goals: list[VPMove]                      # our next (up to) 3 scoring moves, in order
    threats: dict[str, list[VPMove]]         # each opponent's nearest scoring moves
    blocks: dict[tuple, float]               # (kind, target) of OUR action -> denial value
    keep: set[str]                           # cards goals[0] still needs: don't trade them away

    def summary(self) -> str:
        g = ", ".join(map(str, self.goals)) or "-"
        t = "; ".join(f"{k}: {', '.join(map(str, v[:2]))}" for k, v in self.threats.items() if v)
        b = ", ".join(f"{k[0]}" for k in list(self.blocks)[:3])
        return f"goals=[{g}] threats=[{t}] blocks=[{b}] keep={sorted(self.keep)}"


def _rounds(gs: GameState, p: PlayerState, cost: dict) -> float:
    """Rounds until p can pay `cost`: exact hand for us, a share of the card count for others."""
    inc = gs.income(p)
    n = max(2, len(gs.players))
    if p.is_me:
        short = sum(max(0, v - p.hand.get(k, 0)) for k, v in cost.items())
    else:
        short = max(0.0, sum(cost.values()) - 0.4 * p.total_cards())
    if short <= 0:
        return 0.0
    per = max(0.15, sum(inc.get(k, 0.0) for k in cost) * n + 0.25 * sum(inc.values()) * n * 0.25)
    return short / per


def _road_dist(gs: GameState, p: PlayerState, target, limit: int = 3) -> tuple[int, list]:
    """Road segments p needs to reach vertex `target`, and the first edge of that path."""
    b = gs.board
    start = {v for v, _ in p.buildings()}
    for e in p.roads | p.ships:
        start |= set(e)
    if target in start:
        return 0, []
    frontier = {v: [] for v in start}
    seen = set(start)
    for d in range(1, limit + 1):
        nxt = {}
        for v, path in frontier.items():
            o = gs.owner_at(v)
            if o is not None and o is not p:
                continue
            for n in b.neighbors(v):
                if n in seen:
                    continue
                e = ekey(v, n)
                if any(e in q.roads or e in q.ships for q in gs.players):
                    continue
                nxt[n] = path + [e]
                if n == target:
                    return d, nxt[n]
        seen |= set(nxt)
        frontier = nxt
    return 99, []


def player_moves(gs: GameState, p: PlayerState) -> list[VPMove]:
    b = gs.board
    out: list[VPMove] = []
    m = gs.mode
    # settlements: best open spots within 3 roads
    reach = []
    for v in b.vertices:
        if not gs.can_settle(v):
            continue
        d, path = _road_dist(gs, p, v)
        if d > 3:
            continue
        rank = b.vertex_pips(v) - 2 * d
        if p.is_me:
            # owner 2026-10-09: our next spot -- 3 paying hexes, then 2, rarely 1
            from .brain import SETUP_DEFAULTS, hex_count_term
            must = gs.mode.canal and b.harbor_at(v) == "3:1" and not any(
                b.harbor_at(w) == "3:1" for w, _ in p.buildings())
            rank += hex_count_term(gs, v, "settlement", SETUP_DEFAULTS, bool(must))
        reach.append((rank, v, d, path))
    for _, v, d, path in sorted(reach, key=lambda t: t[0], reverse=True)[:2]:
        cost = dict(COSTS["settlement"])
        for k, c in COSTS["road"].items():
            cost[k] = cost.get(k, 0) + c * d
        mv = VPMove(p.name, "settlement", 1, _rounds(gs, p, cost), v, cost)
        mv.path = path               # type: ignore[attr-defined]
        out.append(mv)
    if p.settlements:
        v = max(p.settlements, key=b.vertex_pips)
        out.append(VPMove(p.name, "city", 1, _rounds(gs, p, COSTS["city"]), v, dict(COSTS["city"])))
    # longest road
    holder = next((q for q in gs.players if q.has_longest_road), None)
    need_len = max(5, (holder.longest_road + 1) if holder and holder is not p else 5)
    if not p.has_longest_road:
        k = max(0, need_len - p.longest_road)
        if k <= 4:
            cost = {r: c * k for r, c in COSTS["road"].items()}
            out.append(VPMove(p.name, "longest_road", 2 + (2 if holder else 0) * 0.5,
                              _rounds(gs, p, cost), None, cost))
    if m.ck:
        for tr, lvl in p.improvements.items():
            taken = any(q is not p and q.improvements.get(tr, 0) >= 4 and q.metropolises for q in gs.players)
            if lvl >= 4 or not (p.cities or p.is_me):
                continue
            if any(q is not p and q.improvements.get(tr, 0) >= 5 for q in gs.players):
                continue    # review 0.3.1: a rival at level 5 locks that metropolis (grant_metropolis)
            goal = 5 if taken else 4
            cost: dict = {}
            for L in range(lvl, goal):
                for k, c in improvement_cost(tr, L).items():
                    cost[k] = cost.get(k, 0) + c
            if goal - lvl <= 2:
                out.append(VPMove(p.name, "metropolis", 2, _rounds(gs, p, cost), tr, cost))
        danger = gs.barbarian_pos / max(1, gs.barbarian_len)
        best = max((gs.defence_strength(q) for q in gs.players if q is not p), default=0)
        if danger >= 0.5 and gs.defence_strength(p) >= best:
            out.append(VPMove(p.name, "defender", 1, (1 - danger) * 4 + (0 if gs.defence_strength(p) > best else 1)))
    if m.canal and gs.canal_progress < gs.canal_total and p.knights:
        out.append(_canal_move(gs, p))
    out.sort(key=lambda mv: mv.turns / max(0.5, mv.vp))
    return out


def _canal_move(gs: GameState, p: PlayerState) -> VPMove:
    """Almanac 3.1: a canal hex is dug by two active knights on its corners; 2 VP when both are
    p's, 1 VP each when two owners share it. Pick the undug hex where p scores most for the
    fewest activations; with no knight on any canal hex, a generic 1-VP share."""
    best = None
    for h in gs.board.hexes.values():
        if not gs.board.is_canal(h) or h.coord in gs.canal_dug:
            continue
        corners = set(hex_corners(*h.coord))
        mine = [(v, k) for v, k in p.knights.items() if v in corners]
        if not mine:
            continue
        other = any(v in corners for q in gs.players if q is not p for v in q.knights)
        if len(mine) >= 2:
            vp, use = 2, mine[:2]
        elif other:
            vp, use = 1, mine[:1]
        else:
            continue
        idle = [v for v, k in use if not k.active]
        key = (vp, -len(idle))
        if best is None or key > best[0]:
            best = (key, vp, idle)
    if best is not None:
        _, vp, idle = best
        cost = {r: c * len(idle) for r, c in COSTS["activate"].items()} if idle else {}
        return VPMove(p.name, "canal_chip", vp, _rounds(gs, p, cost) if idle else 0.0,
                      idle[0] if idle else None, cost)
    idle = [v for v, k in p.knights.items() if not k.active]
    return VPMove(p.name, "canal_chip", 1, _rounds(gs, p, COSTS["activate"]) if idle else 2.0,
                  idle[0] if idle else None, dict(COSTS["activate"]) if idle else {})


def build(gs: GameState, threat_probs: dict[str, float] | None = None) -> Outlook:
    me = gs.me()
    # our goals: three cheapest points per round; a chosen move's cost is spent before the next
    mine = player_moves(gs, me)
    goals: list[VPMove] = []
    used: set = set()
    t_acc = 0.0
    for mv in mine:
        key = (mv.kind, mv.target if mv.kind != "settlement" else None)
        if key in used and mv.kind != "settlement":
            continue
        used.add(key)
        t_acc += mv.turns
        goals.append(VPMove(mv.player, mv.kind, mv.vp, t_acc, mv.target, mv.cost))
        if len(goals) == 3:
            break
    keep: set[str] = set()
    if goals:
        keep = {k for k, v in goals[0].cost.items() if v > 0}

    threats: dict[str, list[VPMove]] = {}
    blocks: dict[tuple, float] = {}
    probs = threat_probs or {}
    for p in gs.opponents():
        mv = player_moves(gs, p)
        threats[p.name] = mv[:3]
        w = 0.5 + 2.0 * probs.get(p.name, 1.0 / max(1, len(gs.players)))
        for x in mv[:3]:
            urgency = w * x.vp / (1.0 + x.turns)
            if x.kind == "settlement":
                # take or cut off the spot they're heading for
                blocks[("settlement", x.target)] = blocks.get(("settlement", x.target), 0) + urgency
                for e in getattr(x, "path", [])[-1:]:
                    blocks[("road", e)] = blocks.get(("road", e), 0) + 0.5 * urgency
            elif x.kind == "longest_road":
                blocks[("longest_road", None)] = blocks.get(("longest_road", None), 0) + urgency
            elif x.kind in ("defender", "canal_chip"):
                blocks[("knight", None)] = blocks.get(("knight", None), 0) + 0.5 * urgency
            elif x.kind == "metropolis":
                blocks[("improve", x.target)] = blocks.get(("improve", x.target), 0) + 0.5 * urgency
    blocks = dict(sorted(blocks.items(), key=lambda kv: -kv[1]))
    return Outlook(goals, threats, blocks, keep)


def rank_threats(gs: GameState, o: Outlook, probs: dict[str, float], rise: dict[str, int] | None = None,
                 rise_w: float = 0.0) -> list[tuple[str, float, str]]:
    """Owner 2026-10-02: "see who is the largest threat just before its turn".
    Score = share of the VP target already held + points they can reach soon + model win chance
    (+ rise_w per VP gained over the last own turns -- game 0959: Jean climbed 4 -> 11 unnoticed)."""
    tgt = max(1, gs.target)   # screen value, else 21 (3p) / 18 (4p)
    out = []
    for p in gs.opponents():
        soon = sum(mv.vp / (1.0 + mv.turns) for mv in o.threats.get(p.name, []))
        up = (rise or {}).get(p.name, 0) if rise_w else 0
        sc = p.vp_visible / tgt + 0.15 * soon + 0.5 * probs.get(p.name, 0.0) + rise_w * up
        why = f"vp {p.vp_visible}/{tgt}, next {', '.join(map(str, o.threats.get(p.name, [])[:2])) or '-'}, win {probs.get(p.name, 0):.2f}"
        if up > 0:
            why += f", rise +{up}"
        out.append((p.name, sc, why))
    return sorted(out, key=lambda t: -t[1])


def signature(gs: GameState) -> tuple:
    """Changes whenever anyone builds, scores or moves the barbarians: time to re-plan."""
    return tuple((p.name, p.vp_visible, len(p.settlements), len(p.cities), len(p.roads | p.ships),
                  len(p.knights), sum(k.active for k in p.knights.values()), p.longest_road,
                  tuple(sorted(p.improvements.items())))
                 for p in gs.players) + (gs.barbarian_pos, gs.canal_progress)


__all__ = ["VP_TABLE", "vp_sources", "VPMove", "Outlook", "player_moves", "build", "signature", "rank_threats",
           "IMPROVEMENT_TRACKS"]
