"""Hex board geometry for any board shape (base, Seafarers islands, scenarios).

Hexes use axial coords (q, r) with pointy-top orientation. Vertices and edges get
integer keys derived from their unit-size pixel position, so neighbouring hexes
agree on shared corners without any per-shape bookkeeping.
"""
from __future__ import annotations

import math
from pathlib import Path
from dataclasses import dataclass, field

SQ3 = math.sqrt(3)
# axial neighbour directions, pointy-top: E, NE, NW, W, SW, SE
DIRS = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]

RESOURCES = ("brick", "lumber", "wool", "grain", "ore")
COMMODITIES = ("paper", "cloth", "coin")
# terrain -> resource produced
TERRAIN_RES = {
    "hills": "brick", "forest": "lumber", "pasture": "wool",
    "fields": "grain", "mountains": "ore", "gold": "gold",
}
# C&K: a city on these terrains yields 1 resource + 1 commodity
TERRAIN_COMMODITY = {"forest": "paper", "pasture": "cloth", "mountains": "coin"}
LAND = set(TERRAIN_RES) | {"desert", "canal"}
PIPS = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}

VKey = tuple[int, int]
EKey = tuple[VKey, VKey]


def hex_center(q: int, r: int) -> tuple[float, float]:
    return SQ3 * (q + r / 2), 1.5 * r


def _vkey(x: float, y: float) -> VKey:
    return (round(x / (SQ3 / 2)), round(y * 2))


def hex_corners(q: int, r: int) -> list[VKey]:
    cx, cy = hex_center(q, r)
    out = []
    for k in range(6):
        a = math.radians(-90 + 60 * k)
        out.append(_vkey(cx + math.cos(a), cy + math.sin(a)))
    return out


def ekey(a: VKey, b: VKey) -> EKey:
    return (a, b) if a <= b else (b, a)


@dataclass
class Hex:
    q: int
    r: int
    terrain: str               # hills/forest/pasture/fields/mountains/gold/desert/sea/canal/unknown
    number: int | None = None  # dice number token
    fog: bool = False          # undiscovered (Seafarers)
    barren: bool = False       # Great Canal farmland: token face down, no grain until the canal is done

    @property
    def coord(self) -> tuple[int, int]:
        return (self.q, self.r)

    @property
    def is_land(self) -> bool:
        return self.terrain in LAND


@dataclass
class Harbor:
    edge: EKey
    kind: str  # "3:1" or a resource name (2:1)


@dataclass
class Board:
    hexes: dict[tuple[int, int], Hex] = field(default_factory=dict)
    harbors: list[Harbor] = field(default_factory=list)
    robber: tuple[int, int] | None = None
    pirate: tuple[int, int] | None = None
    canal_hexes: set = field(default_factory=set)   # Great Canal: the 9 chip hexes (they keep producing)

    def canal_cells(self) -> set:
        """Hexes the canal is dug through: the chip hexes, or the old terrain="canal" stand-ins."""
        return set(self.canal_hexes) | {c for c, h in self.hexes.items() if h.terrain == "canal"}

    def is_canal(self, h: "Hex") -> bool:
        return h.coord in self.canal_hexes or h.terrain == "canal"

    # ---- construction -------------------------------------------------
    def add(self, h: Hex) -> None:
        self.hexes[h.coord] = h
        self._cache = None
        self._main = None
        self._home_spots = None

    def _build(self):
        if getattr(self, "_cache", None) is not None:
            return self._cache
        v_hexes: dict[VKey, list[tuple[int, int]]] = {}
        edges: set[EKey] = set()
        e_hexes: dict[EKey, list[tuple[int, int]]] = {}
        for c, h in self.hexes.items():
            cs = hex_corners(*c)
            for i in range(6):
                v_hexes.setdefault(cs[i], []).append(c)
                e = ekey(cs[i], cs[(i + 1) % 6])
                edges.add(e)
                e_hexes.setdefault(e, []).append(c)
        v_adj: dict[VKey, set[VKey]] = {v: set() for v in v_hexes}
        for a, b in edges:
            v_adj[a].add(b)
            v_adj[b].add(a)
        self._vh = {v: [self.hexes[c] for c in cs] for v, cs in v_hexes.items()}
        self._cache = (v_hexes, edges, e_hexes, v_adj)
        return self._cache

    # ---- queries -------------------------------------------------------
    @property
    def vertices(self) -> list[VKey]:
        return list(self._build()[0])

    @property
    def edges(self) -> list[EKey]:
        return list(self._build()[1])

    def vertex_hexes(self, v: VKey) -> list[Hex]:
        self._build()
        return self._vh.get(v, [])          # cached; don't mutate the returned list

    def edge_hexes(self, e: EKey) -> list[Hex]:
        return [self.hexes[c] for c in self._build()[2].get(e, [])]

    def neighbors(self, v: VKey) -> set[VKey]:
        return self._build()[3].get(v, set())

    def vertex_edges(self, v: VKey) -> list[EKey]:
        return [ekey(v, n) for n in self.neighbors(v)]

    def is_land_vertex(self, v: VKey) -> bool:
        return any(h.is_land for h in self.vertex_hexes(v))

    def is_coastal_vertex(self, v: VKey) -> bool:
        hs = self.vertex_hexes(v)
        return any(h.terrain == "sea" for h in hs) or len(hs) < 3

    def main_island(self) -> set[tuple[int, int]]:
        """Hexes of the biggest land component (cached per board). Canal hexes always count
        as main-island land (Great Canal), even where they touch a small island."""
        if getattr(self, "_main", None) is not None:
            return self._main
        land = {c for c, h in self.hexes.items() if h.is_land}
        seen: set = set()
        best: set = set()
        for c in land:
            if c in seen:
                continue
            comp, stack = set(), [c]
            while stack:
                x = stack.pop()
                if x in seen:
                    continue
                seen.add(x)
                comp.add(x)
                stack += [n for dq, dr in DIRS if (n := (x[0] + dq, x[1] + dr)) in land]
            if len(comp) > len(best):
                best = comp
        self._main = best | self.canal_cells()
        return self._main

    def home_spots(self) -> frozenset:
        """Corners where a Great Canal settlement may ever stand: land, main island, not a gold
        corner. Terrain never changes in a game, so this is computed once per board (the sim
        asked can_settle millions of times per run)."""
        if getattr(self, "_home_spots", None) is None:
            self._build()
            self._home_spots = frozenset(
                v for v in self._vh if self.is_land_vertex(v) and not self.is_small_island_vertex(v)
                and not any(h.terrain == "gold" for h in self.vertex_hexes(v)))
        return self._home_spots

    def is_small_island_vertex(self, v: VKey) -> bool:
        """A land corner none of whose land hexes is on the main island."""
        hs = [h for h in self.vertex_hexes(v) if h.is_land]
        main = self.main_island()
        return bool(hs) and not any(h.coord in main for h in hs)

    def is_land_edge(self, e: EKey) -> bool:
        return any(h.is_land for h in self.edge_hexes(e))

    def is_sea_edge(self, e: EKey) -> bool:
        hs = self.edge_hexes(e)
        return any(h.terrain == "sea" for h in hs)

    def harbor_at(self, v: VKey) -> str | None:
        for hb in self.harbors:
            if v in hb.edge:
                return hb.kind
        return None

    def vertex_pips(self, v: VKey) -> int:
        return sum(PIPS.get(h.number or 0, 0) for h in self.vertex_hexes(v)
                   if h.coord != self.robber)

    def pixel_vertex(self, v: VKey) -> tuple[float, float]:
        """Unit-size pixel position of a vertex (for mapping back to the screen)."""
        return v[0] * SQ3 / 2, v[1] / 2

    def pixel_edge(self, e: EKey) -> tuple[float, float]:
        (ax, ay), (bx, by) = self.pixel_vertex(e[0]), self.pixel_vertex(e[1])
        return (ax + bx) / 2, (ay + by) / 2


def _numbers(rng, k):
    pool = [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12]
    out = []
    while len(out) < k:
        rng.shuffle(pool)
        out += pool
    return out[:k]


def _harbors(b: Board, rng, count: int) -> None:
    coast = sorted({e for e in b.edges
                    if len({h.terrain == "sea" for h in b.edge_hexes(e)}) == 2
                    and any(h.is_land for h in b.edge_hexes(e))},
                   key=lambda e: math.atan2(*b.pixel_edge(e)[::-1]))
    kinds = (["3:1"] * 4 + list(RESOURCES)) * 2
    rng.shuffle(kinds)
    step = max(1, len(coast) // count)
    for i in range(min(count, len(coast) // step)):
        b.harbors.append(Harbor(coast[i * step], kinds[i]))



def canal_board(seed: int | None = None) -> Board:
    """The Great Canal approximation: wide island crossed by a row of canal hexes.
    Canal hexes produce nothing; once all are built, grain unlocks (see rules)."""
    import random
    rng = random.Random(seed)
    b = Board()
    land = [(q, r) for r in range(-2, 3) for q in range(-3 - (r < 0) * r, 4 - (r > 0) * r)]
    canal = {(q, 0) for q in range(-3, 4)} | {(-1, 1)} | {(1, -1)}
    nums = _numbers(rng, len(land))
    pool = (["hills", "forest", "pasture", "fields", "mountains"] * 5)[: len(land) - len(canal)]
    rng.shuffle(pool)
    for c in land:
        if c in canal:
            b.add(Hex(c[0], c[1], "canal"))
            continue
        t = pool.pop()
        # rulebook: only the western crop fields are barren, the others produce
        b.add(Hex(c[0], c[1], t, nums.pop(), barren=(t == "fields" and c[0] <= 0)))
    for q in range(-7, 8):
        for r in range(-4, 5):
            if (q, r) not in b.hexes and any((q + dq, r + dr) in b.hexes and b.hexes[(q + dq, r + dr)].terrain != "sea"
                                             for dq, dr in DIRS):
                b.add(Hex(q, r, "sea"))
    b.robber = None
    _harbors(b, rng, 9)
    return b


def real_canal_board(seed: int | None = None, template: str | None = None) -> Board:
    """The Great Canal board (data/boards/canal_template.json, read from live screenshots).
    Fixed: sea, harbours (edges and kinds), deserts, lake, mountains, fields + their numbers,
    gold 8s, and the 9 canal chip hexes. Shuffled: the 13 hills/forest/pasture terrains and the
    other numbers (no two red numbers side by side)."""
    import json, random
    rng = random.Random(seed)
    path = Path(template) if template else Path(__file__).resolve().parent.parent / "data/boards/canal_template.json"
    if not path.exists():
        return canal_board(seed)
    d = json.load(open(path))
    hexes = [Hex(q, r, t, n, barren=bool(br)) for q, r, t, n, br in d["hexes"]]
    free = [h for h in hexes if h.terrain in ("hills", "forest", "pasture")]
    terr = [h.terrain for h in free]
    rng.shuffle(terr)
    for h, t in zip(free, terr):
        h.terrain = t
    keep = set(d.get("fixed_numbers", ["fields", "gold"]))
    numbered = [h for h in hexes if h.number and h.terrain not in keep]
    nums = [h.number for h in numbered]
    by = {h.coord: h for h in hexes}
    for _ in range(500):
        rng.shuffle(nums)
        for h, n in zip(numbered, nums):
            h.number = n
        if not any(h.number in (6, 8) and any((n := by.get((h.q + dq, h.r + dr))) and n.number in (6, 8)
                                              and n.terrain != "gold" for dq, dr in DIRS)
                   for h in hexes if h.terrain != "gold"):
            break
    b = Board()
    for h in hexes:
        b.add(h)
    b.harbors = [Harbor((tuple(e[0]), tuple(e[1])), k) for e, k in d["harbors"]]
    b.canal_hexes = {tuple(c) for c in d.get("canal_hexes", [])}
    b.robber = None
    return b


def board_for_mode(mode_key: str = "great_canal", seed: int | None = None) -> Board:
    if mode_key != "great_canal":
        raise ValueError(f"marlow only plays The Great Canal, not {mode_key!r}")
    return real_canal_board(seed)


def snap_to_canal_template(read: Board, template: str | None = None, min_match: float = 0.7) -> Board | None:
    """Live Great Canal read -> the fixed template (data/boards/canal_template.json).
    Pieces cover hexes and sand reads as desert/canal, so the fixed parts (sea, deserts, lake,
    mountains, fields + numbers, gold 8s, harbours, the 9 canal chip hexes) come from the
    template; hills/forest/pasture and the shuffled numbers come from the read where it saw
    them (an unread shuffled hex keeps the template terrain, no number). Finds the offset
    with the best agreement; None if under min_match (keep the raw read)."""
    import json
    path = Path(template) if template else Path(__file__).resolve().parent.parent / "data/boards/canal_template.json"
    if not path.exists():
        return None
    d = json.load(open(path))
    tpl = {(q, r): (t, n, bool(br)) for q, r, t, n, br in d["hexes"]}
    fixed = {"sea", "desert", "lake", "mountains", "fields", "gold"}
    same = {"canal": "desert"}                       # old reads called the sand hexes "canal"
    best, best_off = -1.0, (0, 0)
    for dq in range(-3, 4):
        for dr in range(-3, 4):
            hit = tot = nhit = ntot = 0
            for (q, r), (t, n, _) in tpl.items():
                h = read.hexes.get((q + dq, r + dr))
                if t not in fixed:
                    continue
                tot += 1
                hit += h is not None and same.get(h.terrain, h.terrain) == t
                if t in ("fields", "gold") and h is not None and h.number:
                    ntot += 1
                    nhit += h.number == n
            if ntot >= 3 and nhit / ntot < min_match:
                continue                                # fixed numbers disagree: wrong offset or bad read
            if tot and hit / tot > best:
                best, best_off = hit / tot, (dq, dr)
    if best < min_match:
        return None
    dq, dr = best_off
    keep = set(d.get("fixed_numbers", ["fields", "gold"]))
    b = Board()
    for (q, r), (t, n, br) in tpl.items():
        h = read.hexes.get((q + dq, r + dr))
        if t in ("hills", "forest", "pasture") and h is not None and h.terrain in ("hills", "forest", "pasture"):
            t = h.terrain                               # shuffled terrain: the read wins
        if n and t not in keep:
            n = h.number if h is not None else None     # shuffled numbers: only what the read saw
        b.add(Hex(q + dq, r + dr, t, n, barren=br))
    c0, c1 = hex_corners(0, 0)[0], hex_corners(dq, dr)[0]
    vo = tuple(y - x for x, y in zip(c0, c1))          # vertex-key shift for the hex offset
    mv = lambda v: tuple(a + o for a, o in zip(v, vo))
    b.harbors = [Harbor(ekey(mv(e[0]), mv(e[1])), k) for e, k in d["harbors"]]
    b.canal_hexes = {(c[0] + dq, c[1] + dr) for c in d.get("canal_hexes", [])}
    b.robber = read.robber if read.robber in b.hexes else None
    b.template_offset = (dq, dr)                       # read coords = template coords + this
    return b
