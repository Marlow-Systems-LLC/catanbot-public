"""Read the board from a screenshot: number tokens -> hex lattice -> terrain, numbers,
legal-spot highlights and pieces.

Works at any capture scale: everything is measured relative to the fitted hex size.
The board is drawn almost top-down; an affine map (unit hex space -> pixels) fits it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from ..board import Board, Hex, hex_center, hex_corners

SQ3 = math.sqrt(3)


# ---------------------------------------------------------------- token detection
def find_tokens(img: np.ndarray, roi: tuple[int, int, int, int] | None = None,
                scale: float | None = None) -> list[tuple]:
    """Cream number-token candidates as (x, y, r, strong). strong=1: round Hough circle;
    0: colour blob only (could be mountain snow) -- used only if it sits on the lattice."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = roi or (0, 0, W, H)
    sub = img[y0:y1, x0:x1]
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    g = cv2.medianBlur(cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY), 3)
    scale = scale or W / 1372.0      # pixels per calib-frame pixel (1372-wide full screen)
    out: list[tuple[float, float, float]] = []

    def cream(x, y, r):
        r = max(3, int(r * 0.7))
        p = hsv[max(0, y - r):y + r, max(0, x - r):x + r].reshape(-1, 3)
        if len(p) == 0:
            return False
        frac = np.mean((p[:, 1] < 100) & (p[:, 2] > 150) & (p[:, 0] < 40))
        return frac > 0.45

    c = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, 1, int(20 * scale), param1=120, param2=22,
                         minRadius=int(8 * scale), maxRadius=int(19 * scale))
    if c is not None:
        for x, y, r in np.round(c[0]).astype(int):
            if cream(x, y, r):
                out.append((x + x0, y + y0, r, 1))
    m = cv2.inRange(hsv, (8, 20, 165), (35, 95, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, st, cen = cv2.connectedComponentsWithStats(m)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if 150 * scale ** 2 < a < 1500 * scale ** 2 and 0.7 < w / h < 1.4:
            cx, cy = cen[i] + (x0, y0)
            if all(math.hypot(cx - ox, cy - oy) > 10 * scale for ox, oy, *_ in out):
                out.append((cx, cy, max(w, h) / 2, 0))
    return out


# ---------------------------------------------------------------- lattice fit
def camera_warp(shape) -> np.ndarray:
    """Screen pre-warp that removes the camera tilt (data/calib/camera.json): after it
    the board is affine. Identity if not calibrated."""
    import json
    from pathlib import Path
    f = Path(__file__).resolve().parents[2] / "data" / "calib" / "camera.json"
    W = np.eye(3)
    if f.exists():
        d = json.loads(f.read_text())
        H, Wd = shape[:2]
        W[2, 0], W[2, 1] = d["a_n"] / Wd, d["b_n"] / H
    return W


def _warp(W, x, y):
    v = W @ np.array([x, y, 1.0])
    return v[0] / v[2], v[1] / v[2]


@dataclass
class Lattice:
    A: np.ndarray          # 2x3 affine: unit hex space -> pre-warped screen
    W: np.ndarray = None   # screen pre-warp (camera tilt); pixel = W^-1(A u)

    def __post_init__(self):
        if self.W is None:
            self.W = np.eye(3)
        self.Wi = np.linalg.inv(self.W)

    def to_px(self, ux: float, uy: float) -> tuple[float, float]:
        x, y = self.A @ np.array([ux, uy, 1.0])
        x, y = _warp(self.Wi, x, y)
        return float(x), float(y)

    def hex_px(self, q: int, r: int) -> tuple[float, float]:
        return self.to_px(*hex_center(q, r))

    def vertex_px(self, v) -> tuple[float, float]:
        return self.to_px(v[0] * SQ3 / 2, v[1] / 2)

    def edge_px(self, e) -> tuple[float, float]:
        a, b = self.vertex_px(e[0]), self.vertex_px(e[1])
        return (a[0] + b[0]) / 2, (a[1] + b[1]) / 2

    @property
    def size(self) -> float:            # pixel length of one hex edge (approx.)
        return float(np.linalg.norm(self.A[:, 0]) + np.linalg.norm(self.A[:, 1])) / 2

    def from_px(self, x: float, y: float) -> tuple[float, float]:
        x, y = _warp(self.W, x, y)
        M = self.A[:, :2]
        return tuple(np.linalg.solve(M, np.array([x, y]) - self.A[:, 2]))

    def nearest_hex(self, x, y) -> tuple[int, int]:
        ux, uy = self.from_px(x, y)
        r = uy / 1.5
        q = ux / SQ3 - r / 2
        return _axial_round(q, r)


def _axial_round(q, r):
    s = -q - r
    rq, rr, rs = round(q), round(r), round(s)
    dq, dr, ds = abs(rq - q), abs(rr - r), abs(rs - s)
    if dq > dr and dq > ds:
        rq = -rr - rs
    elif dr > ds:
        rr = -rq - rs
    return int(rq), int(rr)


def fit_lattice(points: list[tuple[float, float]], W: np.ndarray | None = None) -> tuple[Lattice, list[tuple[int, int]]]:
    """Find the hex lattice through token centres. Returns lattice + axial coord per point.
    Axial (0,0) is assigned to the token nearest the median position."""
    W = np.eye(3) if W is None else W
    P = np.array([_warp(W, x, y) for x, y in points], float)
    if len(P) < 4:
        raise ValueError("need at least 4 tokens")
    D = P[None, :, :] - P[:, None, :]
    L = np.hypot(D[..., 0], D[..., 1])
    np.fill_diagonal(L, np.inf)
    d = float(np.median(L.min(1)))
    diffs = D[(L > 0.75 * d) & (L < 1.5 * d)]
    ang = np.degrees(np.arctan2(diffs[:, 1], diffs[:, 0]))
    e1s = diffs[np.abs(ang) < 15]                         # east neighbour
    e2s = diffs[(ang > 40) & (ang < 75)]                  # south-east neighbour (axial r+1)
    if len(e1s) == 0 or len(e2s) == 0:
        raise ValueError("lattice directions not found")
    e1, e2 = np.median(e1s, 0), np.median(e2s, 0)
    center = np.median(P, 0)
    o = P[np.argmin(np.hypot(*(P - center).T))]
    B = np.column_stack([e1, e2])
    coords = []
    for p in P:
        q, r = np.linalg.solve(B, p - o)
        coords.append((int(round(q)), int(round(r))))
    # least-squares affine from unit space to pixels
    U = np.array([[*hex_center(q, r), 1.0] for q, r in coords])
    A, *_ = np.linalg.lstsq(U, P, rcond=None)
    lat = Lattice(A.T, W)
    # drop outliers (> 0.25 hex from their lattice point) and refit once
    err = [math.dist(_warp(W, *lat.hex_px(*c)), p) for c, p in zip(coords, P)]
    keep = [i for i, e in enumerate(err) if e < 0.25 * d]
    if 4 <= len(keep) < len(P):
        U2, P2 = U[keep], P[keep]
        A, *_ = np.linalg.lstsq(U2, P2, rcond=None)
        lat = Lattice(A.T, W)
    return lat, coords


# ---------------------------------------------------------------- terrain
# reference HSV (OpenCV ranges) measured on calibration captures; nearest wins
TERRAIN_REF = {   # gc_setup1 calibration (median over 4-6 hexes each)
    "sea": (103, 100, 160),
    "fields": (18, 165, 209),
    "hills": (13, 172, 160),
    "forest": (33, 148, 110),
    "pasture": (33, 156, 156),
    "mountains": (14, 66, 173),
    # game2_board (2026-10-02 canal): numberless sand hexes (desert / canal) are paler than
    # fields; gold fields (island 8s with a river) are darker and greyer than mountains
    "desert": (18, 139, 212),
    "gold": (16, 82, 152),
}
# full screen 1920x1080 with video effects off (the server 2026-10-03): every terrain is more
# saturated, so mountains sat next to the old gold and gold next to hills. Nearest of all refs wins.
TERRAIN_REF2 = {
    "fields": [(18, 202, 229)],
    "hills": [(13, 199, 171)],
    "forest": [(33, 184, 107)],
    "pasture": [(33, 207, 154)],
    "mountains": [(14, 82, 171)],
    "desert": [(17, 160, 215)],
    "gold": [(18, 147, 140)],
}
BEACH_V = 195   # land hexes have a bright sand rim (V>215); the wooden table doesn't (<170)


def hex_color(img, hsv, lat: Lattice, q: int, r: int) -> np.ndarray | None:
    """Median HSV of the hex interior, avoiding the centre token and the corners."""
    cx, cy = lat.hex_px(q, r)
    s = lat.size
    pts = []
    for k in range(6):
        a = math.radians(30 + 60 * k)
        for f in (0.45, 0.6):
            x, y = int(cx + math.cos(a) * s * f), int(cy + math.sin(a) * s * f * 0.9)
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                pts.append(hsv[max(0, y - 2):y + 3, max(0, x - 2):x + 3].reshape(-1, 3))
    if not pts:
        return None
    return np.median(np.vstack(pts), 0)


def rim_value(hsv, lat: Lattice, q: int, r: int, f: float = 0.85) -> float:
    cx, cy = lat.hex_px(q, r)
    vals = []
    for v in hex_corners(q, r):
        x, y = lat.vertex_px(v)
        x, y = int(cx + (x - cx) * f), int(cy + (y - cy) * f)
        if 1 <= x < hsv.shape[1] - 1 and 1 <= y < hsv.shape[0] - 1:
            vals.append(hsv[y - 1:y + 2, x - 1:x + 2, 2].ravel())
    return float(np.median(np.concatenate(vals))) if vals else 0.0


def classify_terrain(c: np.ndarray) -> str:
    def dist(ref):
        dh = min(abs(c[0] - ref[0]), 180 - abs(c[0] - ref[0])) * 3
        return math.hypot(dh, (c[1] - ref[1]) * 0.6, (c[2] - ref[2]) * 0.4)
    return min(TERRAIN_REF, key=lambda k: min(dist(r) for r in [TERRAIN_REF[k], *TERRAIN_REF2.get(k, [])]))


# ---------------------------------------------------------------- numbers
def token_is_red(img, x, y, r) -> bool:
    """6 and 8 are printed red."""
    hsv = cv2.cvtColor(img[int(y - r * .6):int(y + r * .6), int(x - r * .6):int(x + r * .6)], cv2.COLOR_BGR2HSV)
    p = hsv.reshape(-1, 3)
    return np.mean(((p[:, 0] < 8) | (p[:, 0] > 170)) & (p[:, 1] > 120) & (p[:, 2] > 90)) > 0.06


def token_glyph(img, x, y, r, size=32) -> np.ndarray:
    """Normalised binary image of the digits on a token (for template matching)."""
    rr = int(r * 0.75)
    crop = img[int(y - rr):int(y + rr), int(x - rr):int(x + rr)]
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return b


class DigitReader:
    """Template matcher for token numbers; templates live in data/calib/tokens.npz and
    are added from labelled calibration captures."""

    def __init__(self, path=None):
        from pathlib import Path
        self.path = Path(path or Path(__file__).resolve().parents[2] / "data" / "calib" / "tokens.npz")
        self.templates: list[tuple[int, np.ndarray]] = []
        if self.path.exists():
            d = np.load(self.path)
            self.templates = list(zip(d["labels"].tolist(), d["glyphs"]))

    def add(self, label: int, glyph: np.ndarray):
        self.templates.append((label, glyph))

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self.path, labels=np.array([l for l, _ in self.templates]),
                 glyphs=np.array([g for _, g in self.templates]))

    def read(self, img, x, y, r) -> tuple[int | None, float]:
        if not self.templates:
            return None, 0.0
        g = token_glyph(img, x, y, r).astype(np.float32) / 255
        red = token_is_red(img, x, y, r)
        best, score = None, -1.0
        for label, t in self.templates:
            if (label in (6, 8)) != red:
                continue
            t = t.astype(np.float32) / 255
            s = float(cv2.matchTemplate(g, t, cv2.TM_CCOEFF_NORMED)[0, 0])
            if s > score:
                best, score = label, s
        return best, score


# ---------------------------------------------------------------- highlights & pieces
def find_highlight_rings(img, lat: Lattice) -> list[tuple[float, float]]:
    """White selection rings the game draws on legal spots (setup / build mode)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (0, 0, 225), (180, 40, 255))
    s = lat.size
    n, _, st, cen = cv2.connectedComponentsWithStats(m)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if 0.25 * s < w < 0.75 * s and 0.25 * s < h < 0.75 * s and a < 0.6 * w * h:
            out.append(tuple(cen[i]))
    return out


RING_MIN = 0.1   # gc_setup1: legal spots 0.12-0.58, others 0.0-0.04


def ring_score(hsv, lat: Lattice, v, edge=None) -> float:
    """Fraction of a small circle around a vertex (or edge midpoint) that is bright white:
    the game's selection ring. Tries a few radii since rings scale with zoom."""
    x, y = lat.edge_px(edge) if edge else lat.vertex_px(v)
    best = 0.0
    for f in (0.12, 0.14, 0.16, 0.19, 0.22, 0.26):
        rad = f * lat.size
        hits = 0
        n = 24
        for k in range(n):
            a = 2 * math.pi * k / n
            px, py = int(x + rad * math.cos(a)), int(y + rad * math.sin(a))
            if 0 <= px < hsv.shape[1] and 0 <= py < hsv.shape[0]:
                p = hsv[max(0, py - 1):py + 2, max(0, px - 1):px + 2].reshape(-1, 3)
                hits += bool(np.any((p[:, 1] < 45) & (p[:, 2] > 215)))
        best = max(best, hits / n)
    return best


PLAYER_HSV = {  # piece colours (hue range, min sat, (min val, max val)); gc_setup4 calibration
    "red": ((0, 6), 170, (165, 255)),       # marlow.systems in solo calib (bright red)
    "brown": ((6, 12), 160, (90, 155)),     # dark red-brown; hills hexes are h13 / v160
    "purple": ((135, 156), 100, (90, 255)),
    "magenta": ((157, 172), 140, (120, 255)),  # marlow.systems, canal game 3 2026-10-02 (h162 s180 v205)
    # Vale, live canal game 2026-10-02 (h95 s110 v80-126); Jean's teal 2026-10-04 h94 s140 v90-130.
    # The sea is h102 s119 v160: hue capped at 99 and value at 145 so open water never reads blue
    "blue": ((88, 99), 90, (45, 145)),
    "orange": ((12, 22), 190, (190, 255)),  # unconfirmed
    "white": (None, 0, (225, 255)),         # unconfirmed; clashes with selection rings
    "green": ((45, 80), 150, (80, 255)),    # unconfirmed
    # marlow.systems in the 4-player canal game 2026-10-02: grey pieces lit warm by the
    # table (h14 s50 v60-150; sand behind is v>200, tokens are s<35)
    "grey": ((8, 22), 38, (60, 160), 68),
}


def color_at(hsv, x, y, rad) -> dict[str, float]:
    p = hsv[max(0, int(y - rad)):int(y + rad), max(0, int(x - rad)):int(x + rad)].reshape(-1, 3)
    if len(p) == 0:
        return {}
    out = {}
    for name, spec in PLAYER_HSV.items():
        hr, smin, (vmin, vmax) = spec[:3]
        smax = spec[3] if len(spec) > 3 else 255
        v = p[:, 2]
        if hr is None:
            ok = (p[:, 1] < 35) & (v >= vmin)
        else:
            ok = (p[:, 0] >= hr[0]) & (p[:, 0] <= hr[1]) & (p[:, 1] >= smin) & (p[:, 1] <= smax) \
                & (v >= vmin) & (v <= vmax)
        out[name] = float(np.mean(ok))
    return out


# ---------------------------------------------------------------- whole board
@dataclass
class BoardRead:
    board: Board
    lattice: Lattice
    tokens: dict[tuple[int, int], int | None]
    highlights: list                     # vertex keys the game highlights


_OCR_CROPS = ((0.45, 3), (0.38, 3), (0.38, 4), (0.55, 4), (0.45, 6), (0.55, 3),
              # 2026-10-03: small single 3s and some 10s only read on a tight crop or binarised
              (0.3, 4), (0.38, 8), (0.38, -4), (0.45, -4))   # fx < 0: Otsu black/white


def read_hex_number(img, lat, c, digits: "DigitReader") -> int | None:
    """Token number of hex c: Apple Vision OCR on a few crops of the digit (above the pips),
    then the glyph templates. Only 6 and 8 are printed red, so a read must match the ink colour."""
    x, y = lat.hex_px(*c)
    red = token_is_red(img, x, y, 0.36 * lat.size)
    votes: list[int] = []
    try:
        from .. import mac
        for k, (f, fx) in enumerate(_OCR_CROPS):
            R = int(f * lat.size)
            crop = img[max(0, int(y - 1.1 * R)):int(y + 0.6 * R), max(0, int(x - R)):int(x + R)]
            if crop.size == 0:
                break
            big = cv2.resize(crop, None, fx=abs(fx), fy=abs(fx), interpolation=cv2.INTER_CUBIC)
            if fx < 0:
                _, bw = cv2.threshold(cv2.cvtColor(big, cv2.COLOR_BGR2GRAY), 0, 255,
                                      cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                big = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
            texts = [o["text"] for o in mac.ocr(big)]
            # the pips under the digit sometimes come back as their own box (". ."): the dot
            # count fixes the pair (2 dots = 3 or 11) when the digit is misread (3 read as 2)
            pips = [t.count(".") + t.count("•") for t in texts if t.strip() and not set(t) - set(". •·")]
            for t in texts:
                d = "".join(ch for ch in t if ch.isdigit())
                if not d or not (2 <= int(d) <= 12) or int(d) == 7:
                    continue
                n = int(d)
                if pips and pips[0] in _PAIR and n not in _PAIR[pips[0]]:
                    n = _PAIR[pips[0]][len(d) - 1]
                if (n in (6, 8)) != red:
                    continue
                if k < _OCR_FIRST:
                    return n
                votes += [n, n] if pips else [n]      # a read backed by its pips counts double
                break
    except Exception:                          # noqa: BLE001 -- no OCR here: templates only
        pass
    if votes:
        # the extra crops are noisier (a 3 once read as 2): take the most common answer
        return max(set(votes), key=lambda v: (votes.count(v), -votes.index(v)))
    # full screen 2026-10-03: blank canal chips matched "3" at 0.18-0.22; real tokens score >= 0.38
    n, score = digits.read(img, x, y, 0.36 * lat.size)
    return n if score >= TEMPLATE_MIN else None


_OCR_FIRST = 6     # the first crops are trusted on their own; later ones vote
TEMPLATE_MIN = 0.3


def read_board(img: np.ndarray, roi=None, digits: DigitReader | None = None, radius: int = 6,
               lattice: Lattice | None = None, canal: bool = False, scale: float | None = None) -> BoardRead:
    """Pass the game's first lattice back in on later reads so vertex keys stay stable.
    canal=True: numberless sand hexes are canal hexes, not a desert with the robber."""
    cand = find_tokens(img, roi, scale=scale)
    strong = [t for t in cand if t[3]]
    lat = lattice or fit_lattice([(x, y) for x, y, *_ in (strong if len(strong) >= 6 else cand)],
                                 camera_warp(img.shape))[0]
    toks, coords = [], []
    for t in cand:
        c = lat.nearest_hex(t[0], t[1])
        if math.dist(lat.hex_px(*c), t[:2]) < (0.3 if t[3] else 0.12) * lat.size:
            toks.append(t[:3]); coords.append(c)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    digits = digits or DigitReader()
    b = Board()
    tokmap: dict[tuple[int, int], int | None] = {}
    cache = getattr(lat, "_numbers", None)
    if cache and set(coords) <= set(cache):
        tokmap = {c: cache[c] for c in set(coords)}
    else:
        # game 2 (2026-10-02): reading every cream candidate let stray ones overwrite the real
        # token (12 of 24 numbers wrong -> terrible setup spots). Read once per hex at the hex
        # centre: OCR first (never wrong in tests), template as fallback. 23/24 on game2_board.
        for c in dict.fromkeys(coords):
            tokmap[c] = read_hex_number(img, lat, c, digits)
        if len(tokmap) >= 10:
            lat._numbers = dict(tokmap)        # numbers don't move during a game
    for q in range(-radius, radius + 1):
        for r in range(-radius, radius + 1):
            x, y = lat.hex_px(q, r)
            if not (0 <= x < img.shape[1] and 0 <= y < img.shape[0]):
                continue
            if roi and not (roi[0] <= x < roi[2] and roi[1] <= y < roi[3]):
                continue                      # player banners / hand bar, not the board
            col = hex_color(img, hsv, lat, q, r)
            if col is None:
                continue
            t = classify_terrain(col)
            if t != "sea" and rim_value(hsv, lat, q, r) < BEACH_V:
                continue                      # wooden table, not a hex
            if t == "sea" and abs(col[0] - 103) > 12:
                continue
            if (q, r) in tokmap and t in ("sea", "table"):
                t = "unknown"
            if t == "desert" and tokmap.get((q, r)):
                t = "fields"                  # deserts/canals never carry a number
            if t == "desert" and canal:
                t = "canal"
            b.add(Hex(q, r, t, tokmap.get((q, r)), barren=(canal and t == "fields")))
            if t == "desert":
                b.robber = (q, r)
    if canal:
        # owner 2026-10-02: in the Great Canal the gold fields always sit at the bottom and are 8s
        land = [h for h in b.hexes.values() if h.is_land]
        low = max((h.coord[1] for h in land), default=0) - 1
        for h in land:
            if h.terrain == "gold" or (h.terrain == "mountains" and h.number == 8 and h.coord[1] >= low):
                h.terrain, h.number = "gold", 8
        # 2026-10-03: pieces hide hexes and sand reads as canal; the fixed layout comes from
        # data/boards/canal_template.json (kept as the raw read if it doesn't line up)
        from ..board import snap_to_canal_template
        snapped = snap_to_canal_template(b)
        off = getattr(snapped, "template_offset", (0, 0))
        if snapped is not None and off != (0, 0) and not getattr(lat, "_fixed", False):
            # live 2026-10-03 game 6: axial (0,0) is the token nearest the middle, so the keys
            # shifted between restarts ((3,7) became (1,7)). Re-anchor a lattice on its first canal
            # read to the fixed template: the same corner gets the same key in every scan/restart.
            A = lat.A.copy()
            A[:, 2] = A[:, 2] + A[:, :2] @ np.array(hex_center(*off))
            return read_board(img, roi, digits, radius, Lattice(A, lat.W), canal, scale)
        b = snapped or b
        lat._fixed = True                              # first canal read decides; keys never move after
    hl = [v for v in b.vertices if b.is_land_vertex(v) and ring_score(hsv, lat, v) > RING_MIN]
    if not b.harbors:
        b.harbors = detect_harbors(img, b, lat)
    return BoardRead(b, lat, tokmap, hl)


def _harbor_kind(patch: np.ndarray) -> str:
    """White disc with '?' = 3:1. Coloured discs = 2:1, by hue (live 2026-10-02 game 4;
    the 2:1 colour map is UNCONFIRMED: brick red, grain yellow, wool light green,
    lumber dark green, ore grey)."""
    white = ((patch[:, 1] < 70) & (patch[:, 2] > 170)).mean()
    if white > 0.35:
        return "3:1"
    h, sat, _ = np.median(patch, axis=0)
    if sat < 90:
        return "ore"
    if h < 15:
        return "brick"
    if h < 30:
        return "grain"
    if h < 42:
        return "wool"
    return "lumber"


def detect_harbors(img: np.ndarray, b: Board, lat: Lattice) -> list:
    """Harbour tokens sit in the sea about halfway between a coast edge and the sea hex's
    centre. Sample a small grid there per coast edge; a token is a non-blue disc. Each token
    goes to the coast edge it sits closest to (that edge's two corners use the harbour)."""
    from ..board import Harbor
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = lat.size
    r = max(3, int(0.22 * s))
    H, W = hsv.shape[:2]
    hits = []
    for e in b.edges:
        hs = b.edge_hexes(e)
        seas = [h for h in hs if h.terrain == "sea"]
        if not seas or not any(h.is_land for h in hs):
            continue
        m = np.mean([lat.vertex_px(v) for v in e], axis=0)
        c = np.array(lat.hex_px(*seas[0].coord))
        a, bv = (np.array(lat.vertex_px(v)) for v in e)
        along = (bv - a) / (np.linalg.norm(bv - a) or 1)
        best = None
        for f in (0.4, 0.55, 0.7):
            for lat_off in (-0.3, -0.15, 0.0, 0.15, 0.3):
                x, y = (m + f * (c - m) + lat_off * s * along).astype(int)
                if not (r <= x < W - r and r <= y < H - r):
                    continue
                patch = hsv[y - r:y + r + 1, x - r:x + r + 1].reshape(-1, 3).astype(int)
                sea = ((patch[:, 0] >= 95) & (patch[:, 0] <= 115) & (patch[:, 1] > 80)).mean()
                if best is None or sea < best[0]:
                    best = (sea, (x, y), patch, np.linalg.norm(m + 0.55 * (c - m) - (x, y)))
        if best and best[0] < 0.15:
            hits.append((best[0], best[3], e, best[1], best[2]))
    out, taken = [], []
    for sea, dist, e, xy, patch in sorted(hits, key=lambda t: (t[0], t[1])):
        if any(np.hypot(xy[0] - x, xy[1] - y) < 0.6 * s for x, y in taken):
            continue
        taken.append(xy)
        out.append(Harbor(e, _harbor_kind(patch)))
    return out


def island_of(board: Board, seeds) -> set[tuple[int, int]]:
    """Land hexes connected (through land) to the hexes touching the seed vertices."""
    from ..board import DIRS
    todo = [h.coord for v in seeds for h in board.vertex_hexes(v) if h.is_land]
    seen = set()
    while todo:
        c = todo.pop()
        if c in seen:
            continue
        seen.add(c)
        for dq, dr in DIRS:
            n = (c[0] + dq, c[1] + dr)
            if n in board.hexes and board.hexes[n].is_land and n not in seen:
                todo.append(n)
    return seen


def setup_options(gs, br: BoardRead) -> list:
    """Legal setup spots: the game's highlights tell us *which island*; the distance rule
    gives every legal vertex on it (faint rings on sand are easy to miss)."""
    if not br.highlights:
        return gs.legal_settlement_spots(gs.me(), setup=True)
    isl = island_of(gs.board, br.highlights)
    legal = gs.legal_settlement_spots(gs.me(), setup=True)
    return [v for v in legal if any(h.coord in isl for h in gs.board.vertex_hexes(v))] or br.highlights


@dataclass
class Pieces:
    """What detect_pieces sees. Unpacks as (blds, roads) for older callers."""
    blds: dict          # vertex -> colour (settlement or city: not told apart yet)
    roads: dict         # edge -> colour
    ships: dict         # edge -> colour (thick boxes on sea/coast edges)
    knights: dict       # vertex -> (colour, active)

    def __iter__(self):
        yield self.blds
        yield self.roads


class _Masks(dict):
    """Colour -> bool mask, plus all masks stacked (H, W, C) so one index reads every colour."""
    stack = None


def _colour_masks(hsv, colors) -> dict[str, np.ndarray]:
    out = _Masks()
    for c in colors:
        spec = PLAYER_HSV.get(c)
        if not spec or spec[0] is None:
            continue
        hr, smin, (vmin, vmax) = spec[:3]
        smax = spec[3] if len(spec) > 3 else 255
        out[c] = cv2.inRange(hsv, np.array((hr[0], smin, vmin)), np.array((hr[1], smax, vmax))) > 0
    if out:
        out.stack = np.dstack(list(out.values())).astype(np.float32)
    return out


_ANNULI: dict = {}


def _ring_px(hsv, x, y, r0, r1):
    """Pixel rows/cols of the annulus r0 <= d <= r1 around (x, y), clipped to the image."""
    H, W = hsv.shape[:2]
    # offsets cached per radius pair and sub-pixel phase (0.25 px): ~550 probes a read
    key = (round(r0, 2), round(r1, 2), round(x % 1, 2) // 0.25, round(y % 1, 2) // 0.25)
    off = _ANNULI.get(key)
    if off is None:
        fx, fy = (x % 1), (y % 1)
        n = int(r1) + 2
        dy, dx = np.mgrid[-n:n + 1, -n:n + 1]
        d = np.hypot(dx - fx, dy - fy)
        m = (d >= r0) & (d <= r1)
        off = _ANNULI[key] = (dy[m], dx[m])
    ys, xs = off[0] + int(np.floor(y)), off[1] + int(np.floor(x))
    m = (xs >= 0) & (ys >= 0) & (xs < W) & (ys < H)
    return ys[m], xs[m]


def _frac(masks, ys, xs) -> dict[str, float]:
    if len(ys) == 0:
        return {c: 0.0 for c in masks}
    if getattr(masks, "stack", None) is not None:
        return dict(zip(masks, masks.stack[ys, xs].mean(axis=0).tolist()))
    return {c: float(np.mean(m[ys, xs])) for c, m in masks.items()}


# live 2026-10-04 (045123_road, 1372x888 frame, hex edge 34 px): a settlement/city is a solid
# block ~15x12 px centred on its corner (centre 0.61-0.92 own colour, disc r<=0.25 hex 0.35+);
# a knight is a coloured ring (r 0.18-0.30 hex: 0.31-0.49) around a beige (inactive) or dark
# (active) disc; road ends at an empty corner fill the centre far less.
BLD_CENTRE, BLD_DISC = 0.45, 0.25
KNIGHT_RING, KNIGHT_CENTRE_MAX = 0.25, 0.2
ROAD_MIN, ROAD_STRONG = 0.35, 0.6
SHIP_WIDTH = 0.17           # coloured width across a coast edge, in hex edges: road ~0.09, ship ~0.25


def _across(masks_c, x, y, ex, ey, half):
    """Longest coloured run along the line through (x, y) across the edge direction (ex, ey)."""
    nx, ny = -ey, ex
    H, W = masks_c.shape
    run = best = 0
    for k in range(-int(half), int(half) + 1):
        px, py = int(round(x + nx * k)), int(round(y + ny * k))
        on = 0 <= px < W and 0 <= py < H and bool(masks_c[py, px])
        run = run + 1 if on else 0
        best = max(best, run)
    return best


def detect_pieces(img, br: BoardRead, colors: list[str], thresh: float = ROAD_MIN,
                  native=None, lay=None) -> Pieces:
    """Buildings and knights per vertex, roads and ships per edge, by player colour.
    native/lay: read on the full-size capture (hex edge 50 px on a 1920x1080 screen instead of 34
    in the frame) -- keys are lattice keys either way. City vs settlement is not separated here:
    the runner tracks it from setup order, popups and its own builds."""
    lat, b = br.lattice, br.board
    s = lat.size
    pos = lambda x, y: (x, y)                                          # noqa: E731
    if native is not None and lay is not None:
        img, s = native, s * lay.s
        pos = lay.to_native
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    masks = _colour_masks(hsv, colors)
    blds, roads, ships, knights = {}, {}, {}, {}
    if not masks:
        return Pieces(blds, roads, ships, knights)
    for v in b.vertices:
        if not b.is_land_vertex(v):
            continue
        x, y = pos(*lat.vertex_px(v))
        centre = _frac(masks, *_ring_px(hsv, x, y, 0, 0.12 * s))
        disc = _frac(masks, *_ring_px(hsv, x, y, 0, 0.25 * s))
        c = max(masks, key=lambda k: centre[k] + disc[k])
        if centre[c] >= BLD_CENTRE and disc[c] >= BLD_DISC:
            blds[v] = c
            continue
        ys, xs = _ring_px(hsv, x, y, 0.18 * s, 0.30 * s)
        ring = _frac(masks, ys, xs)
        c = max(masks, key=lambda k: ring[k])
        if ring[c] >= KNIGHT_RING and max(centre.values()) < KNIGHT_CENTRE_MAX:
            # 6 of 8 sectors must see the ring colour (a road passing by fills only one side)
            ang = np.arctan2(ys - y, xs - x)
            sect = ((ang + np.pi) / (2 * np.pi) * 8).astype(int) % 8
            hit = masks[c][ys, xs]
            if len({int(k) for k in sect[hit]}) >= 6:
                cy_, cx_ = _ring_px(hsv, x, y, 0, 0.12 * s)
                active = float(np.median(hsv[cy_, cx_, 2])) < 110   # active: dark disc; inactive: beige
                knights[v] = (c, active)
    for e in b.edges:
        land, sea = b.is_land_edge(e), b.is_sea_edge(e)
        if not land and not sea:
            continue
        x, y = pos(*lat.edge_px(e))
        sc = _frac(masks, *_ring_px(hsv, x, y, 0, 0.07 * s))
        c = max(masks, key=lambda k: sc[k])
        if sc[c] <= thresh:
            continue
        if sea:
            (ax, ay), (bx, by) = pos(*lat.vertex_px(e[0])), pos(*lat.vertex_px(e[1]))
            n = math.hypot(bx - ax, by - ay) or 1.0
            wide = _across(masks[c], x, y, (bx - ax) / n, (by - ay) / n, 0.3 * s) >= SHIP_WIDTH * s
            if wide or not land:
                ships[e] = (c, sc[c])
                continue
        roads[e] = (c, sc[c])
    # keep a road/ship that joins its colour's network (any own building, knight, road or ship),
    # or reads strongly on its own (beaches never do in these colours)
    own_v = {v: c for v, c in blds.items()}
    for v, (c, _) in knights.items():
        own_v.setdefault(v, c)
    keep: dict = {}
    allr = {**{e: v for e, v in roads.items()}, **{e: v for e, v in ships.items()}}
    grow = True
    while grow:
        grow = False
        for e, (c, score) in allr.items():
            if e in keep:
                continue
            if score >= ROAD_STRONG or any(own_v.get(v) == c for v in e) or any(
                    keep.get(e2) == c for v in e for e2 in b.vertex_edges(v) if e2 != e):
                keep[e] = c
                grow = True
    return Pieces(blds, {e: c for e, c in keep.items() if e in roads},
                  {e: c for e, c in keep.items() if e in ships}, knights)


# ---------------------------------------------------------------- numbers by structure
_PAIR = {1: (2, 12), 2: (3, 11), 3: (4, 10), 4: (5, 9), 5: (6, 8)}


def read_token_number(img, cx, cy, size) -> tuple[int | None, dict]:
    """Number on a token from its structure: pip-dot count -> pair, ink width -> 1 vs 2
    digits, holes -> 6/8 and 5/9. Resolution independent (crop is rescaled)."""
    # re-centre on the cream disc (tokens sit a little off the hex centre)
    R = int(0.6 * size)
    win = img[max(0, int(cy - R)):int(cy + R), max(0, int(cx - R)):int(cx + R)]
    if win.size:
        hw = cv2.cvtColor(win, cv2.COLOR_BGR2HSV)
        m = ((hw[..., 1] < 100) & (hw[..., 2] > 150) & (hw[..., 0] < 40)).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, lab, st, cen = cv2.connectedComponentsWithStats(m)
        if n > 1:
            i = 1 + int(np.argmax(st[1:, 4]))
            cx, cy = cen[i][0] + max(0, int(cx - R)), cen[i][1] + max(0, int(cy - R))
            size = max(st[i][2], st[i][3]) / 0.72      # disc diameter ~0.72 hex size
    r = int(0.38 * size)
    crop = img[max(0, int(cy - r)):int(cy + r), max(0, int(cx - r)):int(cx + r)]
    if crop.size == 0:
        return None, {}
    S = 128
    crop = cv2.resize(crop, (S, S), interpolation=cv2.INTER_CUBIC)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    ink = ((hsv[..., 2] < 110) | ((hsv[..., 1] > 90) & ((hsv[..., 0] < 10) | (hsv[..., 0] > 165))
                                   & (hsv[..., 2] > 80))).astype(np.uint8) * 255
    # restrict to the cream disc
    disc = np.zeros_like(ink)
    cv2.circle(disc, (S // 2, S // 2), int(S * 0.40), 255, -1)
    ink &= disc
    # dots: small blobs in the lower band
    band = ink[int(S * 0.66):int(S * 0.86), :]
    n, _, st, _ = cv2.connectedComponentsWithStats(band)
    dots = sum(1 for i in range(1, n) if 4 <= st[i][4] <= 120 and st[i][3] < S * 0.12)
    # glyph in the middle band
    g = ink[int(S * 0.12):int(S * 0.66), :]
    g = cv2.morphologyEx(g, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    cnts, hier = cv2.findContours(g, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    big = [i for i, c in enumerate(cnts) if cv2.contourArea(c) > 60 and hier[0][i][3] == -1]
    if not big:
        return None, {"dots": dots}
    xs = np.concatenate([cnts[i][:, 0, 0] for i in big])
    width = (xs.max() - xs.min()) / S
    holes = sum(1 for i, c in enumerate(cnts) if hier[0][i][3] in big and cv2.contourArea(c) > 12)
    info = {"dots": dots, "width": round(float(width), 2), "holes": holes, "parts": len(big)}
    two_digit = len(big) >= 2 or width > 0.42
    if dots not in _PAIR:
        # fall back: guess from glyph alone
        return None, info
    lo, hi = _PAIR[dots]
    if dots in (1, 2, 3):
        return (hi if two_digit else lo), info
    if dots == 5:
        return (8 if holes >= 2 else 6), info
    return (9 if holes >= 1 else 5), info
