"""Game-window layout and the game-start scan.

Owner 2026-10-03: run the game full screen for more detail; when a game starts, take one
screenshot, read and annotate everything (players, colours, turn order, hexes, numbers) and
only play once that scan passes.

How any window shape maps onto the calibrated laptop frame (data/calib/ui.json):
Unity scales its UI by s = sqrt(width ratio * height ratio) (canvas scaler, match 0.5) and pins
each control to a corner or edge. Measured 2026-10-03 on a 1920x1080 full screen against the
laptop's 1280x828 pt window: left icon column 60 px apart vs 40.9 in the frame (x1.467 =
sqrt(1.574 * 1.369)); dice, trade and chat buttons land within 7 px. So the capture is scaled by
1/s into a "virtual frame" where every control has its laptop size, and a control only moves by
the extra width/height, depending on its anchor.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REF_W, REF_H = 1280 * 1372 / 1440, 828 * 1372 / 1440     # laptop game content inside the frame
TOP = 54                                                   # content top in the frame

# anchor per ui.json key: (horizontal l/c/r, vertical t/c/b). Unlisted keys: by thirds of the frame.
ANCHORS = {
    "dice_or_end_turn": "rb", "trade_open": "rb",                      # checked full screen 2026-10-03
    "chat_open": "lb", "chat_close": "lb", "info_button": "lb", "quit_gear": "lb",   # checked (icon column)
    "chat_input": "lb", "chat_lines_region": "lb", "quit_check": "lb",
    "setup_confirm": "cb", "place_cancel": "cb",   # checked full screen 2026-10-03 game 6: (654,646)
    "trade_want_row": "cb", "trade_give_row": "cb", "trade_ok": "cb", "trade_cancel": "cb",
    "trade_bank": "cb", "trade_player_select": "cb",   # the trade panel docks on the bottom bar
    "offer_in_decline": "rc", "offer_in_counter": "rc", "offer_in_accept": "rc",   # UNCONFIRMED vertical
    "progress_hand": "cb", "progress_play": "cc", "progress_discard": "cc",          # UNCONFIRMED
    "improve_chart_trade": "cb", "improve_chart_politics": "cb", "improve_chart_science": "cb",
}


@dataclass
class Layout:
    """Native capture (w x h px) <-> virtual frame. Identity for the laptop window."""
    w: int
    h: int

    def __post_init__(self):
        self.s = math.sqrt(self.w / REF_W * self.h / REF_H)    # native px per frame px
        self.vw, self.vh = self.w / self.s, self.h / self.s      # content size in frame px
        self.dx, self.dy = self.vw - REF_W, self.vh - REF_H      # extra room vs the laptop
        self.size = (int(math.ceil(self.vw)), TOP + int(math.ceil(self.vh)))

    @property
    def is_reference(self) -> bool:
        return abs(self.dx) < 2 and abs(self.dy) < 2

    def x(self, x: float, a: str = "l") -> float:
        return x + {"l": 0, "c": self.dx / 2, "r": self.dx}[a]

    def y(self, y: float, a: str = "t") -> float:
        return y + {"t": 0, "c": self.dy / 2, "b": self.dy}[a]

    def xy(self, x, y, anchor="lt"):
        return [int(round(self.x(x, anchor[0]))), int(round(self.y(y, anchor[1])))]

    def frame(self, img: np.ndarray, min_size=(0, 0)) -> np.ndarray:
        """Native capture -> virtual frame (content at (0, TOP), black band above, black pad
        up to min_size so the laptop frame stays 1372x888 exactly)."""
        small = cv2.resize(img, (self.size[0], self.size[1] - TOP), interpolation=cv2.INTER_AREA)
        out = np.zeros((max(self.size[1], min_size[1]), max(self.size[0], min_size[0]), 3), np.uint8)
        out[TOP:self.size[1], :self.size[0]] = small
        return out

    def to_native(self, x, y) -> tuple[float, float]:
        return x * self.s, (y - TOP) * self.s

    def to_frame(self, x, y) -> tuple[float, float]:
        return x / self.s, y / self.s + TOP

    def ui(self, base: dict) -> dict:
        """ui.json with every coordinate moved for this layout."""
        out = {}
        for k, v in base.items():
            if k.startswith("_") or not isinstance(v, (list, dict)) or k == "trade_order":
                out[k] = v
                continue
            a = ANCHORS.get(k)
            if isinstance(v, dict) and "x" in v:
                a = a or "cb"
                out[k] = dict(v, x=[int(round(self.x(x, a[0]))) for x in v["x"]], y=int(round(self.y(v["y"], a[1]))))
            elif isinstance(v, list) and len(v) in (2, 4) and all(isinstance(t, (int, float)) for t in v):
                a = a or _thirds(v[0], v[1])
                pts = [self.xy(v[i], v[i + 1], a) for i in range(0, len(v), 2)]
                out[k] = [c for p in pts for c in p]
            else:
                out[k] = v
        return out


def _thirds(x, y) -> str:
    h = "l" if x < REF_W / 3 else "r" if x > 2 * REF_W / 3 else "c"
    v = "t" if y < TOP + REF_H / 3 else "b" if y > TOP + 2 * REF_H / 3 else "c"
    return h + v


# ---------------------------------------------------------------- game-start scan
@dataclass
class Scan:
    layout: tuple
    players: list = field(default_factory=list)     # [{name, colour, seat, x, y, current}]
    hexes: list = field(default_factory=list)       # [{q, r, terrain, number, x, y}]
    warp: tuple = (0.0, 0.0)
    token_scale: float = 1.0
    hex_px: float = 0.0
    problems: list = field(default_factory=list)
    lattice: object = None

    @property
    def ok(self) -> bool:
        return not self.problems

    def summary(self) -> str:
        pl = ", ".join(f"{p['seat']}.{p['name']}={p['colour']}{'*' if p['current'] else ''}" for p in self.players)
        nums = sorted(h["number"] for h in self.hexes if h["number"])
        land = [h for h in self.hexes if h["terrain"] not in ("sea",)]
        return (f"players [{pl}] | {len(land)} land hexes, {len(nums)} numbers {nums} | hex {self.hex_px:.0f}px"
                f" warp {self.warp[0]:.3f},{self.warp[1]:.3f} | " + ("OK" if self.ok else "PROBLEMS: " + "; ".join(self.problems)))

    def to_json(self) -> dict:
        return {"layout": self.layout, "players": self.players, "hexes": self.hexes, "warp": self.warp,
                "token_scale": self.token_scale, "hex_px": self.hex_px, "problems": self.problems}


NAME_RE = re.compile(r"^[A-Za-z][\w.\-]{2,}$")      # "Lin" is a player (2026-10-03)


def read_players(img, ocr, banner_color) -> list[dict]:
    """Name banners along the top, left to right = seat order. Colour from the banner hue;
    the current player has the gold triangle above its portrait."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    out = []
    for o in sorted((o for o in ocr if TOP <= o["y"] < 130 and NAME_RE.match(o["text"].strip())
                     and len(re.findall(r"[a-z]", o["text"])) >= 2), key=lambda o: o["x"]):
        x, y, w, h = (int(o[k]) for k in ("x", "y", "w", "h"))
        if out and x - out[-1]["x"] < 170:
            continue          # the improvement badges right of a name read as "SrE"/"ELE" (gc_game4)
        reg = hsv[y:y + h, max(0, x - 6):x + w + 60].reshape(-1, 3)
        sat = reg[reg[:, 1] > 80]
        if len(reg) < 300:
            continue
        hh, ss, vv = np.median(sat if len(sat) >= 300 else reg, 0)   # grey banner: few saturated px
        tri = hsv[max(0, y - 40):max(1, y - 8), max(0, x - 110):max(1, x - 10)].reshape(-1, 3)
        gold = float(((tri[:, 0] > 15) & (tri[:, 0] < 32) & (tri[:, 1] > 140) & (tri[:, 2] > 170)).mean()) if len(tri) else 0
        out.append({"name": o["text"].strip(), "colour": banner_color(hh, ss, vv), "x": x, "y": y,
                    "gold": round(gold, 3)})
    if out:
        best = max(out, key=lambda p: p["gold"])
        for i, p in enumerate(out):
            p["seat"] = i + 1
            p["current"] = p is best and p["gold"] > 0.04
    return out


def fit_board(img, roi):
    """Token scale and camera tilt that fit this screen best (any zoom, any window).
    Returns (lattice, token_scale, (a_n, b_n), n_tokens) or None."""
    from .board_reader import find_tokens, fit_lattice, _warp
    prior = _calibrated_warp()
    best = None
    for sc in (0.8, 1.0, 1.25, 1.55, 1.9):
        toks = [t for t in find_tokens(img, roi, scale=sc) if t[3]]
        if len(toks) < 6:
            continue
        pts = [(x, y) for x, y, *_ in toks]
        rad = float(np.median([t[2] for t in toks]))
        H, W = img.shape[:2]
        for a in np.linspace(-0.05, 0.10, 7):
            for b in np.linspace(0.0, 0.6, 13):
                Wm = np.eye(3); Wm[2, 0], Wm[2, 1] = a / W, b / H
                try:
                    lat, coords = fit_lattice(pts, Wm)
                except (ValueError, np.linalg.LinAlgError):
                    continue
                d = lat.size
                if not 0.28 < rad / d < 0.55:
                    continue          # tokens are ~0.4 hex edge (laptop 11/27 px, full screen 20/51)
                err = [math.dist(lat.hex_px(*lat.nearest_hex(*p)), p) / d for p in pts]
                inl = sum(e < 0.15 for e in err)
                # ties (within 0.01 hex) go to the laptop calibration: few tokens over-fit the tilt
                # share of detections on the lattice, not their count: a too-small token size
                # adds stray circles that pull a wrong lattice (gc_turn1: hex 35 px vs 27)
                if inl < 8:
                    continue
                cost = (-round(inl / len(pts), 1), round(float(np.median(err)), 2), math.dist((a, b), prior), -inl)
                if best is None or cost < best[0]:
                    best = (cost, lat, sc, (float(a), float(b)), inl)
    if best is None:
        return None
    return best[1], best[2], best[3], best[4]


def _calibrated_warp():
    try:
        d = json.loads((ROOT / "data" / "calib" / "camera.json").read_text())
        return d["a_n"], d["b_n"]
    except Exception:                            # noqa: BLE001
        return 0.0, 0.3


def scan(img, ocr, banner_color, canal=False, roi=None) -> Scan:
    """Read everything once, in the virtual frame. Problems listed = don't play yet."""
    from .board_reader import read_board
    H, W = img.shape[:2]
    s = Scan(layout=(W, H))
    s.players = read_players(img, ocr, banner_color)
    if len(s.players) < 2:
        s.problems.append(f"only {len(s.players)} player banners read")
    if not any(re.sub(r"\s", "", p["name"]).lower() == "marlow.systems" for p in s.players):
        s.problems.append("marlow.systems banner not found")
    roi = roi or (80, 150, W - 10, H - 40)
    fb = fit_board(img, roi)
    if fb is None:
        s.problems.append("board: fewer than 6 number tokens found")
        return s
    lat, s.token_scale, s.warp, n = fb
    s.lattice, s.hex_px = lat, lat.size
    br = read_board(img, roi, lattice=lat, canal=canal, scale=s.token_scale)
    s.lattice = br.lattice
    for h in br.board.hexes.values():
        x, y = br.lattice.hex_px(*h.coord)
        s.hexes.append({"q": h.coord[0], "r": h.coord[1], "terrain": h.terrain, "number": h.number,
                        "x": round(x), "y": round(y)})
    land = [h for h in s.hexes if h["terrain"] != "sea"]
    # a land hex whose centre is this close to an edge is cut off (or hidden by the bottom bar)
    d = s.hex_px
    edge = [h for h in land if h["x"] < d or h["x"] > W - d or h["y"] < TOP + d or h["y"] > H - 1.6 * d]
    if edge:
        s.problems.append(f"{len(edge)} land hexes at the screen edge: board cut off, zoom out")
    nums = [h["number"] for h in land if h["number"]]
    if len(nums) < 10:
        s.problems.append(f"only {len(nums)} numbers read")
    if nums.count(7) or any(n not in range(2, 13) for n in nums):
        s.problems.append(f"impossible numbers {sorted(nums)}")
    return s


TERRAIN_BGR = {"forest": (40, 120, 40), "hills": (40, 80, 200), "pasture": (90, 220, 140), "fields": (40, 200, 230),
               "mountains": (150, 150, 150), "gold": (0, 215, 255), "desert": (140, 200, 230), "canal": (200, 160, 60),
               "sea": (200, 120, 40), "unknown": (255, 0, 255)}
COLOUR_BGR = {"red": (40, 40, 220), "magenta": (200, 40, 220), "purple": (160, 40, 120), "blue": (200, 100, 30),
              "green": (40, 160, 40), "orange": (20, 140, 240), "brown": (40, 70, 120), "white": (240, 240, 240)}


def annotate(img, s: Scan) -> np.ndarray:
    """Draw what the scan read on top of the screenshot."""
    out = img.copy()
    lat = s.lattice
    if lat is not None:
        for h in s.hexes:
            c = TERRAIN_BGR.get(h["terrain"], (255, 255, 255))
            corners = [lat.to_px(*_unit_corner(h["q"], h["r"], k)) for k in range(6)]
            cv2.polylines(out, [np.int32(corners)], True, c, 2)
            label = f"{h['terrain'][:4]} {h['number'] or ''}".strip()
            x, y = h["x"], h["y"]
            _text(out, label, (x - 28, y + 22), c)
            _text(out, f"{h['q']},{h['r']}", (x - 14, y - 16), (255, 255, 255), 0.35)
    for p in s.players:
        c = COLOUR_BGR.get(p["colour"], (255, 255, 255))
        cv2.rectangle(out, (p["x"] - 8, p["y"] - 6), (p["x"] + 170, p["y"] + 30), c, 3)
        _text(out, f"seat {p['seat']}: {p['colour']}{'  <- turn' if p['current'] else ''}", (p["x"], p["y"] + 48), c, 0.5)
    y0 = out.shape[0] - 12
    _text(out, s.summary()[:180], (8, y0), (0, 255, 255) if s.ok else (0, 0, 255), 0.45)
    return out


def mark_buttons(out, ui: dict):
    for k, v in ui.items():
        if k.startswith("_") or not isinstance(v, list) or len(v) != 2 or not all(isinstance(t, (int, float)) for t in v):
            continue
        x, y = int(v[0]), int(v[1])
        cv2.circle(out, (x, y), 9, (0, 255, 255), 2)
        _text(out, k, (x + 10, y - 6), (0, 255, 255), 0.35)
    for k in ("trade_want_row", "trade_give_row"):
        if k in ui:
            for x in ui[k]["x"]:
                cv2.drawMarker(out, (int(x), int(ui[k]["y"])), (0, 200, 255), cv2.MARKER_CROSS, 10, 1)
    return out


def _unit_corner(q, r, k):
    from ..board import hex_center
    cx, cy = hex_center(q, r)
    a = math.radians(60 * k - 30)
    return cx + math.cos(a), cy + math.sin(a)


def _text(img, t, xy, c, sc=0.45):
    cv2.putText(img, t, xy, cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, t, xy, cv2.FONT_HERSHEY_SIMPLEX, sc, c, 1, cv2.LINE_AA)


def save(img, s: Scan, ui: dict | None = None, d: Path | None = None) -> Path:
    d = d or ROOT / "data" / "popups"
    d.mkdir(parents=True, exist_ok=True)
    out = annotate(img, s)
    if ui:
        mark_buttons(out, ui)
    p = d / "scan_annotated.png"
    cv2.imwrite(str(p), out)
    (d / "scan.json").write_text(json.dumps(s.to_json(), indent=1))
    return p
