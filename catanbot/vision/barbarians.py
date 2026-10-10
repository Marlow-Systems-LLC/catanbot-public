"""Barbarian track (owner 2026-10-05, game 15): "barbarians have a map on the left where they show
when they advance". A small sea island left of the board: the ship starts bottom-left and moves
along 7 circles to the land; after an attack it starts again. Labels beside it give the barbarian
strength (cities on the board) and the active knights of the whole table.
Coordinates are native 1920x1080 frame pixels (game 15 frame data/snaps/042921.png).
Solo probe 2026-10-05: the ship drawn at the start is a fixed picture, not the barbarians -- the
moving piece is a dark ship that sits ON one of the 7 light circles (its strength/knights badges
ride along). Matching the picture always said "start" while an attack came. Now the picture is
only the anchor, and the step is the circle the dark piece covers (free circles ~183 grey, the
covered one ~40-80; 50 probe frames 06:00-06:38 follow 3,5,6,attack,1,3,5,2,4,6,7,1).
Stage 2026-10-09: on the live (1372 px wide) frame step 6 was read as 0 -- the dark 3-D piece sits
up and right of circle 6, so the small patch on the circle's centre stayed light (~128 > 100) and
no circle counted as covered. The step is now the circle with the largest share of dark pixels in
a circle-sized patch (free ~0, the axe on 7 ~0.08, covered 0.37-0.52 at both scales)."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

CALIB = Path(__file__).resolve().parents[2] / "data" / "calib" / "barb_ship.png"
NATIVE_W = 1920
START = (511, 524)
SPOTS = [START, (523, 561), (557, 552), (556, 518), (555, 485), (551, 450), (587, 443), (594, 485)]
REGION = (460, 400, 660, 600)          # x0, y0, x1, y1 searched for the ship
LABELS_AT = (575, 470)
DARK = 70                               # a pixel darker than this belongs to the barbarian piece
PATCH_R = 12                            # native px: about the circle's radius
COVERED = 0.30                          # share of dark pixels that marks the covered circle
_tmpl = None


def _template():
    global _tmpl
    if _tmpl is None and CALIB.exists():
        import cv2
        _tmpl = cv2.imread(str(CALIB))
    return _tmpl


def read_track(img: np.ndarray, min_score: float = 0.55) -> tuple[int, tuple[int, int], float] | None:
    """(position 0..7, barbarian ship in img px, anchor match score) or None without the anchor."""
    import cv2
    t = _template()
    if t is None or img is None or not getattr(img, "size", 0):
        return None
    s = img.shape[1] / NATIVE_W
    if abs(s - 1) > 0.01:
        t = cv2.resize(t, None, fx=s, fy=s)
    x0, y0, x1, y1 = (int(round(c * s)) for c in REGION)
    roi = img[y0:y1, x0:x1]
    if roi.shape[0] < t.shape[0] or roi.shape[1] < t.shape[1]:
        return None
    r = cv2.matchTemplate(roi, t, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(r)
    if score < min_score:
        return None
    ax, ay = x0 + loc[0] + t.shape[1] / 2, y0 + loc[1] + t.shape[0] / 2
    r = max(2, int(round(PATCH_R * s)))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    dark = []
    for x, y in SPOTS[1:]:
        px, py = int(round(ax + (x - START[0]) * s)), int(round(ay + (y - START[1]) * s))
        patch = gray[max(0, py - r):py + r + 1, max(0, px - r):px + r + 1]
        dark.append(float((patch < DARK).mean()) if patch.size else 0.0)
    i = int(np.argmax(dark))
    if dark[i] < COVERED:
        return 0, (int(ax), int(ay)), float(score)          # no circle covered: at the start
    x, y = SPOTS[i + 1]
    return i + 1, (int(round(ax + (x - START[0]) * s)), int(round(ay + (y - START[1]) * s))), float(score)


def read_counts(texts: list[dict]) -> dict:
    """{"strength": n, "active": n} from OCR boxes near the labels (a digit left of each label)."""
    out = {}
    for key, pat in (("strength", r"bar\s*barian"), ("active", r"active\s*kn")):
        lab = next((o for o in texts if re.search(pat, o["text"], re.I)), None)
        if lab is None:
            continue
        m = re.match(r"\s*(\d+)\s", lab["text"])
        if m:
            out[key] = int(m.group(1))
            continue
        digs = [o for o in texts if re.fullmatch(r"\d{1,2}", o["text"].strip())
                and abs(o["y"] - lab["y"]) < 8 and o["x"] < lab["x"]]
        if digs:
            out[key] = int(max(digs, key=lambda o: o["x"])["text"])
    return out


class Track:
    """Stage 2026-10-09: the readings jumped 5 -> 0 -> 2 mid-voyage and a false 0 meant
    "barbarians far". The ship only moves forward; it goes back to the start only after an attack.
    update() turns raw readings into a position that never runs backward on a single read:
      - forward readings are taken (a jump of 4+ steps needs the same reading twice);
      - after step 7 (or attacked(): an attack popup / message / our city lost) any lower reading
        starts the new voyage;
      - otherwise a backward reading is ignored, except that two backward readings 1 <= a < b
        (the ship moving on a new voyage) prove an attack we never saw ("unseen_attack").
    Events: "attack" (step 7 read), "unseen_attack", "ignored", "restart", None."""
    LEN = 7

    def __init__(self) -> None:
        self.pos = 0
        self.reads = 0            # accepted readings (0: the track was never read)
        self.after_attack = False
        self.cand: int | None = None      # a backward reading waiting for the ship to move on
        self.jump: int | None = None      # a big forward jump waiting for a 2nd reading

    def attacked(self) -> None:
        """A confirmed attack (popup, message, a city lost): the ship is back at the start."""
        self.pos, self.after_attack, self.cand, self.jump = 0, True, None, None

    def near_attack(self) -> bool:
        """An attack is plausible now: the ship was last seen 2 steps or less from the coast
        (or the track was never read, so nothing contradicts it)."""
        return self.reads == 0 or self.pos >= self.LEN - 2

    def update(self, raw: int | None) -> tuple[int, str | None]:
        if raw is None:
            return self.pos, None
        if raw == self.pos:
            self.cand = self.jump = None
            self.reads += 1
            return self.pos, None
        if raw > self.pos:
            if raw - self.pos >= 4 and self.reads and self.jump != raw:
                self.jump = raw
                return self.pos, "ignored"
            ev = "attack" if raw == self.LEN and self.reads else None
            self.pos, self.cand, self.jump = raw, None, None
            self.after_attack = False
            self.reads += 1
            return self.pos, ev
        # backward reading
        if self.pos == self.LEN or self.after_attack:
            self.pos, self.cand, self.jump, self.after_attack = raw, None, None, False
            self.reads += 1
            return self.pos, "restart"
        if raw >= 1 and self.cand is not None and self.cand < raw:
            self.pos, self.cand, self.jump = raw, None, None
            self.reads += 1
            return self.pos, "unseen_attack"
        if raw >= 1 and (self.cand is None or raw < self.cand):
            self.cand = raw
        return self.pos, "ignored"
