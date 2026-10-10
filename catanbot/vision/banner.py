"""Player banners (cards in hand per player) and the last dice roll.

read_banner_counts: every player banner across the top has the name at y~98; under its
avatar sit two small badges -- a white oval with the resource-card count, then a card in
the player's colour with the progress-card count (white italic digit). Full-frame OCR
misses these ~10 px digits and Vision will not read a lone enlarged digit either, so:
  1. banners are found by their white resource oval (3 or 4 players, no fixed slots);
  2. each name and each badge digit is binarised (ink black on white);
  3. all names (stacked) and all digits (one spaced line) go into ONE canvas, OCR'd once;
  4. reads are mapped back by position, then cross-checked against grey templates of
     labelled digits (data/calib/banner_digits.npz, built with save_templates): agreement
     or a very strong template match wins, any disagreement -> None.

read_dice: the white and red die at the right edge; pips are counted as dark blobs.

Both return None for any number they cannot read with confidence -- never a guess.
"""
from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

# frame geometry (1372x888 client frames)
NAME_Y = (85, 112)            # y-range of the banner names
NAME_DX = (55, 185)           # name x-range relative to the resource oval's left x
BADGE_Y = (138, 180)          # y-range of the two badges under each avatar
DICE_ROI = (1205, 630, 1320, 695)   # x0, y0, x1, y1

_FIX = str.maketrans({"O": "0", "o": "0", "D": "0", "Q": "0", "l": "1", "I": "1", "i": "1",
                      "|": "1", "!": "1", "]": "1", "Z": "2", "z": "2", "S": "5", "s": "5",
                      "B": "8", "g": "9", "q": "9", "G": "6", "b": "6"})

TEMPLATES = Path(__file__).resolve().parents[2] / "data" / "calib" / "banner_digits.npz"
_SURE = 0.90        # template score that wins without (or against) OCR ...
_MARGIN = 0.10      # ... with this lead over the best other digit
_UNSEEN = 0.78      # below this the glyph fits no labelled digit (OCR may name a new one)

_PAD = 40           # canvas margin / gap between name rows
_DIGIT_SCALE = 3    # digit glyphs are ~10 px tall; 3x reads best (tried 2/3/4/5/6)
_DIGIT_GAP = 60     # >> the gap inside "12", so badges come back as separate tokens


def _ocr_fn(ocr):
    if ocr is not None:
        return ocr
    from catanbot.mac import ocr as mac_ocr
    return mac_ocr


def _white(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return (hsv[..., 2] > 185) & (hsv[..., 1] < 70)


def _find_ovals(img) -> list[tuple[int, int, int, int]]:
    """Frame bboxes (x, y, w, h) of the white resource ovals under the avatars, left to right."""
    y0, y1 = BADGE_Y
    m = _white(img[y0:y1]).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if 11 <= w <= 20 and 18 <= h <= 28 and 50 <= a <= 320 and h > w \
                and abs(y + h / 2 + y0 - 158) <= 8:
            out.append((int(x), int(y0 + y), int(w), int(h)))
    return sorted(out)


class Glyph:
    """A badge's digit ink: tight bool mask + the ink-bright grey crop around it (margin M)."""
    M = 3

    def __init__(self, mask: np.ndarray, gray: np.ndarray):
        self.mask, self.gray = mask, gray

    def digits(self) -> list[np.ndarray]:
        """Grey patch (1 px margin) of each single digit, left to right."""
        out, start = [], None
        cols = list(self.mask.any(axis=0)) + [False]
        for x, c in enumerate(cols):
            if c and start is None:
                start = x
            elif not c and start is not None:
                ys = np.nonzero(self.mask[:, start:x].any(axis=1))[0]
                m = self.M - 1
                out.append(self.gray[ys.min() + m:ys.max() + self.M + 2,
                                     start + m:x + self.M + 1])
                start = None
        return out


def _crop_glyph(mask: np.ndarray, ink: np.ndarray) -> Glyph | None:
    """Glyph from a full-crop mask and ink-bright grey image; None if no plausible digit."""
    if mask is None or mask.sum() < 8:
        return None
    ys, xs = np.nonzero(mask)
    g = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    if not 6 <= g.shape[0] <= 16 or g.shape[1] > 16:
        return None
    M = Glyph.M
    pad = cv2.copyMakeBorder(ink, M, M, M, M, cv2.BORDER_REPLICATE)
    gray = pad[ys.min():ys.max() + 1 + 2 * M, xs.min():xs.max() + 1 + 2 * M].astype(np.float32)
    return Glyph(g, gray)


def _resource_glyph(img, oval) -> np.ndarray | None:
    x, y, w, h = oval
    # the ring can be broken by the digits or cut by overlays: use a nominal ellipse
    # (~15x23) anchored on the bbox's centre x and its bottom edge
    cx, cy = x + w / 2.0, (y + h / 2.0) if h >= 22 else (y + h - 11.5)
    x0, y0 = int(cx - 10), int(cy - 14)
    c = img[y0:y0 + 28, x0:x0 + 20]
    if c.shape[:2] != (28, 20):
        return None
    inside = np.zeros(c.shape[:2], np.uint8)
    cv2.ellipse(inside, (int(round(cx - x0)), int(round(cy - y0))),
                (max(3, int(round(w / 2 - 1))), 10), 0, 0, 360, 1, -1)
    # digit ink: dark grey, or red when over the hand limit = anything not white
    ink = (inside > 0) & ~_white(c)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
    keep = np.zeros_like(ink)
    for i in range(1, n):
        if st[i, 4] >= 5 and st[i, 3] >= 4:
            keep |= lab == i
    # green channel: white paper is bright, grey and red digits are both dark in it
    return _crop_glyph(keep, 255 - c[..., 1])


def _progress_glyph(img, oval) -> np.ndarray | None:
    x, y, w, h = oval
    top = (y + h - 23) + 3          # below the card's white top stripe
    c = img[top:y + h, x + w + 1:x + w + 17]
    if c.size == 0:
        return None
    # the progress badge is a card in the player's colour: no card -> no count
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    coloured = (hsv[..., 1] > 80) | (hsv[..., 2] < 120)
    if coloured.mean() < 0.35:
        return None
    ink = _white(c)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
    keep = np.zeros_like(ink)
    for i in range(1, n):
        bx, by, bw, bh, a = st[i]
        if a >= 4 and bh >= 5 and bx > 0 and bx + bw < c.shape[1]:   # not the card edge
            keep |= lab == i
    return _crop_glyph(keep, c.min(axis=2))


def _name_image(img, oval) -> np.ndarray | None:
    x = oval[0]
    c = img[NAME_Y[0]:NAME_Y[1], max(0, x + NAME_DX[0]):x + NAME_DX[1]]
    if c.size == 0:
        return None
    g = np.where(_white(c), 0, 255).astype(np.uint8)          # white name -> black ink
    return cv2.resize(g, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)


def _ocr_digits(glyphs: list[Glyph], ocr, scale=_DIGIT_SCALE, gap=_DIGIT_GAP):
    """OCR glyphs laid out on one spaced line; returns one int|None per glyph."""
    rs = [cv2.resize(np.where(g.mask, 0, 255).astype(np.uint8), None, fx=scale, fy=scale,
                     interpolation=cv2.INTER_CUBIC) for g in glyphs]
    H = max(r.shape[0] for r in rs)
    canvas = np.full((H + 2 * gap, sum(r.shape[1] + gap for r in rs) + gap), 255, np.uint8)
    spans, x = [], gap
    for r in rs:
        canvas[gap + H - r.shape[0]:gap + H, x:x + r.shape[1]] = r
        spans.append((x, x + r.shape[1]))
        x += r.shape[1] + gap
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
    vals: list[int | None] = [None] * len(glyphs)
    for o in ocr(cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)) or []:
        toks = str(o.get("text", "")).split()
        lo, hi = o["x"] - gap / 2, o["x"] + o["w"] + gap / 2
        idx = [i for i, (a, b) in enumerate(spans) if lo <= (a + b) / 2 <= hi]
        if len(toks) != len(idx):
            continue                       # ambiguous mapping: leave these unread
        for i, t in zip(idx, toks):
            t = t.translate(_FIX)
            if re.fullmatch(r"\d{1,2}", t) and int(t) <= 30:
                vals[i] = int(t)
    return vals


def _ocr_names(names: list[np.ndarray], ocr) -> list[str | None]:
    W = max(n.shape[1] for n in names) + 2 * _PAD
    rows, y = [], _PAD
    for n in names:
        rows.append((y, y + n.shape[0]))
        y += n.shape[0] + _PAD
    canvas = np.full((y, W), 255, np.uint8)
    for n, (a, _) in zip(names, rows):
        canvas[a:a + n.shape[0], _PAD:_PAD + n.shape[1]] = n
    found: list[list[tuple[float, str]]] = [[] for _ in names]
    for o in ocr(cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)) or []:
        cy = o["y"] + o["h"] / 2
        for i, (a, b) in enumerate(rows):
            if a - 5 <= cy <= b + 5:
                found[i].append((o["x"], str(o.get("text", "")).strip()))
    out = []
    for f in found:
        name = "".join(t for _, t in sorted(f)).strip()
        out.append(name if sum(ch.isalpha() for ch in name) >= 2 else None)
    return out


_TPL_CACHE: dict | None = None


def _templates() -> dict[str, list[tuple[int, np.ndarray]]]:
    """{"res"|"prog": [(digit, grey patch)]} from data/calib/banner_digits.npz."""
    global _TPL_CACHE
    if _TPL_CACHE is None:
        _TPL_CACHE = {"res": [], "prog": []}
        try:
            z = np.load(TEMPLATES)
            for k in z.files:                       # "res_5_0" -> kind, digit, n
                kind, dig, _ = k.split("_")
                _TPL_CACHE[kind].append((int(dig), z[k].astype(np.float32)))
        except Exception:                           # noqa: BLE001
            pass
    return _TPL_CACHE


def _ncc(patch: np.ndarray, tpl: np.ndarray) -> float:
    """Best normalised correlation of tpl over patch (+-2 px slack); -1 if it does not fit."""
    q = cv2.copyMakeBorder(patch, 2, 2, 2, 2, cv2.BORDER_REPLICATE)
    if tpl.shape[0] > q.shape[0] or tpl.shape[1] > q.shape[1]:
        return -1.0
    if tpl.std() < 1 or q.std() < 1:
        return -1.0
    return float(cv2.matchTemplate(q, tpl, cv2.TM_CCOEFF_NORMED).max())


def _match(g: Glyph, kind: str) -> tuple[int | None, float, float]:
    """Template read: (value, worst best-score, worst margin to the best other digit)."""
    tpl = _templates().get(kind) or []
    digits = g.digits()
    if not tpl or not 1 <= len(digits) <= 2:
        return None, -1.0, 0.0
    val, score, margin = 0, 1.0, 1.0
    for d in digits:
        best: dict[int, float] = {}
        for v, t in tpl:
            best[v] = max(best.get(v, -1.0), _ncc(d, t))
        ranked = sorted(best.items(), key=lambda kv: -kv[1])
        dig, sc = ranked[0]
        other = ranked[1][1] if len(ranked) > 1 else -1.0
        val, score, margin = val * 10 + dig, min(score, sc), min(margin, sc - other)
    return val, score, margin


def _decide(ocr_val: int | None, g: Glyph, kind: str) -> int | None:
    """Combine the OCR read with the labelled-sample read; disagreement -> None."""
    t_val, score, margin = _match(g, kind)
    if t_val is None:                               # no templates on disk: OCR alone
        return ocr_val
    known = {v for v, _ in _templates().get(kind, [])}
    sure = score >= _SURE and margin >= _MARGIN
    if ocr_val == t_val or sure:
        return t_val
    if ocr_val is None:
        return None
    # OCR says a digit we have never labelled and the glyph fits no sample well
    if score < _UNSEEN and not set(map(int, str(ocr_val))) <= known:
        return ocr_val
    return None


def save_templates(labelled: list[tuple[np.ndarray, dict]], path: Path = TEMPLATES) -> int:
    """Store digit samples from frames with known counts: [(img, {name: (res, prog)})],
    names listed left to right. Only glyphs whose digit count matches the label are kept.
    Returns the number of samples written."""
    out: dict[str, np.ndarray] = {}
    for img, truth in labelled:
        ovals = _find_ovals(img)
        names = list(truth)
        if len(ovals) != len(names):
            continue
        for ov, name in zip(ovals, names):
            for kind, g, v in (("res", _resource_glyph(img, ov), truth[name][0]),
                               ("prog", _progress_glyph(img, ov), truth[name][1])):
                if g is None or v is None:
                    continue
                ds = g.digits()
                if len(ds) != len(str(v)):
                    continue
                for d, ch in zip(ds, str(v)):
                    n = sum(1 for k in out if k.startswith(f"{kind}_{ch}_"))
                    out[f"{kind}_{ch}_{n}"] = d.astype(np.uint8)
    np.savez_compressed(path, **out)
    global _TPL_CACHE
    _TPL_CACHE = None
    return len(out)


def read_banner_counts(img, ocr=None) -> dict[str, tuple[int | None, int | None]]:
    """{player name: (resource cards in hand, progress cards in hand)}; None = unreadable.
    A banner whose name cannot be read is left out."""
    out: dict[str, tuple[int | None, int | None]] = {}
    if img is None or getattr(img, "ndim", 0) != 3 or img.shape[0] < BADGE_Y[1]:
        return out
    ovals = _find_ovals(img)
    if not ovals:
        return out
    ocr = _ocr_fn(ocr)
    names = [_name_image(img, ov) for ov in ovals]
    keep = [i for i, n in enumerate(names) if n is not None]
    if not keep:
        return out
    read = dict(zip(keep, _ocr_names([names[i] for i in keep], ocr)))
    glyphs = {}
    for i in keep:
        glyphs[(i, 0)] = _resource_glyph(img, ovals[i])
        glyphs[(i, 1)] = _progress_glyph(img, ovals[i])
    todo = [k for k, g in glyphs.items() if g is not None]
    vals: dict = {}
    if todo:
        vals = dict(zip(todo, _ocr_digits([glyphs[k] for k in todo], ocr)))
        miss = [k for k in todo if vals[k] is None]
        if miss:                           # second look at another scale, unread ones only
            vals.update(zip(miss, _ocr_digits([glyphs[k] for k in miss], ocr, 4, 90)))
    for i in keep:
        if read.get(i):
            out[read[i]] = tuple(
                _decide(vals.get((i, j)), glyphs[(i, j)], kind) if glyphs[(i, j)] is not None
                else None for j, kind in ((0, "res"), (1, "prog")))
    return out


def _count_pips(face: np.ndarray, dark_thr: int) -> int | None:
    """Pips = small round dark blobs inside a die face (bgr crop, border trimmed)."""
    g = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    m = (g < dark_thr).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(m, 8)
    pips = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if 6 <= a <= 60 and 2 <= w <= 10 and 2 <= h <= 10 and 0.5 <= w / h <= 2:
            pips.append(a)
    if not 1 <= len(pips) <= 6:
        return None
    med = float(np.median(pips))
    if any(p < med * 0.4 or p > med * 2.5 for p in pips):
        return None
    return len(pips)


def _die_faces(roi: np.ndarray):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    white = ((hsv[..., 2] > 200) & (hsv[..., 1] < 40)).astype(np.uint8)
    red = (((hsv[..., 0] < 8) | (hsv[..., 0] > 172)) & (hsv[..., 1] > 150)
           & (hsv[..., 2] > 150)).astype(np.uint8)
    found = {}
    for key, m in (("white", white), ("red", red)):
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, _, st, _ = cv2.connectedComponentsWithStats(m, 8)
        best = None
        for i in range(1, n):
            x, y, w, h, a = st[i]
            if 22 <= w <= 48 and 22 <= h <= 48 and 0.75 <= w / h <= 1.33 and a > 0.6 * w * h:
                if best is None or a > best[4]:
                    best = (x, y, w, h, a)
        if best:
            found[key] = best
    return found


def read_dice(img) -> tuple[int, int] | None:
    """(white die, red die) pips of the roll shown at the right edge; None when not visible."""
    if img is None or getattr(img, "ndim", 0) != 3:
        return None
    x0, y0, x1, y1 = DICE_ROI
    roi = img[y0:y1, x0:x1]
    if roi.shape[0] < 30 or roi.shape[1] < 60:
        return None
    faces = _die_faces(roi)
    if "white" not in faces or "red" not in faces:
        return None
    wx, rx = faces["white"][0], faces["red"][0]
    if wx >= rx:                       # white is drawn left of red
        return None
    vals = []
    for key, thr in (("white", 110), ("red", 70)):
        x, y, w, h, _ = faces[key]
        t = max(2, w // 10)
        v = _count_pips(roi[y + t:y + h - t, x + t:x + w - t], thr)
        if v is None:
            return None
        vals.append(v)
    return vals[0], vals[1]
