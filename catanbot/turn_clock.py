"""Turn clock: how much of the game's turn timer is left (bot 0.4.3, "fast turn").

Owner 2026-10-10: "the timer is a color edge around the next button. bot can also see what the
timer is in the game, the timer does not change throughout a game and bot can time how much time
it has".

Seen in the people games of 2026-10-09 (Rhea + Bramble, stage data/popups + data/probe):
the dice / End-turn button gets a border in the current player's colour. It fills clockwise from
12 o'clock as the turn runs (the coloured part is the time used), linear in angle, full at the
timer's end (about 100 s in those games). Solo games show no border. The border seemed to stop
while a progress-card dialog was open, so the bot's own clock (from the roll) is the main source
and the border only a cross-check that can move the clock forward.
"""
from __future__ import annotations

import math

# stage frame (1372x888 virtual): border centre (1264, 745), half-size 29; the orange button face
# is inside half 24, the yellow glow outside half 32
RING_HALF = 29.0
RING_SAMPLES = 160
LIT_S, LIT_V = 100, 150


def _perimeter(cx: float, cy: float, h: float, n: int):
    """n points on a square border, from 12 o'clock clockwise, with the outward normal."""
    for i in range(n):
        s = i / n * 8 * h
        if s < h:
            yield cx + s, cy - h, 0, -1
        elif s < 3 * h:
            yield cx + h, cy - h + (s - h), 1, 0
        elif s < 5 * h:
            yield cx + h - (s - 3 * h), cy + h, 0, 1
        elif s < 7 * h:
            yield cx - h, cy + h - (s - 5 * h), -1, 0
        else:
            yield cx - h + (s - 7 * h), cy - h, 0, -1


def _angle_frac(run_frac: float, h: float = RING_HALF) -> float:
    """Share of the border (by length, from 12 o'clock) -> share of the full turn (by angle)."""
    if run_frac <= 0:
        return 0.0
    if run_frac >= 1:
        return 1.0
    s = run_frac * 8 * h
    if s < h:
        x, y = s, -h
    elif s < 3 * h:
        x, y = h, -h + (s - h)
    elif s < 5 * h:
        x, y = h - (s - 3 * h), h
    elif s < 7 * h:
        x, y = -h, h - (s - 5 * h)
    else:
        x, y = -h + (s - 7 * h), -h
    return ((math.degrees(math.atan2(x, -y)) + 360) % 360) / 360


def ring_read(img, centre, half: float = RING_HALF, n: int = RING_SAMPLES):
    """(elapsed share 0..1, median hue) of the timer border round the button, or None.

    None = no border (solo game, button covered) or a border that isn't one run from 12 o'clock.
    """
    import cv2
    if img is None:
        return None
    cx, cy = centre
    hgt, wid = img.shape[:2]
    if not (half + 3 <= cx < wid - half - 3 and half + 3 <= cy < hgt - half - 3):
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    bits, hues = [], []
    for x, y, nx, ny in _perimeter(cx, cy, half, n):
        hit = None
        for d in (-2, -1, 0, 1, 2):
            px, py = int(round(x + nx * d)), int(round(y + ny * d))
            h, s, v = hsv[py, px]
            if s > LIT_S and v > LIT_V:
                hit = int(h)
                break
        bits.append(hit)
    lit = [b for b in bits if b is not None]
    if len(lit) < 3:
        return None
    # one run from 12 o'clock, single-sample gaps allowed (anti-aliasing on the corners)
    run, i = 0, 0
    while i < n:
        if bits[i] is not None:
            run = i + 1
        elif not (i + 1 < n and bits[i + 1] is not None and i > 0 and bits[i - 1] is not None):
            break
        i += 1
    if run < 3:
        return None
    stray = sum(1 for b in bits[run:] if b is not None)
    if stray > 4:
        return None                              # lit bits elsewhere: not the timer border
    hues = sorted(b for b in bits[:run] if b is not None)
    med = hues[len(hues) // 2]
    off = sum(1 for h in hues if min(abs(h - med), 180 - abs(h - med)) > 12)
    if off > len(hues) // 4:
        return None                              # mixed colours: not one player's border
    return _angle_frac(run / n, half), med


def phase_cutoffs(timer_s: float, cfg: dict) -> dict:
    """Seconds into our turn at which each phase stops, scaled to the game's timer."""
    t = float(timer_s)
    return {
        "late": cfg.get("turn_late_frac", 0.45) * t,     # past it: trades only for the builds the hand pays for
        "offer": cfg.get("turn_offer_frac", 0.45) * t,   # no new player offer
        "chat": cfg.get("turn_chat_frac", 0.55) * t,     # no chat on our turn
        "end": cfg.get("turn_end_frac", 0.70) * t,       # builds only, then End turn
        "hard": max(0.0, t - cfg.get("turn_hard_margin_s", 6)),  # End turn now
    }
