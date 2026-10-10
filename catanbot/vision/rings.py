"""Selection rings on board corners, read per lattice vertex (live game 13, 2026-10-04).

The runner's ring_spots finds rings as connected blobs of white; a road crossing a ring splits it
into arcs and it is lost ("Select position for Basic knight" stalled 18 minutes with the only
legal ring cut by two roads). Here each land corner is scored by how much of a circle of the
ring's radius is white, minus white inside it (a white piece or text is filled, a ring is not).
Real rings score 0.85-0.95 on the saved frames, other corners <= 0.55."""
from __future__ import annotations

import cv2
import numpy as np

ANG = np.linspace(0, 2 * np.pi, 40, endpoint=False)
THRESHOLD = 0.7


def white_mask(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return (hsv[..., 1] < 55) & (hsv[..., 2] > 200)


def ring_score(wm: np.ndarray, x: float, y: float, r0: float, k: float = 1.0) -> float:
    """Best ring coverage around (x, y) (mask px), searching +-4 native px."""
    hgt, wid = wm.shape
    offs = np.arange(-4, 5, 2) if k > 1.2 else np.arange(-2, 3, 1)     # native vs frame px
    band = 3.0 if k > 1.2 else 2.0
    dx, dy = np.meshgrid(offs, offs)
    dx, dy = dx.ravel()[:, None], dy.ravel()[:, None]                    # (O,1)
    cos, sin = np.cos(ANG)[None, :], np.sin(ANG)[None, :]               # (1,A)
    cov = np.zeros((dx.shape[0], ANG.size), bool)
    for r in np.arange(r0 - band, r0 + band + 0.1, 1.0):
        xs = (x + dx + r * cos).astype(int)
        ys = (y + dy + r * sin).astype(int)
        ok = (xs >= 0) & (ys >= 0) & (xs < wid) & (ys < hgt)
        cov |= np.where(ok, wm[ys.clip(0, hgt - 1), xs.clip(0, wid - 1)], False)
    inner = np.zeros(dx.shape[0])
    for r in (r0 * 0.3, r0 * 0.5):
        xs = (x + dx + r * cos).astype(int)
        ys = (y + dy + r * sin).astype(int)
        ok = (xs >= 0) & (ys >= 0) & (xs < wid) & (ys < hgt)
        inner += np.where(ok, wm[ys.clip(0, hgt - 1), xs.clip(0, wid - 1)], False).sum(1)
    return float((cov.mean(1) - inner / (2 * ANG.size)).max())


def vertex_rings(img, lattice, board, native=None, lay=None, threshold=THRESHOLD, vertices=None):
    """[(score, vertex, (x, y) frame px)] of land corners with a selection ring, best first.
    Reads the native capture when given (thin rings fade in the scaled frame)."""
    if native is not None and lay is not None:
        src, k = native, lay.s
    else:
        src, k = img, 1.0
    wm = white_mask(src)
    r0 = 0.39 * lattice.size * k
    out = []
    for v in (vertices if vertices is not None else board.vertices):
        if not board.is_land_vertex(v):
            continue
        x, y = lattice.vertex_px(v)
        nx, ny = lay.to_native(x, y) if k != 1.0 else (x, y)
        s = ring_score(wm, nx, ny, r0, k)
        if s >= threshold:
            out.append((round(s, 3), v, (int(x), int(y))))
    out.sort(key=lambda t: -t[0])
    return out


EDGE_THRESHOLD = 0.72


def edge_rings(img, lattice, board, native=None, lay=None, threshold=EDGE_THRESHOLD, edges=None, span=4):
    """[(score, edge, (x, y) frame px)] of board edges with a selection ring, best first.
    Stage 2026-10-08 22:17: the Diplomat rings the removable roads with small rings (about 0.33
    of a hex side) at the road's middle; the road through them splits them into arcs, so
    ring_spots read none and the bot clicked the brain's edge, which had no ring, and sat. The
    lattice midpoint can be ~5 px off the drawn ring: search +-4 frame px. Real rings score
    0.75-0.88 on the saved frames, other edges <= 0.65. span: the search radius in frame px (the
    Diplomat's last look widens it to 8, stage 2026-10-09 16:20)."""
    if native is not None and lay is not None:
        src, k = native, lay.s
    else:
        src, k = img, 1.0
    wm = white_mask(src)
    r0 = 0.33 * lattice.size * k
    step = max(2, int(round(2 * k)))
    span = int(round(span * k))
    out = []
    for e in (edges if edges is not None else board.edges):
        x, y = lattice.edge_px(e)
        nx, ny = lay.to_native(x, y) if k != 1.0 else (x, y)
        best, bxy = -1.0, (nx, ny)
        for dx in range(-span, span + 1, step):
            for dy in range(-span, span + 1, step):
                s = ring_score(wm, nx + dx, ny + dy, r0, k)
                if s > best:
                    best, bxy = s, (nx + dx, ny + dy)
        if best >= threshold:
            fx, fy = lay.to_frame(*bxy) if k != 1.0 else bxy
            out.append((round(best, 3), e, (int(fx), int(fy))))
    out.sort(key=lambda t: -t[0])
    # one ring sits between two edge midpoints at most once: drop a weaker hit on the same ring
    kept = []
    for t in out:
        if all(abs(t[2][0] - u[2][0]) + abs(t[2][1] - u[2][1]) > 6 for u in kept):
            kept.append(t)
    return kept
