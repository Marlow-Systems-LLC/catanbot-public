"""Great Canal: which chip hexes are dug, read from the board picture.

Two active knights on a canal hex dig it; the game then paints a dark brown
trench across that hex, edge to edge along the canal path (live game 3,
2026-10-02). Finished canal turns into grey/blue water, also darker than the
grass. We compare the path band of each canal hex with the same pixels in the
first board frame of the game (nothing dug yet) and call a hex dug when it got
clearly darker.

UNCONFIRMED: the -25 threshold comes from 7 frames of one game (dug -33..-80,
undug 0..-4, a few -7..-15 in between). Pieces standing on the band can darken
it too; the median over 12 samples keeps one piece from flipping a hex.
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

TEMPLATE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "boards", "canal_template.json")
BASIN_A, LAKE_B = (-2, -1), (1, 1)          # template ends of the canal path
DUG_DV = -25.0                              # V drop (0-255) that counts as a trench


def _template_path() -> list:
    with open(TEMPLATE) as f:
        return [tuple(c) for c in json.load(f)["canal_hexes"]]


def canal_path(board) -> list:
    """Canal hexes in path order (basin to lake), in this board's coordinates."""
    path = _template_path()
    if not board.canal_hexes:
        return []
    have, tp = sorted(board.canal_hexes), sorted(path)
    dq, dr = have[0][0] - tp[0][0], have[0][1] - tp[0][1]
    seq = [BASIN_A] + path + [LAKE_B]
    return [((c[0] + dq, c[1] + dr), (p[0] + dq, p[1] + dr), (n[0] + dq, n[1] + dr))
            for p, c, n in zip(seq, seq[1:], seq[2:])]


def _v(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2].astype(float)


def canal_darkening(img, ref_img, lattice, board) -> dict:
    """Median V change of each canal hex's path band vs the reference frame."""
    v, rv = _v(img), _v(ref_img)
    hgt, wid = v.shape
    out = {}
    for c, prev, nxt in canal_path(board):
        cx, cy = lattice.hex_px(*c)
        vals = []
        for e in (prev, nxt):
            ex, ey = lattice.hex_px(*e)
            mx, my = (cx + ex) / 2, (cy + ey) / 2          # shared edge midpoint
            for t in np.linspace(0.15, 0.6, 6):             # toward the centre, short of the number chip
                x, y = int(mx + (cx - mx) * t), int(my + (cy - my) * t)
                if 2 <= y < hgt - 2 and 2 <= x < wid - 2:
                    vals.append(v[y - 2:y + 3, x - 2:x + 3].mean() - rv[y - 2:y + 3, x - 2:x + 3].mean())
        out[c] = float(np.median(vals)) if vals else 0.0
    return out


def read_canal_dug(img, ref_img, lattice, board, thresh: float = DUG_DV) -> set:
    """Canal hexes that show a trench (or water) now."""
    return {c for c, dv in canal_darkening(img, ref_img, lattice, board).items() if dv <= thresh}
