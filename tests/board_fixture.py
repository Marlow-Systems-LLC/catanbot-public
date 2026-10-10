"""Shared loader for the 2026-10-04 server frames (tests/fixtures): crops pasted back into a
black 1372x888 frame, and the lattice + board rebuilt from that game's scan."""
import json
import re

import cv2
import numpy as np

from catanbot.board import Board, Hex, hex_center
from catanbot.vision.board_reader import BoardRead, Lattice, _warp

FX = "tests/fixtures/"
COLOURS = ["magenta", "purple", "green", "blue"]      # marlow.systems, Lin, Marianne, Jean


def frame(*names):
    """Crops (x0, y0 in the name) pasted back at their place in one black 1372x888 frame."""
    img = np.zeros((888, 1372, 3), np.uint8)
    for name in names:
        x0, y0 = map(int, re.search(r"_(\d+)_(\d+)\.png$", name).groups())
        crop = cv2.imread(FX + name)
        img[y0:y0 + crop.shape[0], x0:x0 + crop.shape[1]] = crop
    return img


def canvas(name):
    """A crop named <stem>_<W>x<H>_<x0>_<y0>.png pasted back at its place in a black WxH frame
    (the full game frames are 2 MB each and live only on the game hosts)."""
    w, h, x0, y0 = map(int, re.search(r"_(\d+)x(\d+)_(\d+)_(\d+)\.png$", name).groups())
    img = np.zeros((h, w, 3), np.uint8)
    crop = cv2.imread(FX + name)
    img[y0:y0 + crop.shape[0], x0:x0 + crop.shape[1]] = crop
    return img


def board_read() -> BoardRead:
    with open(FX + "scan_2026-10-04.json") as f:
        d = json.load(f)
    W = np.eye(3)
    W[2, 0], W[2, 1] = d["warp"][0] / 1372, d["warp"][1] / 888
    U = [list(hex_center(h["q"], h["r"])) + [1] for h in d["hexes"]]
    P = [_warp(W, h["x"], h["y"]) for h in d["hexes"]]
    A = np.linalg.lstsq(np.array(U), np.array(P), rcond=None)[0].T
    b = Board()
    for h in d["hexes"]:
        b.add(Hex(h["q"], h["r"], h["terrain"], h["number"]))
    return BoardRead(b, Lattice(A, W), {}, [])


def native_of(img):
    """Stand-in for the server's 1920x1080 capture behind a frame (Layout(1920, 1080))."""
    from catanbot.vision.layout import Layout
    return cv2.resize(img[54:54 + 736, 0:1308], (1920, 1080), interpolation=cv2.INTER_CUBIC), Layout(1920, 1080)
