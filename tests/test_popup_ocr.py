"""Live 2026-10-04 (server, 1920x1080 shrunk to the 1308x790 frame): every build popup failed
with "no option for <kind>" -- whole-frame OCR read the small labels as "Build foad",
"Promotekmigit", "Somde kaght?". Replays saved popup crops through Runner.tap_build with the
real Apple Vision OCR and checks the right option icon is clicked."""
import re
import types

import cv2
import numpy as np
import pytest

pytest.importorskip("Vision")

from catanbot import mac
from catanbot import runner as R
from catanbot.brain import Action
from catanbot.state import PlayerState
from catanbot.vision.screens import find_popup_option, popup_label_kind

FX = "tests/fixtures/"


def _frame(name):
    """Fixture crop pasted back at its place (x0, y0 in the name) in a black 1372x888 frame."""
    x0, y0 = map(int, re.search(r"_(\d+)_(\d+)\.png$", name).groups())
    crop = cv2.imread(FX + name)
    img = np.zeros((888, 1372, 3), np.uint8)
    img[y0:y0 + crop.shape[0], x0:x0 + crop.shape[1]] = crop
    return img


def _tap(img, kind, xy, monkeypatch):
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    monkeypatch.setattr(R.cv2, "imwrite", lambda *a, **k: True)
    clicks, notes = [], []
    r = R.Runner.__new__(R.Runner)
    r.br = types.SimpleNamespace(lattice=types.SimpleNamespace(vertex_px=lambda t: xy, edge_px=lambda t: xy))
    # tap_build first looks for a leftover popup (live 2026-10-04): the board is clear before the tap
    frames = [np.zeros_like(img), img]
    r.io = types.SimpleNamespace(snapshot=lambda: frames.pop(0) if len(frames) > 1 else frames[0], ocr=mac.ocr)
    r.click = lambda p, why, *a: clicks.append((why, p))
    r.note = notes.append
    me = PlayerState("marlow.systems", is_me=True); me.hand["grain"] = 1
    r.gs = types.SimpleNamespace(me=lambda: me)
    r.tap_build(Action(kind, (2, -8)))
    return clicks, notes


@pytest.mark.parametrize("name,kind,xy,icon", [
    # popup_045013_road: "Build road" icon ~(720,245), "Build ship" icon ~(847,245)
    ("popup_045013_road_585_176.png", "road", (785, 246), (720, 245)),
    # popup_045020_knight: our knight -- "Activate knight" (red) ~(678,238), "Promote knight" ~(803,238)
    ("popup_045020_knight_542_167.png", "promote", (742, 237), (803, 238)),
])
def test_popup_option_found_on_live_frame(name, kind, xy, icon, monkeypatch):
    clicks, notes = _tap(_frame(name), kind, xy, monkeypatch)
    assert not any("no option" in n for n in notes), notes
    why, (x, y) = clicks[1]
    assert why == f"popup {kind}"
    assert abs(x - icon[0]) < 30 and abs(y - icon[1]) < 25, (x, y)


def test_road_frame_was_missed_by_the_old_regex():
    """Root cause: the whole-frame read has no 'road' at all."""
    img = _frame("popup_045013_road_585_176.png")
    full = [o["text"] for o in mac.ocr(img)]
    assert not any(re.search(R.Runner.POPUP_WORDS["road"], t, re.I) for t in full), full


@pytest.mark.parametrize("text,kind", [
    ("Build foad", "road"), ("Buildboad", "road"), ("Build ahip", "ship"), ("Bald ship", "ship"),
    ("Buld oity", "city"), ("Buld wall", "wall"), ("Baid Knigh", "knight"),
    ("Promotekmigit", "promote"), ("Premofe knigh?", "promote"), ("Promate, Knight", "promote"),
    ("Activate Knight", "activate"), ("Aetivzte kigfy", None)])
def test_ocr_misreads_map_to_their_label(text, kind):
    assert popup_label_kind(text)[0] == kind


def test_tooltip_and_neighbour_labels_are_not_options():
    box = lambda t: {"text": t, "x": 0, "y": 0, "w": 10, "h": 10}
    # live 04:50:15: "road" matched the brick tooltip and the bot clicked it
    assert find_popup_option([box("Needed for: Road, Settlement, City wall.")], "road", r"road") is None
    assert find_popup_option([box("Build ship")], "road", r"road|ship") is None
    assert find_popup_option([box("Promote knight")], "knight", r"knight") is None
