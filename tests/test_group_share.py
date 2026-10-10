"""Group share of end-of-game shots (owner 2026-10-04): multipart body, size cap, caption, off without key."""
from pathlib import Path

import numpy as np
import cv2

from catanbot import group_share as gs

RESULT = {"winner": "Lin", "ranking": ["Lin", "marlow.systems"]}


def _shot(path):
    """A full-size, mostly non-black frame like a real end screen."""
    cv2.imwrite(str(path), np.full((790, 1308, 3), 120, "uint8"))
    return path


def test_caption():
    assert gs.caption_for("marlow.systems", ["marlow.systems", "Lin"]).startswith("Marlow won")
    assert "came 2" in gs.caption_for("Lin", ["Lin", "marlow.systems"])


def test_big_png_becomes_small_jpeg(tmp_path):
    p = tmp_path / "x.png"
    cv2.imwrite(str(p), (np.random.rand(1080, 1920, 3) * 255).astype("uint8"))   # noise: ~6 MB png
    name, data, ctype = gs.image_part(p)
    assert len(data) <= gs.MAX_BYTES and ctype == "image/jpeg" and name.endswith(".jpg")


def test_multipart_shape(tmp_path):
    p = tmp_path / "a.png"
    cv2.imwrite(str(p), np.zeros((10, 10, 3), "uint8"))
    body, ctype = gs.multipart("hi", [p])
    assert ctype.startswith("multipart/form-data; boundary=")
    assert b'name="caption"' in body and b'name="file"; filename="a.png"' in body


def test_off_without_key(tmp_path, monkeypatch):
    monkeypatch.setattr(gs, "KEY_FILE", tmp_path / "none.key")
    notes = []
    assert gs.send("x", [tmp_path / "a.png"], url="https://example.test/send", note=notes.append, **RESULT) is None
    assert "group share off" in notes[0]


def _fake_post(monkeypatch, tmp_path, fail):
    """send() with a fake urlopen: fail = "net" or an HTTP code. Returns the tries and the notes."""
    import urllib.error
    p = _shot(tmp_path / "a.png")
    monkeypatch.setattr(gs, "_key", lambda: "k")
    monkeypatch.setattr(gs, "RETRY_WAITS", (0, 0, 0))
    tries, notes = [], []

    def urlopen(req, timeout):
        tries.append(1)
        if fail == "net":
            raise OSError("down")
        raise urllib.error.HTTPError("u", fail, "m", {}, None)
    monkeypatch.setattr(gs.urllib.request, "urlopen", urlopen)
    gs.send("c", [p], url="https://example.test/send", note=notes.append, wait=True, **RESULT)
    return len(tries), notes


def test_retries_network_and_5xx(monkeypatch, tmp_path):
    for fail in ("net", 503, 429):
        n, notes = _fake_post(monkeypatch, tmp_path, fail)
        assert n == 3 and "gave up after 3 tries" in notes[-1]


def test_4xx_is_final(monkeypatch, tmp_path):
    n, notes = _fake_post(monkeypatch, tmp_path, 403)
    assert n == 1 and "HTTP 403" in notes[-1]


def test_one_key_per_game_whatever_the_screenshots(monkeypatch, tmp_path):
    """owner 2026-10-06: the same game posted twice must carry one Idempotency-Key."""
    k1 = gs.game_key(["A", "marlow"], [13, 9], when=1_790_000_000)
    assert k1 == gs.game_key(["A", "marlow"], [13, 9], when=1_790_000_100)
    assert k1 != gs.game_key(["A", "marlow"], [13, 8], when=1_790_000_000)
    heads = []
    monkeypatch.setattr(gs, "_key", lambda: "k")

    class R:
        status = 200
        def read(self): return b"{}"
        def __enter__(self): return self
        def __exit__(self, *a): return False
    monkeypatch.setattr(gs.urllib.request, "urlopen", lambda req, timeout: heads.append(req.get_header("Idempotency-key")) or R())
    for name in ("a.png", "b.png"):
        p = _shot(tmp_path / name)
        gs.send("c", [p], url="https://example.test/send", note=lambda m: None, wait=True, idem_key=k1, **RESULT)
    assert len(heads) == 2 and heads[0] == heads[1]


BLACK = Path(__file__).parent / "fixtures" / "black_game_mode_toast_151015.png"   # stage 2026-10-10 15:10


def _posts(monkeypatch):
    sent = []
    monkeypatch.setattr(gs, "_key", lambda: "k")
    monkeypatch.setattr(gs, "RETRY_WAITS", (0,))
    monkeypatch.setattr(gs.urllib.request, "urlopen", lambda req, timeout: sent.append(req) or (_ for _ in ()).throw(OSError("x")))
    return sent


def test_black_frame_is_never_sent(monkeypatch, tmp_path):
    """Stage 2026-10-10 15:10: an all-black frame with the macOS 'Game Mode: On' toast was posted."""
    sent, notes = _posts(monkeypatch), []
    black = tmp_path / "black.png"
    cv2.imwrite(str(black), cv2.imread(str(BLACK)))
    toast = tmp_path / "toast.png"                     # what _trim_black left of it: the toast alone
    cv2.imwrite(str(toast), cv2.imread(str(BLACK))[80:130, 1020:1270])
    assert gs.mostly_black(black) and not gs.shareable(black) and not gs.shareable(toast)
    assert gs.send("c", [black, toast], url="https://example.test/send", note=notes.append, wait=True, **RESULT) is None
    assert not sent and any("dropped 2" in n for n in notes)


def test_no_winner_and_no_ranking_is_never_sent(monkeypatch, tmp_path):
    sent, notes = _posts(monkeypatch), []
    p = _shot(tmp_path / "a.png")
    assert gs.send("Game over.", [p], url="https://example.test/send", note=notes.append, wait=True,
                   winner=None, ranking=[]) is None
    assert not sent and "not a real end screen" in notes[-1]


def test_valid_end_screen_is_sent(monkeypatch, tmp_path):
    sent = _posts(monkeypatch)
    end = Path(__file__).parent / "fixtures" / "game_over_014609.png"
    p = tmp_path / "end.png"
    cv2.imwrite(str(p), cv2.imread(str(end)))
    assert gs.shareable(p) and not gs.mostly_black(p)
    gs.send("c", [p], url="https://example.test/send", note=lambda m: None, wait=True,
            winner="Lin", ranking=["Lin", "Marianne", "Jean", "marlow.systems"])
    assert len(sent) == 1
