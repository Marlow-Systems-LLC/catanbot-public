"""Owner 2026-10-09: the record in chat counts every game; chat is answered on our own turn too."""
import json

from catanbot import chat
from catanbot.runner import Runner

REC = {"games": 5, "wins": 0, "losses": 5, "failed": 0, "streak": 0,
       "solo": {"games": 26, "wins": 5, "losses": 20, "failed": 1, "streak": 1}}


from tests.test_round2_0834 import live_layout  # noqa: F401,E402

def test_record_line_counts_every_game():
    line = chat.record_line(REC)
    assert line.startswith("31 games in all")
    assert "vs people 5 (0 won, 5 lost, 0 failed)" in line
    assert "vs the computer 26 (5 won, 20 lost, 1 failed)" in line


def test_intro_and_record_answer_use_full_record(tmp_path, monkeypatch):
    p = tmp_path / "record.json"
    p.write_text(json.dumps(REC))
    monkeypatch.setattr(chat, "RECORD_PATH", p)
    lines = chat.Chat.__new__(chat.Chat)
    lines.sent = []
    out = chat.Chat.intro_more(lines)
    assert any("31 games in all" in t and len(t) <= chat.MAX_LEN for t in out)
    ans = chat._about("marlow what is your record?")
    assert "31 games in all" in ans and "20 lost" in ans


def test_chat_polls_on_own_turn(monkeypatch):
    r = Runner.__new__(Runner)
    r.cfg = {"chat_on_turn": True, "chat_poll_s": 20}
    r._chat_last_poll = 0.0
    r._end_turn_at = 0.0
    seen = []

    def fake_poll(img, ocr, state=None):
        seen.append(state)
        r._chat_last_poll = 1.0

    monkeypatch.setattr(r, "chat_poll", fake_poll, raising=False)
    assert Runner._chat_poll_turn(r, None, []) is True
    assert seen == ["own_turn"]
    r.cfg["chat_on_turn"] = False
    assert Runner._chat_poll_turn(r, None, []) is False


def test_marker_on_our_own_banner_is_our_turn(monkeypatch):
    """live 2026-10-09 19:26: our portrait was banner 1; our robber rings read as a rival's turn."""
    import catanbot.runner as R
    r = Runner.__new__(Runner)
    r.cfg = {"bot_name": "MARLOW.SYSTEMS"}
    r._in_play = True
    r._slot_names = {1: "marlow.systems", 2: "Bramble"}
    monkeypatch.setattr(R, "active_seat", lambda img: 1)
    assert Runner._marker_seat(r, object()) == 0
    monkeypatch.setattr(R, "active_seat", lambda img: 2)
    assert Runner._marker_seat(r, object()) == 2


def test_robber_confirm_hits_the_green_check(monkeypatch, live_layout):
    """live 2026-10-09 19:29: the fixed confirm spot (654,645) sat between check and X; the robber
    on the leader never confirmed. The confirm goes to the green blob."""
    import types

    import cv2
    import numpy as np

    import catanbot.runner as R
    from tests.test_round2_0834 import GREEN, RED

    img = np.zeros((888, 1372, 3), np.uint8)
    cv2.circle(img, (621, 645), 20, GREEN, -1)
    cv2.circle(img, (683, 645), 20, RED, -1)
    r = Runner.__new__(Runner)
    r.cfg = {"confirm_by_colour": True}
    r.io = types.SimpleNamespace(snapshot=lambda: img)
    hexes = {(2, 0): types.SimpleNamespace(is_land=True)}
    board = types.SimpleNamespace(hexes=hexes, robber=(0, 0))
    gs = types.SimpleNamespace(board=board)
    r.ensure_state = lambda i, t: gs
    r.br = types.SimpleNamespace(lattice=types.SimpleNamespace(hex_px=lambda q, s: (799, 372)))
    r.brain = types.SimpleNamespace(choose_robber=lambda g, opts: ((2, 0), "Rhea"))
    r._rings = lambda i: [(799, 372)]
    r.note = lambda *a, **k: None
    clicks = []
    r.click = lambda xy, why="", *a, **k: clicks.append((tuple(xy), why))
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    Runner.on_place_robber(r, img, [], [], {})
    (cx, cy), why = clicks[-1]
    assert why == "confirm robber" and np.hypot(cx - 621, cy - 645) < 18   # on the check, not the gap
