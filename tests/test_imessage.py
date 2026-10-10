import json
import sqlite3

from catanbot import imessage

OWNER = "+15550100"
S = imessage.settings({"imessage": {"chat": "Catan night", "owners": [OWNER]}})


def _db(path):
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE message (ROWID INTEGER PRIMARY KEY, text TEXT, attributedBody BLOB, is_from_me INT, handle_id INT);
        CREATE TABLE handle (ROWID INTEGER PRIMARY KEY, id TEXT);
        CREATE TABLE chat (ROWID INTEGER PRIMARY KEY, guid TEXT, display_name TEXT);
        CREATE TABLE chat_message_join (chat_id INT, message_id INT);
        CREATE TABLE chat_handle_join (chat_id INT, handle_id INT);
        INSERT INTO handle VALUES (1, '+15550100'), (2, '+15550199');
        INSERT INTO chat VALUES (1, 'iMessage;+;chat1', 'Catan night'), (2, 'iMessage;-;+15550100', '');
        INSERT INTO chat_handle_join VALUES (1, 1), (1, 2), (2, 1);
    """)
    return con


def _msg(con, rid, text, handle=1, chat=1, blob=None):
    con.execute("INSERT INTO message VALUES (?, ?, ?, 0, ?)", (rid, text, blob, handle))
    con.execute("INSERT INTO chat_message_join VALUES (?, ?)", (chat, rid))
    con.commit()


def test_off_without_config():
    assert imessage.settings({}) is None and imessage.settings({"imessage": {}}) is None
    assert imessage.start({}, note=lambda s: None) is None


def test_inbox_skips_history_and_reads_new(tmp_path):
    db = tmp_path / "chat.db"
    con = _db(db)
    _msg(con, 1, "old marlow leave the game")
    inbox = imessage.Inbox(db)
    assert inbox.new() == []                       # first read only marks the start
    _msg(con, 2, "marlow score?")
    _msg(con, 3, "marlow leave", chat=2)
    got = inbox.new()
    assert [m["id"] for m in got] == [2, 3]
    assert got[0]["group"] and not got[1]["group"] and got[0]["chat_name"] == "Catan night"
    assert inbox.new() == []


def test_attributed_body_text():
    blob = b"\x04\x0bstreamtyped\x81\xe8\x03\x84\x01@\x84\x84\x84\x12NSAttributedString\x00\x84\x84\x08NSObject\x00\x85\x92\x84\x84\x84\x08NSString\x01\x94\x84\x01+\x0cmarlow score\x86"
    assert imessage.body(None, blob) == "marlow score"
    assert imessage.body("plain", blob) == "plain"
    assert imessage.body(None, None) is None


def _m(text, handle=OWNER, chat="iMessage;+;chat1", name="Catan night", group=True, mine=False):
    return {"id": 9, "text": text, "from_me": mine, "handle": handle, "chat": chat, "chat_name": name, "group": group}


def test_only_owner_naming_the_bot_in_our_chat_or_dm():
    assert imessage.for_us(_m("Marlow, where are we?"), S)
    assert not imessage.for_us(_m("where are we?"), S)                      # bot not named
    assert not imessage.for_us(_m("marlow leave", handle="+15550199"), S)   # a friend, not an owner
    assert not imessage.for_us(_m("marlow leave", mine=True), S)            # the bot's own message
    assert not imessage.for_us(_m("marlow leave", chat="x", name="Other group"), S)
    assert imessage.for_us(_m("marlow leave", chat="dm", name="", group=False), S)
    no_dm = dict(S, dm=False)
    assert not imessage.for_us(_m("marlow leave", chat="dm", name="", group=False), no_dm)


def test_handle_leave_score_and_plan(tmp_path):
    (tmp_path / "data" / "game_logs").mkdir(parents=True)
    (tmp_path / "data" / "game_logs" / "current.json").write_text(json.dumps({
        "gameId": "g1", "turn": 12, "round": 3, "currentPlayer": "Lin", "vp": {"marlow": 5, "Lin": 4},
        "strategy": {"turn": 12, "plan": ["city"], "reason": "ore and grain in hand"}}))
    sent = []
    reply = lambda chat, text, note=None: sent.append((chat, text))  # noqa: E731
    n = imessage.handle([_m("marlow score"), _m("marlow why the city?"), _m("hi all"),
                         _m("marlow, leave the game")], S, note=lambda s: None, root=tmp_path, reply=reply)
    assert n == 3
    assert sent[0] == ("iMessage;+;chat1", "Turn 12 (round 3): Lin to play. Points: marlow 5, Lin 4.")
    assert "city" in sent[1][1] and "ore and grain" in sent[1][1]
    assert sent[2][1] == "Leaving the game." and (tmp_path / "data" / "leave.flag").exists()


def test_send_passes_text_as_argv(monkeypatch):
    calls = []

    class R:
        returncode, stderr = 0, ""

    monkeypatch.setattr(imessage.subprocess, "run", lambda args, **k: calls.append(args) or R())
    assert imessage.send("Catan night", 'he said "end tell"', note=lambda s: None)
    assert calls[0][:2] == ["osascript", "-e"] and calls[0][3:] == ["Catan night", 'he said "end tell"']
