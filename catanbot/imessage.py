"""Local iMessage, on the bot's own Mac, when configured (owner 2026-10-10: "it can do facetime and
imessage too, if configured"). No server needed: Messages.app sends, its database is read for
owner commands.

data/config.json:
  "imessage": {
    "chat": "Catan night",        # group chat name or chat guid: end-of-game shots go here
    "owners": ["+15550100"],      # handles (phone / Apple ID) whose words count as commands
    "name": "marlow",             # a message must name the bot to be read as a command
    "dm": true                    # owners may also DM the bot's Apple ID
  }
No "imessage" block, or no "chat" and no "owners" = off.

Needs on the bot's Mac: Messages signed in to the bot's Apple ID; Automation permission for
Messages (asked on first send); Full Disk Access for the app that runs the bot (to read
~/Library/Messages/chat.db, opened read-only). Owner commands:
  "marlow leave the game"          -> data/leave.flag (same as the server path, commands.py)
  "marlow score" / "where are we"  -> one line from the game log (no hand, no plan)
  "marlow plan" / "why"            -> our plan and its reason (owner's private channels only)
Only owners' words are commands; nobody else in the group can steer the bot. The hand is never sent.
"""
from __future__ import annotations

import os
import re
import shutil
import sqlite3
import subprocess
import threading
import time
from pathlib import Path

from catanbot import commands, game_log

ROOT = Path(__file__).resolve().parent.parent
CHAT_DB = Path(os.environ.get("CATANBOT_CHAT_DB") or Path.home() / "Library/Messages/chat.db")
# Messages only sends files it may read; a copy under ~/Pictures is the reliable spot (UNCONFIRMED
# on every macOS version)
OUTBOX = Path.home() / "Pictures" / "catanbot-share"
POLL_S = 5
SCORE = re.compile(r"\b(score|status|points|where are we|who'?s winning|who is winning)\b", re.I)
PLAN = re.compile(r"\b(plan|strategy|why)\b", re.I)

SEND_SCRIPT = """
on run argv
  set target to item 1 of argv
  set body to item 2 of argv
  tell application "Messages"
    set c to missing value
    try
      set c to first chat whose id is target
    end try
    if c is missing value then set c to first chat whose name is target
    if body is not "" then send body to c
    repeat with i from 3 to count of argv
      send (POSIX file (item i of argv)) to c
    end repeat
  end tell
end run
"""


def settings(cfg: dict | None) -> dict | None:
    s = (cfg or {}).get("imessage") or {}
    if not (s.get("chat") or s.get("owners")):
        return None
    return {"chat": s.get("chat"), "owners": [h.lower() for h in s.get("owners") or []],
            "name": (s.get("name") or "marlow").lower(), "dm": s.get("dm", True)}


def send(target: str, text: str = "", files: list[Path] | None = None, note=print) -> bool:
    """Send text and/or files to a chat (guid or name). Arguments go to osascript as argv, never
    into the script text."""
    paths = []
    for p in files or []:
        OUTBOX.mkdir(parents=True, exist_ok=True)
        dst = OUTBOX / p.name
        shutil.copy2(p, dst)
        paths.append(str(dst))
    try:
        r = subprocess.run(["osascript", "-e", SEND_SCRIPT, target, text, *paths],
                           capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as e:
        note(f"imessage send failed: {type(e).__name__}")
        return False
    if r.returncode != 0:
        note(f"imessage send failed: {r.stderr.strip()[:200]}")
        return False
    note(f"imessage: sent {len(paths)} file(s) to the chat")
    return True


def body(text: str | None, blob: bytes | None) -> str | None:
    """Message text: the text column, or (newer macOS, text is NULL) the string inside the
    attributedBody typedstream."""
    if text:
        return text
    if not blob:
        return None
    i = blob.find(b"NSString")
    if i < 0:
        return None
    b = blob[i + 8:]
    j = b.find(b"+")
    if j < 0 or j + 1 >= len(b):
        return None
    b = b[j + 1:]
    n, b = (int.from_bytes(b[1:3], "little"), b[3:]) if b[0] == 0x81 else (b[0], b[1:])
    return b[:n].decode("utf-8", "replace") or None


QUERY = """
SELECT m.ROWID, m.text, m.attributedBody, m.is_from_me, h.id, c.guid, c.display_name,
       (SELECT COUNT(*) FROM chat_handle_join chj WHERE chj.chat_id = c.ROWID)
FROM message m
JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
JOIN chat c ON c.ROWID = cmj.chat_id
LEFT JOIN handle h ON h.ROWID = m.handle_id
WHERE m.ROWID > ? ORDER BY m.ROWID
"""


class Inbox:
    """New messages since start; the history before the bot started is never replayed."""

    def __init__(self, db: Path = CHAT_DB):
        self.db = db
        self.last = None

    def _open(self):
        return sqlite3.connect(f"file:{self.db}?mode=ro", uri=True, timeout=5)

    def new(self) -> list[dict]:
        with self._open() as con:
            if self.last is None:
                self.last = con.execute("SELECT COALESCE(MAX(ROWID), 0) FROM message").fetchone()[0]
                return []
            rows = con.execute(QUERY, (self.last,)).fetchall()
        out = []
        for rid, text, blob, mine, handle, guid, name, members in rows:
            self.last = max(self.last, rid)
            out.append({"id": rid, "text": body(text, blob) or "", "from_me": bool(mine),
                        "handle": (handle or "").lower(), "chat": guid, "chat_name": name or "",
                        "group": (members or 0) > 1})
        return out


def for_us(msg: dict, s: dict) -> bool:
    """An owner's words, in the configured chat (or a DM when allowed), naming the bot."""
    if msg["from_me"] or msg["handle"] not in s["owners"]:
        return False
    in_chat = s["chat"] and s["chat"] in (msg["chat"], msg["chat_name"])
    if not (in_chat or (s["dm"] and not msg["group"])):
        return False
    return s["name"] in msg["text"].lower()


def reply_for(text: str, logs: Path | None = None) -> str | None:
    """What to answer (None = nothing). Leave is handled by commands.apply first."""
    st = game_log.current_status(logs)
    if PLAN.search(text):
        return game_log.strategy_line(st) or game_log.summary(st)
    if SCORE.search(text):
        return game_log.summary(st)
    return None


def handle(msgs: list[dict], s: dict, note=print, root: Path = ROOT, reply=send) -> int:
    n = 0
    for m in msgs:
        if not for_us(m, s):
            continue
        n += 1
        if commands.intent({"text": m["text"]}) == "leave":
            commands.apply([{"id": m["id"], "command": "leave"}], note=note, root=root)
            reply(m["chat"], "Leaving the game.", note=note)
            continue
        ans = reply_for(m["text"], root / "data" / "game_logs")
        if ans:
            reply(m["chat"], ans, note=note)
    return n


def start(cfg: dict, note=print, every: float = POLL_S) -> threading.Thread | None:
    s = settings(cfg)
    if not s or not s["owners"]:
        return None
    inbox = Inbox()

    def loop():
        failing = False
        while True:
            try:
                handle(inbox.new(), s, note)
                failing = False
            except Exception as e:                  # noqa: BLE001 -- keep polling
                if not failing:
                    note(f"imessage: can't read Messages ({type(e).__name__}); "
                         "grant Full Disk Access to the app running the bot")
                failing = True
            time.sleep(every)
    t = threading.Thread(target=loop, name="imessage-commands", daemon=True)
    t.start()
    note("imessage: listening for owner commands")
    return t
