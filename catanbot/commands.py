"""Owner commands for the bot from iMessage (owner 2026-10-05: "I can ask marlow in dm or in the catan
imessage group to exit the current game and it should understand what to do and do that").

marlow.systems reads the owner's DM / Catan-group messages (owner's own messages only) and queues a
command; this module polls GET the configured endpoint (Bearer: the Catan send key) and acts:
  leave -> touch data/leave.flag; the runner walks out of the game with the game's own buttons
           (never kills the app) and waits at the front page for the next invite.
Server contract: {"commands": [{"id": "<str>", "command": "leave", "text": "<owner's words>"}]};
the bot acks with POST .../commands {"ack": ["<id>", ...]}. The key and texts are never logged.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.request
from pathlib import Path

from catanbot.group_share import _key

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_URL = os.environ.get("CATANBOT_COMMANDS_URL", "")  # "" = off
POLL_S = 20
UA = "marlow-catanbot/0.2 (+https://marlow.systems)"   # Cloudflare 1010 blocks urllib's own
LEAVE = re.compile(r"\b(exit|leave|quit|get out of|drop out of|abandon)\b.*\b(game|match|it)\b|^\s*(exit|leave|quit)\s*[.!]?\s*$", re.I)


def intent(cmd: dict) -> str | None:
    """The server may send a parsed command, raw owner text, or both; text is the fallback."""
    c = (cmd.get("command") or "").lower()
    if c in ("leave", "exit", "quit"):
        return "leave"
    if LEAVE.search(cmd.get("text") or ""):
        return "leave"
    return None


def apply(cmds: list[dict], note=print, root: Path = ROOT) -> list[str]:
    done = []
    for c in cmds:
        what = intent(c)
        if what == "leave":
            (root / "data" / "leave.flag").write_text(f"owner via iMessage {time.strftime('%H:%M:%S')}\n")
            note("owner command (iMessage): leave the game")
        elif what is None:
            note("owner command (iMessage) not understood; ignored")
        done.append(str(c.get("id")))
    return done


def _call(url: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                 method="POST" if body is not None else "GET",
                                 headers={"authorization": f"Bearer {_key()}", "user-agent": UA,
                                          "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read() or b"{}")


def poll_once(url: str = DEFAULT_URL, note=print) -> int:
    if not url or not _key():
        return 0
    cmds = _call(url).get("commands") or []
    if cmds:
        _call(url, {"ack": apply(cmds, note)})
    return len(cmds)


def start(url: str = DEFAULT_URL, note=print, every: float = POLL_S) -> threading.Thread:
    """Background poller; quiet on network errors (logs the first of a run only)."""
    def loop():
        failing = False
        while True:
            try:
                poll_once(url, note)
                failing = False
            except Exception as e:                  # noqa: BLE001 -- keep polling
                if not failing:
                    note(f"owner commands: poll failed ({type(e).__name__}); retrying quietly")
                failing = True
            time.sleep(every)
    t = threading.Thread(target=loop, name="owner-commands", daemon=True)
    t.start()
    return t
