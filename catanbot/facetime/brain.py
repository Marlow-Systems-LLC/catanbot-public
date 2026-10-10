"""What the bot says on a call. Local only: an OpenAI-compatible LLM when configured, else rules.

Context comes from catanbot.game_log (the public status line, the plan line and the recent decisions,
already scrubbed of the hand). "Leave the game" is detected with catanbot.commands.intent and acted on
with catanbot.commands.apply, the same data/leave.flag the iMessage command writes, before any LLM.
The hand is never spoken: every reply passes game_log.scrub and a card-count filter.
"""
from __future__ import annotations

import json
import re
import time
import urllib.request
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from catanbot import game_log

ROOT = Path(__file__).resolve().parent.parent.parent

_GAME_WORD = re.compile(r"\b(game|match)\b", re.I)
_SCORE = re.compile(r"\b(score|points?|vp|who(?:'s| is) (?:winning|ahead|leading)|winning|ahead)\b", re.I)
_WHERE = re.compile(r"\b(where are we|what turn|which turn|status|how(?:'s| is) (?:it|the game) going|what(?:'s| is) happening|round)\b", re.I)
_PLAN = re.compile(r"\b(plan|strategy|what(?:'s| is| are) (?:you|your) (?:doing|going|next|goal)|next move|goals?)\b", re.I)
_WHY = re.compile(r"\b(why|de ce|reason)\b", re.I)
# "you have 2 ore", "my hand is ...", "{'ore': 2}": the hand never leaves the bot
_HAND_TALK = re.compile(r"\{[^}]*\}|\b(?:my|your|our) hand\b[^.]*\.?|\b(?:i|we) (?:have|hold|got)\s+\d+\s+\w+[^.]*\.?", re.I)

HELP = "I can tell you the score, where we are, the plan, or why I made a move. Say leave the game and I will."


def safe(text: str) -> str:
    """Strip anything that could be the hand before it is spoken."""
    t = game_log.scrub(text or "")
    t = _HAND_TALK.sub("", t)
    return re.sub(r"\s+", " ", t).strip()


def wants_leave_game(text: str) -> bool:
    """The owner's leave intent (catanbot.commands.intent), about the game: "hang up" is the call."""
    from catanbot import commands  # lazy: commands pulls in the network bridge module
    return commands.intent({"text": text or ""}) == "leave" and bool(_GAME_WORD.search(text or ""))


def leave_game(text: str, root: Path = ROOT, note: Callable[[str], None] = print) -> None:
    """Exactly what catanbot/commands.py apply does for a leave: write data/leave.flag."""
    from catanbot import commands
    (Path(root) / "data").mkdir(parents=True, exist_ok=True)
    commands.apply([{"id": f"facetime-{int(time.time())}", "command": "leave", "text": text}], note=note, root=Path(root))


def context(log_dir: Optional[Path] = None) -> dict:
    s = game_log.current_status(log_dir)
    st = (s or {}).get("strategy") or {}
    return {
        "status": game_log.summary(s),
        "plan": game_log.strategy_line(s),
        "decisions": [game_log.scrub(x) for x in (st.get("decisions") or [])[-8:]],
        "me": (s or {}).get("me"),
        "ended": bool((s or {}).get("ended")),
    }


def rule_reply(text: str, ctx: dict) -> str:
    t = text or ""
    if _WHY.search(t):
        ds = ctx.get("decisions") or []
        if ds:
            return f"Last key move: {ds[-1]}."
        return ctx.get("plan") or "No decisions logged yet this game."
    if _PLAN.search(t):
        return ctx.get("plan") or "No plan yet. " + ctx.get("status", "")
    if _SCORE.search(t) or _WHERE.search(t):
        return ctx.get("status") or "No game on right now."
    return HELP


class Brain:
    def __init__(self, cfg, root: Path = ROOT, log_dir: Optional[Path] = None,
                 note: Callable[[str], None] = print, opener=None):
        self.cfg = cfg
        self.root = Path(root)
        self.log_dir = log_dir
        self.note = note
        self.opener = opener or urllib.request.urlopen
        self.history: List[Tuple[str, str]] = []

    def reset(self) -> None:
        self.history = []

    def reply(self, text: str, kind: str = "dm") -> Tuple[str, Optional[str]]:
        """(what to say, action) for one addressed utterance. action: "leave_game" or None."""
        if wants_leave_game(text):
            leave_game(text, self.root, self.note)
            return "Okay, leaving the game.", "leave_game"
        ctx = context(self.log_dir)
        out = None
        if getattr(self.cfg, "llm", None):
            try:
                out = self.llm(text, ctx, kind)
            except Exception as e:  # noqa: BLE001 - a dead LLM falls back to rules
                self.note(f"llm failed ({type(e).__name__}); rule reply")
        if not out:
            out = rule_reply(text, ctx)
        out = safe(out) or HELP
        self.history = (self.history + [(text, out)])[-6:]
        return out, None

    def llm(self, text: str, ctx: dict, kind: str) -> Optional[str]:
        from catanbot.facetime.config import read_key
        llm = self.cfg.llm
        name = getattr(self.cfg, "name", "marlow")
        system = (
            f"You are {name}, a bot playing Catan Universe, talking on a FaceTime "
            f"{'group ' if kind == 'group' else ''}voice call. Answer in one to three short spoken sentences, "
            "no lists, no markdown. Answer in the caller's language. Use only the game context below; if it "
            "does not say, say you don't know. Never say which cards are in your hand or how many of each.\n"
            "Game context: " + json.dumps(ctx, ensure_ascii=False))
        msgs = [{"role": "system", "content": system}]
        for q, a in self.history[-4:]:
            msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        msgs.append({"role": "user", "content": text})
        body = json.dumps({"model": llm["model"], "messages": msgs, "max_tokens": 160, "temperature": 0.4}).encode()
        headers = {"content-type": "application/json"}
        key = read_key(llm.get("key_file", ""))
        if key:
            headers["authorization"] = f"Bearer {key}"
        req = urllib.request.Request(llm["url"].rstrip("/") + "/chat/completions", data=body, headers=headers, method="POST")
        with self.opener(req, timeout=30) as r:
            d = json.loads(r.read())
        return ((d.get("choices") or [{}])[0].get("message") or {}).get("content", "").strip() or None
