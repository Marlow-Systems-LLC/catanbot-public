"""Live game feed for Marlow chat (owner 2026-10-04, via the chat session): one small snapshot
per roll plus one at the end, POSTed to marlow.systems. Public state only: never our hand,
progress cards, plan, trade or rob targets, or estimates of other hands.

Off by default. Switched on by data/feed.json (not committed) or env:
  {"enabled": true, "url": "https://...", "key": "..."}   /  CATAN_FEED_URL + CATAN_FEED_KEY
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from pathlib import Path

CONF = Path(__file__).resolve().parent.parent / "data" / "feed.json"
PLAYER_KEYS = {"name", "colour", "vp", "cardCount", "settlements", "cities", "metropolises", "walls",
               "roads", "ships", "knights", "improvements", "defenderVp", "progressCardCount"}
TOP_KEYS = {"v", "gameId", "status", "at", "turn", "currentPlayer", "players", "longestRoad", "barbarian",
            "canal", "metropolisOwners", "robber", "recentDice", "events", "result"}
MAX_BYTES = 16 * 1024


def config() -> dict | None:
    """URL + key when the feed is on, else None. Never logged."""
    conf = {}
    try:
        conf = json.loads(CONF.read_text())
    except (OSError, ValueError):
        pass
    url = os.environ.get("CATAN_FEED_URL") or conf.get("url")
    key = os.environ.get("CATAN_FEED_KEY") or conf.get("key")
    on = os.environ.get("CATAN_FEED_URL") is not None or conf.get("enabled") is True
    return {"url": url, "key": key} if on and url and key else None


def _player(p) -> dict:
    kn = list(p.knights.values())
    return {"name": p.name, "colour": p.color, "vp": p.vp_visible, "cardCount": p.hand_count,
            "settlements": len(p.settlements), "cities": len(p.cities), "metropolises": len(p.metropolises),
            "walls": len(p.walls), "roads": len(p.roads), "ships": len(p.ships),
            "knights": {"total": len(kn), "active": sum(k.active for k in kn),
                        "strength": sum(k.level for k in kn if k.active)},
            "improvements": dict(p.improvements), "defenderVp": p.defender_vp,
            "progressCardCount": p.progress_cards}


def snapshot(gs, game_id: str, status: str, dice: list, events: list, result: dict | None = None) -> dict:
    """Public state only. Built field by field from the allowlist, never from the bot's log."""
    holder = next((p for p in gs.players if p.has_longest_road), None)
    cur = gs.players[gs.current].name if gs.players and 0 <= gs.current < len(gs.players) else None
    snap = {"v": 1, "gameId": game_id, "status": status,
            "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "turn": len(dice),
            "currentPlayer": cur, "players": [_player(p) for p in gs.players],
            "longestRoad": {"holder": holder.name if holder else None,
                            "length": holder.longest_road if holder else None},
            "barbarian": {"position": gs.barbarian_pos, "stepsToAttack": gs.barbarian_len - gs.barbarian_pos},
            "canal": {"progress": gs.canal_progress, "of": gs.canal_total},
            "metropolisOwners": {t: (gs.metropolis_owner.get(t) or (None,))[0]
                                 for t in ("trade", "politics", "science")},
            "robber": ",".join(map(str, gs.board.robber)) if gs.board.robber else None,
            "recentDice": list(dice)[-10:], "events": list(events)[-10:]}
    if status == "finished":
        snap["result"] = result or {}
    assert set(snap) <= TOP_KEYS and all(set(p) == PLAYER_KEYS for p in snap["players"])
    return snap


def post(snap: dict, conf: dict | None = None, note=print) -> threading.Thread | None:
    """Fire and forget in a thread (3 s timeout); a feed failure never touches play."""
    conf = conf or config()
    if conf is None:
        return None
    body = json.dumps(snap, separators=(",", ":")).encode()
    if len(body) > MAX_BYTES:
        note(f"feed: snapshot {len(body)} B > 16 KB, not sent")
        return None

    def send():
        req = urllib.request.Request(conf["url"], data=body, method="POST", headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {conf['key']}"})
        try:
            with urllib.request.urlopen(req, timeout=3) as r:
                if r.status >= 300:
                    note(f"feed: HTTP {r.status}")
        except Exception as e:                       # noqa: BLE001 -- never break play
            note(f"feed: {type(e).__name__}")

    t = threading.Thread(target=send, daemon=True)
    t.start()
    return t
