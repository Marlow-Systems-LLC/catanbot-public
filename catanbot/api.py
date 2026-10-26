"""Read-only LAN API for the Catan bot (owner-approved 2026-10-03).

Routes (GET only, bearer token required except /health):
  /health            {"ok": true, "version", "commit"}
  /version           {"version", "commit"}: the running catanbot version and its git short sha
  /rag?q=...&k=3     RAG search over the strategy docs and game lessons
  /record            live win/loss record
  /games             list of finished live games (id, winner, mode)
  /games/<id>        one game: winner, mode and its decision log
  /live              whether a live game is running + the last runner log lines
  /game              the current game's ledger: status (incl. strategy), summary (public line),
                     strategy (owner-private line) and its events; never the hand
  /train            self-play training rounds (last 20)
/health, /version, /game and /live carry "version" and "commit" (owner 2026-10-09).

The token lives in ~/.catanbot-api.key (chmod 600, outside the repo). Only private-network
clients are served. Nothing under ~/Library, no prefs, no credentials.
"""
from __future__ import annotations

import hmac
import ipaddress
import json
import os
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
# CATANBOT_API_KEY_FILE / CATANBOT_API_PORT override these (the Docker image sets both)
KEY_PATH = Path(os.environ.get("CATANBOT_API_KEY_FILE") or Path.home() / ".catanbot-api.key")
PORT = int(os.environ.get("CATANBOT_API_PORT") or 8940)
GAME_ID = re.compile(r"^live_\d{8}_\d{6}$")
SECRET_WORDS = re.compile(r"authtoken|password|passwd|token=|nakama|secret", re.I)


def _token() -> str:
    return KEY_PATH.read_text().strip()


def _tail(path: Path, n: int) -> list[str]:
    if not path.exists():
        return []
    with path.open("rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 200_000))
        lines = f.read().decode("utf-8", "replace").splitlines()[-n:]
    return [ln for ln in lines if not SECRET_WORDS.search(ln)]


def _games() -> list[dict]:
    out = []
    for p in sorted((DATA / "games").glob("live_*.json")):
        try:
            g = json.loads(p.read_text())
        except Exception:
            continue
        out.append({"id": p.stem, "winner": g.get("winner"), "mode": g.get("mode")})
    return out


def _game_file(gid: str) -> Path | None:
    # Resolve against the fixed games directory and refuse anything that lands outside it
    # (a symlinked live_*.json pointing elsewhere included).
    base = (DATA / "games").resolve()
    for p in base.glob("live_*.json"):
        if p.stem != gid:
            continue
        real = p.resolve()
        if real.parent == base and real.is_relative_to(base) and real.is_file():
            return real
        return None
    return None


def _ver() -> dict:
    """owner 2026-10-09: marlow says what version it runs (commit '' off a git checkout)."""
    import catanbot
    return {"version": catanbot.__version__, "commit": catanbot.COMMIT}


def route(path: str, query: dict) -> tuple[int, object]:
    if path == "/health":
        return 200, {"ok": True, **_ver()}
    if path == "/version":
        return 200, _ver()
    if path == "/rag":
        q = (query.get("q") or [""])[0].strip()
        if not q:
            return 400, {"error": "missing q"}
        k = max(1, min(10, int((query.get("k") or ["3"])[0] or 3)))
        from catanbot import rag
        hits = rag.search(q[:500], k=k)
        return 200, [{"source": h.get("source"), "title": h.get("title"), "text": h.get("text")} for h in hits]
    if path == "/record":
        from catanbot import chat
        return 200, chat.record()
    if path == "/games":
        return 200, _games()
    if path.startswith("/games/"):
        gid = path.split("/", 2)[2]
        # The file path comes from the games directory listing, never from the URL:
        # the id only selects one of the files that are already there.
        p = _game_file(gid) if GAME_ID.fullmatch(gid) else None
        if p is None:
            return 404, {"error": "no such game"}
        g = json.loads(p.read_text())
        log = [ln for ln in g.get("log", []) if not SECRET_WORDS.search(ln)]
        return 200, {"id": gid, "winner": g.get("winner"), "mode": g.get("mode"), "log": log}
    if path == "/game":
        # owner 2026-10-08: every turn of the current game, for the voice (never the hand).
        # owner 2026-10-09: with our strategy, for the owner's private channels only (the
        # pusher sends it to marlow.systems, which shows it in the owner's DM / call alone)
        from catanbot import game_log
        st = game_log.current_status()
        return 200, {"status": st, "summary": game_log.summary(st), "strategy": game_log.strategy_line(st),
                     "turns": [r for r in game_log.turns(last=200)
                               if not SECRET_WORDS.search(json.dumps(r))], **_ver()}
    if path == "/live":
        from catanbot import game_log
        log = DATA / "runner_stage.log"
        age = time.time() - log.stat().st_mtime if log.exists() else None
        st = game_log.current_status()
        return 200, {"game": {"summary": game_log.summary(st), "status": st},
                     "stopped": (DATA / "STOP").exists(),
                     "alert": _tail(DATA / "ALERT", 3),
                     "log_age_s": round(age) if age is not None else None,
                     "running": age is not None and age < 120 and not (DATA / "STOP").exists(),
                     # owner 2026-10-09: never the hand ('me: hand=...', offer answers)
                     "tail": [game_log.scrub(ln) for ln in _tail(log, 40)], **_ver()}
    if path == "/train":
        rows = []
        for ln in _tail(DATA / "train" / "loop.jsonl", 20):
            try:
                rows.append(json.loads(ln))
            except Exception:
                pass
        return 200, rows
    return 404, {"error": "unknown route"}


class Handler(BaseHTTPRequestHandler):
    server_version = "catanbot-api"
    sys_version = ""

    def _send(self, code: int, body: object) -> None:
        data = json.dumps(body, indent=1).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        try:
            ip = ipaddress.ip_address(self.client_address[0])
        except ValueError:
            return self._send(403, {"error": "forbidden"})
        if not (ip.is_private or ip.is_loopback):
            return self._send(403, {"error": "forbidden"})
        u = urlparse(self.path)
        if u.path != "/health":
            auth = self.headers.get("Authorization", "")
            if not hmac.compare_digest(auth.encode(), f"Bearer {_token()}".encode()):
                return self._send(401, {"error": "unauthorized"})
        try:
            code, body = route(u.path.rstrip("/") or "/", parse_qs(u.query))
        except Exception as e:  # never leak a traceback
            code, body = 500, {"error": type(e).__name__}
        self._send(code, body)

    def log_message(self, fmt, *args) -> None:   # one line per request, no headers
        sys.stderr.write(f"{time.strftime('%H:%M:%S')} {self.client_address[0]} {fmt % args}\n")


def main() -> None:
    if not KEY_PATH.exists():
        sys.exit(f"missing {KEY_PATH}")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
