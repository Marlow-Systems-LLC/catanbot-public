"""Owner 2026-10-04: after each game, send the end-of-game screenshots to the Catan iMessage group.

Posts to the marlow.systems Catan endpoint (POST the configured endpoint, multipart: caption + 1-4
image files, <= 2 MiB each; the server fixes the target to the Catan group). The key is read from
~/.catanbot-share.key (the owner puts it there); without it, sharing is off. Never logs the key.
"""
from __future__ import annotations

import os
import threading
import time
import urllib.request
import uuid
from pathlib import Path

import cv2

# CATANBOT_SHARE_KEY_FILE / CATANBOT_SHARE_URL override these; no key or no URL = sharing off
KEY_FILE = Path(os.environ.get("CATANBOT_SHARE_KEY_FILE") or Path.home() / ".catanbot-share.key")
DEFAULT_URL = os.environ.get("CATANBOT_SHARE_URL", "")
MAX_BYTES = 2 * 1024 * 1024
MAX_FILES = 4
RETRY_WAITS = (0, 10, 30)    # seconds before each try


def _key() -> str | None:
    try:
        k = KEY_FILE.read_text().strip()
        return k or None
    except OSError:
        return None


def image_part(path: Path) -> tuple[str, bytes, str]:
    """PNG as is when small enough, else JPEG, shrinking until it fits under 2 MiB."""
    data = path.read_bytes()
    if len(data) <= MAX_BYTES and path.suffix.lower() == ".png":
        return path.name, data, "image/png"
    img = cv2.imread(str(path))
    for width, q in ((1920, 85), (1372, 80), (1024, 70)):
        h, w = img.shape[:2]
        if w > width:
            img = cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok and len(buf) <= MAX_BYTES:
            return path.with_suffix(".jpg").name, buf.tobytes(), "image/jpeg"
    raise ValueError(f"{path.name}: can't fit under 2 MiB")


def multipart(caption: str, files: list[Path]) -> tuple[bytes, str]:
    b = uuid.uuid4().hex
    out = [f'--{b}\r\nContent-Disposition: form-data; name="caption"\r\n\r\n{caption[:1000]}\r\n'.encode()]
    for p in files[:MAX_FILES]:
        name, data, ctype = image_part(p)
        out.append(f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="{name}"\r\n'
                   f"Content-Type: {ctype}\r\n\r\n".encode() + data + b"\r\n")
    out.append(f"--{b}--\r\n".encode())
    return b"".join(out), f"multipart/form-data; boundary={b}"


def caption_for(winner: str | None, ranking: list[str], me: str = "marlow.systems") -> str:
    if winner and winner.lower() == me.lower():
        head = "Marlow won this one!"
    elif winner:
        head = f"{winner} won this one. Marlow came {ranking.index(me) + 1 if me in ranking else '?'}."
    else:
        head = "Game over."
    return f"{head} Ranking: {', '.join(ranking)}" if ranking else head


def game_key(ranking, vps, when=None) -> str:
    """One key per finished game, the same on every host and process: the UTC day plus the
    final scoreline (who placed where, with VP). Screenshot names differ per process."""
    day = time.strftime("%Y-%m-%d", time.gmtime(when if when is not None else time.time()))
    return "catan-game|" + day + "|" + ",".join(f"{n}:{v}" for n, v in zip(ranking, vps))


def send(caption: str, files: list[Path], url: str = DEFAULT_URL, note=print, wait: bool = False,
         idem_key: str | None = None):
    """Fire-and-forget POST (a failure never holds up the restart). Returns the thread or None.
    idem_key names the game: a second post of the same game (another runner, a restart on the
    end screen) carries the same Idempotency-Key, so the server drops it."""
    key = _key()
    files = [p for p in files if p.exists()][:MAX_FILES]
    if not url:
        return None
    if not key:
        note(f"group share off: no {KEY_FILE.name} ({len(files)} shot(s) kept in data/endgame)")
        return None
    if not files:
        note("group share: no end-of-game screenshots to send")
        return None

    idem = uuid.uuid5(uuid.NAMESPACE_URL, idem_key or caption + "|" + "|".join(p.name for p in files)).hex

    def go():
        for n, pause in enumerate(RETRY_WAITS, 1):
            time.sleep(pause)
            if once(n):
                return
        note(f"group share gave up after {len(RETRY_WAITS)} tries ({len(files)} shot(s) kept in data/endgame)")

    def once(n: int) -> bool:
        """One post. True = done (sent, or a final 4xx); False = worth another try."""
        try:
            body, ctype = multipart(caption, files)
            req = urllib.request.Request(url, data=body, method="POST", headers={"User-Agent": "marlow-catanbot/0.2 (+https://marlow.systems)",
                "Authorization": f"Bearer {key}", "Content-Type": ctype,
                # owner 2026-10-06: the group got each post twice (one POST logged); the same key on
                # every try of one game lets the server drop repeats
                "Idempotency-Key": idem})
            with urllib.request.urlopen(req, timeout=30) as r:
                note(f"group share: sent {len(files)} shot(s), HTTP {r.status} (try {n})")
                return True
        except urllib.error.HTTPError as e:
            note(f"group share failed (try {n}): HTTP {e.code} {e.read()[:200].decode(errors='replace')}")
            return e.code < 500 and e.code != 429
        except Exception as e:                    # noqa: BLE001
            note(f"group share failed (try {n}): {type(e).__name__}: {e}")
            return False

    t = threading.Thread(target=go, daemon=True)
    t.start()
    if wait:
        t.join(sum(RETRY_WAITS) + 40 * len(RETRY_WAITS))
    return t


def shots_since(folder: Path, since: float) -> list[Path]:
    return sorted(p for p in folder.glob("*.png") if p.stat().st_mtime >= since - 1)
