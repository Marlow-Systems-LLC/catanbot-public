"""Probe capture + per-action scorecard (owner 2026-10-04, HANDOFF_v02 Part A).

Every screen or prompt the runner doesn't handle, and every action that didn't land, saves the
native frame, the board scan and a JSON note to data/probe/<time>_<kind>.{png,json} (cap per game).
The scorecard counts tries / landed / failed / unverified per action kind and is written to
data/probe/scorecard_<game>.json (every few minutes and at game over).
"""
from __future__ import annotations

import collections
import json
import threading
import time
from pathlib import Path

from catanbot import shots


class Probe:
    CAP = 400                 # saved frames per game
    DUMP_EVERY_S = 300

    def __init__(self, root: Path, enabled: bool = True):
        self.dir = Path(root) / "data" / "probe"
        self.enabled = enabled
        self._lock = threading.Lock()                 # save() runs on the caller and on the worker
        self._cv = threading.Condition()
        self._q: collections.deque = collections.deque()
        self._busy = False
        self._worker = None
        self.dropped = 0
        self.reset()

    def reset(self, game: str | None = None):
        self.game = game or time.strftime("%Y%m%d_%H%M%S")
        self.n = 0
        self.card: dict[str, dict] = {}
        self.events: list[dict] = []
        self._dumped_at = time.time()

    # -- scorecard
    def _row(self, kind):
        return self.card.setdefault(kind, {"tries": 0, "landed": 0, "failed": 0, "unverified": 0})

    def tried(self, kind: str):
        self._row(kind)["tries"] += 1

    def result(self, kind: str, ok: bool | None, why: str = ""):
        """ok True = landed, False = failed, None = could not be checked."""
        row = self._row(kind)
        row["landed" if ok else "failed" if ok is False else "unverified"] += 1
        if ok is False:
            self.events.append({"t": time.strftime("%H:%M:%S"), "kind": kind, "why": why})
            self.events = self.events[-300:]

    def clean(self) -> bool:
        """Every action used in the game landed at least once and none failed outright."""
        return all(r["failed"] == 0 for r in self.card.values())

    def summary(self) -> str:
        return ", ".join(f"{k} {r['landed']}/{r['tries']}" + (f" F{r['failed']}" if r["failed"] else "")
                         + (f" ?{r['unverified']}" if r["unverified"] else "")
                         for k, r in sorted(self.card.items()))

    def dump(self, final: bool = False, extra: dict | None = None) -> Path | None:
        if not self.enabled:
            return None
        self.dir.mkdir(parents=True, exist_ok=True)
        p = self.dir / f"scorecard_{self.game}.json"
        data = {"game": self.game, "final": final, "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "clean": self.clean(), "frames_saved": self.n, "actions": self.card,
                "failures": self.events[-100:]}
        if extra:
            data.update(extra)
        p.write_text(json.dumps(data, indent=1))
        self._dumped_at = time.time()
        return p

    def maybe_dump(self):
        if time.time() - self._dumped_at > self.DUMP_EVERY_S:
            self.dump()

    # -- frames
    def save(self, kind: str, img, native=None, note: dict | None = None, texts=None, scan=None) -> Path | None:
        if not self.enabled or img is None or self.n >= self.CAP:
            return None
        with self._lock:
            return self._save(kind, img, native, note, texts, scan)

    def _save(self, kind, img, native, note, texts, scan):
        if self.n >= self.CAP:
            return None
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            stem = self.dir / f"{time.strftime('%H%M%S')}_{self.n:03d}_{kind}"
            shots.imwrite(str(stem) + ".png", img)
            if native is not None and native is not img:
                shots.imwrite(str(stem) + "_native.png", native)
            j = {"kind": kind, "t": time.strftime("%Y-%m-%d %H:%M:%S"), "game": self.game}
            if note:
                j["note"] = note
            if texts is not None:
                j["texts"] = list(texts)[:120]
            if scan is not None:
                j["scan"] = scan
            Path(str(stem) + ".json").write_text(json.dumps(j, indent=1, default=str))
            self.n += 1
            return stem
        except Exception:                            # noqa: BLE001 -- capture must never stop play
            return None

    # -- off the turn (owner 2026-10-09 "play every move way faster": a probe save cost ~4 s of
    # our turn, 11.8 s/turn live). Frames are copied here; the PNG/JSON writes and the scan
    # (rings) run on one worker thread. Bounded: the oldest waiting frame is dropped first.
    QUEUE_CAP = 20

    def save_async(self, kind: str, img, native=None, note: dict | None = None, texts=None,
                   scan_fn=None, done=None) -> bool:
        """Queue a save; scan_fn(img, native) builds the scan dict on the worker, done(path) is
        called there after the write. True = queued."""
        if not self.enabled or img is None or self.n >= self.CAP:
            return False
        job = (kind, img.copy(), native.copy() if native is not None and native is not img else None,
               dict(note) if isinstance(note, dict) else note, list(texts) if texts is not None else None,
               scan_fn, done)
        self._put(job)
        return True

    def call_async(self, fn) -> None:
        """Any other write off the turn (data/unread) on the same worker and queue."""
        self._put(("__call__", fn))

    def _put(self, job):
        with self._cv:
            if len(self._q) >= self.QUEUE_CAP:
                self._q.popleft()
                self.dropped += 1
            self._q.append(job)
            self._cv.notify()
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._run, name="probe-writer", daemon=True)
                self._worker.start()

    def _run(self):
        while True:
            with self._cv:
                while not self._q:
                    self._busy = False
                    self._cv.notify_all()
                    self._cv.wait()
                job = self._q.popleft()
                self._busy = True
            if job[0] == "__call__":
                try:
                    job[1]()
                except Exception:                     # noqa: BLE001
                    pass
                continue
            kind, img, native, note, texts, scan_fn, done = job
            try:
                scan = scan_fn(img, native) if scan_fn is not None else None
            except Exception as e:                    # noqa: BLE001 -- capture must never stop play
                scan = {"error": f"{type(e).__name__}: {e}"}
            p = self.save(kind, img, native, note, texts, scan)
            if done is not None:
                try:
                    done(p)
                except Exception:                     # noqa: BLE001
                    pass

    def pending(self) -> int:
        with self._cv:
            return len(self._q) + (1 if self._busy else 0)

    def flush(self, timeout: float = 10.0) -> bool:
        """Wait until every queued frame is written. True = the queue is empty."""
        end = time.time() + timeout
        with self._cv:
            while self._q or self._busy:
                left = end - time.time()
                if left <= 0:
                    return False
                self._cv.wait(left)
        return True
