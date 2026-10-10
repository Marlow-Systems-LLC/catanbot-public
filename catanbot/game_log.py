"""One log per game, every turn (owner 2026-10-08: "create a log file for each game, with every turn
logged and recorded so both the voice and the chatbot know every turn in the game and where we
are. once a new game starts, a new log starts and the old one is archived for training and record
keeping").

data/game_logs/current.jsonl   one JSON event per line (start, setup, turn, roll, action, note,
                               plan, decision, end)
data/game_logs/current.json    where we are now: turn, round, whose turn, last roll, points,
                               recent, strategy
data/game_logs/archive/<id>.jsonl + <id>.json   a finished (or abandoned) game, kept for training

Turns are counted from rolls: our own roll screen anchors "our turn"; any other new roll moves
the turn to the next seat. Two identical rolls in a row look like one (UNCONFIRMED turns are
re-anchored at our next turn). Our hand goes into the event file for training only; the status,
the chat line and the API never carry it.

Strategy (owner 2026-10-09: "first to log the game and strategy, second to be able to discuss the
game live"): our plan with a one-line reason (a "plan" event whenever it changes in a turn), the
outlook goals and threats (kept current in the status, no event of their own: they are re-planned
after every move), and key decisions such as knight first, save the city, trades and the robber
target ("decision" events, the last ~50 in the status). Each decision line carries its why
("T119 bank trade ore 3 -> brick 1 for settlement (1, -2) (goal settlement+1)") and each plan
change one "T<n> plan: ... because ..." line, so the owner can ask why we did X on turn N. It is for the owner's private channels
(WhatsApp DM, FaceTime) through the API; the in-game chat line (summary()) never carries it, and
none of it carries the hand.
"""
from __future__ import annotations

import ast
import json
import re
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "data" / "game_logs"
PRIVATE = {"hand"}                      # never leaves the event file
# owner 2026-10-09: "we gave it the ledger so that we can talk to it about the game, strategy,
# moves ... it can tell us why it makes a decision". Every decision line carries its why; the
# list covers the game's key moves within a few KB (marlow.systems clips the ledger at ~64 KB)
KEEP_DECISIONS = 50                     # decision lines kept in the status (all in the event file)
DECISIONS_BYTES = 6144                  # and at most this many characters of them
LINE_MAX = 160                          # one decision line, "T<turn> " included
# click reasons that are decisions the owner may ask about (trades, robber, progress cards)
DECISION_ACTIONS = ("robber on", "bank trade", "send offer", "accept", "counter", "monopoly pick",
                    "progress select", "steal")
# dropped first when the list is full: plan lines and trades are the moves worth asking about
LOW_PRIORITY = ("robber on", "steal", "progress select", "progress deck", "threat before our turn",
                "situational:", "monopoly pick")
# runner notes that say why the next trades happen (their label becomes those trades' why)
CONTEXTS = ("knight first", "save the city", "spend the turn", "opening", "city lost",
            "barbarians close", "hand ")
TRADE = re.compile(r"^(bank trade|send offer) (\{[^}]*\}) ?-> ?(\{[^}]*\})")
PLAN_TRADE = re.compile(r"^(?:bank|trade) (.+?) -> (.+)$")
HAND = re.compile(r",? ?\bhand=(\{[^}]*\}|\S+)")
DICT = re.compile(r"\{(?:'[a-z_]+': ?\d+(?:, ?'[a-z_]+': ?\d+)*)?\}")


def scrub(text) -> str:
    """No hand in anything that reaches the status or the API (offer notes carry hand={...})."""
    return HAND.sub("", str(text))


def _hand_of(text: str) -> str | None:
    m = HAND.search(str(text))
    return m.group(1) if m else None


def _cards(text: str) -> str:
    """{'ore': 3, 'wool': 1}->{'brick': 1}  ->  ore 3 + wool 1 -> brick 1"""
    def one(m):
        try:
            d = ast.literal_eval(m.group(0))
        except Exception:                                   # noqa: BLE001
            return m.group(0)
        return " + ".join(f"{k} {v}" for k, v in d.items() if v) or "nothing"
    return re.sub(r" ?-> ?", " -> ", DICT.sub(one, str(text)))


def _res(text: str) -> frozenset:
    return frozenset(re.findall(r"[a-z]{3,}", text)) - {"bank", "trade", "send", "offer", "nothing"}


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


class GameLog:
    def __init__(self, root: Path | None = None):
        self.dir = Path(root) if root else DIR
        self.cur = self.dir / "current.jsonl"
        self.stat = self.dir / "current.json"
        self.arch = self.dir / "archive"
        self.reset_state()

    def reset_state(self):
        self.game_id = None
        self.mode, self.me, self.target = None, None, None
        self.seats: list[str] = []
        self.turn, self.current, self.rolled = 0, None, False
        self.our_turns = 0
        self.last_roll = None
        self.vp: dict[str, int] = {}
        self.recent: list[str] = []
        self.ended = False
        self.winner = None
        self.strategy = self._no_strategy()
        self._plan_sig = None
        self.stance = None                  # the brain's stance this turn (race/block); not stored
        self._plan_lines: tuple = (None, set(), None)   # (turn, plans logged this turn, last plan)
        self._ctx: list[tuple[int, str]] = []           # (turn, context note) this turn
        self._offer_why = None
        self.failures: list[dict] = []      # owner 2026-10-09: every failure in the record
        self.fail_count = 0
        self._fail_seen: tuple = (None, set())          # (turn, failure shapes logged this turn)

    @staticmethod
    def _no_strategy() -> dict:
        return {"turn": 0, "plan": [], "reason": None, "goals": [], "threats": {}, "decisions": []}

    # ---- lifecycle -------------------------------------------------------------------------
    def active(self) -> bool:
        return self.game_id is not None and not self.ended

    def resume(self) -> bool:
        """Runner restarted mid-game: pick the open log back up. False if none or it ended."""
        if not self.stat.exists() or not self.cur.exists():
            return False
        try:
            s = json.loads(self.stat.read_text())
        except Exception:                                   # noqa: BLE001
            return False
        if s.get("ended"):
            return False
        self.game_id, self.mode, self.me, self.target = s.get("gameId"), s.get("mode"), s.get("me"), s.get("target")
        self.seats = list(s.get("seats") or [])
        self.turn, self.current = int(s.get("turn") or 0), s.get("currentPlayer")
        self.rolled = bool(s.get("rolled"))
        self.our_turns = int(s.get("round") or 0)
        self.last_roll, self.vp = s.get("lastRoll"), dict(s.get("vp") or {})
        self.recent = [scrub(x) for x in s.get("recent") or []]
        self.strategy = {**self._no_strategy(), **(s.get("strategy") or {})}
        self.strategy["decisions"] = [scrub(x) for x in self.strategy["decisions"]]
        self.failures = list(s.get("failures") or [])
        self.fail_count = int(s.get("failCount") or len(self.failures))
        self.event("resume", note="runner restarted")
        return True

    def archive(self):
        """Move the current game's files to archive/ (kept for training and the record)."""
        if not self.cur.exists():
            return None
        gid = None
        try:
            gid = json.loads(self.stat.read_text()).get("gameId")
        except Exception:                                   # noqa: BLE001
            pass
        gid = gid or time.strftime("%Y%m%d_%H%M%S", time.localtime(self.cur.stat().st_mtime))
        self.arch.mkdir(parents=True, exist_ok=True)
        dst = self.arch / f"{gid}.jsonl"
        n = 1
        while dst.exists():
            n += 1
            dst = self.arch / f"{gid}_{n}.jsonl"
        shutil.move(str(self.cur), dst)
        if self.stat.exists():
            shutil.move(str(self.stat), dst.with_suffix(".json"))
        return dst

    def start(self, mode: str | None, seats: list[str], me: str | None, target: int | None = None,
              game_id: str | None = None):
        """A new game: archive the old log, open a fresh one."""
        self.archive()
        self.reset_state()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.game_id = game_id or time.strftime("%Y%m%d_%H%M%S")
        self.mode, self.me, self.target = mode, me, target
        self.seats = list(seats)
        self.event("start", mode=mode, seats=self.seats, me=me, target=target)

    def end(self, winner: str | None, ranking: list[str] | None = None, vp: dict | None = None):
        if self.game_id is None or self.ended:
            return
        if vp:
            self.vp = dict(vp)
        self.winner = winner
        self.ended = True
        self.event("end", winner=winner, ranking=ranking or [], vp=self.vp,
                   note=f"game over: {winner or '?'} won after {self.turn} turns")

    # ---- turns -----------------------------------------------------------------------------
    def _next_seat(self) -> str | None:
        if not self.seats:
            return None
        if self.current not in self.seats:
            return self.seats[0]
        return self.seats[(self.seats.index(self.current) + 1) % len(self.seats)]

    def our_turn(self, vp: dict | None = None, hand: dict | None = None):
        """Our roll screen is up: this is our turn (re-anchors the seat order)."""
        if self.game_id is None or self.ended or (self.current == self.me and not self.rolled):
            return
        if vp:
            self.vp = dict(vp)
        self.turn += 1
        self.our_turns += 1
        self.current, self.rolled = self.me, False
        self.event("turn", player=self.me, vp=self.vp, hand=dict(hand or {}),
                   note=f"turn {self.turn}: our turn (round {self.our_turns})")

    def roll(self, dice: tuple | list, vp: dict | None = None):
        """A new roll on screen: ours if our turn is open and not rolled yet, else the next seat's."""
        if self.game_id is None or self.ended:
            return
        if vp:
            self.vp = dict(vp)
        total = sum(dice)
        if not (self.current == self.me and not self.rolled):
            self.turn += 1
            self.current = self._next_seat()
            self.event("turn", player=self.current, vp=self.vp, note=f"turn {self.turn}: {self.current or '?'} to play")
        self.rolled = True
        self.last_roll = total
        self.event("roll", player=self.current, dice=list(dice), total=total,
                   note=f"{self.current or '?'} rolled {total}" + (" (robber)" if total == 7 else ""))

    def action(self, what: str, **data):
        if self.game_id is None or self.ended:
            return
        self.event("action", player=self.me, what=what, **data)
        if what.startswith(DECISION_ACTIONS):
            self._decide(what, log=False)                   # the action row is its event

    # ---- strategy (owner-private: the API and the owner's channels, never the in-game chat) -----
    def plan(self, plan: list, reason: str | None = None, goals: list | None = None,
             threats: dict | None = None, hand: dict | None = None, stance: str | None = None):
        """Our plan for this turn: one "plan" event each time it changes within a turn, and one
        "plan: ... because ..." decision line when it differs from the last one logged (a plan
        already logged this turn is not logged again). The hand goes into the event only
        (PRIVATE), never into the status."""
        if self.game_id is None or self.ended:
            return
        plan = [scrub(a)[:60] for a in plan][:10]
        reason = scrub(reason or "").strip()[:200] or None
        st = self.strategy
        st.update(turn=self.turn, plan=plan, reason=reason)
        self.stance = stance
        if goals is not None or threats is not None:
            self.outlook(goals, threats, write=False)
        self._plan_line(plan, reason)
        sig = (self.turn, tuple(plan))
        if sig == self._plan_sig:
            self.write_status()
            return
        self._plan_sig = sig
        self.event("plan", player=self.me, plan=plan, reason=reason, goals=st["goals"],
                   threats=st["threats"], hand=dict(hand or {}))

    def _plan_line(self, plan: list, reason: str | None):
        turn, seen, last = self._plan_lines
        if turn != self.turn:
            seen = set()
        key = tuple(plan)
        self._plan_lines = (self.turn, seen | {key}, key)
        if key == last or key in seen:
            return                                          # re-planned to the same plan
        text = f"plan: {', '.join(plan) or 'hold'}" + (f" because {reason}" if reason else "")
        self._add(f"T{self.turn} {text}"[:LINE_MAX])

    def _why(self, text: str) -> str:
        """Why this move, from what we know right now: the build the trade is for (this turn's
        plan), the runner's context (knight first, save the city, hand over the limit ...), the
        brain's stance for the robber, and the next goal. Empty when we know none of them."""
        st, parts, target = self.strategy, [], None
        plan = (st.get("plan") or []) if st.get("turn") == self.turn else []
        m = TRADE.match(text)
        if m:
            give, get = _res(_cards(m.group(2))), _res(_cards(m.group(3)))
            for i, item in enumerate(plan):
                pm = PLAN_TRADE.match(item)
                if pm and _res(pm.group(1)) == give and _res(pm.group(2)) == get:
                    target = next((b for b in plan[i + 1:] if not PLAN_TRADE.match(b)), None)
                    break
            ctx = next((c for t, c in reversed(self._ctx)
                        if t == self.turn and give and get and give | get <= _res(c)), None)
            if ctx:
                parts.append(ctx.split(":")[0].split(",")[0].strip())
        elif text.startswith("accept") and self._offer_why and self._offer_why[0] == self.turn:
            return self._offer_why[1]                       # the yes to our own offer: its why
        elif text.startswith(("robber on", "steal")) and self.stance:
            parts.append(f"stance {self.stance}")
        if st.get("goals"):
            parts.append(f"goal {str(st['goals'][0]).split('@')[0]}")
        why = (f" for {target[:30]}" if target else "") + (f" ({'; '.join(parts)})" if parts else "")
        if m and text.startswith("send offer"):
            self._offer_why = (self.turn, why)
        return why

    def outlook(self, goals: list | None, threats: dict | None, write: bool = True):
        """The latest outlook, into the status only (it is re-planned after every move)."""
        if self.game_id is None or self.ended:
            return
        if goals is not None:
            self.strategy["goals"] = [str(g)[:40] for g in goals][:3]
        if threats is not None:
            self.strategy["threats"] = {str(k)[:40]: [str(x)[:40] for x in (v or [])][:2]
                                        for k, v in threats.items() if v}
        if write:
            self.write_status()

    def decision(self, text: str):
        """A key decision (knight first, save the city, a trade, the robber target)."""
        if self.game_id is None or self.ended:
            return
        self._decide(text, log=True)

    def _decide(self, text: str, log: bool):
        raw = scrub(text).strip()
        text = _cards(raw)
        if log and text.startswith(CONTEXTS):
            self._ctx = [c for c in self._ctx if c[0] == self.turn][-5:] + [(self.turn, text)]
            why = ""                                        # the runner's note says its own why
        else:
            why = self._why(raw)
        head = f"T{self.turn} "
        line = (head + text[:max(40, LINE_MAX - len(head) - len(why))] + why)[:LINE_MAX]
        if not self._add(line):
            return                                          # the same decision re-noted this turn
        if log:
            self.event("decision", player=self.me, decision=line[len(head):])
        else:
            self.write_status()

    def _add(self, line: str) -> bool:
        """One decision line into the status, the list kept to KEEP_DECISIONS lines and
        DECISIONS_BYTES characters: robber taps, steals and progress picks go first, then the
        oldest. False if it repeats the last line."""
        ds = list(self.strategy["decisions"])
        if ds and ds[-1] == line:
            return False
        ds.append(line)
        while len(ds) > KEEP_DECISIONS or sum(len(d) + 4 for d in ds) > DECISIONS_BYTES:
            low = next((i for i, d in enumerate(ds[:-1])
                        if d.split(" ", 1)[-1].startswith(LOW_PRIORITY)), 0)
            ds.pop(low)
        self.strategy["decisions"] = ds
        return True

    def failure(self, text: str):
        """Owner 2026-10-09 "every failure must be recorded in the logs": a step that did not do
        what it was meant to (a click not landing, a card play not completed, a read that failed).
        One line per failure shape per turn (numbers ignored), so a retry loop can't flood it."""
        if self.game_id is None or self.ended:
            return
        line = scrub(text).strip()
        shape = re.sub(r"\d+", "#", line)
        turn, seen = self._fail_seen
        if turn != self.turn:
            seen = set()
        if shape in seen:
            return
        seen.add(shape)
        self._fail_seen = (self.turn, seen)
        self.fail_count += 1
        self.failures = (self.failures + [{"turn": self.turn, "what": line}])[-50:]
        self.event("failure", note=f"FAILED: {line}")
        self.write_status()

    def setup(self, what: str):
        if self.game_id is None or self.ended:
            return
        self.event("setup", player=self.me, what=what)

    def note(self, text: str, **data):
        """A note for the record; a hand=... in it (offer answers) moves to the private field."""
        if self.game_id is None or self.ended:
            return
        hand = _hand_of(text)
        if hand is not None:
            data["hand"] = hand
        self.event("note", note=scrub(text), **data)

    # ---- files -----------------------------------------------------------------------------
    def event(self, kind: str, **data):
        row = {"t": _now(), "kind": kind, "turn": self.turn, "round": self.our_turns, **data}
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            with self.cur.open("a") as f:
                f.write(json.dumps(row, default=str) + "\n")
        except Exception:                                   # noqa: BLE001 -- the log never stops play
            pass
        line = data.get("note") or (f"{kind}: {data.get('what')}" if data.get("what") else None)
        if line:
            self.recent = (self.recent + [f"T{self.turn} {scrub(line)}"])[-15:]
        self.write_status()

    def status(self) -> dict:
        return {"gameId": self.game_id, "mode": self.mode, "me": self.me, "target": self.target,
                "seats": self.seats, "turn": self.turn, "round": self.our_turns,
                "currentPlayer": self.current, "rolled": self.rolled, "lastRoll": self.last_roll,
                "vp": self.vp, "recent": self.recent, "ended": self.ended, "winner": self.winner,
                "strategy": self.strategy, "failures": self.failures, "failCount": self.fail_count,
                "updated": _now()}

    def write_status(self):
        try:
            tmp = self.stat.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.status(), indent=1))
            tmp.replace(self.stat)
        except Exception:                                   # noqa: BLE001
            pass


def summary(s: dict | None) -> str:
    """One public line: where we are (no hand, no plan)."""
    if not s or not s.get("gameId"):
        return "No game on right now."
    vp = s.get("vp") or {}
    pts = ", ".join(f"{n} {v}" for n, v in sorted(vp.items(), key=lambda kv: -kv[1])) if vp else "not read yet"
    if s.get("ended"):
        return f"Game over after {s.get('turn')} turns: {s.get('winner') or '?'} won. Points: {pts}."
    if not s.get("turn"):
        return f"Setting up the board. Points: {pts}."
    who = s.get("currentPlayer") or "?"
    roll = f", last roll {s['lastRoll']}" if s.get("lastRoll") else ""
    return f"Turn {s['turn']} (round {s.get('round') or 0}): {who} to play{roll}. Points: {pts}."


def strategy_line(s: dict | None) -> str | None:
    """Owner-private: our plan and its reason, goals and threats in one line. Never for the
    in-game chat (owner rule: the chat jokes about the next move, never reveals the plan)."""
    st = (s or {}).get("strategy") or {}
    if not s or not s.get("gameId") or s.get("ended") or not (st.get("plan") or st.get("goals")):
        return None
    parts = []
    if st.get("plan"):
        parts.append(f"Plan (turn {st.get('turn')}): {', '.join(st['plan'])}"
                     + (f" -- {st['reason']}" if st.get("reason") else ""))
    if st.get("goals"):
        parts.append(f"Goals: {', '.join(st['goals'])}")
    if st.get("threats"):
        parts.append("Threats: " + "; ".join(f"{k} {', '.join(v)}" for k, v in st["threats"].items()))
    return ". ".join(parts) + "."


def current_status(root: Path | None = None) -> dict | None:
    p = (Path(root) if root else DIR) / "current.json"
    try:
        s = json.loads(p.read_text())
    except Exception:                                       # noqa: BLE001
        return None
    if isinstance(s, dict):                                 # a status written before the scrub
        s["recent"] = [scrub(x) for x in s.get("recent") or []]
        if isinstance(s.get("strategy"), dict):
            s["strategy"]["decisions"] = [scrub(x) for x in s["strategy"].get("decisions") or []]
    return s


def turns(root: Path | None = None, last: int = 40) -> list[dict]:
    """The current game's events without private fields (for the API / voice)."""
    p = (Path(root) if root else DIR) / "current.jsonl"
    out = []
    try:
        for ln in p.read_text().splitlines()[-last:]:
            row = json.loads(ln)
            out.append({k: scrub(v) if isinstance(v, str) else v
                        for k, v in row.items() if k not in PRIVATE})
    except Exception:                                       # noqa: BLE001
        pass
    return out
