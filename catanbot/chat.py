"""marlow's in-game chat: classify a message, answer from the local RAG store, filter hard.

Strict topic limit: Catan (rules, strategy, what marlow learned from games), marlow's own name,
and the company / maker / site from data/persona.json (owner-given values only).
Everything else gets a fixed deflection. No network, no external API.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from . import rag

ROOT = Path(__file__).resolve().parents[1]
PERSONA_PATH = ROOT / "data" / "persona.json"
MAX_LEN = 200
MIN_GAP_S = 10.0


def persona() -> dict:
    return json.loads(PERSONA_PATH.read_text())


RECORD_PATH = ROOT / "data" / "record.json"


def record() -> dict:
    try:
        return json.loads(RECORD_PATH.read_text())
    except (OSError, ValueError):
        return {"games": 0, "wins": 0, "streak": 0, "best_streak": 0}


def record_line(r: dict | None = None) -> str:
    """Owner 2026-10-09 "the records of games it mentions is not complete": every game counts --
    people and computer games, wins, losses and failed games (failures count until it works)."""
    r = record() if r is None else r
    s = r.get("solo") or {}
    g = lambda b, k: int(b.get(k, 0) or 0)                       # noqa: E731
    total = g(r, "games") + g(s, "games")
    part = lambda b: f"{g(b, 'games')} ({g(b, 'wins')} won, {g(b, 'losses')} lost, {g(b, 'failed')} failed)"  # noqa: E731
    return (f"{total} games in all: vs people {part(r)}, vs the computer {part(s)}; "
            f"win streak {g(r, 'streak')}")


def _bucket(r: dict, solo: bool) -> dict:
    """Live games vs people sit at the top level of record.json (games/wins/streak, quoted in the
    chat intro); solo games vs the computer sit in r["solo"]."""
    b = r.setdefault("solo", {}) if solo else r
    for k in ("games", "wins", "losses", "failed", "streak", "best_streak"):
        b.setdefault(k, 0)
    return b


def _save_record(r: dict):
    RECORD_PATH.write_text(json.dumps(r, indent=1))


def record_game_open(game_id: str | None, solo: bool = False, seats: list[str] | None = None,
                     mode: str | None = None) -> dict:
    """Owner 2026-10-09: every game is counted -- a win, a loss or a failure (we left it, crashed
    or never saw its end screen). A game opening while another is still open fails the old one,
    unless we left the old one and this is the same match again (review r1: a quit can pull us
    back into the old match, which opens a new game id -- that one game must not count twice)."""
    r = record()
    old = r.get("open")
    if old and old.get("id") != game_id:
        if (old.get("left_at") and time.time() - old["left_at"] < REJOIN_S and seats
                and old.get("seats") == list(seats) and old.get("mode") == mode):
            old = None                       # the same match: carry on under the new id
        else:
            _record_failed(r, old, old.get("left") or "a new game opened before its end screen")
    if game_id:
        r["open"] = {"id": game_id, "solo": bool(solo), "seats": list(seats or []), "mode": mode,
                     "at": time.strftime("%Y-%m-%d %H:%M:%S")}
    _save_record(r)
    return r


REJOIN_S = 900


def record_left(game_id: str | None, reason: str) -> bool:
    """We left the open game without its end screen. It counts as failed when the next game opens,
    unless that is the same match again (record_game_open)."""
    r = record()
    old = r.get("open")
    if not old or (game_id and old.get("id") != game_id) or old.get("left"):
        return False
    old["left"], old["left_at"] = reason, time.time()
    _save_record(r)
    return True


def _record_failed(r: dict, old: dict, reason: str):
    r.pop("open", None)
    if any(g.get("id") == old.get("id") for g in r.get("log", [])):
        return
    b = _bucket(r, old.get("solo", False))
    b["games"] += 1
    b["failed"] += 1
    b["streak"] = 0
    r.setdefault("log", []).append({"id": old.get("id"), "solo": old.get("solo", False), "outcome": "failed",
                                    "reason": reason, "at": time.strftime("%Y-%m-%d %H:%M:%S")})


def record_result(won: bool, solo: bool = False, game_id: str | None = None, place: int | None = None,
                  players: int | None = None, vp: int | None = None) -> dict:
    """Live games vs people count in games/wins/streak; solo games vs the computer in r["solo"]
    (owner 2026-10-06). One entry per game id: a second game-over screen of the same game
    (stage 2026-10-05 06:38 + 06:41) is not counted again."""
    r = record()
    if game_id and any(g.get("id") == game_id for g in r.get("log", [])):
        b = _bucket(r, solo)
        return {**r, **{k: b[k] for k in ("games", "wins", "streak")}, "duplicate": True}
    if r.get("open") and (game_id is None or r["open"].get("id") == game_id):
        r.pop("open", None)
    b = _bucket(r, solo)
    b["games"] += 1
    b["wins"] += int(bool(won))
    b["losses"] += int(not won)
    b["streak"] = b["streak"] + 1 if won else 0
    b["best_streak"] = max(b["best_streak"], b["streak"])
    r.setdefault("log", []).append({"id": game_id, "solo": bool(solo), "outcome": "win" if won else "loss",
                                    "place": place, "players": players, "vp": vp,
                                    "at": time.strftime("%Y-%m-%d %H:%M:%S")})
    _save_record(r)
    return {**r, **{k: b[k] for k in ("games", "wins", "streak")}}


# ---- classify --------------------------------------------------------------------------
_INJECTION = re.compile(
    r"(ignore|forget|disregard|override|bypass)\b.{0,40}\b(rules?|instructions?|prompt|previous|above|everything)|"
    r"system prompt|your (instructions|prompt|config|settings|source)|jailbreak|developer mode|\bDAN\b|"
    r"pretend (to be|you)|act as|roleplay|you are now|new rules|"
    r"\b(files?|folders?|director(y|ies)|paths?|repo(sitory)?|source code|codebase|terminal|shell|command|"
    r"sudo|admin|root|password|api ?key|access token|auth token|credentials?|env(ironment)?|\.env|ssh|script)\b|"
    r"\b(run|execute|eval|print|reveal|leak|dump|cat|ls)\b.{0,20}\b(code|this|that|command|file|it)\b|"
    r"[/\\~]\w|https?://|<\w+>|\{|\}|```", re.I)

# personal or off-limits probes that look like questions about marlow
_PERSONAL = re.compile(
    r"\b(owner|who (owns|runs|operates) you|real name|surname|last name|address|phone|email|"
    r"where (do )?you live|where are you|location|age|how old|salary|family|girlfriend|boyfriend|wife|husband|"
    r"other (projects?|work|games? you)|what else (do|does)|computer|machine|laptop|mac\b|server|ip\b|"
    r"model are you|gpt|llm|claude|openai|anthropic|chatgpt)\b", re.I)

_ABOUT = re.compile(
    r"\b(who are you|what are you|your name|are you (a )?(bot|ai|human|real|computer)|marlow|"
    r"who (made|built|created|wrote|programmed) you|made by|built by|company|maker|creator|website|site|"
    r"what (did|have) you learn|learned|learnt|how many games|your (strategy|training|knowledge|record)|"
    r"how do you (play|learn|decide|think)|you a bot)\b", re.I)

_GREETING = re.compile(r"^\W*(hi|hello|hey|yo|hiya|howdy|gg|glhf|gl|hf|good luck|have fun|good game|wp|"
                       r"well played|thanks|thank you|ty|bye|cya)\b[\s\w!.,]*$", re.I)

CATAN_TERMS = set("""catan robber knight knights barbarian barbarians city cities settlement settlements road roads
ship ships harbor harbors harbour port ports trade trades trading dice roll rolled vp vps victory points point card cards
progress commodity commodities paper coin cloth canal canals metropolis metropolises aqueduct longest army pirate gold
island islands wool sheep wheat grain ore lumber wood brick strategy rules rule discard seven improvement improvements
build building upgrade wall walls bank merchant fleet alchemist bishop spy monopoly hex hexes tile tiles number token
tokens desert sea seafarers scenario greater expansion opening setup placement resource resources production produce
politics science trade crane defence defense defender win winning chips 7 2:1 3:1 4:1 fortress""".split())


def classify(msg: str) -> str:
    """game_question | about_marlow | greeting | offtopic | injection"""
    m = (msg or "").strip()
    if not m:
        return "offtopic"
    if _INJECTION.search(m):
        return "injection"
    if _PERSONAL.search(m):
        return "offtopic"
    if _ABOUT.search(m):
        return "about_marlow"
    words = set(re.findall(r"[a-z0-9:]+", m.lower()))
    if words & CATAN_TERMS:
        return "game_question"
    if _GREETING.match(m):
        return "greeting"
    return "offtopic"


# ---- output filter ---------------------------------------------------------------------
_BAD = re.compile(r"[/\\~]\w|\w[/\\]|https?://|www\.|\b[\w.+-]+@[\w-]+\.\w+|\.(py|md|json|jsonl|sh|txt|log)\b|"
                  r"[{}<>`=$#|]|\bdef |\bimport |\w\(\)|\b(file|folder|directory|password|token|api)\b", re.I)
_vocab_cache: tuple[int, set] | None = None


def _vocab() -> set:
    """Every word marlow may say: words from the RAG chunks and the persona text."""
    global _vocab_cache
    st = rag.store()
    key = st.meta["version"]
    if _vocab_cache and _vocab_cache[0] == key:
        return _vocab_cache[1]
    words = set()
    for c in st.chunks:
        words |= set(re.findall(r"[a-z]+", (c["title"] + " " + c["text"]).lower()))
    words |= set(re.findall(r"[a-z]+", json.dumps(persona()).lower()))
    words |= {"from", "games", "game", "i", "ve", "learned", "so", "far", "s", "plus", "notes", "here",
              "my", "home", "is", "by", "a", "and", "the", "with", "for", "of"}
    _vocab_cache = (key, words)
    return words


def safe(reply: str) -> bool:
    """Hard output filter. False = do not send this text."""
    if not reply or len(reply) > MAX_LEN:
        return False
    p = persona()
    site = (p.get("site") or "").strip()
    probe = reply.replace(site, " ") if site else reply
    if _BAD.search(probe):
        return False
    vocab = _vocab()
    return all(w in vocab for w in re.findall(r"[a-z]+", probe.lower()))


# ---- answer ----------------------------------------------------------------------------
def deflection(i: int = 0) -> str:
    d = persona()["deflections"]
    return d[i % len(d)]


def _fit(sents: list[str], limit: int = MAX_LEN) -> str:
    out = ""
    for s in sents:
        s = s.strip()
        cand = (out + " " + s).strip()
        if len(cand) <= limit:
            out = cand
        elif not out:
            cut = s[:limit - 1].rsplit(" ", 1)[0].rstrip(",;:")
            out = cut + "."
        if len(out) > limit * 0.6:
            break
    return out


def _best_sentences(q: str, hits: list[dict], n: int = 2) -> list[str]:
    qt = set(rag._tokens(q))
    st = rag.store()
    N = max(len(st.chunks), 1)
    idf = {t: 1.0 / (1 + st.df.get(t, 0) / N * 10) for t in qt}
    scored = []
    for rank, h in enumerate(hits):
        for i, s in enumerate(h["sentences"]):
            ov = sum(idf[t] for t in qt & set(rag._tokens(s)))
            if ov > 0:
                scored.append((ov + 0.3 / (rank + 1) - 0.01 * i, s))
    scored.sort(key=lambda x: -x[0])
    return list(dict.fromkeys(s for _, s in scored))[:n]


def _about(msg: str) -> str:
    p, games = persona(), rag.stats()["games"]
    low = msg.lower()
    if re.search(r"record|how many games|wins|won|lost|losses|streak", low):
        return _fit([f"Record: {record_line()}."])
    if re.search(r"learn|games|training|knowledge|record|strategy|how do you", low):
        game_hits = [h for h in rag.search(msg + " game won built trade", 6) if h["source"] == "game"]
        lesson = game_hits[0]["sentences"][0] if game_hits else ""
        base = p["about_learned"].format(games=games)
        return _fit([base, lesson] if lesson else [base])
    if re.search(r"made|built|created|wrote|programmed|maker|creator|company|who.{0,10}behind", low):
        return p["about_maker"]
    if re.search(r"website|site|url|link", low):
        return p["about_site"]
    return p["about_name"].format(games=games)


# owner 2026-10-03: the chat called the game "Greater Catan" (a different scenario in the same
# box). The game we play is "The Great Canal": Catan Universe > Solo > Special Scenarios;
# scenario 5 of Treasures, Dragons & Adventurers (C&K + ships), 18 VP with 4 players, 21 with 3.
WHICH_GAME = re.compile(r"\b(what|which)\b.{0,25}\b(game|scenario|mode|map|version)\b|\bscenario\b\??$", re.I)

# owner 2026-10-09: "what version are you / which version do you run / marlow version?" asks for
# the bot's own version; WHICH_GAME also matches "version", so this is checked first and leaves
# any question naming the game, Catan or the scenario to the scenario line
VERSION_ASK = re.compile(r"\b(what|which)\b.{0,25}\bversion\b|\bversion\b.{0,20}\b(are you|you run|do you|running)\b"
                         r"|^\W*(your\s+|marlow'?s?\s+)?version\W*$|\byour\s+version\b", re.I)
GAME_WORDS = re.compile(r"\b(game|catan|scenario|map|mode)\b", re.I)


def version_line() -> str:
    import catanbot
    c = f" ({catanbot.COMMIT})" if catanbot.COMMIT else ""
    return f"I'm running marlow {catanbot.__version__}{c}."


WHERE_ARE_WE = re.compile(r"\b(what|which)\s+(turn|round)\b|\bwho(se|'?s| is)\s+turn\b|\bwhere are we\b"
                          r"|\bwho('?s| is)\s+(winning|ahead|leading)\b|\b(score|standings?)\b|\bhow many turns\b", re.I)


def scenario_line(mode, target: int | None = None) -> str:
    """`target` is gs.target (screen value, else 21 with 3 players / 18 with 4)."""
    return (f"This is The Great Canal, a Cities & Knights scenario: {target or mode.vp_target} VP to win. "
            f"Two active knights dig the canal; grain comes back when it is finished.")


def _other_scenario(sentence: str, mode) -> bool:
    """A sentence about another Catan Universe scenario (e.g. Greater Catan): marlow only plays
    The Great Canal, so the chat never quotes it."""
    from .rules import other_scenario
    return other_scenario(sentence) is not None


def answer(msg: str, gs=None) -> str:
    """One short reply (<= 200 chars), always on-topic or a deflection."""
    kind = classify(msg)
    p = persona()
    if kind != "injection" and VERSION_ASK.search(msg) and not GAME_WORDS.search(msg):
        return version_line()                      # our own fixed line: the vocab filter would drop the sha
    if kind != "injection" and (mode := getattr(gs, "mode", None)) is not None and WHICH_GAME.search(msg):
        return scenario_line(mode, getattr(gs, "target", None))
    if kind != "injection" and WHERE_ARE_WE.search(msg):
        # owner 2026-10-08: the chat knows every turn (data/game_logs/current.json)
        from . import game_log
        out = game_log.summary(game_log.current_status())
        if safe(out):
            return out[:200]
    if kind in ("offtopic", "injection"):
        return deflection(0 if kind == "offtopic" else 1)
    if kind == "greeting":
        out = p["greeting"]
    elif kind == "about_marlow":
        out = _about(msg)
    else:
        hits = rag.search(msg, 4)
        sents = _best_sentences(msg, hits, 4) if hits and hits[0]["score"] > 1.5 else []
        sents = [s for s in sents if safe(s) and not _other_scenario(s, getattr(gs, "mode", None))][:2]
        out = _fit(sents) if sents else p["no_answer"]
    return out if safe(out) else deflection()


# ---- chat session: rate limit, self-filter, line parsing -------------------------------
_LINE = re.compile(r"^\s*\[(?:Game|game)\]\s*\[([^\]]{1,40})\]\s*:?\s*(.*)$")


def parse_lines(texts: list[str]) -> list[tuple[str, str]]:
    """OCR lines (top to bottom) -> [(sender, message)]. A message starts with '[Game] ['; others continue it."""
    out: list[list[str]] = []
    for t in texts:
        m = _LINE.match(t)
        if m:
            out.append([m.group(1).strip(), m.group(2).strip()])
        elif out and t.strip():
            out[-1][1] = (out[-1][1] + " " + t.strip()).strip()
    return [(a, b) for a, b in out]


class Chat:
    def __init__(self, me: str = "marlow.systems", min_gap_s: float = MIN_GAP_S, clock=time.time):
        self.me = me.lower()
        self.min_gap = min_gap_s
        self.clock = clock
        self.last_reply = -1e9
        self.sent: list[str] = []
        self.seen: set[tuple[str, str]] = set()
        self.color: str | None = None        # our colour this game, set by the runner

    # owner 2026-10-02: marlow answers only messages naming it or its colour
    _COLOR_WORDS = {"magenta": ["magenta", "pink", "purple"], "purple": ["purple", "magenta", "pink"],
                    "pink": ["pink", "magenta", "purple"]}

    def addressed(self, msg: str) -> bool:
        low = msg.lower()
        if re.search(r"\bmarlow", low):
            return True
        if self.color:
            words = self._COLOR_WORDS.get(self.color.lower(), [self.color.lower()])
            return any(re.search(rf"\b{re.escape(w)}\b", low) for w in words)
        return False

    def is_me(self, sender: str, msg: str = "") -> bool:
        s = (sender or "").lower().strip()
        if s and (s == self.me or s in self.me or self.me.split(".")[0] == s.split(".")[0]):
            return True
        return any(msg.strip() and msg.strip()[:40] == x[:40] for x in self.sent)

    def respond(self, sender: str, msg: str, gs=None) -> str | None:
        if self.is_me(sender, msg) or not msg.strip() or not self.addressed(msg):
            return None
        now = self.clock()
        if now - self.last_reply < self.min_gap:
            return None
        # 2026-10-03: "marlow" is what addresses us, and it also matched _ABOUT, so every
        # question ("marlow how do knights work") got the who-am-I line. Classify the rest.
        q = re.sub(r"(?i)@?\bmarlow(\.systems)?\b[\s,:!?]*", " ", msg).strip()
        reply = answer(q if len(q) > 2 else msg, gs)
        self.last_reply = now
        self.sent = (self.sent + [reply])[-20:]
        return reply

    def new_messages(self, texts: list[str], prime: bool = False) -> list[tuple[str, str]]:
        """Messages not seen before. prime=True marks the current backlog as seen without returning it."""
        fresh = [m for m in parse_lines(texts) if m not in self.seen]
        self.seen.update(fresh)
        return [] if prime else fresh

    def intro(self) -> str:
        c = self.color
        t = persona()["intro"].format(color=f" ({c})" if c else "", or_color=f" or \"{c}\"" if c else "")
        self.sent.append(t)
        return t

    def intro_more(self) -> list[str]:
        """Owner 2026-10-02: after the intro, a couple of sentences on what marlow can do, how
        many games, its win streak and its rules toward the other players."""
        rec = record()
        lines = persona().get("intro_more", [])
        out = [t.format(record=record_line(rec), **{k: rec.get(k, 0) for k in ("games", "wins", "streak")})
               for t in ([lines] if isinstance(lines, str) else lines)]
        self.sent += out
        return out
