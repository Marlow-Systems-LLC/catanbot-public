"""Local retrieval store for marlow's in-game chat (BM25, pure python + numpy, no network).

Store layout (data/rag/):
  chunks.jsonl  one chunk per line: {id, source, title, text, sentences}
  index.json    BM25 stats: document frequencies, chunk lengths, avg length
  meta.json     version counter, games ingested, game keys already seen

Only whitelisted docs are ingested (DOC_GLOBS); never code, never other folders.
Game lessons are short factual sentences with in-game names only.

CLI:  python -m catanbot.rag ingest-docs | ingest-game <file> [winner] | search "<q>" | stats
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
STORE = ROOT / "data" / "rag"
DOC_NAMES = ("canal_playbook.md", "progress_cards.md", "progress_cards_playbook.md", "rules_research.md",
             "great_canal_rules_research.md", "great_canal_official_rules.md")   # the game we play
DOC_GLOB = "strategy_*.md"
K1, B = 1.5, 0.75

STOP = set("""a an the and or of to in on at for is are was were be been it its this that these those with as by
from if then than so but not no do does did can could should would will i you he she we they my your our their me
what which who whom how when where why there here about into over under up down out off just only also very
any all some each per vs""".split())

# sentences that talk about how the docs were made, people or tooling: never stored
_META = re.compile(r"\b(owner|task brief|brief|agent|research(ed)?|transcript|repo|github|python|script|file|folder|"
                   r"directory|code|api|claude|anthropic|email|password|machine|computer|laptop|todo|"
                   r"memory|prompt|source:|sources?\b|see also|wikipedia|rulebook above|steam|reddit|forum|bgg|"
                   r"boardgamegeek|cdn|pdf|http|www\.|\.com|\.md|\.py|\.json|~/|(?:^|\s)/\w|\w+/\w+/\w+)", re.I)


# implementation details of the bot (weights, model internals): not chat material
_IMPL = re.compile(r"(our model|the model|weight|bonus|heuristic|penalt|\bx\d|[+-]\d+\.\d|\bpips? weight|"
                   r"brain|runner|calib|ocr|unity|screen|click|pixel|mismatch|\bfix\b|implement|parameter|"
                   r"\w+_\w+|\[|\]|\{|\}|=|->|<-)", re.I)


# trailing "(owner, live 2026-10-09 21:12)"-style source tag on a heading
_DATE_TAG = re.compile(r"\s*\([^()]*\b20\d\d-\d\d-\d\d[^()]*\)\s*$")

def _tokens(s: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in STOP and len(t) > 1]


def clean_text(s: str) -> str:
    """Markdown -> plain prose. URLs, links, code spans and UNCONFIRMED markers removed."""
    s = re.sub(r"\[([^\]]+)\]\((?:https?://)?[^)]*\)", r"\1", s)        # [text](url) -> text
    s = re.sub(r"\(\s*(?:https?://|www\.)[^)]*\)", "", s)                # (url)
    s = re.sub(r"(?:https?://|www\.)\S+", "", s)
    s = re.sub(r"`[^`]*`", "", s)
    s = re.sub(r"\[?UNCONFIRMED\]?", " UNCONFIRMED ", s)             # sentence is dropped later
    s = re.sub(r"\[(?:S|S:|source|src)[^\]]{0,60}\]", "", s)           # citation tags like [S: TDE]
    s = re.sub(r"[*_#>|]+", " ", s)
    s = re.sub(r"\(\s*[;,]?\s*\)", "", s)
    s = re.sub(r"\s+([,.;:])", r"\1", s)
    return re.sub(r"\s+", " ", s).strip(" -;,")


def split_sentences(s: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", s)
    out = []
    for p in parts:
        p = p.strip()
        if len(p) < 12 or _META.search(p) or _IMPL.search(p) or "UNCONFIRMED" in p or len(_tokens(p)) < 3:
            continue
        out.append(p if p[-1] in ".!?" else p + ".")
    return out


def doc_paths() -> list[Path]:
    ps = [DOCS / n for n in DOC_NAMES] + sorted(DOCS.glob(DOC_GLOB))
    return [p for p in ps if p.is_file() and p.parent == DOCS and p.suffix == ".md"]


def chunk_markdown(text: str, source: str, max_sent: int = 3) -> list[dict]:
    chunks, title, buf, skip = [], "", [], False

    def flush():
        sents = []
        for para in buf:
            sents += split_sentences(clean_text(para))
        for i in range(0, len(sents), max_sent):
            part = sents[i:i + max_sent]
            chunks.append({"source": source, "title": clean_text(title), "sentences": part, "text": " ".join(part)})
        buf.clear()

    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            flush()
            # a dated tag such as "(owner, live 2026-10-09 21:12)" says where a lesson came from,
            # not what it is about: drop it so owner lessons are not skipped as meta
            title = _DATE_TAG.sub("", line.lstrip("# ").strip())
            skip = bool(_META.search(title) or _IMPL.search(title))
        elif skip or line.lstrip().startswith("|"):        # meta sections and tables are not chat prose
            continue
        elif line.strip().startswith("```"):
            continue
        elif line.strip():
            buf.append(line.strip())
        else:
            buf.append("")
    flush()
    return chunks


# ---- store ---------------------------------------------------------------------------
class Store:
    def __init__(self, root: Path = STORE):
        self.root = Path(root)
        self.chunks: list[dict] = []
        self.meta = {"version": 0, "games": 0, "game_keys": [], "docs": []}
        self.df: dict[str, int] = {}
        self.lens: list[int] = []
        self._load()

    # io
    def _load(self):
        cp, mp = self.root / "chunks.jsonl", self.root / "meta.json"
        if cp.exists():
            self.chunks = [json.loads(l) for l in cp.read_text().splitlines() if l.strip()]
        if mp.exists():
            self.meta.update(json.loads(mp.read_text()))
        self._reindex()

    def save(self):
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "chunks.jsonl").write_text("".join(json.dumps(c) + "\n" for c in self.chunks))
        n = len(self.lens)
        (self.root / "index.json").write_text(json.dumps(
            {"n": n, "avgdl": (sum(self.lens) / n) if n else 0.0, "df": self.df, "lens": self.lens}))
        (self.root / "meta.json").write_text(json.dumps(self.meta, indent=1))

    def _reindex(self):
        self._tf = [Counter(_tokens(c["title"] + " " + c["text"])) for c in self.chunks]
        self.lens = [sum(t.values()) for t in self._tf]
        df = Counter()
        for t in self._tf:
            df.update(t.keys())
        self.df = dict(df)
        for i, c in enumerate(self.chunks):
            c["id"] = i

    # ingest
    def ingest_docs(self) -> int:
        self.chunks = [c for c in self.chunks if c["source"] == "game"]
        names = []
        for p in doc_paths():
            self.chunks += chunk_markdown(p.read_text(), p.stem)
            names.append(p.name)
        self.meta["docs"] = names
        self.meta["version"] += 1
        self._reindex()
        self.save()
        return sum(1 for c in self.chunks if c["source"] != "game")

    def add_game(self, sentences: list[str], key: str) -> bool:
        if key in self.meta["game_keys"]:
            return False
        self.meta["games"] += 1
        self.meta["game_keys"].append(key)
        self.meta["version"] += 1
        title = f"game {self.meta['games']} lessons"
        for i in range(0, len(sentences), 3):
            part = sentences[i:i + 3]
            self.chunks.append({"source": "game", "title": title, "sentences": part, "text": " ".join(part)})
        self._reindex()
        self.save()
        return True

    # query
    def search(self, q: str, k: int = 3) -> list[dict]:
        qt = _tokens(q)
        n = len(self.chunks)
        if not qt or not n:
            return []
        import numpy as np
        avg = (sum(self.lens) / n) or 1.0
        lens = np.asarray(self.lens, float)
        score = np.zeros(n)
        for t in set(qt):
            d = self.df.get(t, 0)
            if not d:
                continue
            idf = math.log(1 + (n - d + 0.5) / (d + 0.5))
            tf = np.asarray([x.get(t, 0) for x in self._tf], float)
            score += idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * lens / avg))
        order = np.argsort(-score)[:k]
        return [dict(self.chunks[i], score=float(score[i])) for i in order if score[i] > 0]

    def stats(self) -> dict:
        return {"version": self.meta["version"], "games": self.meta["games"], "chunks": len(self.chunks),
                "doc_chunks": sum(1 for c in self.chunks if c["source"] != "game")}


_STORE: Store | None = None


def store() -> Store:
    global _STORE
    if _STORE is None:
        _STORE = Store()
    return _STORE


def ingest_docs() -> int:
    return store().ingest_docs()


def search(q: str, k: int = 3) -> list[dict]:
    return store().search(q, k)


def stats() -> dict:
    return store().stats()


# ---- per-game lessons ----------------------------------------------------------------
_NAME = re.compile(r"[^A-Za-z0-9._ -]")


def _name(s) -> str | None:
    if not s:
        return None
    s = _NAME.sub("", str(s)).strip()[:20]
    return s or None


def _read_lines(log_path_or_lines) -> tuple[list[str], dict]:
    extra = {}
    if isinstance(log_path_or_lines, (list, tuple)):
        return [str(x) for x in log_path_or_lines], extra
    p = Path(log_path_or_lines)
    raw = p.read_text(errors="replace")
    if p.suffix == ".json":
        d = json.loads(raw)
        extra = {k: d.get(k) for k in ("winner", "mode", "vp", "improvements", "me")}
        return [str(x) for x in d.get("log", [])], extra
    return raw.splitlines(), extra


def game_lessons(lines: list[str], result=None, mode=None, me: str = "marlow") -> list[str]:
    """Short factual sentences. Only in-game names, resource words and counts; no paths."""
    winner = result.get("winner") if isinstance(result, dict) else result
    mode = (result.get("mode") if isinstance(result, dict) else None) or mode
    winner = _name(winner)
    out = []
    mode_s = f" of {str(mode).replace('_', ' ')}" if mode and re.fullmatch(r"[a-z_ ]{2,30}", str(mode)) else ""
    if winner:
        won = me.lower() in winner.lower()
        out.append(f"In a game{mode_s}, {'I won' if won else winner + ' won'}.")
    built = Counter()
    for l in lines:
        m = re.search(r"click (?:\[dry\] )?(?:build|confirm|setup) (settlement|city|road|ship|knight|wall|"
                      r"city_wall|improvement|bridge|development_card)\b", l)
        if m:
            built[m.group(1).replace("_", " ")] += 1
    if built:
        out.append("I built " + ", ".join(f"{n} {k}{'s' if n > 1 and not k.endswith('s') else ''}"
                                          for k, n in built.most_common()) + ".")
    acc, dec = [], Counter()
    for l in lines:
        m = re.search(r"offer from ([^:]{1,30}): .*-> give (\{[^}]*\}) get (\{[^}]*\}): (ACCEPT|decline)", l)
        if not m:
            continue
        who = _name(m.group(1)) or "a player"
        res = lambda s: ", ".join(f"{v} {k}" for k, v in re.findall(r"'([a-z]+)': (\d+)", s)) or "nothing"
        give, get = res(m.group(2)), res(m.group(3))
        if m.group(4) == "ACCEPT":
            acc.append(f"I accepted a trade with {who}: gave {give} for {get}.")
        else:
            dec[(who, give, get)] += 1
    out += list(dict.fromkeys(acc))[:6]
    for (who, give, get), n in dec.most_common(4):
        out.append(f"I declined {'a trade' if n == 1 else str(n) + ' trades'} from {who} asking {give} for {get}.")
    r = result if isinstance(result, dict) else {}
    out += _score_lessons(r, winner, me)
    out += _play_lessons(lines)
    if len(out) <= 1:
        out.append("I played a full game and recorded the result.")
    return out


def _score_lessons(r: dict, winner, me: str) -> list[str]:
    """Owner 2026-10-06: game lessons were trades only. Add the final score, the VP gap and the
    improvement levels reached."""
    out = []
    vp = {(_name(k) or ""): v for k, v in (r.get("vp") or {}).items() if isinstance(v, (int, float))}
    vp.pop("", None)
    if vp:
        out.append("Final score: " + ", ".join(f"{k} {v}" for k, v in sorted(vp.items(), key=lambda x: -x[1])) + ".")
        mine = next((v for k, v in vp.items() if me.lower()[:6] in k.lower()), None)
        top = max(vp.values())
        if mine is not None and winner and me.lower() not in winner.lower():
            out.append(f"I finished {top - mine} VP behind the winner.")
    lv = r.get("improvements")
    if isinstance(lv, dict) and lv:
        out.append("My city improvements ended at " + ", ".join(f"{t} {int(lv.get(t, 0))}" for t in
                                                                  ("trade", "politics", "science")) + ".")
        if max(int(v) for v in lv.values()) < 3:
            out.append("No improvement track reached level 3, so I had no trading house, fortress or aqueduct.")
    return out


def _play_lessons(lines: list[str]) -> list[str]:
    out = []
    failed = Counter(m.group(1) for l in lines if (m := re.search(r"^(\w+) at \S.* already failed this turn", l)))
    if failed:
        out.append("Actions that kept failing: " + ", ".join(f"{k} {n} time{'s' if n > 1 else ''}" for k, n in failed.most_common(4)) + ".")
    bank = Counter()
    for l in lines:
        m = re.search(r"bank trade landed.*->\{'([a-z]+)'", l)
        if m:
            bank[m.group(1)] += 1
    if bank:
        n = sum(bank.values())
        out.append(f"I made {n} bank trade{'s' if n > 1 else ''}, mostly for {bank.most_common(1)[0][0]}.")
    imp = Counter(m.group(1) for l in lines if (m := re.search(r"improve (trade|politics|science): level", l)))
    if imp:
        out.append("Improvements that landed: " + ", ".join(f"{k} {n}" for k, n in imp.most_common()) + ".")
    leader = [l for l in lines if l.startswith("threat before our turn: ")]
    if leader:
        m = re.match(r"threat before our turn: (.{1,30}?) \(vp (\d+)/(\d+)", leader[-1])
        if m and _name(m.group(1)):
            out.append(f"Late in the game {_name(m.group(1))} led with {m.group(2)} of {m.group(3)} VP.")
    return out


def ingest_game(log_path_or_lines, result=None) -> dict:
    lines, extra = _read_lines(log_path_or_lines)
    if result is None:
        result = extra
    elif isinstance(result, str) or result is None:
        result = {"winner": result, "mode": extra.get("mode")}
    sents = game_lessons(lines, result)
    key = hashlib.sha1(("\n".join(lines) + json.dumps(result, sort_keys=True, default=str)).encode()).hexdigest()
    store().add_game(sents, key)
    return stats()


def main(argv=None):
    a = list(sys.argv[1:] if argv is None else argv)
    if not a:
        print(__doc__)
        return 1
    if a[0] == "ingest-docs":
        print(f"doc chunks: {ingest_docs()}; {stats()}")
    elif a[0] == "ingest-game" and len(a) >= 2:
        print(ingest_game(a[1], a[2] if len(a) > 2 else None))
    elif a[0] == "search" and len(a) >= 2:
        for h in search(" ".join(a[1:]), 3):
            print(round(h["score"], 2), h["title"], "|", h["text"][:160])
    elif a[0] == "stats":
        print(stats())
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
