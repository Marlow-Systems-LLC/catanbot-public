"""FaceTime voice add-on: the pure parts (no audio devices, no network, no macOS).

Wake rule, energy VAD, sentence split, WAV/PCM helpers, the multipart body,
call-title -> caller mapping, the local allow-list and AX-tree classification.
Unit-tested in tests/test_facetime*.py with synthetic data; imports only numpy
and the standard library, so the tests run on Linux.
"""
from __future__ import annotations

import io
import json
import re
import time
import uuid
import wave
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

VERSION = "catanbot-facetime-0.1.0"
RATE = 16000  # pipeline rate: VAD, STT
FRAME = 480   # 30 ms at 16 kHz

# ---------------------------------------------------------------------------
# Wake rule (speech). The bot's name comes from config ("name", default
# "marlow"). For "marlow", Whisper writes Marlow / Marlo / Marlowe / "Mar low"
# and Romanian voices Marlău / Marlau / Mărlou; any other name is matched as
# written (case-insensitive, optional "'s"). People open with hey/hi/ok/yo or,
# in Romanian, bă/mă/hai/salut/bună/auzi/zi. Addressed when:
#   - the name opens the utterance (after an optional opener), or
#   - the name is anywhere AND the utterance is a question ("?"), or
#   - the name closes the utterance (a vocative: "your turn, Marlow"), or
#   - it starts inside the follow-up window after the bot last spoke.
# ---------------------------------------------------------------------------
MARLOW = r"(?:m[aă]r[\s-]?l(?:owe?|ou|ău|au|o)(?:'s)?)"
OPENERS = r"(?:hey|hi|hello|ok|okay|yo|so|and|but|well|wait|listen|um|uh|ba|bă|ma|mă|hai|salut|buna|bună|auzi|zi|spune|deci|și|si|păi|pai|ia|ascultă|asculta)"


def name_pattern(name: str = "marlow") -> str:
    """Regex for the bot's spoken name."""
    n = (name or "marlow").strip()
    if n.casefold() in ("marlow", "marlowe", "marlo"):
        return MARLOW
    parts = [re.escape(p) for p in n.split()]
    return r"(?:" + r"[\s-]?".join(parts) + r"(?:'s)?)"


NAME = MARLOW
_NEVER = re.compile(r"(?!x)x")
_NAME_AT_START = _NAME_AT_END = _NAME_ANYWHERE = _GOODBYE = _LEAVE = _DM_LEAVE = _NEVER


def set_name(name: str = "marlow") -> str:
    """(Re)build the wake regexes for the configured name; returns the pattern."""
    global NAME, _NAME_AT_START, _NAME_AT_END, _NAME_ANYWHERE, _GOODBYE, _LEAVE, _DM_LEAVE
    NAME = name_pattern(name)
    _NAME_AT_START = re.compile(rf"^\W*(?:{OPENERS}[\s,.!]+){{0,2}}{NAME}\b", re.IGNORECASE)
    # The name closing the utterance is a vocative ("ce zici, Marlow"): Whisper often drops the "?".
    _NAME_AT_END = re.compile(rf"\b{NAME}\W*$", re.IGNORECASE)
    _NAME_ANYWHERE = re.compile(rf"\b{NAME}\b", re.IGNORECASE)
    _GOODBYE = re.compile(rf"\b(?:bye|goodbye|thanks|thank you|mulțumesc|multumesc|pa)\b[\s,]*{NAME}\b|\b{NAME}\b[\s,]*(?:bye|goodbye|leave(?: the call)?|hang up|you can go)\b", re.IGNORECASE)
    _LEAVE = re.compile(rf"\b{NAME}\b[\s,]*(?:please\s+)?(?:leave(?: the call)?|hang up|drop off)\b", re.IGNORECASE)
    # A 1:1 call: the caller is talking to the bot, so a farewell that is the WHOLE
    # utterance ends the call without the name: "hang up", "goodbye", "bye Marlow",
    # "thanks, bye", "can you hang up?", "end the call", "good night". Inside a
    # sentence ("say goodbye to grandma") it stays a turn. A lone "Bye." too: in a
    # 1:1 call is_junk lets it through (in a group _JUNK still drops it).
    dm_open = rf"(?:(?:ok|okay|alright|all right|well|so|thanks|thank you|{NAME})[\s,.!]+)*"
    dm_act = r"(?:(?:can|could|would|will) you\s+)?(?:please\s+)?(?:hang up(?: the call)?|end the call|good\s?bye|bye(?:[\s-]bye)?|good\s?night)"
    dm_tail = rf"(?:[\s,]+(?:now|then|for now|please))?(?:[\s,]+{NAME})?(?:[\s,.!]+good\s?night)?"
    _DM_LEAVE = re.compile(rf"^\W*{dm_open}{dm_act}{dm_tail}\W*$", re.IGNORECASE)
    return NAME


set_name("marlow")


_QUESTION = re.compile(r"\?\s*$|^\W*(?:and\s+|so\s+)?(?:who|what|when|where|why|how|which|is|are|can|could|do|does|did|should|will|would|cine|ce|cât|cat|când|cand|unde|de ce|cum)\b", re.IGNORECASE)


def wake(text: str, now: float, follow_until: float, group: bool = True) -> Optional[str]:
    """'direct' | 'mention' | 'follow' | None for one transcribed utterance.
    In a group, a follow-up must be a question: table talk right after
    the bot spoke ("pass me the sheep") is not for it. In a 1:1 call (group=False)
    the caller is talking to the bot: every utterance is addressed (the caller
    filters junk with is_junk first)."""
    t = (text or "").strip()
    if not t:
        return None
    if _NAME_AT_START.search(t):
        return "direct"
    if _NAME_ANYWHERE.search(t) and ("?" in t or _QUESTION.search(t) or _NAME_AT_END.search(t)):
        return "mention"
    if now <= follow_until and (not group or _QUESTION.search(t)):
        return "follow"
    if not group:
        return "direct"
    return None


_RO_WORDS = re.compile(r"\b(?:și|si|este|sunt|nu|da|că|ca|pe|cu|în|in|la|de|ce|cum|care|mai|foarte|acum|aici|tu|eu|noi|voi|ai|am|are|vrei|poți|poti|bine|mulțumesc|multumesc|salut|bună|buna)\b", re.IGNORECASE)


def looks_romanian(text: str) -> bool:
    """A reply to speak with the Romanian voice (an English-only TTS voice would mangle it): Romanian
    letters, or several common Romanian words (the brain answers in the caller's language)."""
    t = text or ""
    if re.search(r"[ăâîșțşţĂÂÎȘȚŞŢ]", t):
        return True
    words = re.findall(r"\w+", t)
    return len(words) >= 3 and len(_RO_WORDS.findall(t)) >= max(2, len(words) // 4)


def near_name(text: str) -> bool:
    """A word like the name that the wake rule did not take (logged as a flag, never the words)."""
    return bool(re.search(r"\bm[aă]r\w*l|\bmar\s", text or "", re.IGNORECASE))


def is_goodbye(text: str) -> bool:
    return bool(_GOODBYE.search(text or ""))


def wants_leave(text: str, group: bool = True) -> bool:
    """The bot should hang up. In a group only with the name ("Marlow, hang up");
    in a 1:1 call also a bare farewell (_DM_LEAVE)."""
    t = text or ""
    return bool(_LEAVE.search(t) or (not group and _DM_LEAVE.search(t)))


# Whisper hallucinations on near-silence / noise: never a turn.
_JUNK = re.compile(r"^\W*(?:thanks? (?:you )?for watching|subtitles by|you|bye|\.+|music|\[[^\]]*\])\W*$", re.IGNORECASE)


def is_junk(text: str, group: bool = True) -> bool:
    """In a 1:1 call a lone farewell ("Bye.") is not junk: the caller is talking to
    the bot, and it ends the call as "goodbye" does (wants_leave, _DM_LEAVE)."""
    t = (text or "").strip()
    if not group and _DM_LEAVE.search(t):
        return False
    return len(t) < 2 or bool(_JUNK.match(t))


# ---------------------------------------------------------------------------
# Energy VAD: the satellite's gate (max(noise*2.5, 250) on int16 RMS) with a
# shorter tail. Feed 30 ms frames at 16 kHz; get whole utterances back.
# ---------------------------------------------------------------------------
@dataclass
class Vad:
    tail_s: float = 0.7
    min_speech_s: float = 0.35
    max_utt_s: float = 15.0
    floor: float = 250.0
    k: float = 2.5
    pre_s: float = 0.3
    noise: float = 100.0
    _buf: List[np.ndarray] = field(default_factory=list)
    _pre: List[np.ndarray] = field(default_factory=list)
    _speech: int = 0
    _silence: int = 0
    _active: bool = False

    @property
    def active(self) -> bool:
        return self._active

    def threshold(self) -> float:
        return max(self.noise * self.k, self.floor)

    def feed(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """One int16 frame in; a finished utterance (int16) out, or None."""
        rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2))) if len(frame) else 0.0
        loud = rms >= self.threshold()
        fs = len(frame) / RATE
        if not self._active:
            if not loud:
                # Slow noise-floor tracker, only while nobody talks.
                self.noise = 0.95 * self.noise + 0.05 * rms
                self._pre.append(frame)
                while len(self._pre) * fs > self.pre_s:
                    self._pre.pop(0)
                return None
            self._active = True
            self._buf = list(self._pre)
            self._pre = []
            self._speech = 0
            self._silence = 0
        self._buf.append(frame)
        if loud:
            self._speech += 1
            self._silence = 0
        else:
            self._silence += 1
        dur = sum(len(b) for b in self._buf) / RATE
        if self._silence * fs >= self.tail_s or dur >= self.max_utt_s:
            utt = np.concatenate(self._buf)
            speech_s = self._speech * fs
            self._active = False
            self._buf = []
            if speech_s < self.min_speech_s:
                return None
            # Trim the trailing silence (keep 0.2 s).
            keep = len(utt) - max(0, int((self._silence * fs - 0.2) * RATE))
            return utt[:keep]
        return None

    def reset(self) -> None:
        self._buf, self._pre, self._active, self._speech, self._silence = [], [], False, 0, 0


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------
def resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    """Linear resample of mono int16 (good enough for speech in a call)."""
    if src == dst or len(x) == 0:
        return x.astype(np.int16)
    n = int(round(len(x) * dst / src))
    t = np.linspace(0, len(x) - 1, n)
    return np.interp(t, np.arange(len(x)), x.astype(np.float32)).astype(np.int16)


def to_mono_int16(block: np.ndarray) -> np.ndarray:
    """(frames, channels) float32 or int16 from a device -> mono int16."""
    a = block
    if a.ndim == 2:
        a = a.astype(np.float32).mean(axis=1)
    if a.dtype != np.int16:
        a = np.clip(a * 32767.0, -32768, 32767) if np.abs(a).max(initial=0) <= 1.5 else a
        a = a.astype(np.int16)
    return a


def mix_live(block: np.ndarray) -> np.ndarray:
    """(frames, channels) -> mono, averaging only the channels that carry
    signal. FaceTime may write its voice to one channel of BlackHole 16ch; a
    plain mean with a silent neighbour halves it (-6 dB) under the VAD gate."""
    if block.ndim != 2:
        return block
    if block.shape[1] < 2:
        return block[:, 0]
    live = np.abs(block).max(axis=0) > 0
    if not live.any():
        return block[:, 0]
    return block[:, live].astype(np.float32).mean(axis=1).astype(block.dtype)


def wav_bytes(pcm: np.ndarray, rate: int = RATE) -> bytes:
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.astype(np.int16).tobytes())
    return b.getvalue()


def read_wav(data: bytes) -> Tuple[np.ndarray, int]:
    """WAV bytes -> (mono int16, rate). 16-bit PCM only (what speech servers return)."""
    with wave.open(io.BytesIO(data), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError("only 16-bit WAV")
        ch, rate = w.getnchannels(), w.getframerate()
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    if ch > 1:
        a = a.reshape(-1, ch).astype(np.float32).mean(axis=1).astype(np.int16)
    return a, rate


def multipart(fields: Dict[str, str], file_field: str, filename: str, data: bytes, mime: str = "audio/wav") -> Tuple[bytes, str]:
    boundary = "----catanbot" + uuid.uuid4().hex
    out = io.BytesIO()
    for k, v in fields.items():
        out.write(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    out.write(f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\nContent-Type: {mime}\r\n\r\n'.encode())
    out.write(data)
    out.write(f"\r\n--{boundary}--\r\n".encode())
    return out.getvalue(), f"multipart/form-data; boundary={boundary}"


# ---------------------------------------------------------------------------
# Speech shaping
# ---------------------------------------------------------------------------
_SENT = re.compile(r"(?<=[.!?…])\s+(?=[^\s])")


def sentences(text: str, max_chars: int = 220) -> List[str]:
    """Split a reply into speakable chunks: TTS starts on the first sentence."""
    t = re.sub(r"\s+", " ", (text or "")).strip()
    if not t:
        return []
    out: List[str] = []
    for s in _SENT.split(t):
        while len(s) > max_chars:
            cut = s.rfind(",", 0, max_chars)
            cut = cut if cut > 40 else s.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            out.append(s[: cut + 1].strip())
            s = s[cut + 1 :].strip()
        if s:
            out.append(s)
    return out


# ---------------------------------------------------------------------------
# Call -> caller. macOS 27 shows no group name for a group call: the banner
# says "To +1 (415) 555-0100, Ana Pop, & bo@x.com" and the Messages window is
# titled by the same participant list. A 1:1 call shows only the caller's
# contact name, so it maps only through the configured callers
# ({"Ana Pop": "+15555550100"}), never a guess.
# ---------------------------------------------------------------------------
_MARKS = re.compile("[\u200e\u200f\u202a-\u202e\u2066-\u2069\ufe0f]")


def norm_title(s: str) -> str:
    return re.sub(r"\s+", " ", _MARKS.sub("", s or "")).strip().casefold()


def norm_handle(s: str) -> Optional[str]:
    """+digits for a phone, lowercase for an email, None for a contact name."""
    t = _MARKS.sub("", s or "").strip()
    if re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", t):
        return t.lower()
    if re.fullmatch(r"\+?[\d\s().\-\u2010-\u2015]{7,}", t):
        d = re.sub(r"\D", "", t)
        if len(d) == 10 and not t.startswith("+"):
            d = "1" + d  # US number shown without country code
        return "+" + d if len(d) >= 7 else None
    return None


def parse_people(s: str) -> List[str]:
    """"To A, B, & C" / "A, B & C" -> ["A", "B", "C"]."""
    t = re.sub(r"\s+", " ", _MARKS.sub("", s or "")).strip()
    t = re.sub(r"^to ", "", t, flags=re.IGNORECASE)
    return [x.strip() for x in re.split(r"\s*(?:,\s*&|,|&)\s*", t) if x.strip()]


def ring_key(people: List[str]) -> str:
    """Identifies a call across polls: its handles plus how many names. Contact
    names vary by surface ("Ana Pop" on the banner, "Ana" in Messages)."""
    hs = sorted(h for h in (norm_handle(p) for p in people) if h)
    if len(people) == 1 and not hs:
        hs = [norm_title(people[0])]  # a 1:1 call: the caller's name is all there is
    return f"{len(people)}:" + "|".join(hs)


def group_participants(imsg_chats_jsonl: str) -> List[Tuple[str, List[str]]]:
    """`imsg chats --json` lines -> [(chat_id:<n>, [normalized handles])] for groups."""
    out: List[Tuple[str, List[str]]] = []
    for line in (imsg_chats_jsonl or "").splitlines():
        try:
            c = json.loads(line)
        except ValueError:
            continue
        parts = c.get("participants") or []
        if c.get("is_group") and c.get("id") is not None and parts:
            out.append((f"chat_id:{c['id']}", [norm_handle(str(x)) or str(x).strip().lower() for x in parts]))
    return out


def parse_callers(spec: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for part in (spec or "").split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            if k.strip() and v.strip():
                out[norm_title(k)] = norm_handle(v) or v.strip()
    return out


def resolve_call(people: List[str], groups: List[Tuple[str, List[str]]], callers: Dict[str, str]) -> List[Tuple[str, str]]:
    """Candidate (conversation id, kind)s for a ringing call's participant list;
    [] = let it ring. Several iMessage threads can have the same members (an old
    duplicate thread), and a FaceTime call belongs to the member set, not to a
    thread. That holds only for identical member sets: if groups with
    different members fit (a contact name could be either), it's []. Names-only
    lists never match: no guesses."""
    if not people:
        return []
    if len(people) == 1:
        h = callers.get(norm_title(people[0]))
        return [(h, "dm")] if h else []
    handles: List[str] = []
    for p in people:
        h = norm_handle(p) or callers.get(norm_title(p))
        if h:
            handles.append(h)
    if not handles or len(set(handles)) != len(handles):
        return []
    fits = [(cid, parts) for cid, parts in groups if len(parts) == len(people) and all(h in parts for h in handles)]
    if len({tuple(sorted(parts)) for _, parts in fits}) > 1:
        return []  # different member sets fit: a contact name is not evidence of which
    return [(cid, "group") for cid, _ in fits]


def new_call_id() -> str:
    return time.strftime("ft-%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]


HOLDING_LINES = ["One moment.", "Still thinking, one second.", "Bear with me."]


# ---------------------------------------------------------------------------
# Call state from facetime-ax.js `detect`. What macOS 27 shows for an
# incoming group call (seen 2026-10-03):
#   NotificationCenter: texts "<caller>", "To <participants>", "Join FaceTime
#     Call"; its buttons have no names, so the banner is never pressed.
#   Messages: the chat's window, titled by the participant list, carries an
#     AXButton described "Join FaceTime" (toolbar) and one described "Join"
#     (the call bubble). The bot presses one of those, in that window only.
# Exact labels only: "FaceTime", "FaceTime Audio" or "Call" would place a call.
# ---------------------------------------------------------------------------
JOIN_BUTTONS = ("Join FaceTime", "Join", "Accept", "Answer")
LEAVE_BUTTONS = ("Leave", "End")


_INCOMING = re.compile(r"^(?:incoming )?facetime(?: video| audio)?(?: call)?$")
_NOISE = re.compile(
    r"^(?:(?:incoming |group )?facetime(?: video| audio)?(?: call)?|accept|answer|decline|ignore|remind me|message"
    r"|now|just now|notification center|\d+:\d+.*|today|yesterday|monday|tuesday|wednesday|thursday|friday|saturday|sunday"
    r"|\d+\s*(?:s|sec|secs|seconds?|m|min|mins|minutes?|h|hr|hrs|hours?|d|days?|w|wk|wks|weeks?)\s*ago)$"
)


@dataclass
class CallState:
    state: str  # "ringing" | "in-call" | "idle"
    people: List[str] = field(default_factory=list)  # the call's participant list, as shown
    join: Optional[str] = None  # which button answers it (None: seen, but nothing safe to press)
    window: Optional[str] = None  # the window that button is in
    caller: Optional[str] = None  # a 1:1 ring seen by facetime-nc: press join through it, for this caller only


def classify(detect: dict) -> CallState:
    procs = (detect or {}).get("procs", {})
    nc = procs.get("NotificationCenter", {})
    texts = [t.strip() for t in nc.get("texts", []) if t and t.strip()]
    banner = any(re.fullmatch(r"join facetime call", norm_title(t)) for t in texts)
    banner_people = next((parse_people(t) for t in texts if norm_title(t).startswith("to ")), [])
    for w in procs.get("Messages", {}).get("wins", []):
        btns = w.get("buttons", [])
        # The bare "Join" (the call bubble) only alongside the toolbar's "Join
        # FaceTime" or the call banner: other Messages invites may say "Join" too.
        join = "Join FaceTime" if "Join FaceTime" in btns else ("Join" if "Join" in btns and banner else None)
        people = parse_people(w.get("title", ""))
        if join and people:
            if banner_people and ring_key(banner_people) != ring_key(people):
                continue  # the open chat is not the one ringing
            return CallState("ringing", people, join, w.get("title"))
    if banner:
        return CallState("ringing", banner_people, None)
    # A 1:1 call (unconfirmed on macOS 27): the caller is read ONLY from the
    # banner that carries the incoming-FaceTime line, and only when exactly one
    # other text in it is not noise. Anything less certain rings with no people
    # (never guessed). Accept/Answer must be named in that banner and in no
    # other one, since click() presses the first match. Not "Missed ...".
    if any(_INCOMING.match(norm_title(t)) for t in texts):
        groups = [g for g in nc.get("groups", []) if isinstance(g, dict)]
        rings = [g for g in groups if any(_INCOMING.match(norm_title(t)) for t in g.get("texts", []) if t)]
        if len(rings) != 1:
            return CallState("ringing", [], None)
        g = rings[0]
        names = [t.strip() for t in g.get("texts", []) if t and t.strip() and not _NOISE.match(norm_title(t))]
        if len(names) != 1:
            return CallState("ringing", [], None)
        answers = [b for b in ("Accept", "Answer") if b in g.get("buttons", [])]
        elsewhere = any(b in o.get("buttons", []) for o in groups if o is not g for b in ("Accept", "Answer"))
        return CallState("ringing", names, answers[0] if answers and not elsewhere else None)
    for pn in ("FaceTime", "Messages"):
        for w in procs.get(pn, {}).get("wins", []):
            if any(b in w.get("buttons", []) for b in LEAVE_BUTTONS):
                return CallState("in-call", parse_people(w.get("title", "")))
    return CallState("idle")


# ---------------------------------------------------------------------------
# The 1:1 ring on macOS 27, from facetime-nc (native AX: System Events sees
# only empty groups there). Captured 2026-10-04 (name anonymised):
#   AXGroup id=FACETIME_NOTIFICATION
#     AXGenericElement desc="<U+202A>Ana Pop<U+202C>, FaceTime<U+00A0>Audio"
#     AXButton desc="Live Voicemail" | "Decline with Message" | "Decline" | "Answer"
# `facetime-nc detect` -> {"ok": true, "groups": [{"callers": [...], "buttons": [...]}]}.
# Only Answer (or Accept) is ever pressed; Live Voicemail and Decline never are.
# ---------------------------------------------------------------------------
NC_ANSWER = ("Answer", "Accept")
_CALL_SUFFIX = re.compile(r", FaceTime (?:Audio|Video)$")


def _flat(s: str) -> str:
    return re.sub(r"\s+", " ", _MARKS.sub("", s or "")).strip()


def caller_name(label: str) -> str:
    """Banner label -> caller: bidi marks out, whitespace runs (macOS writes a
    no-break space) to one space, the ", FaceTime Audio/Video" suffix off.
    Same rule as callerName() in facetime-nc.swift."""
    return _CALL_SUFFIX.sub("", _flat(label)).strip()


def classify_native(nc: dict) -> Optional[CallState]:
    """None: facetime-nc failed or shows no FaceTime banner (the JXA path decides).
    Otherwise a 1:1 ring. It names the caller only when there is exactly one
    FaceTime banner, with exactly one caller label, and that label reads
    "<name>, FaceTime Audio|Video"; it carries a button to press only when
    exactly one Answer (else Accept) is in that banner. Anything less certain
    rings with no people: never guessed."""
    if not isinstance(nc, dict) or nc.get("ok") is not True:
        return None
    groups = [g for g in nc.get("groups") or [] if isinstance(g, dict)]
    if not groups:
        return None
    unsure = CallState("ringing", [], None)
    if len(groups) != 1:
        return unsure
    labels = [x for x in groups[0].get("callers") or [] if isinstance(x, str) and x.strip()]
    if len(labels) != 1 or not _CALL_SUFFIX.search(_flat(labels[0])):
        return unsure  # not "<name>, FaceTime Audio": a missed call, a group ring, something else
    name = caller_name(labels[0])
    if not name or re.match(r"(?i)to\s", name):
        return unsure
    buttons = [b for b in groups[0].get("buttons") or [] if isinstance(b, str)]
    join = next((b for b in NC_ANSWER if buttons.count(b) == 1), None)
    return CallState("ringing", [name], join, None, name if join else None)


# ---------------------------------------------------------------------------
# Is the call still up? `facetime-nc incall` (native AX + CoreAudio, no JXA):
#   {"ok": true, "incall": <an End/Leave button in FaceTime / InCallService /
#    Messages>, "truncated": <walk hit its budget>, "mic": <FaceTime's
#    microphone device (BlackHole 2ch) is running somewhere, or null>}
# The mic: during a call FaceTime holds its input device open; the bot opens that
# device only while it speaks, so "running" while the bot is silent is the call.
# ---------------------------------------------------------------------------
def incall_state(d: dict, playing: bool = False) -> Optional[bool]:
    """True = the call is up; False = a complete look found no call;
    None = unknown (a failed or partial check proves nothing)."""
    if not isinstance(d, dict) or d.get("ok") is not True:
        return None
    if d.get("incall") is True:
        return True
    mic = d.get("mic")
    if mic is True:
        return None if playing else True  # while the bot plays, the device is busy anyway
    if d.get("truncated"):
        return None
    return False


# ---------------------------------------------------------------------------
# Local allow-list (stands in for a server's approval). A call is answered
# only when:
#   - 1:1: the caller's shown name is a key of "callers" (or the shown text is
#     a handle that is a value of it);
#   - group: every participant shown resolves to a configured handle (a
#     handle in the callers list, or a caller name), or the set fits one of
#     the configured "groups".
# Anything else rings out. Never guessed, never fuzzy.
# ---------------------------------------------------------------------------
def _h(v: str) -> str:
    return norm_handle(v) or v.strip().lower()


def callers_map(callers: Dict[str, str]) -> Dict[str, str]:
    """Config {"Ana Pop": "+1 555 555 0100"} -> {"ana pop": "+15555550100"}."""
    out: Dict[str, str] = {}
    for k, v in (callers or {}).items():
        if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip():
            out[norm_title(k)] = _h(v)
    return out


def allowed(people: List[str], callers: Dict[str, str],
            groups: Optional[Dict[str, List[str]]] = None) -> Optional[Tuple[str, str]]:
    """(label, kind) to answer this call with, or None = let it ring.
    kind is "dm" or "group"; label is the caller's configured name or the group's."""
    if not people:
        return None
    cmap = callers_map(callers)
    label = {h: k for k, h in ((k, _h(v)) for k, v in (callers or {}).items()
                               if isinstance(k, str) and isinstance(v, str) and v.strip())}
    handles = [norm_handle(p) or cmap.get(norm_title(p)) for p in people]
    if len(people) == 1:
        h = handles[0]
        return (label[h], "dm") if h and h in label else None
    if any(h is None for h in handles) or len(set(handles)) != len(handles):
        return None
    hs = set(handles)
    for gname, members in (groups or {}).items():
        ms = {norm_handle(m) or cmap.get(norm_title(m)) or m.strip().lower()
              for m in (members or []) if isinstance(m, str) and m.strip()}
        # The list FaceTime shows leaves out this Mac's own account: the group may have one more.
        if ms and hs <= ms and len(ms) - len(hs) <= 1:
            return str(gname), "group"
    if all(h in label for h in handles):
        return "group call", "group"
    return None
