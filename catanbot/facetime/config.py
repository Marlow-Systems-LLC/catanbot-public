"""The "facetime" block of data/config.json. Missing block = off; "answer": false = log only.

{
  "facetime": {
    "answer": false,                       # true = press Answer/Join for allowed calls
    "name": "marlow",                      # the spoken wake name
    "callers": {"Ana Pop": "+15555550100"},# shown name -> phone/email; only these are answered
    "groups": {"Catan": ["+15555550100", "bo@example.com"]},   # optional: group calls by member set
    "stt_url": "http://127.0.0.1:8080/v1/audio/transcriptions",
    "stt_model": "whisper-1",
    "llm": {"url": "http://127.0.0.1:11434/v1", "model": "llama3.1", "key_file": "~/.config/catanbot/llm.key"},
    "tts": "say",                          # or an OpenAI-compatible /v1/audio/speech URL
    "tts_model": "tts-1",
    "voice": "Samantha",
    "in_device": "BlackHole 16ch",
    "out_device": "BlackHole 2ch"
  }
}
Read straight from the file (catanbot/config.py stays untouched by this add-on).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent.parent
PATH = ROOT / "data" / "config.json"

# "tuning": {...} knobs and their defaults (seconds unless named otherwise)
TUNING: Dict[str, object] = {
    "vad_tail_s": 0.7,          # silence that closes an utterance
    "vad_floor": 250.0,         # int16 RMS gate floor; quiet callers may need ~40
    "barge_floor": 150.0,       # input RMS that counts as talking over the bot
    "barge_s": 0.6,             # that long over the floor stops playback
    "followup_s": 8.0,          # after the bot speaks, a follow-up needs no name
    "brain_deadline_s": 12.0,   # then a holding line is spoken
    "brain_max_s": 45.0,        # then the turn is dropped
    "silent_end_s": 120.0,      # no audio this long: leave the call
    "incall_misses": 40.0,      # in-call checks in a row without call controls: end
    "ax_timeout_s": 10.0,       # one JXA walk
    "ax_via": "ssh-localhost",  # run AX helpers under `ssh localhost` (sshd holds Accessibility), or "direct"
    "nc_bin": "~/.catanbot-facetime/facetime-nc",
    "stt_language": "",
    "stt_prompt": "",           # biases Whisper's spelling of the name, e.g. "Marlow"
    "ro_voice": "Ioana",        # macOS voice for Romanian replies; "off" = never switch
    "dm_greeting": "",          # "" = "Hi, it's <Name>."; "off" = none
    "group_greeting": "",
    "leave_on_start": True,     # quit FaceTime at start: a call held from before has no audio session
    "quit_to_leave": True,      # no Leave/End button found: quit FaceTime to end the call
    "level_log_s": 5.0,         # input level line every N s of call audio (0 = off)
}


@dataclass
class FaceTimeConfig:
    enabled: bool = False          # a "facetime" block exists
    answer: bool = False           # press the answer button (else: log rings only)
    name: str = "marlow"
    callers: Dict[str, str] = field(default_factory=dict)
    groups: Dict[str, List[str]] = field(default_factory=dict)
    stt_url: str = ""
    stt_model: str = "whisper-1"
    stt_key_file: str = ""
    llm: Optional[dict] = None     # {"url", "model", "key_file"}
    tts: str = "say"
    tts_model: str = "tts-1"
    tts_key_file: str = ""
    voice: str = ""
    in_device: str = "BlackHole 16ch"
    out_device: str = "BlackHole 2ch"
    greeting: bool = True
    max_call_s: float = 3600.0
    tuning: Dict[str, object] = field(default_factory=dict)   # optional knobs, see TUNING

    def t(self, key: str):
        """A tuning knob: the config's value when it has the default's type, else the default."""
        d = TUNING[key]
        v = self.tuning.get(key, d)
        if isinstance(d, bool):
            return v if isinstance(v, bool) else d
        if isinstance(d, (int, float)):
            return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else float(d)
        return v if isinstance(v, str) else d

    @property
    def live(self) -> bool:
        """Answers calls: the block exists, answer is on and at least one caller is allowed."""
        return self.enabled and self.answer and bool(self.callers or self.groups)


def from_dict(block) -> FaceTimeConfig:
    if not isinstance(block, dict):
        return FaceTimeConfig()
    c = FaceTimeConfig(enabled=True)
    c.answer = block.get("answer") is True   # only a real true turns it on
    if isinstance(block.get("name"), str) and block["name"].strip():
        c.name = block["name"].strip()
    if isinstance(block.get("callers"), dict):
        c.callers = {str(k): str(v) for k, v in block["callers"].items() if str(k).strip() and str(v).strip()}
    if isinstance(block.get("groups"), dict):
        c.groups = {str(k): [str(x) for x in v] for k, v in block["groups"].items() if isinstance(v, list)}
    for k in ("stt_url", "stt_model", "stt_key_file", "tts", "tts_model", "tts_key_file", "voice", "in_device", "out_device"):
        if isinstance(block.get(k), str):
            setattr(c, k, block[k])
    llm = block.get("llm")
    if isinstance(llm, dict) and llm.get("url") and llm.get("model"):
        c.llm = {"url": str(llm["url"]), "model": str(llm["model"]), "key_file": str(llm.get("key_file") or "")}
    if isinstance(block.get("greeting"), bool):
        c.greeting = block["greeting"]
    if isinstance(block.get("max_call_s"), (int, float)) and block["max_call_s"] > 0:
        c.max_call_s = float(block["max_call_s"])
    if isinstance(block.get("tuning"), dict):
        c.tuning = {k: v for k, v in block["tuning"].items() if k in TUNING}
    return c


def load(path: Path | str | None = None) -> FaceTimeConfig:
    p = Path(path) if path else PATH
    try:
        data = json.loads(p.read_text())
    except (OSError, ValueError):
        return FaceTimeConfig()
    return from_dict(data.get("facetime") if isinstance(data, dict) else None)


def read_key(path: str) -> str:
    """A bearer key from a file ("" when unset or unreadable). Never logged."""
    if not path:
        return ""
    try:
        return Path(path).expanduser().read_text().strip()
    except OSError:
        return ""
