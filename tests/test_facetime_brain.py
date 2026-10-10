"""catanbot.facetime.config and .brain: config parsing, rule replies from the game log, the LLM call
through a fake opener, "leave the game" writing data/leave.flag, and the hand never being spoken."""
from __future__ import annotations

import io
import json

import pytest

from catanbot.facetime import brain as B
from catanbot.facetime import config as C

STATUS = {
    "gameId": "g-1", "turn": 14, "round": 5, "currentPlayer": "Ana", "lastRoll": 8,
    "me": "marlow", "vp": {"Ana": 7, "marlow": 6, "Bo": 4},
    "hand": {"ore": 3, "grain": 2, "wool": 1},
    "strategy": {"turn": 14, "plan": ["city"], "reason": "ore and grain on a 6", "goals": ["metropolis"],
                 "decisions": ["T12 bank trade {'wool': 3} -> {'grain': 1}", "T13 robber on Ana's 6, hand={'ore': 3}"]},
}
HAND_WORDS = ("'ore': 3", "'grain': 2", "hand=")  # trades stay (game_log keeps them); the hand never


@pytest.fixture
def logs(tmp_path):
    d = tmp_path / "game_logs"
    d.mkdir()
    (d / "current.json").write_text(json.dumps(STATUS))
    return d


def no_hand(text):
    assert not any(w in text for w in HAND_WORDS), text


# ── config ───────────────────────────────────────────────────────────────


def test_no_block_or_no_file_is_off(tmp_path):
    assert not C.load(tmp_path / "missing.json").enabled
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"other": 1}))
    assert not C.load(p).enabled
    p.write_text("not json")
    assert not C.load(p).enabled


def test_answer_only_for_a_literal_true(tmp_path):
    p = tmp_path / "config.json"
    for v, want in ((True, True), ("true", False), (1, False), (None, False)):
        p.write_text(json.dumps({"facetime": {"answer": v, "callers": {"Ana Pop": "+15555550100"}}}))
        c = C.load(p)
        assert c.enabled and c.answer is want and c.live is want, v
    p.write_text(json.dumps({"facetime": {"answer": True}}))
    assert not C.load(p).live  # nobody to answer


def test_fields_and_tuning_types():
    c = C.from_dict({"name": "Pip", "llm": {"url": "http://127.0.0.1:11434/v1", "model": "m"}, "tts": "say",
                     "groups": {"Catan": ["a@example.com"], "bad": "x"}, "max_call_s": 600,
                     "tuning": {"vad_floor": 40, "barge_s": "fast", "leave_on_start": "no", "unknown": 1}})
    assert (c.name, c.tts, c.max_call_s) == ("Pip", "say", 600.0)
    assert c.llm == {"url": "http://127.0.0.1:11434/v1", "model": "m", "key_file": ""}
    assert c.groups == {"Catan": ["a@example.com"]}
    assert c.t("vad_floor") == 40.0
    assert c.t("barge_s") == C.TUNING["barge_s"]
    assert c.t("leave_on_start") is True
    assert "unknown" not in c.tuning
    assert C.from_dict({"llm": {"url": "x"}}).llm is None


def test_read_key(tmp_path):
    k = tmp_path / "k"
    k.write_text("  secret-test-value\n")
    assert C.read_key(str(k)) == "secret-test-value"
    assert C.read_key("") == ""
    assert C.read_key(str(tmp_path / "none")) == ""


# ── rule replies ─────────────────────────────────────────────────────────


def brain(tmp_path, logs, **block):
    notes = []
    b = B.Brain(C.from_dict(block), root=tmp_path, log_dir=logs, note=notes.append, opener=block.pop("opener", None))
    b.notes = notes
    return b


def test_rule_replies(tmp_path, logs):
    b = brain(tmp_path, logs)
    score, _ = b.reply("marlow who's winning?")
    assert "Ana 7" in score and "Turn 14" in score
    plan, _ = b.reply("marlow what's the plan?")
    assert "city" in plan and "metropolis" in plan
    why, _ = b.reply("marlow why did you do that")
    assert why.startswith("Last key move:")
    other, _ = b.reply("marlow sing a song")
    assert other == B.HELP
    for t in (score, plan, why):
        no_hand(t)


def test_no_game(tmp_path):
    b = brain(tmp_path, tmp_path / "nothing")
    assert b.reply("marlow what's the score")[0] == "No game on right now."


def test_safe_strips_hand_talk():
    for t in ("I have 3 ore and 2 grain.", "My hand is ore ore grain.", "Trade {'ore': 3} now"):
        out = B.safe(t)
        assert "3 ore" not in out and "ore ore" not in out and "{" not in out, out


# ── leaving the game ─────────────────────────────────────────────────────


def test_leave_the_game_writes_the_flag(tmp_path, logs):
    b = brain(tmp_path, logs)
    out, action = b.reply("marlow, leave the game")
    assert (out, action) == ("Okay, leaving the game.", "leave_game")
    assert (tmp_path / "data" / "leave.flag").exists()


def test_hang_up_is_not_leaving_the_game(tmp_path, logs):
    assert not B.wants_leave_game("marlow, hang up")
    assert not B.wants_leave_game("leave the call")
    assert not B.wants_leave_game("should I leave the robber there")
    assert B.wants_leave_game("quit the match please")
    b = brain(tmp_path, logs)
    assert b.reply("marlow, leave the call")[1] is None
    assert not (tmp_path / "data" / "leave.flag").exists()


# ── LLM ──────────────────────────────────────────────────────────────────


class FakeResp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_llm_gets_context_and_its_reply_is_scrubbed(tmp_path, logs):
    key = tmp_path / "llm.key"
    key.write_text("test-key")
    seen = []

    def opener(req, timeout=0):
        seen.append(req)
        content = "Ana leads with 7. I have 3 ore and 2 grain."
        return FakeResp(json.dumps({"choices": [{"message": {"content": content}}]}).encode())

    b = brain(tmp_path, logs, llm={"url": "http://127.0.0.1:9/v1/", "model": "m", "key_file": str(key)}, opener=opener)
    out, action = b.reply("marlow who's winning", kind="group")
    assert action is None
    assert out == "Ana leads with 7."
    req = seen[0]
    assert req.full_url == "http://127.0.0.1:9/v1/chat/completions"
    assert req.get_header("Authorization") == "Bearer test-key"
    body = json.loads(req.data)
    system = body["messages"][0]["content"]
    assert "Turn 14" in system and "metropolis" in system
    no_hand(system)
    assert body["messages"][-1] == {"role": "user", "content": "marlow who's winning"}


def test_llm_failure_falls_back_to_rules(tmp_path, logs):
    def opener(req, timeout=0):
        raise OSError("down")

    b = brain(tmp_path, logs, llm={"url": "http://127.0.0.1:9/v1", "model": "m"}, opener=opener)
    out, _ = b.reply("marlow what's the score")
    assert "Ana 7" in out
    assert any("llm failed" in n for n in b.notes)


def test_leave_game_skips_the_llm(tmp_path, logs):
    calls = []
    b = brain(tmp_path, logs, llm={"url": "http://127.0.0.1:9/v1", "model": "m"}, opener=lambda *a, **k: calls.append(1))
    assert b.reply("marlow exit the game")[1] == "leave_game"
    assert calls == []
