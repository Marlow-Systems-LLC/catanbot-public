"""catanbot.facetime.lib: wake rule, VAD, audio helpers, ring classification, the local allow-list.
Synthetic data only; runs anywhere numpy does (no macOS imports)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from catanbot.facetime import lib as L

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(autouse=True)
def _default_name():
    L.set_name("marlow")
    yield
    L.set_name("marlow")


def tone(seconds: float, amp: int = 3000) -> np.ndarray:
    t = np.arange(int(seconds * L.RATE)) / L.RATE
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.int16)


def silence(seconds: float) -> np.ndarray:
    return (np.random.default_rng(0).normal(0, 30, int(seconds * L.RATE))).astype(np.int16)


def run_vad(pcm: np.ndarray, **kw) -> list:
    v = L.Vad(**kw)
    out = []
    for i in range(0, len(pcm) - L.FRAME + 1, L.FRAME):
        u = v.feed(pcm[i : i + L.FRAME])
        if u is not None:
            out.append(u)
    return out


def _nc(*groups):
    """A NotificationCenter detect with per-banner groups (texts, buttons)."""
    gs = [{"texts": list(t), "buttons": list(b)} for t, b in groups]
    return {"procs": {"NotificationCenter": {"texts": [x for g in gs for x in g["texts"]],
                                             "buttons": [x for g in gs for x in g["buttons"]], "groups": gs}}}


def _msgs(title, *buttons):
    return {"Messages": {"wins": [{"title": title, "buttons": list(buttons)}], "texts": [], "buttons": []}}


# ── wake rule ────────────────────────────────────────────────────────────


def test_name_first():
    for t in ["Marlow, who's winning?", "Hey Marlo what's the longest road rule", "ok marlowe, roll for me",
              "Mar low, are you there", "Bă Marlow, cine câștigă?", "hai marlo", "Salut, Marlow!"]:
        assert L.wake(t, 0, -1) == "direct", t


def test_mention_needs_a_question():
    assert L.wake("what do you think Marlow?", 0, -1) == "mention"
    assert L.wake("I told Marlow about it yesterday", 0, -1) is None


def test_chatter_is_ignored_unless_follow_up():
    assert L.wake("pass me the dice", 5, 4) is None
    assert L.wake("and what about ore?", 5, 6) == "follow"
    assert L.wake("Pass me the sheep, please.", 5, 6) is None
    assert L.wake("how many points do I need", 5, 6) == "follow"
    assert L.wake("Pass me the sheep, please.", 5, 6, group=False) == "follow"
    assert L.wake("", 5, 6) is None


def test_romanian_spellings_and_openers():
    for t in ["Marlău, ce faci?", "marlau zi-mi", "Mărlou, cine câștigă", "Mar-low, roll", "deci Marlow ce zici",
              "păi bă Marlow", "ok so Marlow, your turn", "Marlow's turn now"]:
        assert L.wake(t, 0, -1) == "direct", t


def test_mention_without_the_question_mark():
    assert L.wake("ce zici Marlow", 0, -1) == "mention"
    assert L.wake("what do you think Marlow", 0, -1) == "mention"
    assert L.wake("so the robber goes there, right Marlau.", 0, -1) == "mention"
    assert L.wake("cine are cele mai multe puncte marlow", 0, -1) == "mention"
    assert L.wake("Vlad told Marlow about it yesterday", 0, -1) is None


def test_romanian_reply_voice():
    for t in ["Ai nevoie de grâu.", "Da, eu am cele mai multe puncte acum", "nu, tu ai drumul cel mai lung"]:
        assert L.looks_romanian(t), t
    for t in ["You need grain.", "I have the longest road, Vlad", "Okay", "Dan, a 6 or an 8 is best"]:
        assert not L.looks_romanian(t), t


def test_near_name_flag():
    assert L.near_name("Marla, ce faci")
    assert L.near_name("mar lo")
    assert not L.near_name("pass me the sheep")


def test_no_false_names():
    for t in ["marmalade please", "Marlboro is a cigarette", "the marlin swam?"]:
        assert L.wake(t, 0, -1) is None, t


def test_goodbye_and_leave():
    assert L.is_goodbye("thanks Marlow")
    assert L.is_goodbye("Marlow, bye")
    assert not L.is_goodbye("Marlow, how many points?")
    assert L.wants_leave("Marlow, leave the call")
    assert L.wants_leave("marlo please hang up")
    assert not L.wants_leave("Marlow, should I leave the robber here?")


def test_dm_leave_without_the_name():
    for t in ["hang up", "Hang up please.", "please hang up", "Goodbye.", "good bye", "bye Marlow", "Bye, Marlo!",
              "bye bye", "bye now", "good night", "Good night, Marlow.", "okay, goodbye", "thank you, bye Marlowe",
              "thanks, bye", "alright bye", "okay thanks bye", "Thank you. Goodbye.", "Thanks Marlow, bye",
              "Bye Marlow, good night", "Marlow, can you hang up now?", "could you hang up please",
              "please end the call", "hang up the call", "end the call"]:
        assert L.wants_leave(t, group=False), t
        if not re.search(L.NAME, t, re.IGNORECASE):
            assert not L.wants_leave(t), t  # a group still needs the name
    for t in ["say goodbye to grandma for me", "hang up on him", "can you end the call with Mo first?",
              "that's all", "should I hang up the laundry?", "good night was a great movie", "bye mom",
              "goodbye is a sad word, isn't it"]:
        assert not L.wants_leave(t, group=False), t
    # a lone "bye" is junk in a group (Whisper hears it in near-silence), a farewell 1:1
    for t in ["bye", "Bye.", "Bye!", "BYE", " bye... "]:
        assert L.is_junk(t), t
        assert not L.is_junk(t, group=False), t
        assert L.wants_leave(t, group=False), t
        assert not L.wants_leave(t), t
    for t in ["you", "Thanks for watching!", "[Music]", "..."]:
        assert L.is_junk(t, group=False), t


def test_marlo_spelling():
    for t in ["Marlo, what time is it?", "hey marlo", "marlowe bye"]:
        assert L.wake(t, 0, -1) or L.is_goodbye(t), t
    assert L.wants_leave("Marlo, hang up")
    assert L.wants_leave("marlo leave the call")
    assert L.is_goodbye("bye marlo")


def test_junk():
    for t in ["", " ", "Thanks for watching!", "you", "[Music]", "..."]:
        assert L.is_junk(t), t
    assert not L.is_junk("Marlow who won")


def test_custom_wake_name():
    L.set_name("Pip")
    assert L.wake("Pip, who's winning?", 0, -1) == "direct"
    assert L.wake("hey pip what is the plan", 0, -1) == "direct"
    assert L.wake("Marlow, who's winning?", 0, -1) is None
    assert L.wake("pipe the water", 0, -1) is None
    assert L.wants_leave("Pip, hang up")
    assert not L.wants_leave("Marlow, hang up")
    assert L.is_goodbye("thanks Pip")
    L.set_name("marlow")
    assert L.wake("Marlo, who's winning?", 0, -1) == "direct"


# ── VAD and audio ────────────────────────────────────────────────────────


def test_vad_one_utterance_with_tail():
    utts = run_vad(np.concatenate([silence(1), tone(1.5), silence(1.2)]))
    assert len(utts) == 1
    assert abs(len(utts[0]) / L.RATE - (1.5 + 0.3 + 0.2)) <= 0.15


def test_vad_short_pause_does_not_split():
    assert len(run_vad(np.concatenate([silence(1), tone(1), silence(0.4), tone(1), silence(1)]))) == 1


def test_vad_two_utterances():
    assert len(run_vad(np.concatenate([silence(1), tone(1), silence(1.2), tone(1), silence(1)]))) == 2


def test_vad_click_is_not_speech():
    assert run_vad(np.concatenate([silence(1), tone(0.1), silence(1)])) == []


def test_vad_max_length_cuts():
    utts = run_vad(np.concatenate([silence(0.5), tone(20)]), max_utt_s=15)
    assert len(utts) == 1
    assert len(utts[0]) / L.RATE <= 15.1


def test_wav_roundtrip_and_resample():
    x = tone(1)
    y, r = L.read_wav(L.wav_bytes(x, 16000))
    assert r == 16000
    assert np.array_equal(x, y)
    assert len(L.resample(x, 16000, 48000)) == 48000
    assert len(L.resample(L.resample(x, 16000, 24000), 24000, 16000)) == 16000


def test_to_mono():
    st = np.stack([np.full(10, 0.5, np.float32), np.full(10, -0.5, np.float32)], axis=1)
    assert (L.to_mono_int16(st) == 0).all()


def test_multipart():
    body, ctype = L.multipart({"model": "m"}, "file", "u.wav", b"RIFFxx")
    b = ctype.split("boundary=")[1]
    assert f"--{b}--".encode() in body
    assert b'name="model"\r\n\r\nm\r\n' in body


def test_sentences():
    assert L.sentences("Ana leads. Then you, with 7! Ok?") == ["Ana leads.", "Then you, with 7!", "Ok?"]
    assert all(len(s) <= 221 for s in L.sentences("word " * 120))
    assert L.sentences("  ") == []


# ── people, handles, group resolution ────────────────────────────────────

IMSG = "\n".join([
    '{"id": 8, "is_group": true, "display_name": "", "name": "chat1234", "participants": ["+14155550100", "+15555550199", "bo@example.com"]}',
    '{"id": 12, "is_group": true, "participants": ["bo@example.com", "+15555550199", "+14155550100"]}',
    '{"id": 7, "is_group": true, "display_name": "Bayne", "participants": ["+14155550100", "+15555550177"]}',
    '{"id": 3, "is_group": false, "name": "+15550100", "participants": ["+15550100"]}',
    '{"id": 21, "is_group": true, "participants": ["+14155550111", "+15555550199", "ana@example.com"]}',
    '{"id": 22, "is_group": true, "participants": ["+14155550111", "+15555550188", "ana@example.com"]}',
    "not json",
])

BANNER = ["Dan Ionescu", "To +1 (415) 555-0100, Dan Ionescu, & Bo@Example.com", "Join FaceTime Call"]
CATAN_WIN = "Dan, +1 (415) 555-0100 & bo@example.com"


def _gid(g, n):
    """The conversation id group_participants gives the n-th group chat in IMSG."""
    return g[n][0]


def test_people_and_handles():
    assert L.parse_people(BANNER[1]) == ["+1 (415) 555-0100", "Dan Ionescu", "Bo@Example.com"]
    assert L.parse_people(CATAN_WIN) == ["Dan", "+1 (415) 555-0100", "bo@example.com"]
    assert L.norm_handle("‪+1 (415) 555‑0100‬") == "+14155550100"
    assert L.norm_handle("(415) 555-0100") == "+14155550100"
    assert L.norm_handle("Bo@Example.com") == "bo@example.com"
    assert L.norm_handle("Dan Ionescu") is None


def test_participants_map_to_their_group_chats():
    g = L.group_participants(IMSG)
    want = [(_gid(g, 0), "group"), (_gid(g, 1), "group")]
    assert L.resolve_call(L.parse_people(BANNER[1]), g, {}) == want
    assert L.resolve_call(L.parse_people(CATAN_WIN), g, {}) == want


def test_ambiguous_or_unknown_rings():
    g = L.group_participants(IMSG)
    assert L.resolve_call(["Ana", "+1 415 555 0111", "Someone"], g, {}) == []
    assert L.resolve_call(["+14155550100", "bo@example.com"], g, {}) == []
    assert L.resolve_call(["+14155550100", "Dan", "zed@example.com"], g, {}) == []
    assert L.resolve_call(["Ana", "Bo", "Cy"], g, {}) == []
    assert L.resolve_call([], g, {}) == []


def test_callers_disambiguate_names():
    g = L.group_participants(IMSG)
    c = L.parse_callers("Ana=Ana@Example.com, Mo=+1 555 555 0188")
    assert L.resolve_call(["Ana", "+1 415 555 0111", "Mo"], g, c) == [(_gid(g, 4), "group")]


def test_dm_only_through_callers():
    c = L.parse_callers("Dan Ionescu=+15550100, Ana=Ana@X.com,bad")
    assert L.resolve_call(["dan ionescu"], [], c) == [("+15550100", "dm")]
    assert L.resolve_call(["Ana"], [], c) == [("ana@x.com", "dm")]
    assert L.resolve_call(["+15550100"], [], {}) == []


# ── local allow-list ─────────────────────────────────────────────────────

CALLERS = {"Ana Pop": "+1 555 555 0100", "Bo": "Bo@Example.com", "Cy": "+15555550133"}


def test_allowed_one_to_one():
    assert L.allowed(["Ana Pop"], CALLERS) == ("Ana Pop", "dm")
    assert L.allowed(["ana pop"], CALLERS) == ("Ana Pop", "dm")
    assert L.allowed(["‪Ana Pop‬"], CALLERS) == ("Ana Pop", "dm")
    assert L.allowed(["+1 (555) 555-0100"], CALLERS) == ("Ana Pop", "dm")
    assert L.allowed(["bo@example.com"], CALLERS) == ("Bo", "dm")
    assert L.allowed(["Stranger"], CALLERS) is None
    assert L.allowed(["+15555550999"], CALLERS) is None
    assert L.allowed([], CALLERS) is None
    assert L.allowed(["Ana Pop"], {}) is None


def test_allowed_group_of_known_callers():
    assert L.allowed(["Ana Pop", "Bo"], CALLERS) == ("group call", "group")
    assert L.allowed(["+15555550100", "bo@example.com", "Cy"], CALLERS) == ("group call", "group")
    # one unknown participant: the whole call rings out
    assert L.allowed(["Ana Pop", "Bo", "Someone"], CALLERS) is None
    assert L.allowed(["Ana Pop", "zed@example.com"], CALLERS) is None
    # the same person twice is not two callers
    assert L.allowed(["Ana Pop", "+15555550100"], CALLERS) is None


def test_allowed_named_group():
    groups = {"Catan": ["+15555550100", "bo@example.com", "+15555550177"]}
    callers = {"Ana Pop": "+15555550100"}
    # FaceTime leaves this Mac's own account out: one member fewer is still the group
    assert L.allowed(["Ana Pop", "Bo@Example.com"], callers, groups) == ("Catan", "group")
    assert L.allowed(["+15555550100", "bo@example.com", "+1 555 555 0177"], callers, groups) == ("Catan", "group")
    # two members missing: not this group, and not all known callers either
    assert L.allowed(["Ana Pop", "Bo@Example.com"], callers, {"Big": groups["Catan"] + ["x@example.com"]}) is None
    # a participant outside the group
    assert L.allowed(["Ana Pop", "zed@example.com"], callers, groups) is None
    # names only resolve through callers
    assert L.allowed(["Ana Pop", "Bo"], callers, groups) is None
    assert L.allowed(["Ana Pop", "Bo"], callers, {"Catan": ["Ana Pop", "Bo"]}) is None


def test_ring_key_is_order_free():
    assert L.ring_key(["Ana Pop", "Bo"]) == L.ring_key(["bo", "ana pop"])


# ── classify (JXA detect) ────────────────────────────────────────────────


def test_macos27_group_ring():
    d = {"procs": {"NotificationCenter": {"texts": BANNER, "buttons": []}, **_msgs(CATAN_WIN, "Join", "Join FaceTime")}}
    st = L.classify(d)
    assert (st.state, st.join, st.window) == ("ringing", "Join FaceTime", CATAN_WIN)


def test_join_bubble_only_with_the_banner():
    st = L.classify({"procs": {"NotificationCenter": {"texts": BANNER}, **_msgs(CATAN_WIN, "Join")}})
    assert (st.state, st.join) == ("ringing", "Join")
    assert L.classify({"procs": _msgs(CATAN_WIN, "Join")}).state == "idle"


def test_banner_without_messages_join_is_not_pressable():
    st = L.classify({"procs": {"NotificationCenter": {"texts": BANNER, "buttons": ["", ""]}}})
    assert (st.state, st.join) == ("ringing", None)
    assert len(st.people) == 3


def test_open_chat_is_not_the_ringing_one():
    d = {"procs": {"NotificationCenter": {"texts": BANNER}, **_msgs("Ana & +1 (415) 555-0111", "Join FaceTime")}}
    st = L.classify(d)
    assert (st.state, st.join) == ("ringing", None)


def test_one_to_one_ring():
    st = L.classify(_nc((["FaceTime Video", "Dan Ionescu"], ["Accept"])))
    assert (st.state, st.people, st.join, st.window) == ("ringing", ["Dan Ionescu"], "Accept", None)
    st = L.classify(_nc((["Dan Ionescu", "Incoming FaceTime Audio", "now"], [])))
    assert (st.state, st.people, st.join) == ("ringing", ["Dan Ionescu"], None)
    assert L.classify(_nc((["Dan Ionescu", "Missed FaceTime Video"], []))).state == "idle"


def test_one_to_one_caller_only_from_the_ringing_banner():
    st = L.classify(_nc((["Ella", "hey can you help"], []), (["FaceTime Video", "Stranger Danger"], ["Accept"])))
    assert (st.people, st.join) == (["Stranger Danger"], "Accept")
    st = L.classify(_nc((["FaceTime Video", "1m ago", "Dan Ionescu"], ["Accept"])))
    assert st.people == ["Dan Ionescu"]
    for stamp in ("Yesterday", "5 min ago", "2h ago", "Just now", "Friday", "10:41 PM"):
        assert L.classify(_nc((["FaceTime Video", stamp, "Ana"], []))).people == ["Ana"], stamp


def test_one_to_one_never_guesses():
    for d in (_nc((["FaceTime Video", "Dan Ionescu", "hello there"], ["Accept"])),
              {"procs": {"NotificationCenter": {"texts": ["FaceTime Video", "Dan Ionescu"], "buttons": ["Accept"]}}},
              _nc((["FaceTime Video", "Ana"], ["Accept"]), (["FaceTime Audio", "Bo"], ["Accept"]))):
        st = L.classify(d)
        assert (st.state, st.people, st.join) == ("ringing", [], None)
    st = L.classify(_nc((["FaceTime Video", "Ana"], ["Accept"]), (["Calendar", "Standup"], ["Accept"])))
    assert (st.people, st.join) == (["Ana"], None)


def test_facetime_button_is_never_a_join():
    assert L.classify({"procs": _msgs(CATAN_WIN, "FaceTime", "FaceTime Audio", "Call")}).state == "idle"


def test_non_facetime_banner_is_not_a_call():
    d = {"procs": {"NotificationCenter": {"texts": ["Calendar", "To Standup"], "buttons": ["Join"]}}}
    assert L.classify(d).state == "idle"


def test_in_call_and_idle():
    assert L.classify({"procs": {"FaceTime": {"wins": [{"title": "", "buttons": ["Leave"]}]}}}).state == "in-call"
    assert L.classify({}).state == "idle"


# ── the 1:1 ring through facetime-nc (native AX) ─────────────────────────


def nc_detect_from_tree(tree):
    """What `facetime-nc detect` reports for an AX tree (mirrors collect/find in facetime-nc.swift)."""
    out = []

    def collect(node, b):
        for k in node.get("children", []):
            if k.get("id") == "FACETIME_NOTIFICATION":
                banner(k)
                continue
            if k.get("role") == "AXGenericElement" and k.get("desc"):
                b["callers"].append(k["desc"])
            if k.get("role") == "AXButton" and k.get("desc"):
                b["buttons"].append(k["desc"])
            collect(k, b)

    def banner(g):
        b = {"callers": [], "buttons": []}
        out.append(b)
        collect(g, b)

    def find(node):
        for k in node.get("children", []):
            banner(k) if k.get("id") == "FACETIME_NOTIFICATION" else find(k)

    find(tree)
    return {"ok": True, "truncated": False, "groups": out}


CAPTURED = json.loads((FIXTURES / "facetime_nc_ring_1to1.json").read_text(encoding="utf-8"))["tree"]
CAPTURED_NC = nc_detect_from_tree(CAPTURED)
ANA_LABEL = "‪Ana Pop‬, FaceTime Audio"


def ring(*labels, buttons=("Live Voicemail", "Decline with Message", "Decline", "Answer")):
    return {"ok": True, "groups": [{"callers": list(labels), "buttons": list(buttons)}]}


def test_captured_tree():
    assert CAPTURED_NC["groups"] == [{"callers": [ANA_LABEL],
                                      "buttons": ["Live Voicemail", "Decline with Message", "Decline", "Answer"]}]
    st = L.classify_native(CAPTURED_NC)
    assert (st.state, st.people, st.join, st.window, st.caller) == ("ringing", ["Ana Pop"], "Answer", None, "Ana Pop")


def test_caller_name():
    assert L.caller_name(ANA_LABEL) == "Ana Pop"
    assert L.caller_name("⁨Ana Pop⁩, FaceTime Video") == "Ana Pop"
    assert L.caller_name("‫إلفيس‬, FaceTime Audio") == "إلفيس"
    assert L.caller_name("+1 (555) 010-0100, FaceTime Audio") == "+1 (555) 010-0100"
    assert L.caller_name("Ana Pop") == "Ana Pop"


def test_native_accept_is_the_alternative():
    st = L.classify_native(ring(ANA_LABEL, buttons=("Decline", "Accept")))
    assert (st.join, st.caller) == ("Accept", "Ana Pop")


def test_native_nothing_to_see():
    for d in ({}, {"ok": False, "error": "timeout"}, {"ok": True, "groups": []}, {"ok": True}, None,
              {"ok": True, "truncated": True, "groups": "x"}):
        assert L.classify_native(d) is None, d


def test_native_never_guesses():
    for d in ({"ok": True, "groups": ring(ANA_LABEL)["groups"] * 2},
              ring(ANA_LABEL, "‪Bo‬, FaceTime Video"),
              ring(),
              ring("Ana Pop"),
              ring("Missed FaceTime Audio"),
              ring("To Ana, Bo, FaceTime Video"),
              ring(", FaceTime Audio")):
        st = L.classify_native(d)
        assert (st.state, st.people, st.join, st.caller) == ("ringing", [], None, None), d


def test_native_only_answer_is_ever_pressed():
    for buttons in (("Live Voicemail", "Decline with Message", "Decline"), ("Decline",), ("Live Voicemail",), (),
                    ("Answer", "Answer"), ("Answer ",)):
        st = L.classify_native(ring(ANA_LABEL, buttons=buttons))
        assert (st.people, st.join, st.caller) == (["Ana Pop"], None, None), buttons
    assert L.NC_ANSWER == ("Answer", "Accept")


def test_incall_state():
    assert L.incall_state({"ok": True, "incall": True}) is True
    assert L.incall_state({"ok": True, "mic": True}) is True
    assert L.incall_state({"ok": True, "mic": True}, playing=True) is None
    assert L.incall_state({"ok": True, "truncated": True}) is None
    assert L.incall_state({"ok": True}) is False
    for d in ({}, {"ok": False}, None, "x"):
        assert L.incall_state(d) is None
