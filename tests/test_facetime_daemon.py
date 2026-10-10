"""catanbot.facetime.daemon without FaceTime, audio devices or macOS: the answer decision (tick),
the in-call voice commands (Session.handle), and source checks that no code path places a call."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from catanbot.facetime import config as C
from catanbot.facetime import daemon as D
from catanbot.facetime import lib as L

PKG = Path(D.__file__).resolve().parent
CALLERS = {"Ana Pop": "+15555550100", "Bo": "bo@example.com", "Dan": "+15555550133", "Cy": "+14155550100"}

BANNER = ["Dan Ionescu", "To +1 (415) 555-0100, Dan Ionescu, & Bo@Example.com", "Join FaceTime Call"]
CATAN_WIN = "Dan, +1 (415) 555-0100 & bo@example.com"


def _msgs(title, *buttons):
    return {"Messages": {"wins": [{"title": title, "buttons": list(buttons)}], "texts": [], "buttons": []}}


def _nc(*groups):
    gs = [{"texts": list(t), "buttons": list(b)} for t, b in groups]
    return {"procs": {"NotificationCenter": {"texts": [x for g in gs for x in g["texts"]],
                                             "buttons": [x for g in gs for x in g["buttons"]], "groups": gs}}}


GROUP_RING = {"procs": {"NotificationCenter": {"texts": BANNER}, **_msgs(CATAN_WIN, "Join", "Join FaceTime")}}
NATIVE_RING = {"ok": True, "groups": [{"callers": ["‪Ana Pop‬, FaceTime Audio"],
                                       "buttons": ["Live Voicemail", "Decline with Message", "Decline", "Answer"]}]}
IN_CALL = {"procs": {"FaceTime": {"wins": [{"title": "", "buttons": ["Leave"]}]}}}


class FakeAx:
    def __init__(self, detect=None, nc=None, nc_ok=True):
        self.d, self.nc, self.nc_ok = detect or {}, nc or {}, nc_ok
        self.clicks, self.nc_clicks, self.detects = [], [], []
        self.incall = None  # None = no facetime-nc helper: the in-call check falls back to JXA
        self.quits = 0

    def detect(self, procs=None):
        self.detects.append(procs)
        return self.d

    def click(self, label, window=None):
        assert label in L.JOIN_BUTTONS + L.LEAVE_BUTTONS, label
        self.clicks.append((label, window))
        return label != "End"

    def hang_up(self):
        return self.click("Leave") or self.click("End")

    def quit_facetime(self):
        self.quits += 1
        return True

    def nc_detect(self):
        return self.nc

    def nc_incall(self):
        return self.incall

    def nc_click(self, label, caller):
        self.nc_clicks.append((label, caller))
        return self.nc_ok


class FakeBrain:
    def __init__(self, reply="Ana leads with 7.", action=None):
        self.out, self.action, self.asked = reply, action, []

    def reply(self, text, kind="dm"):
        self.asked.append((text, kind))
        return self.out, self.action

    def reset(self):
        pass


class FakeSpeech:
    def __init__(self, text=""):
        self.text, self.said = text, []

    def stt(self, pcm):
        return self.text

    def tts(self, text):
        self.said.append(text)
        return np.zeros(160, dtype=np.int16), 16000


class FakePlayer:
    playing = False

    def __init__(self):
        self.played, self.stops = 0, 0

    def play(self, pcm, rate):
        self.played += 1
        return True

    def stop(self):
        self.stops += 1


def cfg(**block):
    base = {"answer": True, "callers": dict(CALLERS)}
    base.update(block)
    return C.from_dict(base)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(D.time, "sleep", lambda s: None)
    yield
    L.set_name("marlow")


def make(detect=None, nc=None, nc_ok=True, **block):
    d = D.Daemon(cfg(**block), brain=FakeBrain(), speech=FakeSpeech(), ax=FakeAx(detect, nc, nc_ok))
    d.started = []

    def start(label, kind):
        d.started.append((label, kind))
        d.session = D.Session(d.cfg, d.brain, d.speech, FakePlayer(), label, kind, on_leave=d.leave)

    d.start_session = start
    return d


# ── answering ────────────────────────────────────────────────────────────


def test_group_of_known_callers_is_joined_in_its_window():
    d = make(GROUP_RING)
    d.tick()
    assert d.ax.clicks == [("Join FaceTime", CATAN_WIN)]
    assert d.started == [("group call", "group")]
    assert d.ax.detects == [["NotificationCenter", "Messages"]]


def test_named_group_is_joined_under_its_name():
    d = make(GROUP_RING, callers={"Dan": "+15555550133"},
             groups={"Catan": ["+15555550133", "+14155550100", "bo@example.com"]})
    d.tick()
    assert d.started == [("Catan", "group")]


def test_listed_one_to_one_is_answered_in_its_banner():
    d = make(nc=NATIVE_RING)
    d.tick()
    assert d.ax.nc_clicks == [("Answer", "Ana Pop")]
    assert (d.ax.clicks, d.ax.detects) == ([], [])  # no JXA walk on a native ring
    assert d.started == [("Ana Pop", "dm")]


def test_jxa_one_to_one_ring_is_accepted():
    d = make(_nc((["FaceTime Video", "Ana Pop"], ["Accept"])))
    d.tick()
    assert d.ax.clicks == [("Accept", None)]
    assert d.started == [("Ana Pop", "dm")]


def test_answer_false_never_clicks():
    for d in (make(GROUP_RING, answer=False), make(nc=NATIVE_RING, answer=False), make(nc=NATIVE_RING, answer="yes")):
        d.tick()
        d.tick()
        assert (d.ax.clicks, d.ax.nc_clicks, d.started) == ([], [], [])


def test_unknown_callers_ring_out():
    for d in (make(nc=NATIVE_RING, callers={"Bo": "bo@example.com"}),
              make(_nc((["FaceTime Video", "Someone"], ["Accept"]))),
              make(GROUP_RING, callers={"Dan": "+15555550133", "Cy": "+14155550100"}),   # Bo is not a caller
              make(GROUP_RING, callers={})):
        d.tick()
        assert (d.ax.clicks, d.ax.nc_clicks, d.started) == ([], [], [])


def test_stranger_next_to_a_listed_callers_message_rings_out():
    d = make(_nc((["Ana Pop", "hey can you help"], []), (["FaceTime Video", "Stranger Danger"], ["Decline", "Accept"])))
    d.tick()
    assert (d.ax.clicks, d.started) == ([], [])


def test_unsure_native_banner_overrides_a_jxa_guess():
    two = {"ok": True, "groups": NATIVE_RING["groups"] * 2}
    d = make(_nc((["FaceTime Video", "Ana Pop"], ["Accept"])), nc=two)
    d.tick()
    assert (d.ax.clicks, d.ax.nc_clicks, d.started) == ([], [], [])


def test_failed_press_starts_nothing():
    d = make(nc=NATIVE_RING, nc_ok=False)
    d.tick()
    assert d.ax.nc_clicks == [("Answer", "Ana Pop")]
    assert d.started == []


def test_audio_failure_after_answering_hangs_up():
    d = make(nc=NATIVE_RING)

    def boom(label, kind):
        raise OSError("no BlackHole")

    d.start_session = boom
    d.tick()
    assert d.session is None
    assert d.ax.clicks == [("Leave", None)]


# ── leaving ──────────────────────────────────────────────────────────────


def test_voice_leave_does_not_rejoin():
    d = make(GROUP_RING)
    d.tick()
    d.last_sound = 1e12
    d.leave()  # "Marlow, hang up"
    d.tick()
    assert d.session is None
    d.ax.d = {}
    d.tick()
    d.ax.d = GROUP_RING
    d.tick()
    assert [c[0] for c in d.ax.clicks].count("Join FaceTime") == 1
    assert d.started == [("group call", "group")]


def test_silence_leaves_and_waits_for_the_ring_to_go():
    d = make(GROUP_RING)
    d.tick()
    d.last_sound = 0.0
    d.tick()
    assert d.session is None
    assert d.ax.clicks[1:] == [("Leave", None)]
    d.tick()  # the Join button is still there: stay out
    assert len(d.ax.clicks) == 2
    d.ax.d = {"procs": {"NotificationCenter": {"texts": [], "buttons": []}}}
    for _ in range(D.LEFT_GONE_POLLS):
        d.tick()
    assert d.left == ""


def test_idle_ends_only_after_in_call_was_seen():
    d = make(GROUP_RING)
    d.tick()
    d.last_sound = 1e12
    d.ax.d = {}
    d.tick()
    assert d.session is not None
    d.ax.d = IN_CALL
    d.tick()
    d.ax.d = {}
    d.tick()
    assert d.session is None


def test_native_gone_is_unknown_while_people_talk():
    d = make(nc=NATIVE_RING)
    d.tick()
    d.ax.incall = {"ok": True, "incall": True}
    d.tick()
    d.ax.incall = {"ok": True}
    d.last_sound = D.time.time()  # audio just now: hidden controls are not the end
    d.tick()
    assert d.session is not None
    d.last_sound = D.time.time() - D.IDLE_QUIET_S - 1
    d.tick()
    assert d.session is None
    assert d.ax.clicks == [("Leave", None)]


def test_watch_only_start_never_quits_facetime(monkeypatch):
    d = make(answer=False)
    monkeypatch.setattr(d, "tick", lambda: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        d.run()
    assert d.ax.quits == 0


# ── in-call voice commands ───────────────────────────────────────────────


def session(text, kind="dm", reply="Ana leads with 7.", action=None, name="marlow"):
    c = cfg(name=name)
    L.set_name(c.name)
    left = []
    s = D.Session(c, FakeBrain(reply, action), FakeSpeech(text), FakePlayer(), "Ana Pop", kind,
                  on_leave=lambda: left.append(1))
    return s, left


PCM = np.zeros(16000, dtype=np.int16)


def test_hang_up_ends_the_call():
    s, left = session("Marlow, hang up")
    assert s.handle(PCM, 0.0) is None
    assert left == [1]
    assert s.brain.asked == []


def test_leave_the_game_is_not_a_hang_up():
    s, left = session("Marlow, leave the game", reply="Okay, leaving the game.", action="leave_game")
    rec = s.handle(PCM, 0.0)
    assert left == []
    assert s.brain.asked == [("Marlow, leave the game", "dm")]
    assert rec["action"] == "leave_game"
    assert "Okay, leaving the game." in s.speech.said


def test_question_is_answered_and_unaddressed_talk_ignored():
    s, _ = session("Marlow, who's winning?", kind="group")
    assert s.handle(PCM, 0.0)["how"] == "direct"
    assert s.speech.said == ["Ana leads with 7."]
    s, _ = session("pass me the sheep", kind="group")
    assert s.handle(PCM, 0.0) is None
    assert s.brain.asked == []


def test_custom_name_wakes_the_session():
    s, _ = session("Pip, who's winning?", kind="group", name="Pip")
    assert s.handle(PCM, 0.0)["how"] == "direct"
    s, _ = session("Marlow, who's winning?", kind="group", name="Pip")
    assert s.handle(PCM, 0.0) is None


def test_dm_greeting():
    s, _ = session("")
    assert s.greeting == "Hi, it's Marlow."
    c = cfg(tuning={"dm_greeting": "off"})
    assert D.Session(c, FakeBrain(), FakeSpeech(), FakePlayer(), "Ana Pop", "dm").greeting == ""
    assert D.Session(cfg(), FakeBrain(), FakeSpeech(), FakePlayer(), "x", "group").greeting == ""


# ── never dials ──────────────────────────────────────────────────────────


def _code(text, comment):
    return "\n".join(ln for ln in text.splitlines() if not ln.strip().startswith(comment))


def test_ax_script_only_presses_answer_and_leave():
    js = (PKG / "macos" / "facetime-ax.js").read_text()
    m = re.search(r"const CLICKABLE = \[([^\]]*)\]", js)
    assert m
    assert sorted(re.findall(r'"([^"]+)"', m.group(1))) == sorted(["Join FaceTime", "Join", "Accept", "Answer", "Leave", "End"])
    code = _code(js, "//")
    for bad in ["facetime://", "facetime-audio://", "tel:", "openLocation", "keystroke", "doShellScript", ".openURL",
                "Start Call", '"Call"', '"FaceTime Audio"']:
        assert bad not in code, bad


def test_native_helper_only_presses_answer():
    sw = (PKG / "macos" / "facetime-nc.swift").read_text()
    assert re.search(r'let PRESSABLE: Set<String> = \["Answer", "Accept"\]', sw)
    code = _code(sw, "//")
    for bad in ["facetime://", "facetime-audio://", "tel:", "openURL", "NSWorkspace", '"FaceTime Audio"']:
        assert bad not in code, bad


def test_daemon_never_opens_urls_or_dials():
    src = (PKG / "daemon.py").read_text()
    for bad in ["facetime://", "facetime-audio://", "tel:", '"open"', "open -a", "openURL"]:
        assert bad not in src, bad
    for label in re.findall(r'self\.(?:ax\.)?click\("([^"]+)"', src):
        assert label in L.JOIN_BUTTONS + L.LEAVE_BUTTONS
    assert set(L.JOIN_BUTTONS) | set(L.LEAVE_BUTTONS) == {"Join FaceTime", "Join", "Accept", "Answer", "Leave", "End"}
    assert not (PKG / "outbound.py").exists()


def test_no_audio_written_to_disk():
    src = (PKG / "daemon.py").read_text()
    # the only file writes: --out in synthetic mode, and `say`'s temp file (deleted at once)
    assert src.count("write_bytes(") == 1
    assert "os.unlink(path)" in src


def test_sounddevice_is_imported_lazily():
    for f in ("daemon.py", "lib.py", "brain.py", "config.py"):
        assert not re.search(r"^(?:import|from) (?:sounddevice|AppKit|Quartz|Foundation)", (PKG / f).read_text(), re.M), f
