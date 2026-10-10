"""The FaceTime voice daemon: answer allow-listed calls, listen, answer questions about the game.

    FaceTime group ring (macOS 27 AX: NotificationCenter banner + the Messages window, facetime-ax.js)
    FaceTime 1:1 ring (native AX, facetime-nc): "<caller>, FaceTime Audio" banner
      -> lib.allowed(people, callers, groups)  (local allow-list; anything else rings out)
      -> "answer": true: press Join FaceTime / Join (group) or Answer (1:1, only if the banner still
         names that caller); "answer": false: log "would answer" and let it ring
    in the call:
      BlackHole 16ch (FaceTime's speaker) -> energy VAD (0.7 s tail)
      -> STT (OpenAI-compatible /audio/transcriptions) -> wake rule (lib.wake)
      -> brain.Brain (local LLM or rules; game_log context; "leave the game" writes data/leave.flag)
      -> TTS (OpenAI-compatible /audio/speech, or macOS `say`) per sentence
      -> BlackHole 2ch (FaceTime's microphone); someone talking over the bot stops it.

The bot never places a call: facetime-ax.js can only press Join FaceTime/Join/Accept/Answer/Leave/End,
facetime-nc only Answer/Accept. Audio lives in memory for one utterance and is never written to disk;
transcripts and replies are never logged (lengths and counts only).

sounddevice is imported inside functions: importing this module needs only numpy.
"""
from __future__ import annotations

import json
import os
import queue
import random
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from catanbot.facetime import lib as L
from catanbot.facetime.config import FaceTimeConfig, read_key

HERE = Path(__file__).resolve().parent
MACOS = HERE / "macos"
UA = "catanbot-facetime/" + L.VERSION


def log(*a: object) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def http(url: str, body: Optional[bytes], ctype: str, timeout: float, bearer: str = "") -> Tuple[int, bytes]:
    h = {"content-type": ctype, "user-agent": UA}
    if bearer:
        h["authorization"] = f"Bearer {bearer}"
    req = urllib.request.Request(url, data=body, headers=h, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read() or b""


# -- speech ------------------------------------------------------------------


class Speech:
    """STT over an OpenAI-compatible transcription endpoint; TTS over an OpenAI-compatible speech
    endpoint or macOS `say`."""

    def __init__(self, cfg: FaceTimeConfig):
        self.cfg = cfg
        self.stt_key = read_key(cfg.stt_key_file)
        self.tts_key = read_key(cfg.tts_key_file)

    def stt(self, pcm: np.ndarray) -> Optional[str]:
        """The transcript; None when the request failed ("" = nothing said)."""
        if not self.cfg.stt_url:
            return None
        fields = {"model": self.cfg.stt_model or "whisper-1", "response_format": "json"}
        if self.cfg.t("stt_language"):
            fields["language"] = self.cfg.t("stt_language")
        if self.cfg.t("stt_prompt"):
            fields["prompt"] = self.cfg.t("stt_prompt")
        body, ctype = L.multipart(fields, "file", "u.wav", L.wav_bytes(pcm))
        try:
            s, b = http(self.cfg.stt_url, body, ctype, 15, self.stt_key)
            if s == 200:
                return str(json.loads(b).get("text", "")).strip()
            log("stt http", s)
        except Exception as e:  # noqa: BLE001
            log("stt failed:", type(e).__name__)
        return None

    def tts(self, text: str) -> Optional[Tuple[np.ndarray, int]]:
        ro = self.cfg.t("ro_voice")
        if ro != "off" and L.looks_romanian(text):
            a = self.say(text, ro)
            if a is not None:
                return a
        url = self.cfg.tts or "say"
        if url != "say":
            body = json.dumps({"model": self.cfg.tts_model or "tts-1", "voice": self.cfg.voice or "alloy",
                               "input": text, "response_format": "wav"}).encode()
            try:
                s, b = http(url, body, "application/json", 15, self.tts_key)
                if s == 200 and b[:4] == b"RIFF":
                    return L.read_wav(b)
                log("tts http", s)
            except Exception as e:  # noqa: BLE001 - `say` below
                log("tts failed:", type(e).__name__)
            return self.say(text)
        return self.say(text, self.cfg.voice or None)

    @staticmethod
    def say(text: str, voice: Optional[str] = None) -> Optional[Tuple[np.ndarray, int]]:
        """macOS `say` rendered to a temp WAV (deleted at once)."""
        if sys.platform != "darwin":
            return None
        import tempfile

        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            subprocess.run(["say"] + (["-v", voice] if voice else []) + ["--file-format=WAVE", "--data-format=LEI16@22050", "-o", path, text], check=True, timeout=30)
            return L.read_wav(Path(path).read_bytes())
        except Exception as e:  # noqa: BLE001
            log("say failed:", type(e).__name__)
            return None
        finally:
            os.unlink(path)


# -- audio out ---------------------------------------------------------------


def find_device(name: Optional[str], kind: str):
    if not name:
        return None
    import sounddevice as sd

    for i, d in enumerate(sd.query_devices()):
        if d["name"] == name and d[f"max_{kind}_channels"] > 0:
            return i
    raise RuntimeError(f"no {kind} device named {name!r}")


class Player:
    """Plays int16 chunks to FaceTime's microphone (BlackHole 2ch). stop() = barge-in."""

    def __init__(self, device: Optional[str]):
        import sounddevice as sd

        self.sd = sd
        dev = find_device(device, "output")
        self.rate = int(sd.query_devices(dev, "output")["default_samplerate"]) if dev is not None else 48000
        self.dev = dev
        self._stop = threading.Event()
        self.playing = False

    def play(self, pcm: np.ndarray, rate: int) -> bool:
        """Blocking; False when interrupted."""
        x = L.resample(pcm, rate, self.rate)
        self._stop.clear()
        self.playing = True
        try:
            with self.sd.OutputStream(samplerate=self.rate, channels=1, dtype="int16", device=self.dev) as s:
                step = self.rate // 20
                for i in range(0, len(x), step):
                    if self._stop.is_set():
                        return False
                    s.write(x[i : i + step].reshape(-1, 1))
            return True
        finally:
            self.playing = False

    def stop(self) -> None:
        self._stop.set()


class FilePlayer:
    """Synthetic mode: the bot's audio goes into a buffer (and --out)."""

    def __init__(self) -> None:
        self.rate = 24000
        self.chunks: List[np.ndarray] = []
        self.playing = False

    def play(self, pcm: np.ndarray, rate: int) -> bool:
        self.chunks.append(L.resample(pcm, rate, self.rate))
        return True

    def stop(self) -> None:
        pass


# -- one call ----------------------------------------------------------------


class Session:
    """The audio side of one answered call: utterances in, spoken answers out."""

    def __init__(self, cfg: FaceTimeConfig, brain, speech, player, label: str, kind: str,
                 clock: Callable[[], float] = time.monotonic, on_leave: Optional[Callable[[], None]] = None):
        self.cfg, self.brain, self.speech, self.player = cfg, brain, speech, player
        self.label, self.kind = label, kind
        self.clock = clock
        self.on_leave = on_leave
        self.call_id = L.new_call_id()
        self.vad = L.Vad(tail_s=cfg.t("vad_tail_s"), floor=cfg.t("vad_floor"))
        g = cfg.t("dm_greeting" if kind == "dm" else "group_greeting")
        if g == "off" or not cfg.greeting:
            g = ""
        elif not g and kind == "dm":
            g = f"Hi, it's {cfg.name.title()}."
        self.greeting = g
        self.level_every = int(cfg.t("level_log_s") * L.RATE / L.FRAME)
        self.lvl_frames, self.lvl_peak, self.lvl_sq, self.utts = 0, 0, 0.0, 0
        self.follow_s = cfg.t("followup_s")
        self.deadline_s = cfg.t("brain_deadline_s")
        self.max_s = cfg.t("brain_max_s")
        self.follow_until = -1.0
        self.seq = 0
        # (audio, when the VAD closed it, whether the bot was silent then): an utterance closed
        # while the bot played is maybe its own echo, so it never supersedes a reply.
        self.q: "queue.Queue[Optional[Tuple[np.ndarray, float, bool]]]" = queue.Queue(maxsize=4)
        self.loud_frames = 0
        # Barge-in has its own, higher floor than the VAD: the bot's own voice echoing back, or
        # line noise, passes a low VAD floor and would cut every reply. The loudest input frame
        # of each reply is logged ("playback: input rms max") to tune it.
        self.barge_floor = cfg.t("barge_floor")
        self.play_rms_max, self.was_playing = 0.0, False
        self.utt_quiet = True
        self.barge_s = cfg.t("barge_s")
        self.timings: List[dict] = []
        self.closed = False

    def frame(self, f: np.ndarray) -> None:
        """Every 30 ms frame (capture thread or the file feeder)."""
        utt = self.vad.feed(f)
        if utt is not None:
            self.utts += 1
        if self.level_every > 0:
            self.level(f)
        playing = bool(self.player.playing)
        rms = float(np.sqrt(np.mean(f.astype(np.float32) ** 2))) if len(f) and playing else 0.0
        if playing:
            self.play_rms_max = max(self.play_rms_max, rms)
        elif self.was_playing:
            log(f"playback: input rms max {self.play_rms_max:.0f} (barge floor {self.barge_floor:.0f})")
            self.play_rms_max = 0.0
        self.was_playing = playing
        self.utt_quiet = (self.utt_quiet and not playing) if (self.vad.active or utt is not None) else True
        if playing and self.vad.active:
            if rms >= self.barge_floor:
                self.loud_frames += 1
            if self.loud_frames * len(f) / L.RATE >= self.barge_s:
                log(f"barge-in: stopping playback (rms {rms:.0f})")
                self.player.stop()
                self.loud_frames = 0
        else:
            self.loud_frames = 0
        if utt is not None:
            try:
                self.q.put_nowait((utt, self.clock(), self.utt_quiet))
            except queue.Full:
                log("busy: dropped an utterance")
            self.utt_quiet = True

    def level(self, f: np.ndarray) -> None:
        x = f.astype(np.float32)
        self.lvl_frames += 1
        self.lvl_peak = max(self.lvl_peak, int(np.abs(x).max(initial=0)))
        self.lvl_sq += float(np.mean(x * x)) if len(x) else 0.0
        if self.lvl_frames >= self.level_every:
            rms = (self.lvl_sq / self.lvl_frames) ** 0.5
            log(f"level: peak {self.lvl_peak} rms {rms:.0f} gate {self.vad.threshold():.0f} utterances {self.utts}")
            self.lvl_frames, self.lvl_peak, self.lvl_sq = 0, 0, 0.0

    def greet(self) -> None:
        g = (self.greeting or "").strip()
        if not g or self.closed:
            return
        log(f"greeting: {len(g)} chars")
        self.say_now(g)
        self.follow_until = self.clock() + self.follow_s

    def worker(self) -> None:
        self.greet()
        while True:
            item = self.q.get()
            if item is None or self.closed:
                return
            self.handle(*item)

    def handle(self, pcm: np.ndarray, t_end: float, quiet: bool = True) -> Optional[dict]:
        """One utterance -> maybe one spoken answer. t_end = when the VAD closed it."""
        t0 = time.monotonic()
        text = self.speech.stt(pcm)
        t_stt = time.monotonic() - t0
        heard = f"heard {len(pcm) / L.RATE:.1f}s:"  # lengths and counts only, never the words
        if text is None:
            log(heard, "stt failed")
            return None
        group = self.kind == "group"
        if L.is_junk(text, group=group):
            log(heard, f"stt {len(text)} chars, junk")
            return None
        how = L.wake(text, t_end, self.follow_until, group=group)
        if not how:
            log(heard, f"stt {len(text)} chars, not addressed (wake)" + (", name-like word" if L.near_name(text) else ""))
            return None
        log(heard, f"stt {len(text)} chars, addressed ({how})")
        from catanbot.facetime.brain import wants_leave_game
        if not wants_leave_game(text) and L.wants_leave(text, group=group) and self.on_leave:
            self.say_now("Okay, bye!")
            self.on_leave()
            return None
        self.seq += 1
        del pcm
        t1 = time.monotonic()
        reply, action = self.ask(text)
        t_brain = time.monotonic() - t1
        if action:
            log("action:", action)
        if self.closed:
            log("call over before the reply; not spoken")
            return None
        if not reply:
            log("no reply for seq", self.seq)
            self.follow_until = -1.0 if L.is_goodbye(text) else self.follow_until
            return None
        if self.superseded():
            log("superseded: a newer utterance is waiting; reply not spoken")
            self.follow_until = self.clock() + self.follow_s
            return None
        t_first, done = self.speak(reply, fresh=True)
        rec = {"how": how, "stt_s": round(t_stt, 2), "brain_s": round(t_brain, 2), "tts_first_s": round(t_first, 2),
               "latency_s": round(self.vad.tail_s + t_stt + t_brain + t_first, 2), "interrupted": not done, "action": action}
        log("turn", json.dumps(rec))
        self.timings.append(rec)
        self.follow_until = -1.0 if L.is_goodbye(text) else self.clock() + self.follow_s
        return rec

    def superseded(self) -> bool:
        """A newer utterance, spoken while the bot was silent, is already queued.
        Never on a group call: table talk queued meanwhile isn't a question to the bot."""
        if self.kind == "group":
            return False
        with self.q.mutex:
            return any(it is not None and it[2] for it in self.q.queue)

    def ask(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """The brain's reply, with a spoken holding line if it passes the deadline."""
        box: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
        t = threading.Thread(target=lambda: box.__setitem__("r", self.safe_turn(text)), daemon=True)
        t.start()
        t.join(self.deadline_s)
        if t.is_alive() and not self.closed:
            self.say_now(random.choice(L.HOLDING_LINES))
            t.join(max(0.0, self.max_s - self.deadline_s))
        return box.get("r", (None, None))

    def safe_turn(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        try:
            return self.brain.reply(text, self.kind)
        except Exception as e:  # noqa: BLE001
            log("brain failed:", type(e).__name__)
            return None, None

    def say_now(self, text: str) -> None:
        if self.closed:
            return
        a = self.speech.tts(text)
        if a and not self.closed:
            self.player.play(*a)

    def speak(self, reply: str, fresh: bool = False) -> Tuple[float, bool]:
        """TTS sentence by sentence; synthesis of n+1 overlaps playback of n.
        fresh: a reply to the caller; dropped if they spoke again during synthesis."""
        parts = L.sentences(reply)
        t0 = time.monotonic()
        first = 0.0
        pending: "queue.Queue[Optional[Tuple[np.ndarray, int]]]" = queue.Queue()

        def synth() -> None:
            for p in parts:
                a = self.speech.tts(p)
                if a is not None:
                    pending.put(a)
            pending.put(None)

        threading.Thread(target=synth, daemon=True).start()
        n = 0
        while True:
            a = pending.get()
            if a is None:
                return first, True
            if n == 0:
                first = time.monotonic() - t0
                if fresh and self.superseded():
                    log("superseded during synthesis; reply not spoken")
                    return first, False
            n += 1
            if self.closed or not self.player.play(*a):
                return first, False

    def close(self) -> None:
        self.closed = True
        self.vad.reset()
        self.player.stop()
        try:
            self.q.put_nowait(None)
        except queue.Full:
            pass


# -- macOS accessibility -----------------------------------------------------


def run_capped(cmd: List[str], timeout: float) -> str:
    """Run cmd in its own process group; on timeout kill the whole group, reap it, and raise
    TimeoutExpired (a bare subprocess.run timeout killed only ssh)."""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, start_new_session=True)
    try:
        out, _ = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        p.communicate()
        raise
    return (out or "").strip()


class Ax:
    """facetime-ax.js (JXA: group rings, Join, Leave/End) and facetime-nc (native AX: 1:1 rings,
    Answer, in-call check). Both can only press the labels listed in lib.JOIN_BUTTONS,
    lib.LEAVE_BUTTONS and lib.NC_ANSWER."""

    def __init__(self, cfg: FaceTimeConfig):
        self.cfg = cfg
        self.script = str(MACOS / "facetime-ax.js")
        self.nc_bin = str(Path(str(cfg.t("nc_bin"))).expanduser())
        self.via = cfg.t("ax_via")
        self.ax_timeout = cfg.t("ax_timeout_s")
        self.mic = cfg.out_device or "BlackHole 2ch"

    def _cmd(self, cmd: List[str], cap: int) -> List[str]:
        if self.via != "ssh-localhost":
            return cmd
        import shlex

        # The remote side is sshd's child, out of our process group: perl's alarm (kept across
        # exec) kills it there after `cap` seconds.
        remote = ["/usr/bin/perl", "-e", "alarm shift; exec @ARGV", str(cap), *cmd]
        return ["/usr/bin/ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "-T", "localhost", " ".join(shlex.quote(c) for c in remote)]

    def _run(self, *args: str, timeout: Optional[float] = None) -> str:
        timeout = timeout if timeout is not None else self.ax_timeout
        cmd = ["/usr/bin/osascript", "-l", "JavaScript", self.script, *args]
        return run_capped(self._cmd(cmd, int(timeout) + 1), timeout)

    def _nc(self, *args: str) -> dict:
        return json.loads(run_capped(self._cmd([self.nc_bin, *args], 6), 8) or "{}")

    def detect(self, procs: Optional[List[str]] = None) -> dict:
        try:
            return json.loads(self._run("detect", *([",".join(procs)] if procs else [])) or "{}")
        except Exception as e:  # noqa: BLE001
            log("ax detect failed:", type(e).__name__)
            return {}

    def nc_detect(self) -> dict:
        if not os.path.exists(self.nc_bin):
            return {}
        try:
            d = self._nc("detect")
        except Exception as e:  # noqa: BLE001
            log("nc detect failed:", type(e).__name__)
            return {}
        if d.get("ok") is not True:
            log("nc detect:", str(d.get("error", "?"))[:80])
        return d

    def nc_incall(self) -> Optional[dict]:
        """None = the helper isn't installed (the JXA path decides); {} = it failed."""
        if not os.path.exists(self.nc_bin):
            return None
        try:
            return self._nc("incall", self.mic)
        except Exception as e:  # noqa: BLE001
            log("nc incall failed:", type(e).__name__)
            return {}

    def nc_click(self, label: str, caller: str) -> bool:
        """Press Answer/Accept in the one FaceTime banner, only if it names caller."""
        if label not in L.NC_ANSWER:
            return False
        try:
            r = self._nc("click", label, caller)
        except Exception as e:  # noqa: BLE001
            log("nc click failed:", type(e).__name__)
            return False
        log("nc click", label, r.get("ok"), r.get("error", ""))
        return r.get("ok") is True

    def hang_up(self) -> bool:
        """Leave the call: JXA Leave/End, else quit FaceTime (macOS 27's call controls may be
        invisible to JXA; quitting FaceTime ends any call it holds)."""
        if self.click("Leave") or self.click("End"):
            return True
        return self.quit_facetime()

    def quit_facetime(self) -> bool:
        if not self.cfg.t("quit_to_leave"):
            return False
        try:
            run_capped(self._cmd(["/usr/bin/osascript", "-e", 'tell application "FaceTime" to quit'], 15), 20)
            ok = True
        except Exception as e:  # noqa: BLE001
            log("quit FaceTime failed:", type(e).__name__)
            ok = False
        log("left by quitting FaceTime", ok)
        return ok

    def click(self, label: str, window: Optional[str] = None) -> bool:
        if label not in L.JOIN_BUTTONS + L.LEAVE_BUTTONS:
            return False
        try:
            r = json.loads(self._run("click", label, *([window] if window else [])) or "{}")
        except Exception as e:  # noqa: BLE001
            log("ax click failed:", type(e).__name__)
            return False
        log("ax click", label, r.get("ok"), r.get("error", ""))
        return bool(r.get("ok"))


LEFT_GONE_POLLS = 20   # ~30 s of polls without the call we left before it may ring us in again
IDLE_QUIET_S = 20.0    # "no call controls" ends a session only after this long without audio


class Daemon:
    def __init__(self, cfg: FaceTimeConfig, brain=None, speech=None, ax=None):
        self.cfg = cfg
        L.set_name(cfg.name)
        if brain is None:
            from catanbot.facetime.brain import Brain
            brain = Brain(cfg, note=lambda m: log(m))
        self.brain = brain
        self.speech = speech or Speech(cfg)
        self.ax = ax or Ax(cfg)
        self.answer = cfg.answer
        self.session: Optional[Session] = None
        self.declined: Dict[str, float] = {}
        self.key = ""
        self.left = ""
        self.left_gone = 0
        self.seen_in_call = False
        self.leaving = False
        self.call_started = 0.0
        self.incall_misses = 0
        self.last_incall: object = "?"
        self.last_sound = 0.0
        self.silent_end_s = cfg.t("silent_end_s")

    def start_session(self, label: str, kind: str) -> None:
        import sounddevice as sd

        player = Player(self.cfg.out_device or "BlackHole 2ch")
        if hasattr(self.brain, "reset"):
            self.brain.reset()
        s = Session(self.cfg, self.brain, self.speech, player, label, kind, on_leave=self.leave)
        dev = find_device(self.cfg.in_device or "BlackHole 16ch", "input")
        in_rate = int(sd.query_devices(dev, "input")["default_samplerate"])
        carry = np.zeros(0, dtype=np.int16)

        def cb(indata, frames, t, status) -> None:  # noqa: ANN001 - PortAudio thread
            nonlocal carry
            if np.abs(indata).max(initial=0.0) > 0.003:
                self.last_sound = time.time()
            x = np.concatenate([carry, L.resample(L.to_mono_int16(L.mix_live(indata[:, :2])), in_rate, L.RATE)])
            n = len(x) // L.FRAME * L.FRAME
            for i in range(0, n, L.FRAME):
                s.frame(x[i : i + L.FRAME])
            carry = x[n:]

        stream = sd.InputStream(samplerate=in_rate, channels=min(2, sd.query_devices(dev, "input")["max_input_channels"]),
                                dtype="float32", device=dev, callback=cb, blocksize=in_rate // 50)
        stream.start()
        s.stream = stream  # type: ignore[attr-defined]
        threading.Thread(target=s.worker, daemon=True).start()
        self.session = s
        log("in call:", kind, s.call_id)

    def end_session(self) -> None:
        if self.session:
            try:
                self.session.stream.stop()  # type: ignore[attr-defined]
                self.session.stream.close()  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass
            self.session.close()
            log("call over:", self.session.call_id, len(self.session.timings), "answers")
        self.session, self.leaving = None, False

    def leave(self) -> None:
        self.left, self.left_gone = self.key, 0
        self.declined[self.key] = time.time()
        if not self.ax.hang_up():
            log("leave: no End/Leave button pressed")
        self.leaving = True

    def hang_up_at_end(self) -> None:
        """Every session end hangs up too: a session closed while FaceTime still holds the call
        would leave the bot in it, deaf. Not a "leave": the next ring is still answered."""
        if not self.ax.hang_up():
            log("call end: no End/Leave button pressed")

    def observe(self) -> Tuple[L.CallState, bool]:
        """The call state, and whether the poll saw anything (a failed detect proves nothing).
        Out of a call: facetime-nc first; a 1:1 ring it names is final. Otherwise facetime-ax.js over
        NotificationCenter + Messages (the group-call Join). In a call: facetime-nc incall, or,
        without the helper, facetime-ax.js over all apps (Leave/End)."""
        if self.session is not None:
            d = self.ax.nc_incall()
            if d is not None:
                playing = bool(getattr(getattr(self.session, "player", None), "playing", False))
                v = L.incall_state(d, playing)
                if v is False and time.time() - self.last_sound < IDLE_QUIET_S:
                    v = None  # controls may be hidden while people still talk: not proof the call ended
                if v != self.last_incall:
                    log("in-call check:", {True: "live", False: "gone", None: "unknown"}[v], "mic", d.get("mic"))
                    self.last_incall = v
                return L.CallState({True: "in-call", False: "idle", None: "unknown"}[v]), v is not None
        nat = L.classify_native(self.ax.nc_detect()) if self.session is None else None
        if nat is not None and nat.people:
            return nat, True
        raw = self.ax.detect(["NotificationCenter", "Messages"] if self.session is None else None)
        st = L.classify(raw)
        if nat is not None and st.state == "ringing" and st.window is None and len(st.people) <= 1:
            st = nat
        return st, bool(raw.get("procs")) or nat is not None

    def tick(self) -> None:
        st, seen = self.observe()
        now = time.time()
        key = L.ring_key(st.people)
        if self.left and seen:
            self.left_gone = 0 if st.state == "ringing" and key == self.left else self.left_gone + 1
            if self.left_gone >= LEFT_GONE_POLLS:
                self.left = ""
        if st.state == "ringing" and self.session is None:
            if key == self.left or now - self.declined.get(key, 0) < 60:
                return
            ok = L.allowed(st.people, self.cfg.callers, self.cfg.groups)
            if not ok:
                log("ringing:", len(st.people), "participants, not on the allow-list; letting it ring")
                self.declined[key] = now
                return
            label, kind = ok
            if not st.join:
                log("ringing: allowed, but no Join/Answer button to press; letting it ring")
                self.declined[key] = now
                return
            if not self.answer:
                log("would answer (\"answer\": false):", kind, st.join)
                self.declined[key] = now
                return
            pressed = self.ax.nc_click(st.join, st.caller) if st.caller else self.ax.click(st.join, st.window)
            if pressed:
                time.sleep(1.5)
                self.key, self.seen_in_call, self.last_sound = key, False, time.time()
                self.call_started, self.incall_misses, self.last_incall = time.time(), 0, "?"
                try:
                    self.start_session(label, kind)
                except Exception as e:  # noqa: BLE001 - never sit in a call deaf and mute
                    log("answered but audio failed, leaving:", type(e).__name__, str(e)[:200])
                    self.ax.hang_up()
                    self.session = None
                    self.declined[key] = now
            else:
                self.declined[key] = now
        elif self.session is not None:
            if st.state == "in-call":
                self.seen_in_call, self.incall_misses = True, 0
            else:
                self.incall_misses += 1
            if self.leaving:
                log("left the call")
                self.end_session()
            elif st.state == "idle" and self.seen_in_call:
                self.hang_up_at_end()
                self.end_session()
            elif self.incall_misses >= int(self.cfg.t("incall_misses")):
                log("no in-call controls for", self.incall_misses, "checks; ending the session")
                self.hang_up_at_end()
                self.end_session()
            elif now - self.call_started > self.cfg.max_call_s:
                log("call longer than max_call_s; leaving")
                self.leave()
                self.end_session()
            elif now - self.last_sound > self.silent_end_s:
                log("call silent for", int(now - self.last_sound), "s; leaving")
                self.leave()
                self.end_session()

    def run(self) -> None:
        log("catanbot-facetime", L.VERSION, "answer" if self.answer else "watch-only",
            len(self.cfg.callers), "callers", len(self.cfg.groups), "groups")
        if self.answer and self.cfg.t("leave_on_start"):
            # a call FaceTime holds from before this start has no audio session: leave it
            self.ax.quit_facetime()
        while True:
            try:
                self.tick()
            except Exception as e:  # noqa: BLE001 - keep watching
                log("tick failed:", type(e).__name__, str(e)[:200])
            time.sleep(1.5)


# -- tools -------------------------------------------------------------------


class StubBrain:
    def __init__(self, delay: float = 0.0):
        self.delay = delay
        self.asked: List[str] = []

    def reply(self, text: str, kind: str = "dm") -> Tuple[str, None]:
        self.asked.append(text)
        time.sleep(self.delay)
        return "Sheep is worth trading for wood right now.", None


def synthetic(cfg: FaceTimeConfig, wav: str, out: Optional[str], kind: str, brain) -> dict:
    """Feed a WAV through the VAD, STT, wake rule, brain and TTS; no devices, no FaceTime."""
    L.set_name(cfg.name)
    pcm, rate = L.read_wav(Path(wav).read_bytes())
    pcm = L.resample(pcm, rate, L.RATE)
    player = FilePlayer()
    t = {"now": 0.0}
    s = Session(cfg, brain, Speech(cfg), player, "synthetic", kind, clock=lambda: t["now"])
    for i in range(0, len(pcm) - L.FRAME + 1, L.FRAME):
        t["now"] = (i + L.FRAME) / L.RATE
        s.frame(pcm[i : i + L.FRAME])
        while not s.q.empty():
            s.handle(*s.q.get())
    for _ in range(int(1.0 * L.RATE / L.FRAME)):
        s.frame(np.zeros(L.FRAME, dtype=np.int16))
        while not s.q.empty():
            s.handle(*s.q.get())
    if out and player.chunks:
        Path(out).write_bytes(L.wav_bytes(np.concatenate(player.chunks), player.rate))
    return {"answers": s.timings}


def mic_check(cfg: FaceTimeConfig, log_dir: Path) -> int:
    """Is the microphone allowed? Plays a quiet 2 s tone into the out device (BlackHole 2ch) and
    records the same device, which loops back. A denied microphone records pure zeros, without an
    error. Run it once through the app bundle so macOS shows the Allow prompt; never during a call."""
    import sounddevice as sd

    name = cfg.out_device or "BlackHole 2ch"
    try:
        dev = find_device(name, "input")
        rate = int(sd.query_devices(dev, "input")["default_samplerate"])
        t = np.arange(int(rate * 2.0)) / rate
        tone = (0.05 * np.sin(2 * np.pi * 440 * t)).astype("float32")[:, None]
        rec = sd.playrec(tone, samplerate=rate, channels=1, dtype="float32", device=(dev, dev))
        sd.wait()
        peak = float(np.abs(rec).max(initial=0.0))
        verdict = "ALLOWED" if peak > 0.001 else "DENIED (recorded silence)"
    except Exception as e:  # noqa: BLE001
        peak, verdict = 0.0, f"FAILED: {type(e).__name__}: {str(e)[:200]}"
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} mic-check {name}: peak={peak:.4f} {verdict}"
    print(line, flush=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "mic-check.log", "a") as f:
        f.write(line + "\n")
    return 0 if verdict == "ALLOWED" else 1


def level_meter(cfg: FaceTimeConfig, seconds: float, log_dir: Path) -> int:
    """Peak and RMS (int16 scale) of the in device once per second, per channel and as the daemon
    mixes it. Numbers only; safe during a call (it only reads)."""
    import sounddevice as sd

    name = cfg.in_device or "BlackHole 16ch"
    log_dir.mkdir(parents=True, exist_ok=True)
    out = open(log_dir / "level.log", "a")

    def say(line: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {line}"
        print(line, flush=True)
        out.write(line + "\n")
        out.flush()

    try:
        dev = find_device(name, "input")
        info = sd.query_devices(dev, "input")
        rate, ch = int(info["default_samplerate"]), min(2, int(info["max_input_channels"]))
        say(f"level {name}: {ch} of {int(info['max_input_channels'])} channels at {rate} Hz, VAD floor {cfg.t('vad_floor'):.0f}")
        with sd.InputStream(samplerate=rate, channels=ch, dtype="float32", device=dev) as st:
            for _ in range(max(1, int(seconds))):
                block, _ = st.read(rate)
                per = " ".join(f"ch{i + 1} peak {int(np.abs(block[:, i]).max(initial=0) * 32767)}" for i in range(block.shape[1]))
                m = L.to_mono_int16(L.mix_live(block)).astype(np.float32)
                say(f"level: {per} | mixed peak {int(np.abs(m).max(initial=0))} rms {float(np.sqrt(np.mean(m * m))):.0f}")
        return 0
    except Exception as e:  # noqa: BLE001
        say(f"level FAILED: {type(e).__name__}: {str(e)[:200]}")
        return 1
    finally:
        out.close()
