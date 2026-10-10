"""Live loop: look at the game window, decide with the brain, click.

All coordinates are in the 1372x888 calibration frame (data/calib/ui.json). Every image,
live or saved, is first normalised to that frame, so saved captures replay exactly.

Hard rules (from the owner): private games only, never matchmaking; clicks only inside the
game window; stop when data/STOP exists or the mouse sits in the top-left corner.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import config
from .brain import Brain
from .learning import Model
from .rules import IMPROVEMENT_TRACKS, can_afford, discard_limit
from .rules_reader import parse_rules
from .state import GameState, Knight, PlayerState, new_hand
from .vision.layout import REF_H, REF_W, TOP, Layout
from .vision.screens import FORBIDDEN_BUTTONS, active_seat, classify, find_popup_option, find_text, ocr_near, offline_menu

ROOT = Path(__file__).resolve().parent.parent
UI = json.loads((ROOT / "data" / "calib" / "ui.json").read_text())
STOP = ROOT / "data" / "STOP"
# SPEED_PLAN 5/7 (owner 2026-10-09 "play every move way faster"): clicks that keep the full
# click_delay_s (a popup confirm, a dialog's card rows clicked back to back, drags) ...
SETTLE_CLICK = re.compile(r"confirm|green check|popup|accept|\bok\b|^(bank )?(want|give) |^setup |"
                          r"^send offer|^bank trade|^progress (select|play)|^monopoly|^steal|^choose ", re.I)
# ... and our clicks after which the game animates (dice, trade, hand-over): transition frames
KEY_CLICK = re.compile(r"^(roll dice|end turn)\b", re.I)
FRAME = (1372, 888)
# prompts that each answer in seconds: one of these on screen for 15 min means we're stuck
STALL_SCREENS = {"pick_progress", "take_cards", "choose_player", "choose_deck", "discard", "discard_progress", "place_robber", "place_pirate", "steal", "place_ship",
                 "promote_knight", "place_knight", "place_metropolis", "place_wall", "remove_city", "lose_knight", "choose_resource",
                 "harbor_player"}
BOARD_ROI = (90, 180, 1000, 820)   # game 2: the canal board runs left/below the old box
HAND_XS = [395, 450, 506, 555, 605, 660, 713, 765]      # hand bar digit centres, y ~828
HAND_Y = (814, 842)
CARD_ORDER = UI["trade_order"]
BAR_BOX = (812, 846, 380, 790)     # hand bar: y0, y1, x0, x1
_BASE = dict(UI=copy.deepcopy(UI), HAND_XS=list(HAND_XS), HAND_Y=HAND_Y, BOARD_ROI=BOARD_ROI, BAR_BOX=BAR_BOX)
LAY = Layout(REF_W, REF_H)         # the laptop window: identity


def set_layout(lay: Layout) -> None:
    """Move every fixed frame coordinate for this window shape (see vision/layout.py)."""
    global LAY, HAND_XS, HAND_Y, BOARD_ROI, BAR_BOX
    LAY = lay
    UI.clear()
    UI.update(lay.ui(_BASE["UI"]))
    HAND_XS = [int(round(lay.x(x, "c"))) for x in _BASE["HAND_XS"]]
    HAND_Y = tuple(int(round(lay.y(y, "b"))) for y in _BASE["HAND_Y"])
    y0, y1, x0, x1 = _BASE["BAR_BOX"]
    BAR_BOX = (int(lay.y(y0, "b")), int(lay.y(y1, "b")), int(lay.x(x0, "c")), int(lay.x(x1, "c")))
    BOARD_ROI = _BASE["BOARD_ROI"] if lay.is_reference else (80, 150, lay.size[0] - 10, lay.size[1] - 40)
    Runner.RESP_XS = tuple(int(round(lay.x(x, "c"))) for x in Runner._RESP_XS0)


def to_frame(img: np.ndarray) -> np.ndarray:
    if (img.shape[1], img.shape[0]) == FRAME:
        return img
    return cv2.resize(img, FRAME, interpolation=cv2.INTER_CUBIC)


# ---- screen I/O ----------------------------------------------------------------------

def _norm(name: str) -> str:
    return (name or "").lower().replace(" ", "")

class ReplayIO:
    """Feeds saved captures; records clicks instead of sending them."""

    def __init__(self, paths: list[str]):
        self.paths, self.i, self.clicks = list(paths), 0, []

    def snapshot(self) -> np.ndarray | None:
        if self.i >= len(self.paths):
            return None
        img = cv2.imread(self.paths[self.i])
        self.i += 1
        return to_frame(img)

    def click(self, x: float, y: float, why: str = "") -> None:
        self.clicks.append((int(x), int(y), why))

    def drag(self, p0, p1, why: str = "") -> None:
        self.clicks.append((int(p0[0]), int(p0[1]), why))

    def move(self, x, y, why: str = "") -> None:
        self.__dict__.setdefault("moves", []).append((int(x), int(y), why))

    def type_text(self, text: str, enter: bool = True) -> None:
        self.clicks.append((-1, -1, f"type {text!r}" + (" + Return" if enter else "")))

    def ocr(self, img):
        from . import mac
        return mac.ocr(img)

    def paused(self) -> bool:
        return False


class LiveIO:
    """Quartz capture of the game window, mapped into the calibration frame."""

    def __init__(self):
        from . import mac
        self.mac = mac
        self.win = None
        self.lay = None                  # Layout of the current game window
        self.native = self._frame = None
        self.layout_ok = False

    def snapshot(self) -> np.ndarray:
        """The game window's content as a "virtual frame": scaled so Unity's controls have the
        size they have in the laptop calibration (vision/layout.py). Owner 2026-10-03: full screen
        for more detail; OCR still reads the full-resolution capture."""
        img, self.win = self.mac.capture()
        h, w = img.shape[:2]
        if self.lay is None or (self.lay.w, self.lay.h) != (w, h):
            self.lay = Layout(w, h)
            set_layout(self.lay)
            print(f"LAYOUT game window {w}x{h} px: UI x{self.lay.s:.3f}, frame {self.lay.size}, "
                  f"extra {self.lay.dx:+.0f},{self.lay.dy:+.0f} px vs the laptop", flush=True)
            try:                                     # audit 2026-10-03: one 1920x1022 capture went by unnoticed
                sw, sh = self.mac.screen_px()
                if (w, h) != (sw, sh) and (w, h) != (sw * 2, sh * 2):
                    print(f"LAYOUT WARNING: game window {w}x{h} is not full screen ({sw}x{sh}); "
                          f"clicks are scaled but set the game to full screen", flush=True)
            except Exception:                        # noqa: BLE001
                pass
        self.layout_ok = True
        self.native = img
        self._frame = self.lay.frame(img, FRAME)
        return self._frame

    def click(self, x: float, y: float, why: str = "") -> None:
        if not self.layout_ok:
            raise RuntimeError(f"layout not known yet: not clicking ({why})")
        self.mac.click(self.win, *self.lay.to_native(x, y))   # raises if outside the game window

    def move(self, x: float, y: float, why: str = "") -> None:
        if self.layout_ok:
            self.mac.move(self.win, *self.lay.to_native(x, y))

    def drag(self, p0, p1, why: str = "") -> None:
        if not self.layout_ok:
            raise RuntimeError(f"layout not known yet: not dragging ({why})")
        fw, fh = FRAME
        if not all(0 <= p[0] < fw and 0 <= p[1] < fh for p in (p0, p1)):
            raise RuntimeError(f"drag outside the game frame: {p0}->{p1} ({why})")
        self.mac.drag(self.win, self.lay.to_native(*p0), self.lay.to_native(*p1))

    def type_text(self, text: str, enter: bool = True) -> None:
        """Type into the focused game field (chat). Brings the game to front first, like click."""
        for _ in range(3):
            if self.mac.frontmost_is_game():
                break
            self.mac.activate()
        if not self.mac.frontmost_is_game():
            raise RuntimeError("game window not frontmost: not typing")
        if not self.layout_ok:
            raise RuntimeError("layout not calibrated: not typing")
        self.mac.type_text(text)
        if enter:
            time.sleep(0.05)
            self.mac.key(36)             # Return sends (the chat has no send button)

    def ocr(self, img):
        # not the full-resolution capture: Apple Vision found 7 lines and no player names on the
        # 1920x1080 screen vs all of them on the frame (2026-10-03)
        return self.mac.ocr(img)

    def paused(self) -> bool:
        """Mouse in the main screen's top-left corner = pause (step() does nothing until it moves).
        review 0.3.1: the screen list is NSScreen's -- NSEvent has no `screens`, so the old
        hasattr guard made this always False."""
        from AppKit import NSEvent, NSScreen
        screens = NSScreen.screens()
        if not screens:
            return False
        p = NSEvent.mouseLocation()             # Cocoa: origin at the main screen's bottom-left
        return p.x < 3 and p.y > screens[0].frame().size.height - 3


# ---- small readers ---------------------------------------------------------------------
def _patch(hsv, x, y, r=18):
    return hsv[y - r:y + r, x - r:x + r].reshape(-1, 3).astype(float)


def _orange(hsv, xy) -> float:
    p = _patch(hsv, *xy)
    return float(((p[:, 0] > 8) & (p[:, 0] < 30) & (p[:, 1] > 120) & (p[:, 2] > 150)).mean())


def turn_phase(img: np.ndarray) -> str:
    """'main' (rolled, trade+build enabled), 'roll' (dice waiting) or 'idle'.
    Calibrated on gc_turn1_* / gc_other_turn / gc_trade_*: grey buttons = not our action."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if _orange(hsv, UI["trade_open"]) > 0.4:
        return "main"
    if _orange(hsv, UI["dice_or_end_turn"]) > 0.4:
        return "roll"
    return "idle"


class HandReader:
    """Reads the 8 card counts on the bottom bar. Zero is drawn dimmed; other digits are
    matched against glyph templates labelled from captures (data/calib/hand_digits.npz).
    Unknown glyphs read as None so the caller can fall back to its own tracking."""
    PATH = ROOT / "data" / "calib" / "hand_digits.npz"

    def __init__(self):
        self.t: dict[int, list[np.ndarray]] = {}
        if self.PATH.exists():
            d = np.load(self.PATH)
            for k in d.files:
                self.t[int(k.split("_")[0])] = self.t.get(int(k.split("_")[0]), []) + [d[k]]

    @staticmethod
    def glyph(img, x) -> np.ndarray:
        g = cv2.cvtColor(img[HAND_Y[0]:HAND_Y[1], x - 12:x + 10], cv2.COLOR_BGR2GRAY)
        return cv2.resize(g, (22, 28)).astype(np.float32)

    def add(self, n: int, g: np.ndarray):
        self.t.setdefault(n, []).append(g)

    def save(self):
        np.savez(self.PATH, **{f"{n}_{i}": g for n, gs in self.t.items() for i, g in enumerate(gs)})

    BAR_LEFT0 = 360          # purple bar's left edge on the laptop calibration frame

    @staticmethod
    def bar_left(img) -> int | None:
        """Left edge of the purple hand bar. The bar is centred and grows when progress cards
        are held (live 2026-10-03: it moved 55 px left and every count shifted one card)."""
        y = int(round(LAY.y(818, "b")))
        if not 0 <= y < img.shape[0]:
            return None
        hsv = cv2.cvtColor(img[y:y + 1], cv2.COLOR_BGR2HSV)[0]
        m = (hsv[:, 0] > 135) & (hsv[:, 0] < 170) & (hsv[:, 1] > 120) & (hsv[:, 2] > 60)
        xs = np.where(m)[0]
        if len(xs) < 300:
            return None
        # icons and digits break the purple up: join gaps up to 60 px, keep the widest span
        runs = np.split(xs, np.where(np.diff(xs) > 60)[0] + 1)
        run = max(runs, key=lambda r: r[-1] - r[0])
        return int(run[0]) if run[-1] - run[0] > 400 else None

    def xs(self, img) -> list[int]:
        left = self.bar_left(img)
        if left is None:
            return list(HAND_XS)
        return [left + x - self.BAR_LEFT0 for x in _BASE["HAND_XS"]]

    TWO_DIGIT_SHIFT = 8      # probe 2026-10-04: "10 wool" pushed every later count 8 px right

    @staticmethod
    def _second_digit(img, x) -> bool:
        """Two-digit count: right of the first digit, a second glyph after a gap of at most 5 px
        (inside "10" the gap is ~4 px; a digit and its card icon are 7+ px apart)."""
        g = cv2.cvtColor(img[HAND_Y[0]:HAND_Y[1], max(0, x - 8):x + 22], cv2.COLOR_BGR2GRAY)
        if g.size == 0:
            return False
        cols = ((g > 175).sum(0) >= 4).astype(int).tolist()
        runs, start = [], None
        for i, c in enumerate(cols + [0]):
            if c and start is None:
                start = i
            elif not c and start is not None:
                runs.append((start, i - 1))
                start = None
        if len(runs) < 2:
            return False
        (a0, a1), (b0, b1) = runs[0], runs[1]
        return a0 <= 12 and b0 - a1 - 1 <= 5 and b1 - b0 >= 3

    def read(self, img) -> dict[str, int | None]:
        out = {}
        shift = 0
        for name, x0 in zip(CARD_ORDER, self.xs(img)):
            x = x0 + shift
            if self.ocr_fn is not None and self._second_digit(img, x):
                n = self.ocr_digit(img, x, two=True)
                for dx in (-3, 3):               # a 1 is narrow: nudge the crop once each way
                    if n is None:
                        n = self.ocr_digit(img, x + dx, two=True)
                out[name] = n                    # None = unread, but the later slots still shift
                shift += self.TWO_DIGIT_SHIFT
                continue
            g = self.glyph(img, x)
            if g.max() < 175:                      # dimmed digit = 0
                out[name] = 0
                continue
            best, bs = None, 0.0
            for n, gs in self.t.items():
                for tg in gs:
                    s = float(cv2.matchTemplate(g, tg, cv2.TM_CCOEFF_NORMED).max())
                    if s > bs:
                        best, bs = n, s
            if bs <= 0.8 and self.ocr_fn is not None:
                n = self.ocr_digit(img, x)
                if n is not None:
                    # live 2026-10-02: only 0-4 had templates, so 5+ read as unknown -> the
                    # bot saw an empty hand with 12 cards. Learn the glyph from OCR.
                    self.add(n, g)
                    try:
                        self.save()
                    except Exception:            # noqa: BLE001
                        pass
                    best, bs = n, 1.0
            out[name] = best if bs > 0.8 else None
        return out

    ocr_fn = None

    def ocr_digit(self, img, x, two: bool = False) -> int | None:
        # tested live 2026-10-02: grey>170, x2 scale, a "VP" label in front makes the OCR read
        # a lone digit ("VP 2"); a wider crop catches the yellow dot and fails
        c = img[HAND_Y[0]:HAND_Y[1], x - 13:x + (17 if two else 8)]
        m = (cv2.cvtColor(c, cv2.COLOR_BGR2GRAY) > 170).astype(np.uint8) * 255
        if m.mean() < 5:
            return None
        r = cv2.resize(255 - m, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        mm = cv2.copyMakeBorder(r, 40, 40, 140, 60, cv2.BORDER_CONSTANT, value=255)
        cv2.putText(mm, "VP", (10, 40 + r.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 2, 0, 5)
        boxes = self.ocr_fn(cv2.cvtColor(mm, cv2.COLOR_GRAY2BGR))
        if two:                                  # "10" often comes back as its own box
            nums = [int(d) for t in boxes for d in re.findall(r"\d{2}", t["text"]) if 10 <= int(d) <= 30]
            return max(nums) if nums else None
        for t in boxes:
            d = re.search(r"VP\s*(\d{1,2})\b", t["text"])
            if d and int(d.group(1)) <= 30:
                return int(d.group(1))
        return None


# ---- the runner ------------------------------------------------------------------------
class _ByColour:
    """cfg confirm_by_colour: on an instance the switch picks the colour search or the fixed
    point; on the class (and with no cfg) the colour search."""

    def __init__(self, colour: str, fixed: str):
        self.colour, self.fixed = colour, fixed

    def __get__(self, obj, cls):
        cfg = (obj.__dict__.get("cfg") or {}) if obj is not None else {}
        return getattr(cls, self.colour if cfg.get("confirm_by_colour", True) else self.fixed)


@dataclass
class Runner:
    from .probe import Probe as _Probe
    probe = _Probe(ROOT, enabled=False)   # replaced per instance in __post_init__ (tests may skip it)
    _pend = None
    # probe game 2 2026-10-04: a runner started mid-game never passed the home/lobby reset, and the
    # first barbarian text raised AttributeError on every step for 20 min (then the stall restart)
    barbarians_attacked = False
    _in_play = False
    io: object
    cfg: dict = field(default_factory=config.load)
    log: list = field(default_factory=list)

    def __post_init__(self):
        from .probe import Probe
        # HANDOFF_v02 Part A: probe frames + per-action scorecard (off in frame replays)
        self.probe = Probe(ROOT, enabled=not isinstance(self.io, ReplayIO) or bool(self.cfg.get("probe_in_replay")))
        self._pend = None
        from . import progress_cards as _pc
        _pc.INVENTOR_BARREN = bool(self.cfg.get("inventor_barren", True))
        self.model = Model()
        self.brain = Brain(self.model)
        self.brain.note = self.note               # brain notes (held progress cards, spy) into our log
        self._brain_cfg()
        self.hand = HandReader()
        self.hand.ocr_fn = getattr(self.io, "ocr", None) if isinstance(self.io, LiveIO) else None
        self.gs: GameState | None = None
        self.mode = None
        self.lattice = None          # fixed after the first board read of a game
        self._harbors_best = None
        self._harbors_tpl = False
        self._harbors_seen, self._harbors_denied = None, set()
        self._canal_ref, self._canal_seen = None, set()
        self.offers_this_turn = 0
        self.turn_started = None
        from .chat import Chat
        self.chat = Chat(self.cfg.get("bot_name", "marlow.systems"))
        self._intro_sent = False          # one intro per game; reset with game_colors
        self._load_game_flags()
        self._saw_game_start = False      # False after a mid-game restart -> no repeat intro
        self._chat_primed = False
        self._chat_last_poll = 0.0
        self._feed_dice, self._feed_events = [], []     # live feed (public rolls + events)
        self._end_turn_at = 0.0

    # -- plumbing
    def note(self, msg: str):
        self.log.append(msg)
        print(time.strftime("%H:%M:%S"), msg, flush=True)
        if self.GLOG_FAIL.search(msg) and not msg.startswith(self.GLOG_FAIL_SKIP) \
                and not any(k in msg for k in self.GLOG_FAIL_REPEAT):
            gl = self._glog(start=False)
            if gl is not None:
                gl.failure(msg[:200])
        if msg.startswith(self.GLOG_NOTES):
            gl = self._glog(start=False)
            if gl is not None:
                gl.note(msg[:200])
        elif msg.startswith(self.GLOG_DECISIONS) or self.GLOG_LIMIT.match(msg):
            gl = self._glog(start=False)
            if gl is not None:
                gl.decision(msg[:160])

    # -- speed (owner 2026-10-09 "figure out how the bot can play every move way faster. it runs
    # out of time almost always when we have a timer (usually 100s)")
    def _live_waits(self) -> bool:
        """Real sleeps: live, or a test IO that asks for them (real_sleeps) to count them."""
        io = self.__dict__.get("io")
        return not isinstance(io, ReplayIO) or bool(getattr(io, "real_sleeps", False))

    def _tcount(self, key: str, n: int = 1) -> None:
        """Per own turn counters for the 'turn done:' line."""
        tc = self.__dict__.get("_turn_counts")
        if tc is not None:
            tc[key] = tc.get(key, 0) + n

    def _turn_begin(self) -> None:
        self._turn_counts = {"clicks": 0, "reads": 0, "probes": 0, "chat polls": 0}
        self._phases, self._phase_cur = {}, ("roll", time.time())

    def _phase(self, name: str) -> None:
        """0.4.3 step 8: the time since the last mark goes to the phase that was running."""
        ph = self.__dict__.get("_phases")
        if ph is None:
            return
        cur, t = self.__dict__.get("_phase_cur") or (name, time.time())
        now = time.time()
        ph[cur] = ph.get(cur, 0.0) + now - t
        self._phase_cur = (name, now)

    def _turn_done(self, how: str) -> None:
        """One line per own turn: how long and what it spent (SPEED_PLAN 'measure after')."""
        tc, t0 = self.__dict__.get("_turn_counts"), self.__dict__.get("turn_started")
        if tc is None or t0 is None:
            return
        self._turn_counts = None
        cfg = self.__dict__.get("cfg") or {}
        if self.__dict__.get("_phases") is not None:
            self._phase("end")
            ph, self._phases = self._phases, None
            if cfg.get("log_phases", True):
                timer = self._turn_timer_s()
                tail = f"s; clock {self._turn_used_s():.0f} of {timer:.0f}" if timer else "s"
                self.note("phase: " + " ".join(f"{k} {v:.0f}" for k, v in ph.items() if v >= 0.5) + f" ({tail})")
        if not cfg.get("log_timing", True):
            return
        self.note(f"turn done: {time.time() - t0:.0f} s, clicks {tc.get('clicks', 0)}, reads {tc.get('reads', 0)}, "
                  f"probes {tc.get('probes', 0)}, chat polls {tc.get('chat polls', 0)}, "
                  f"offers {self.__dict__.get('offers_this_turn', 0)}, batched builds {tc.get('batched builds', 0)}, "
                  f"preplan hits {tc.get('preplan hits', 0)} ({how})")

    def _key_click_recent(self, s: float) -> bool:
        """Our own roll / trade OK / End turn click less than s seconds ago (transition frames follow)."""
        k = self.__dict__.get("_key_click")
        return k is not None and time.time() - k[1] < s

    def _our_turn_open(self) -> bool:
        ts = self.__dict__.get("turn_started")
        closed = max(self.__dict__.get("_end_turn_at", 0.0), self.__dict__.get("_turn_closed_at", 0.0))
        return bool(ts) and closed < ts and time.time() - ts < 600

    def _loop_sleep_s(self) -> float:
        """Main loop wait: 0.25 s right after our click, poll_s_turn on our own turn, poll_s otherwise."""
        cfg = self.cfg
        acted = time.time() - self.__dict__.get("_last_click_at", 0) < 1.5
        if acted:
            return min(0.25, cfg["poll_s"])
        if cfg.get("fast_transitions", True) and self.__dict__.get("_step_state") == "unknown" \
                and self._key_click_recent(3.0):
            return min(0.25, cfg["poll_s"])
        if cfg.get("offer_poll_s") and self._our_turn_open() and self.__dict__.get("offer_sent") \
                and time.time() - self.offer_sent < self.offer_wait_s() + 2:
            # 0.4.3 step 6: our offer is out -- look for the answers often, they decide the next step
            return min(float(cfg["offer_poll_s"]), cfg["poll_s"])
        if cfg.get("poll_s_turn") and self._our_turn_open():
            return min(float(cfg["poll_s_turn"]), cfg["poll_s"])
        return cfg["poll_s"]

    # ---- bot 0.4.3 turn clock (config turn_clock / turn_ring_read; catanbot/turn_clock.py) ----
    def _turn_timer_s(self) -> float | None:
        """The game's turn timer: the rules panel's Round Timer (one value a game), else
        turn_timer_default_s. None = the old fixed budgets (switch off, or a solo game with no
        timer read -- solo games show no timer border)."""
        cfg = self.__dict__.get("cfg") or {}
        if not cfg.get("turn_clock", True):
            return None
        t = getattr(self.__dict__.get("mode"), "turn_timer_s", None)
        if t:
            return float(t)
        if cfg.get("solo"):
            return None
        return float(cfg.get("turn_timer_default_s", 100) or 100)

    def _turn_used_s(self) -> float:
        """Seconds of our turn used: from the first roll frame of the turn, moved forward when the
        timer border showed more used (the border is the game's own count)."""
        ts = self.__dict__.get("turn_started")
        t0 = self.__dict__.get("_clock_t0") if self.__dict__.get("_clock_turn") == ts else None
        t0 = t0 or ts
        return time.time() - t0 if t0 else 0.0

    def _turn_cut(self, name: str) -> float | None:
        timer = self._turn_timer_s()
        if timer is None:
            return None
        from .turn_clock import phase_cutoffs
        return phase_cutoffs(timer, self.cfg)[name]

    def _turn_past(self, name: str) -> bool:
        """Our turn is past this phase cutoff (late / offer / chat / end / hard)."""
        if not self.__dict__.get("turn_started"):
            return False
        cut = self._turn_cut(name)
        if cut is None:
            return False
        return self._turn_used_s() > cut

    def _turn_late(self) -> bool:
        if self._turn_timer_s() is not None:
            return self._turn_past("late")
        ts = self.__dict__.get("turn_started")
        return bool(ts) and time.time() - ts > self.cfg["turn_budget_s"]

    def _ring_check(self, img) -> None:
        """Owner 2026-10-10 "the timer is a color edge around the next button": read the border's
        used share; when it is ahead of our clock by > 3 s, our clock moves to it (never back --
        the border seemed to stop while a card dialog was open)."""
        if img is None or not self.cfg.get("turn_ring_read", True) or not self.__dict__.get("turn_started"):
            return
        timer = self._turn_timer_s()
        if timer is None:
            return
        from .turn_clock import ring_read
        xy = UI.get("dice_or_end_turn")
        if not xy:
            return
        try:
            r = ring_read(img, (int(round(xy[0] + 1)), int(round(xy[1] - 2))))
        except Exception:                            # noqa: BLE001 -- the clock never stops play
            return
        if r is None:
            return
        frac, hue = r
        ring_used = frac * timer
        used = self._turn_used_s()
        # two reads that agree on the turn's start (within 4 s) before the clock moves: one odd
        # frame (a banner over the button) never ends a turn early
        start = time.time() - ring_used
        prev, self._ring_start = self.__dict__.get("_ring_start"), (start, self.turn_started)
        if prev is None or prev[1] != self.turn_started or abs(prev[0] - start) > 4:
            return
        if ring_used > used + 3:
            self._clock_t0, self._clock_turn = time.time() - ring_used, self.turn_started
            self.note(f"timer border: {frac:.2f} used (hue {hue}) = {ring_used:.0f} s, our clock said {used:.0f} s: moved")

    # owner 2026-10-08: one log per game, every turn (catanbot/game_log.py)
    LOBBY_CONFIRM = 3       # live 2026-10-08: one false lobby frame mid-game wiped the game
    GLOG_NOTES = ("offer from", "counter offer", "wildcard from", "game over", "barbarian", "steal",
                  "robber", "played ", "monopoly", "lost a city", "metropolis", "canal")
    # owner 2026-10-09: the strategy too -- key decisions into the ledger. Never the hand or our
    # progress cards: 'bank set: 2 wool ...', 'me: hand=...', 'progress ...: holding <card>' and
    # 'spy: took <card>' stay out; the plan goes in through gl.plan (hand in the event file only)
    GLOG_DECISIONS = ("knight first", "save the city", "spend the turn", "opening:", "city lost",
                      "situational:", "barbarians close", "threat before our turn", "progress deck")
    GLOG_LIMIT = re.compile(r"^hand \d+ (>|near) limit")
    # owner 2026-10-09 "every failure must be recorded in the logs" (game_log.failure)
    GLOG_FAIL = re.compile(r"fail|not seen|did not land|not completed|timed out|gave up|giving up|stuck|"
                           r"unreadable|refused|no effect|didn't|could not|couldn't|not found|missed|"
                           r"\bUNVERIFIED\b|traceback|error", re.I)
    GLOG_FAIL_SKIP = ("click ", "drag ", "probe")    # the click itself; its outcome is its own note
    GLOG_FAIL_REPEAT = ("already failed this turn", "refused earlier this game")   # a skip, logged once already

    def _glog(self, start: bool = True, ending: bool = False):
        """The game's turn log; opened at the first in-game hook after a lobby, resumed after a
        runner restart mid-game, archived when the next game opens one. ending=True: the open
        game's log even after leaving it (the game-over screen), to record its end."""
        if (isinstance(getattr(self, "io", None), ReplayIO) and not getattr(self, "_glog_force", False)) \
                or (getattr(self, "cfg", None) or {}).get("dry_run"):
            return None
        gl = self.__dict__.get("_glog_obj")
        if gl is None:
            from .game_log import GameLog
            gl = self._glog_obj = GameLog()
            if ending or not getattr(self, "_glog_new", False):
                gl.resume()
        if gl.active() and (ending or not getattr(self, "_glog_new", False)):
            return gl
        if ending:
            return None
        if not start or not (getattr(self, "_glog_new", False) or gl.game_id is None):
            return None
        seats = [n for n, _ in (getattr(self, "game_colors", None) or [])]
        gs = getattr(self, "gs", None)
        if not seats or gs is None:
            return None
        try:
            gid = self._game_file("x").parent.name
        except Exception:                             # noqa: BLE001
            gid = None
        gl.start(getattr(gs.mode, "key", None), seats, gs.me().name, gs.target, game_id=gid)
        self._glog_new = False
        self.note(f"game log: new game {gl.game_id}, seats {seats}")
        if not isinstance(self.io, ReplayIO):
            from .chat import record_game_open
            record_game_open(gl.game_id, solo=bool((self.cfg or {}).get("solo")), seats=seats,
                             mode=getattr(gs.mode, "key", None))
        # live 2026-10-08 21:45:40: the new game kept the old game's harbours and never logged
        # 'harbours read' -- every new game reads (and logs) its harbours again
        self._reset_harbors(f"game {gl.game_id}")
        return gl

    def _glog_vp(self) -> dict:
        gs = getattr(self, "gs", None)
        return {p.name: p.vp_visible for p in gs.players} if gs is not None else {}

    def _click_pause(self, xy, why: str, settle=None) -> float:
        """SPEED_PLAN 5: a click the next screen read checks anyway (open/close panels, tabs, roll,
        end turn) waits click_delay_fast_s; popup confirms, trade OK and the card rows of a dialog
        (several clicks with no read between) keep click_delay_s."""
        slow = float(self.cfg["click_delay_s"])
        fast = self.cfg.get("click_delay_fast_s")
        if fast is None:
            return slow
        if settle is None:
            settle = bool(SETTLE_CLICK.search(why or "")) or any(
                tuple(xy) == tuple(UI.get(k, ())) for k in ("trade_ok", "setup_confirm"))
        return slow if settle else min(float(fast), slow)

    def click(self, xy, why: str, ocr=None, settle=None):
        if ocr is not None:
            # never press a matchmaking/public button, whatever the screen says
            near = [o for o in ocr if abs(o["x"] + o["w"] / 2 - xy[0]) < 60 and abs(o["y"] + o["h"] / 2 - xy[1]) < 25]
            if any(FORBIDDEN_BUTTONS.search(o["text"]) for o in near):
                self.note(f"REFUSED click {why}: forbidden button nearby")
                return
        if self._click_loop(xy, why):
            return
        # review 0.3.1: stamped only for a click that happens -- a refused or loop-blocked click
        # made _after_handler think the handler acted (the stall watch and the pending prompt)
        self._last_click_at = time.time()
        gl = self._glog(start=True) if getattr(self, "gs", None) is not None else None
        if gl is not None and why:
            (gl.setup if why.startswith("setup") else gl.action)(why[:120])
        if self.cfg.get("dry_run"):
            self.note(f"[dry] click {why} at {tuple(int(a) for a in xy)}")
            if isinstance(self.io, ReplayIO):
                self.io.click(*xy, why)
            return
        self.note(f"click {why} at {tuple(int(a) for a in xy)}")
        self.io.click(*xy, why)
        self._tcount("clicks")
        if KEY_CLICK.search(why) or tuple(xy) == tuple(UI.get("trade_ok", ())):
            self._key_click = (why, time.time())
        if self._live_waits():
            time.sleep(self._click_pause(xy, why, settle))
        if xy[1] >= 690 and not isinstance(self.io, ReplayIO):
            # game 7 12:19: the pointer left on the hand bar after a trade showed a card's big
            # preview over the trade button; the next "open trade" hit the card (4 bank trades
            # failed). Clicks down there park the pointer on empty table.
            try:
                self.io.move(*UI.get("pointer_park", self.PARK_SPOT), "park")
            except Exception:                        # noqa: BLE001
                pass

    # -- probe capture (HANDOFF_v02 Part A)
    PROMPT_KINDS = {"pick_progress", "take_cards", "choose_player", "choose_deck", "discard", "discard_progress", "place_robber", "place_pirate", "steal", "place_ship",
                    "promote_knight", "place_knight", "place_metropolis", "place_wall", "remove_city", "lose_knight", "choose_resource", "roll",
                    "setup_settlement", "setup_city", "setup_road"}
    RING_FALLBACK = {"place_robber", "place_pirate", "place_knight", "promote_knight", "place_metropolis", "place_wall", "lose_knight",
                     "select_token"}

    # failure probes stay written inside the turn (SPEED_PLAN 1): they are rare and explain a loss
    PROBE_SYNC = ("bank_failed", "harbor_after_", "progress_once_", "stall_")

    def _probe(self, kind, img=None, note=None, texts=None):
        """Save the frame, its native capture, a board/state summary and a note to data/probe."""
        probe = getattr(self, "probe", None)
        if probe is None or not probe.enabled:
            return None
        if img is None:
            img = getattr(self, "_last_img", None)
        io = self.io
        native = getattr(io, "native", None) if getattr(io, "_frame", None) is img else None
        if texts is None and img is getattr(self, "_last_img", None):
            texts = [o["text"] for o in (getattr(self, "_last_ocr", None) or [])]
        self._tcount("probes")
        cfg = self.__dict__.get("cfg") or {}
        if cfg.get("probe_async", True) and img is not None and not kind.startswith(self.PROBE_SYNC) \
                and hasattr(probe, "save_async"):
            # owner 2026-10-09 "play every move way faster": ~4 s per save came off our turn.
            # The game state is copied now; rings, edge rings and the PNG/JSON run on the worker.
            base = self._probe_scan_base()
            lay = getattr(io, "lay", None) if native is not None else None
            br = getattr(self, "br", None)
            boxes = [dict(o) for o in (getattr(self, "_last_ocr", None) or []) if len(o["text"]) >= 8] \
                if img is getattr(self, "_last_img", None) else []

            def scan_fn(im, nat, base=base, lay=lay, br=br, boxes=boxes):
                if base is None or "error" in base:
                    return base
                out = dict(base)
                rings = self.ring_spots(im, nat, lay) if nat is not None and lay is not None else self.ring_spots(im)
                out["rings"] = [tuple(map(int, r)) for r in rings
                                if not any(o["x"] - 8 <= r[0] <= o["x"] + o["w"] + 8 and
                                           o["y"] - 8 <= r[1] <= o["y"] + o["h"] + 8 for o in boxes)]
                try:
                    from .vision.rings import EDGE_THRESHOLD, edge_rings
                    out["edge_rings"] = sum(1 for t in edge_rings(im, br.lattice, br.board, nat, lay, threshold=0.0)
                                            if t[0] >= EDGE_THRESHOLD) if br is not None else 0
                except Exception as e:                # noqa: BLE001
                    out["edge_rings"] = f"error {type(e).__name__}"
                out["turn_phase"] = turn_phase(im)
                return out

            def done(p):
                if p:                                 # the worker's line (no game-log hooks off the main thread)
                    msg = f"probe: saved {p.name}"
                    self.log.append(msg)
                    print(time.strftime("%H:%M:%S"), msg, flush=True)
            return True if probe.save_async(kind, img, native, note, texts, scan_fn, done) else None
        scan = self._probe_scan_base()
        if scan is not None and "error" not in scan:
            try:
                scan.update({"rings": [tuple(map(int, r)) for r in self._rings(img)] if img is not None else [],
                             # r4: game A's probes showed rings [] for 10 min with the road rings on screen
                             "edge_rings": len(self._edge_rings(img)) if img is not None else 0,
                             "turn_phase": turn_phase(img) if img is not None else None})
            except Exception as e:                    # noqa: BLE001
                scan = {"error": f"{type(e).__name__}: {e}"}
        p = probe.save(kind, img, native, note, texts, scan)
        if p:
            self.note(f"probe: saved {p.name}")
        return p

    def _probe_scan_base(self):
        """The game-state part of a probe's scan (cheap; copied on the main thread)."""
        gs = getattr(self, "gs", None)
        if gs is None:
            return None
        try:
            me = gs.me()
            return {"mode": getattr(self.mode, "name", str(self.mode)),
                    "me": {"settlements": sorted(map(str, me.settlements)), "cities": sorted(map(str, me.cities)),
                           "knights": sorted(map(str, me.knights)), "roads": len(me.roads),
                           "hand": dict(me.hand)},
                    "robber": str(gs.board.robber),
                    "my_cities_seen": sorted(map(str, getattr(self, "_my_cities", set()))),
                    "my_knights_seen": sorted(map(str, getattr(self, "_my_knights", {})))}
        except Exception as e:                        # noqa: BLE001
            return {"error": f"{type(e).__name__}: {e}"}

    def _track_pending(self, state):
        """A prompt we clicked on landed once the game shows another screen (an "unknown" frame
        only counts after 20 s: animations sit between a click and the prompt coming back)."""
        p = self._pend
        if p is None:
            return
        kind, st, t0, rounds, failed = p
        if state == st:
            return
        # other_turn is what most of those in-between frames read as since the turn marker (2026-10-09)
        if state in ("unknown", "other_turn") and kind != "end_turn" and time.time() - t0 < 20:
            return
        if not failed:
            self.probe.result(kind, True)
        self._pend = None
        self.__dict__.get("_fb_tried", {}).pop(st, None)

    def _after_handler(self, state, clicked: bool, img):
        if state in self.RING_FALLBACK:
            self._prompt_clock(state)
        if state not in self.PROMPT_KINDS:
            return
        same = self.__dict__.setdefault("_same_n", [None, 0])
        same[1] = same[1] + 1 if same[0] == state else 1
        same[0] = state
        p = self._pend
        if clicked:
            if p and p[1] == state:
                kind, st, t0, rounds, failed = p
                rounds += 1
                if rounds == 4:
                    self._probe(f"stuck_{state}", img, note={"rounds": rounds})
                if rounds >= 8 and not failed:
                    self.probe.result(kind, False, f"{state}: {rounds} rounds without leaving the prompt")
                    failed = True
                self._pend = (kind, st, t0, rounds, failed)
            else:
                self.probe.tried(state)
                self._pend = (state, state, time.time(), 1, False)
        n = same[1]
        fbn = self.__dict__.get("_promote_fb_n", 0)
        acts = (state == "promote_knight" and fbn >= 2) or (state in self.RING_FALLBACK and n >= 6 and n % 3 == 0)
        if clicked and acts and getattr(self, "io", None) is not None:
            # review 0.3.1: the handler clicked, so `img` is the screen from before its click (an
            # old ring could be tapped on a board whose prompt already closed) -- look again
            img = self.io.snapshot()
            if img is None:
                return
        if state == "promote_knight" and fbn >= 2 and self._red_x(img):
            # game 9 13:35: the ringed knight never gives a green check, only the red X: the game
            # won't promote it. Cancel once and stop planning promotions this turn (cfg off: this game).
            self._probe("promote_cancel_x", img)
            self.probe.tried("promote_cancel_x")
            self.click(self._x_at(img), "cancel strong-knight prompt (red X)")
            if self._by_colour():
                # game 0834: the latch dropped both mighty promotions of the game -- the red X alone
                # frees this turn only; promotions are planned again next turn
                self._promote_skip_turn, self._promote_fb_n = self.__dict__.get("_turn_n", 0), 0
            else:
                self._promote_disabled, self._promote_fb_n = True, 0
            return
        if state == "promote_knight" and fbn >= 6 and img is not None and turn_phase(img) == "main":
            self._promote_fb_n = 0
            # game 8 12:57: Smith's 2nd promotion rang a knight the game would not take (no strong
            # knight piece left?) and the prompt never left; End turn skips the optional rest
            self._probe("promote_skip_end_turn", img)
            self.probe.tried("skip_promote_end_turn")
            self.click(UI["dice_or_end_turn"], "end turn (skip the stuck strong-knight prompt)")
            return
        # forced prompt the handler can't clear (or doesn't click): click any legal ringed target
        if state in self.RING_FALLBACK and n >= 6 and n % 3 == 0:
            self._forced_fallback(img, state)

    def _fallback_rings(self, img):
        # game 8 12:56: the green check's ring was taken for a target (clicked as "ringed target")
        # game 0834: the check and the X sit apart from the fixed point -- skip the buttons found
        btn = [b for b in (self._check_xy(img), self._bar_blob(img, "red")) if b] if self._by_colour() else []
        btn = btn or [tuple(UI["setup_confirm"])]
        rings = [r for r in self._rings(img) if all(math.hypot(r[0] - bx, r[1] - by) > 30 for bx, by in btn)]
        rings = [r for r in rings if self._on_board(r)]
        # game 13 16:35: roads crossing a ring split it into arcs -- add rings read per corner
        rings += [xy for _, xy in self._vertex_rings(img)
                  if all(math.hypot(xy[0] - r[0], xy[1] - r[1]) > 12 for r in rings)]
        if self._edge_rings_on():
            # stage 2026-10-09 16:20: the Diplomat's road rings (split by the road) were in no list,
            # so the textless fallback and the stall rescue saw nothing for 10 min
            for _, xy in self._edge_rings(img):
                if all(math.hypot(xy[0] - r[0], xy[1] - r[1]) > 12 for r in rings):
                    rings.append(xy)
        return rings

    def _edge_rings_on(self) -> bool:
        return bool((self.__dict__.get("cfg") or {}).get("edge_rings_in_fallback", True)) \
            and getattr(self, "br", None) is not None

    def _on_board(self, xy) -> bool:
        """game 0959 10:29/10:33: a ring at (173,404) -- the piece supply left of the board, inside
        BOARD_ROI. With a lattice: within 1.3 hex sizes of a board hex centre; else inside BOARD_ROI."""
        cfg = self.__dict__.get("cfg") or {}
        if not cfg.get("ring_fallback_our_turn", True):
            return True
        x0, y0, x1, y1 = BOARD_ROI
        if not (x0 <= xy[0] <= x1 and y0 <= xy[1] <= y1):
            return False
        lat = getattr(getattr(self, "br", None), "lattice", None)
        gs = getattr(self, "gs", None)
        if lat is None or gs is None:
            return True
        try:
            lim = 1.3 * lat.size
            return any(math.hypot(xy[0] - px, xy[1] - py) <= lim
                       for px, py in (lat.hex_px(*h.coord) for h in gs.board.hexes.values()))
        except Exception:                            # noqa: BLE001 -- a filter only
            return True

    def _forced_fallback(self, img, state, ocr=None, texts=None, info=None):
        """Forced prompt (pirate, knight placement, Smith promotion, robber): click a ringed
        (highlighted) target the game offers, one per call, then the confirm check if the
        prompt is still up. Never stalls the game on a prompt the runner can't plan."""
        if img is None:
            return
        self._prompt_clock(state)
        tried = self.__dict__.setdefault("_fb_tried", {}).setdefault(state, set())
        if self._prompt_over_cap(state) or (state == "ringed_prompt" and self._cap_s() and len(tried) >= 3):
            # game 0959 10:34-10:43: the Inventor prompt took 21 ring taps in 9 min -- past the
            # cap (or 3 tries at a textless prompt) no more rings: leave by the exit ladder
            self._prompt_exit(img, state)
            return
        rings = self._fallback_rings(img)
        todo = [r for r in rings if (int(r[0]) // 12, int(r[1]) // 12) not in tried]
        if not todo and rings:
            tried.clear()
            todo = rings
        if not todo:
            if time.time() - self.__dict__.get("_fb_saved", {}).get(state, 0) > 60:
                self.__dict__.setdefault("_fb_saved", {})[state] = time.time()
                self._probe(f"noring_{state}", img)
            self.note(f"fallback {state}: no ringed target on screen")
            return
        xy = todo[0]
        tried.add((int(xy[0]) // 12, int(xy[1]) // 12))
        if state == "promote_knight":            # counted per turn (screen flicker resets step counts)
            self._promote_fb_n = self.__dict__.get("_promote_fb_n", 0) + 1
        self.probe.tried(f"fallback_{state}")
        self.click(xy, f"fallback {state}: ringed target ({len(rings)} rings)")
        time.sleep(0.8)
        im2 = self.io.snapshot()
        if im2 is None:
            return
        o2 = self.io.ocr(im2)
        st2, _ = classify([o["text"] for o in o2], self.cfg["bot_name"].lower())
        if state == "ringed_prompt" and st2 == "unknown":
            st2 = state                              # no prompt text: the same screen, needs the check
        if (st2 == state or find_text(o2, r"confirm")) and not self.cfg.get("fallback_blind_confirm", False) \
                and not self._green_check(im2):
            # game 0959 10:34: the Inventor prompt has no check at all -- 19 "confirm at (654,645)"
            # taps hit empty table. Look once more; with still no check, don't tap the bar
            time.sleep(0.8)
            im2 = self.io.snapshot()
            if im2 is None or not self._green_check(im2):
                self.note(f"fallback {state}: no check, not tapping the bar")
                self.probe.result(f"fallback_{state}", False, f"{state}: no check after a ringed click")
                return
        if st2 == state or find_text(o2, r"confirm"):
            self.click(self._check_at(im2), f"fallback {state}: confirm")
            time.sleep(0.8)
            im3 = self.io.snapshot()
            st2 = classify([o["text"] for o in self.io.ocr(im3)], self.cfg["bot_name"].lower())[0] if im3 is not None else st2
            if state == "ringed_prompt":
                # review 0.3.1: count the after-frame the same way (ring spots + per-corner rings)
                n3 = len(self._fallback_rings(im3)) if im3 is not None else 0
                st2 = state if (n3 and n3 >= len(rings)) else "done"
        self.probe.result(f"fallback_{state}", st2 != state, f"{state} still up after a ringed click")

    # -- prompt cap (game 0959: the Inventor prompt took 550 s of one turn)
    def _cap_s(self) -> float:
        return float((self.__dict__.get("cfg") or {}).get("prompt_cap_s", 90) or 0)

    def _prompt_clock(self, state):
        """Start the clock of a prompt the first time a fallback or ring handler works on it."""
        if self._cap_s():
            since = self.__dict__.setdefault("_prompt_since", {})
            if state not in since:
                since[state] = time.time()
                # a fresh prompt: the ring cells tried at an earlier one of the same kind do not count
                self.__dict__.get("_fb_tried", {}).pop(state, None)

    def _prompt_over_cap(self, state) -> bool:
        cap = self._cap_s()
        since = self.__dict__.get("_prompt_since", {}).get(state)
        return bool(cap) and since is not None and time.time() - since > cap

    def _prompt_clock_clear(self, state):
        """The prompt is gone once the screen reads our main phase, a rival's turn or the dice."""
        # the live loop reads our main phase as my_turn, never main (r3 review)
        if str(state).split(":")[0] in ("main", "my_turn", "other_turn", "roll"):
            self.__dict__.pop("_token_pairs", None)
            self.__dict__.pop("_progress_blocked", None)
            if self.__dict__.get("_prompt_since"):
                self._prompt_since = {}
                self.__dict__.pop("_prompt_exit_at", None)

    def _prompt_exit(self, img, state):
        """Exit ladder for a prompt past its cap, at most once per 30 s: the red X, else an
        OK/Close/Cancel/Done button, else End turn when the turn buttons show our main phase."""
        last = self.__dict__.setdefault("_prompt_exit_at", {})
        if time.time() - last.get(state, 0) < 30:
            return
        last[state] = time.time()
        self._probe(f"prompt_cap_{state}", img)
        self.probe.tried(f"prompt_cap_{state}")
        tried = self.__dict__.setdefault("_fb_tried", {}).setdefault(state, set())
        if img is not None and "exit_road" not in tried and self._prompt_cap_unknown() and self._edge_rings(img):
            # stage 2026-10-09 game 1548: the cap must not skip a ring it never tried -- the
            # Diplomat's road rings sat on screen for 10 min with zero taps (once per prompt)
            tried.add("exit_road")
            self.note(f"prompt cap {state}: road rings shown -> pick one")
            self._diplomat_pick(img=img)
            return
        if img is not None and self._bar_blob(img, "red") is not None:
            self.click(self._x_at(img), f"prompt cap {state}: red X")
            return
        ocr = self.io.ocr(img) if img is not None else []
        btn = find_text(ocr or [], r"^(ok|close|cancel|done)$")
        if btn:
            self.click((btn["x"] + btn["w"] / 2, btn["y"] + btn["h"] / 2), f"prompt cap {state}: {btn['text']}")
            return
        phase = turn_phase(img) if img is not None else None
        if phase == "main":
            self.click(UI["dice_or_end_turn"], "prompt cap: ending the turn")
            return
        if phase == "idle" and "exit_end_idle" not in tried and self._prompt_cap_unknown() \
                and not self._marker_seat(img):
            # game 1548: End turn stayed grey under the textless Diplomat prompt; once per prompt
            tried.add("exit_end_idle")
            self.click(UI["dice_or_end_turn"], "prompt cap: End turn grey, pressing once "
                                               "(UNCONFIRMED whether the game takes it)")
            return
        self.note(f"prompt cap {state}: no way out on screen (no X, no button, not our main phase)")

    def _prompt_cap_unknown(self) -> bool:
        return bool((self.__dict__.get("cfg") or {}).get("prompt_cap_unknown", True))

    def _looks_like_our_prompt(self, img) -> bool:
        """stage 2026-10-09 game 1548: 10 min of a textless Diplomat prompt read 'unknown', outside
        the prompt clock. Unknown on our turn (marker on us or unread) with rings on the board, a
        grey End turn under our own marker, or a card dragged in the last 10 min = our prompt."""
        if img is None or not getattr(self, "_in_play", False):
            return False
        seat = self._marker_seat(img)
        if seat not in (0, None):
            return False
        if time.time() - self.__dict__.get("_progress_click_at", 0) < 600:
            return True
        if seat == 0 and turn_phase(img) == "idle":
            return True
        return bool(self._rings(img) or self._vertex_rings(img) or self._edge_rings(img))

    DECKS = ("trade", "politics", "science")     # left to right: yellow, blue, green

    def on_choose_deck(self, img, ocr, texts, info):
        """'Defender of Catan -- you receive a progress card': three decks (yellow trade, blue
        politics, green science) showing the cards left. Click the deck we want (the track we
        are best in: its cards are the ones our plan uses; science when level), by the count
        label on each deck, else at the deck's spot."""
        nums = [o for o in ocr if re.fullmatch(r"\d{1,2}", o["text"].strip()) and 250 < o["y"] < 650]
        gs = getattr(self, "gs", None)
        slots = [UI.get(f"deck_{d}", (512 + 174 * k, 432)) for k, d in enumerate(self.DECKS)]
        counts = {}
        for d, sl in zip(self.DECKS, slots):
            near = [o for o in nums if abs(o["x"] + o["w"] / 2 - sl[0]) < 70]
            counts[d] = int(near[0]["text"].strip()) if near else None
        if gs is not None:
            want = self.brain.choose_progress_deck(gs, counts)
        else:
            want = "science"
        i = self.DECKS.index(want)
        # the count label sits on its deck: use it when read (probe 2026-10-04: 2 of 3 read)
        hit = [o for o in nums if abs(o["x"] + o["w"] / 2 - slots[i][0]) < 70]
        xy = (hit[0]["x"] + hit[0]["w"] / 2, hit[0]["y"] + hit[0]["h"] / 2) if hit else slots[i]
        if not hit:
            self._probe("choose_deck_unread", img, note={"nums": [o["text"] for o in nums]})
        self.click(xy, f"progress deck {want}")
        time.sleep(1.0)
        im2 = self.io.snapshot()
        if im2 is not None:
            o2 = self.io.ocr(im2)
            self._probe("choose_deck_after", im2, texts=[o["text"] for o in o2])
            if find_text(o2, r"confirm"):
                self.click(UI["setup_confirm"], "confirm progress deck")

    def on_pick_progress(self, img, ocr, texts, info):
        """Spy: 'Select progress card' shows the victim's cards face up, titles on top. Click the
        card we value most (keep_value), on its picture under the title; a confirm/check if one
        shows; if the dialog stays, the next call tries the next card."""
        from . import progress_cards as pc
        hdr = find_text(ocr, r"select (a )?progress car[a-z]?")
        y0 = hdr["y"] if hdr else 250
        cards = []
        for o in ocr:
            if not (y0 < o["y"] < y0 + 90):
                continue
            try:
                cards.append((pc.card_key(o["text"]), o))
            except KeyError:
                continue
        if not cards:
            self._probe("pick_progress_unread", img, texts=texts)
            self.note(f"pick progress: no card titles read in {texts[:20]}")
            return
        gs = getattr(self, "gs", None)
        me = gs.me() if gs is not None and any(p.is_me for p in gs.players) else None
        tried = self.__dict__.setdefault("_pick_tried", set())
        todo = [c for c in cards if c[0] not in tried] or cards
        # owner 2026-10-06: pick by a real value -- playable now first, the leader's best counts
        phase = getattr(self, "_progress_last_phase", None) or "main"
        try:
            pick, scores = pc.spy_pick(gs, me, [k for k, _ in todo], getattr(self, "_spy_victim", None),
                                       phase, bool(getattr(self, "barbarians_attacked", False)))
            key, box = next(c for c in todo if c[0] == pick)
        except Exception:                            # noqa: BLE001
            key, box, scores = todo[0][0], todo[0][1], {}
        tried.add(key)
        self.note(f"pick progress: {key} of {[k for k, _ in cards]} scores={ {k: round(v, 1) for k, v in scores.items()} }")
        self.click((box["x"] + box["w"] / 2, box["y"] + 60), f"pick progress card {key}")
        time.sleep(1.0)
        im2 = self.io.snapshot()
        if im2 is None:
            return
        o2 = self.io.ocr(im2)
        self._probe("pick_progress_after", im2, texts=[o["text"] for o in o2])
        btn = find_text(o2, r"^(ok|confirm|take|select)$")
        if btn:
            self.click((btn["x"] + btn["w"] / 2, btn["y"] + btn["h"] / 2), f"pick progress: {btn['text']}", o2)
        if not find_text(o2, r"select (a )?progress car[a-z]?"):
            self._pick_tried = set()
            self._spy_stolen(gs, key, phase)

    def _spy_stolen(self, gs, key, phase):
        """The stolen card is ours: re-plan the turn (brain.after_spy) and let play_progress try
        it this turn even if a copy was already tried (owner 2026-10-06)."""
        self._spy_victim = None
        tried = self.__dict__.get("_progress_tried", set())
        tried -= {t for t in tried if t[1] == key}
        if gs is not None and any(p.is_me for p in gs.players):
            try:
                self.brain.after_spy(gs, key, phase)
            except Exception as e:                   # noqa: BLE001
                self.note(f"spy: re-plan failed ({e})")

    def on_take_cards(self, img, ocr, texts, info):
        """Master Merchant 'Take 2 resources from other player.': the victim's cards are the top
        row of the trade dialog. Click the n cards we value most, then the check; if the dialog
        stays (a card they don't hold), the next call tries the next cards in value order."""
        m = re.search(r"take (\d+)", " ".join(texts), re.I)
        n = int(m.group(1)) if m else 2
        gs = getattr(self, "gs", None)
        try:
            val = self.brain.resource_value(gs, gs.me()) if gs is not None else {}
        except Exception:                            # noqa: BLE001
            val = {}
        order = sorted(CARD_ORDER, key=lambda r: -val.get(r, 1.0))
        k = self.__dict__.get("_take_round", 0)
        pick = (order[k * n:(k + 1) * n] + order)[:n] if k * n < len(order) else order[:n]
        self._take_round = k + 1 if k * n < len(order) else 0
        self._probe("take_cards", img, texts=texts, note={"pick": pick, "round": k})
        for r in pick:
            self.click((UI["trade_want_row"]["x"][CARD_ORDER.index(r)], UI["trade_want_row"]["y"]), f"take {r}")
        self.click(UI["trade_ok"], f"take {n} cards: confirm")
        time.sleep(1.0)
        im2 = self.io.snapshot()
        if im2 is not None and not re.search(r"take \d+ (resources|cards)", " ".join(o["text"] for o in self.io.ocr(im2)), re.I):
            self._take_round = 0

    PARK_SPOT = (1140, 400)              # empty table right of the board (60,160 showed a banner tooltip)
    PAN_MIN_PX = 6
    BOARD_REF = (365, 655)               # sea area's left / bottom edge, 1920x1080 frame (probe frames)

    # stage 2026-10-09 game 0834: the knight prompt's bar shows the green check at (621,645) and the
    # red X at (683,647) when both are up, so the fixed point (654,645) read neither and the ring
    # loop ran ~300 s; with the red X alone it sits at (653,647). Found by colour in the bar band
    # (offsets from UI setup_confirm: x -114..+126, y -35..+45; the hand bar and board stay out).
    CHECK_BAND = (-114, 126, -35, 45)
    CHECK_MIN_PX = 500

    @staticmethod
    def _bar_blob(img, colour: str):
        """Centroid (x, y) of the largest green / red blob (>= CHECK_MIN_PX px) in the button bar."""
        if img is None:
            return None
        cx, cy = map(int, UI["setup_confirm"])
        bx0, bx1, by0, by1 = Runner.CHECK_BAND
        x0, y0 = max(0, cx + bx0), max(0, cy + by0)
        crop = img[y0:max(0, cy + by1), x0:max(0, cx + bx1)]
        if crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        hue = (h > 45) & (h < 85) if colour == "green" else (h < 8) | (h > 170)
        m = (hue & (s > 120) & (v > 90)).astype(np.uint8)
        n, _, st, cen = cv2.connectedComponentsWithStats(m)
        if n < 2:
            return None
        i = 1 + int(np.argmax(st[1:, 4]))
        if st[i, 4] < Runner.CHECK_MIN_PX:
            return None
        return int(round(cen[i][0])) + x0, int(round(cen[i][1])) + y0

    @staticmethod
    def _check_xy(img):
        """Where the green confirm check is (None = no check showing)."""
        return Runner._bar_blob(img, "green")

    @staticmethod
    def _green_check_colour(img) -> bool:
        return Runner._check_xy(img) is not None

    @staticmethod
    def _red_x_colour(img) -> bool:
        """The red cancel X alone (no green check beside it)."""
        return Runner._bar_blob(img, "red") is not None and Runner._check_xy(img) is None

    @staticmethod
    def _green_check_fixed(img) -> bool:
        """The green confirm check of a placement / selection prompt is showing (UI setup_confirm)."""
        if img is None:
            return False
        x, y = map(int, UI["setup_confirm"])
        hsv = cv2.cvtColor(img[max(0, y - 12):y + 12, max(0, x - 12):x + 12], cv2.COLOR_BGR2HSV)
        return hsv.size > 0 and float(((hsv[..., 0] > 45) & (hsv[..., 0] < 85) & (hsv[..., 1] > 120)
                                         & (hsv[..., 2] > 90)).mean()) > 0.4

    @staticmethod
    def _red_x_fixed(img) -> bool:
        """The red cancel X of a placement prompt is showing (same spot as the green check)."""
        if img is None:
            return False
        x, y = map(int, UI["setup_confirm"])
        hsv = cv2.cvtColor(img[max(0, y - 12):y + 12, max(0, x - 12):x + 12], cv2.COLOR_BGR2HSV)
        return hsv.size > 0 and float((((hsv[..., 0] < 8) | (hsv[..., 0] > 170)) & (hsv[..., 1] > 120)
                                         & (hsv[..., 2] > 90)).mean()) > 0.3

    _green_check = _ByColour("_green_check_colour", "_green_check_fixed")
    _red_x = _ByColour("_red_x_colour", "_red_x_fixed")

    def _by_colour(self) -> bool:
        return bool((self.__dict__.get("cfg") or {}).get("confirm_by_colour", True))

    def _check_at(self, img):
        """Click point of the green check: found by colour, else the fixed UI setup_confirm."""
        xy = self._check_xy(img) if self._by_colour() else None
        return xy or UI["setup_confirm"]

    def _x_at(self, img):
        """Click point of the red X: found by colour, else the fixed UI setup_confirm."""
        xy = self._bar_blob(img, "red") if self._by_colour() else None
        return xy or UI["setup_confirm"]

    def _confirm_if_check(self, why):
        time.sleep(0.8)
        im = self.io.snapshot()
        if self._green_check(im):
            self.click(self._check_at(im), f"confirm {why} (green check)")
            time.sleep(0.8)
            return True
        return False

    @staticmethod
    def _sea_box(img):
        """Bounding box (x0, y0, x1, y1, area) of the largest sea-blue area on the board rows."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        m = ((hsv[..., 0] > 95) & (hsv[..., 0] < 115) & (hsv[..., 1] > 90) & (hsv[..., 2] > 120)).astype(np.uint8)
        m[:120] = 0
        m[700:] = 0
        n, _, st, _ = cv2.connectedComponentsWithStats(m)
        if n < 2:
            return None
        i = 1 + int(np.argmax(st[1:, 4]))
        x, y, w, h, a = st[i]
        return int(x), int(y), int(x + w), int(y + h), int(a)

    def _recenter_board(self, img) -> bool:
        """Final probe game 10:18: a drag that missed its card panned the board up-left and every
        scan after it failed. The sea area keeps its width; drag the table back by the offset of
        its left and bottom edges from where they sit in a normal frame."""
        if img is None or time.time() - getattr(self, "_recenter_t", 0) < 20:
            return False
        box = self._sea_box(img)
        if box is None or box[4] < 15000 or not 330 < box[2] - box[0] < 420:
            return False
        rx, ry = getattr(self, "_board_ref", None) or self.BOARD_REF
        dx, dy = rx - box[0], ry - box[3]
        # solo 2026-10-06 02:05: a 13 px pan (sea box 365,655 -> 352,642) under the old 25 px
        # gate made every board tap miss for the rest of the game; the box is pixel-stable
        if abs(dx) < self.PAN_MIN_PX and abs(dy) < self.PAN_MIN_PX:
            return False
        self._recenter_t = time.time()
        p0 = (750, 380)
        p1 = (max(5, min(1360, p0[0] + dx)), max(130, min(740, p0[1] + dy)))
        self._probe("board_panned", img, note={"sea_box": box, "ref": (rx, ry), "drag": (p0, p1)})
        self.note(f"board panned by {(-dx, -dy)}: dragging it back")
        self.drag(p0, p1, "recenter board")
        self.lattice = self.scan = None
        self._park_pointer("after recenter")
        return True

    def _park_pointer(self, why=""):
        """Move the pointer off the hand cards: hovering a card shows its big preview."""
        try:
            self.io.move(*UI.get("pointer_park", self.PARK_SPOT), why)
        except Exception as e:                       # noqa: BLE001
            self.note(f"pointer park failed: {e}")
        self.note(f"pointer parked ({why})")

    @staticmethod
    def _enlarged_card(ocr):
        """(card key, title box) of an enlarged progress card (title ~ frame y 500-530, right of
        the board centre), or None. The small hand cards' titles sit at y ~715."""
        from . import progress_cards as pc
        if find_text(ocr, r"hide overlay"):
            return None                      # a dialog, not a card view
        for o in ocr:
            if 430 < o["y"] < 600 and 600 < o["x"] < 1150 and len(o["text"]) >= 5:
                try:
                    return pc.card_key(o["text"]), o
                except KeyError:
                    continue
        return None

    def on_place_pirate(self, img, ocr, texts, info):
        self._forced_fallback(img, "place_pirate")

    def _vertex_rings(self, img):
        """[(vertex, (x, y))] of board corners with a selection ring, read per lattice corner (a
        road crossing a ring splits it, so ring_spots misses it -- game 13 16:35, 18 min stall)."""
        br = getattr(self, "br", None)
        if img is None or br is None:
            return []
        from .vision.rings import vertex_rings
        io = self.io
        native = lay = None
        if getattr(io, "_frame", None) is img:
            native, lay = getattr(io, "native", None), getattr(io, "lay", None)
        try:
            found = vertex_rings(img, br.lattice, br.board, native, lay)
        except Exception as e:                        # noqa: BLE001 -- vision must never stop play
            self.note(f"vertex rings failed: {type(e).__name__}: {e}")
            return []
        boxes = [o for o in (getattr(self, "_last_ocr", None) or []) if len(o["text"]) >= 8] \
            if img is getattr(self, "_last_img", None) else []
        cx, cy = UI["setup_confirm"]
        return [(v, xy) for _, v, xy in found
                if math.hypot(xy[0] - cx, xy[1] - cy) > 30
                and not any(o["x"] - 8 <= xy[0] <= o["x"] + o["w"] + 8 and o["y"] - 8 <= xy[1] <= o["y"] + o["h"] + 8
                            for o in boxes)]

    def _edge_rings(self, img, br=None, span=4, log=False):
        """[(edge, (x, y))] of board edges with a small selection ring at their middle (the
        Diplomat's removable roads), best first; none near the confirm spot or inside text.
        stage 2026-10-09 16:20: live it found none for 10 min where the offline replay finds 6-8,
        and no line said why -- log=True notes the count, top scores, lattice and source."""
        br = br or getattr(self, "br", None)
        if img is None or br is None:
            return []
        from .vision.rings import EDGE_THRESHOLD, edge_rings
        io = self.io
        native = lay = None
        if getattr(io, "_frame", None) is img:
            native, lay = getattr(io, "native", None), getattr(io, "lay", None)
        cache = self.__dict__.get("_edge_cache")
        if not log and cache is not None and cache[0] is img and cache[1] is br and cache[2] == span:
            return list(cache[3])                 # one detector run per frame (~1 s each)
        try:
            # threshold 0 + the cut here = the same rings as at EDGE_THRESHOLD (best-first de-dup)
            scored = edge_rings(img, br.lattice, br.board, native, lay, threshold=0.0, span=span)
        except Exception as e:                        # noqa: BLE001 -- vision must never stop play
            self.note(f"edge rings failed: {type(e).__name__}: {e}")
            return []
        found = [t for t in scored if t[0] >= EDGE_THRESHOLD]
        boxes = [o for o in (getattr(self, "_last_ocr", None) or []) if len(o["text"]) >= 8] \
            if img is getattr(self, "_last_img", None) else []
        cx, cy = UI["setup_confirm"]
        out = [(e, xy) for _, e, xy in found
               if math.hypot(xy[0] - cx, xy[1] - cy) > 30
               and not any(o["x"] - 8 <= xy[0] <= o["x"] + o["w"] + 8 and o["y"] - 8 <= xy[1] <= o["y"] + o["h"] + 8
                           for o in boxes)]
        self._edge_cache = (img, br, span, list(out))
        if log:
            top = ", ".join(f"{s:.3f} {e}" for s, e, _ in scored[:3])
            try:
                size = f"{br.lattice.size:.1f}"
            except Exception:                         # noqa: BLE001 -- a log line only
                size = "?"
            self.note(f"edge rings: {len(found)} >= {EDGE_THRESHOLD}, top {top}, lattice {size}, "
                      f"src {'native' if native is not None and lay is not None else 'frame'}, "
                      f"ocr-dropped {len(found) - len(out)}" + (f", span {span}" if span != 4 else ""))
        return out

    def _diplomat_order(self, rings, want=None, own=False):
        """Ringed edges in the order to try: the brain's edge if it is ringed, then (own move)
        our roads, else rivals' roads by owner VP (highest first), then nearest the wanted edge."""
        gs = getattr(self, "gs", None)
        me = gs.me() if gs is not None else None
        lat = getattr(getattr(self, "br", None), "lattice", None)
        wset = frozenset(want) if want else None
        wxy = lat.edge_px(want) if (want and lat is not None) else None

        def owner_vp(e):
            fe = frozenset(e)
            for p in (gs.players if gs is not None else []):
                if any(frozenset(r) == fe for r in (p.roads | p.ships)):
                    return p, p.vp_visible
            return None, -1

        def key(item):
            e, xy = item
            p, vp = owner_vp(e)
            exact = 0 if (wset is not None and frozenset(e) == wset) else 1
            mine = p is not None and p is me
            side = (0 if mine else 1) if own else (1 if mine else 0)
            dist = math.hypot(xy[0] - wxy[0], xy[1] - wxy[1]) if wxy is not None else 0.0
            return (exact, side, -vp, dist)
        return sorted(rings, key=key)

    def _diplomat_pick(self, want=None, own=False, img=None) -> bool:
        """Stage 2026-10-08 22:33: the Diplomat clicked the brain's edge (653,471), which had no
        ring, and the bot sat on the prompt for minutes. Tap a ring that is shown (edge rings,
        3 looks), confirm with the green check, try the next ring when no check comes; nothing
        works -> the red X if it shows, else leave it to the stall rescue."""
        rings, im = [], img
        for i in range(3):
            im = img if (i == 0 and img is not None) else self.io.snapshot()
            rings = self._edge_rings(im, log=True)
            if rings:
                break
            time.sleep(0.6)
        if not rings and im is not None and (self.__dict__.get("cfg") or {}).get("diplomat_reread", True):
            rings = self._diplomat_reread(im)
        if not rings:
            self.note(f"diplomat: no road ring on screen (wanted {want})")
        for e, xy in self._diplomat_order(rings, want, own)[:4]:
            self.click(xy, f"diplomat ringed road {e}")
            if self._confirm_if_check("diplomat road"):
                self._diplomat_played_t = 0.0       # done: no textless recovery needed
                self.__dict__.pop("_progress_blocked", None)
                return True
            self.note(f"diplomat: no green check after road {e}; trying the next ring")
        im = self.io.snapshot()
        if self._red_x(im):
            self.click(UI.get("place_cancel", UI["setup_confirm"]), "diplomat: cancel (red X)")
        else:
            self.note("diplomat: no ring took and no cancel shown; leaving it to the stall rescue")
        # stage 2026-10-09 16:21:10: the Inventor was dragged onto the open Diplomat prompt --
        # no more cards this turn while it may still be up (the prompt clock clears it)
        self._progress_blocked = ("diplomat", time.time())
        return False

    def _diplomat_reread(self, im):
        """stage 2026-10-09 16:20: live looks found no road ring for 10 min where the replay of the
        same frame finds 6-8 (lattice off by >= 6 px is one UNCONFIRMED cause). Read the board once
        more from this frame with a fresh lattice and look again, then once with a wider search."""
        from .vision.board_reader import read_board
        br = None
        try:
            br = read_board(im, BOARD_ROI, canal=True, lattice=None)
        except Exception as e:                        # noqa: BLE001 -- vision must never stop play
            self.note(f"diplomat: board re-read failed: {type(e).__name__}: {e}")
        old = getattr(self, "br", None)
        rings = []
        if br is not None:
            rings = self._edge_rings(im, br=br, log=True)
            # keep the fresh read only when it names the same hexes (vertex keys must stay stable)
            same = old is not None and set(br.board.hexes) == set(old.board.hexes)
            self.note(f"diplomat: board re-read, {len(rings)} road rings"
                      + (" (lattice replaced)" if rings and same else ""))
            if rings and same:
                self.br = br
        if not rings:
            rings = self._edge_rings(im, br=br if br is not None and old is None else None, span=8, log=True)
        return rings

    def _diplomat_place(self, want, img=None) -> bool:
        """Diplomat moving our own road, 2nd tap: the game rings the legal new spots. Tap the
        brain's edge when it is ringed, else the ring nearest to it (stage 2026-10-09: the lattice
        position of the wanted edge was tapped even when no ring sat there). No ring in 3 looks
        -> the lattice tap as a last resort (the road is already lifted)."""
        rings = []
        for i in range(3):
            im = img if (i == 0 and img is not None) else self.io.snapshot()
            rings = self._edge_rings(im) if im is not None else []
            if rings:
                break
            time.sleep(0.6)
        lat = getattr(getattr(self, "br", None), "lattice", None)
        if not rings:
            self.note(f"diplomat move: no ring for the new spot (wanted {want})")
            if lat is None or want is None:
                return False
            self.click(lat.edge_px(want), f"diplomat move: new spot {want} (lattice, no ring)")
            return self._confirm_if_check("diplomat move")
        wset = frozenset(want) if want else None
        wxy = lat.edge_px(want) if (want and lat is not None) else None

        def key(item):
            e, xy = item
            exact = 0 if (wset is not None and frozenset(e) == wset) else 1
            return (exact, math.hypot(xy[0] - wxy[0], xy[1] - wxy[1]) if wxy is not None else 0.0)
        for e, xy in sorted(rings, key=key)[:3]:
            self.click(xy, f"diplomat move: new spot ringed {e}")
            if self._confirm_if_check("diplomat move"):
                return True
            self.note(f"diplomat move: no green check after {e}; trying the next ring")
        return False

    def _ringed_pick(self, img, texts, state, choose, why):
        """Shared flow of a ringed board choice: green check up -> confirm; else the legal ringed
        corners -> the brain's choice (not tried yet for this prompt) -> click -> confirm check ->
        verified when the prompt text is gone. No corner ring read -> the generic ring fallback."""
        if self._green_check(img):
            self.click(self._check_at(img), f"confirm {why} (green check)")
            return True
        gs = self.ensure_state(img, texts) if self.mode is not None else None
        vr = self._vertex_rings(img) if gs is not None else []
        tried = self.__dict__.setdefault("_ring_pick_tried", {}).setdefault(state, set())
        opts = [v for v, _ in vr if v not in tried]
        if not opts and vr:
            tried.clear()
            opts = [v for v, _ in vr]
        if not opts:
            self._forced_fallback(img, state)        # any ringed target (saves a probe frame)
            return False
        try:
            v = choose(gs, opts)
        except Exception as e:                        # noqa: BLE001
            self.note(f"{state}: brain choice failed ({type(e).__name__}: {e}), first ring")
            v = None
        if v not in opts:
            v = opts[0]
        tried.add(v)
        xy = dict(vr)[v]
        self.probe.tried(state)
        self.click(xy, f"{why} at {v} ({len(vr)} ringed)")
        self._confirm_if_check(why)
        time.sleep(0.4)
        im2 = self.io.snapshot()
        still = im2 is not None and bool(find_text(self.io.ocr(im2), r"select ?position|remove|knight to"))
        ok = im2 is not None and not still
        self.probe.result(state, ok, f"{why}: prompt still up after clicking {v}")
        if ok:
            tried.clear()
        return ok

    def on_place_knight(self, img, ocr, texts, info):
        """'Select position for (Basic) knight' we did not build (Deserter, any free knight):
        the brain picks among the ringed corners as for a paid knight (gold, open canal tile,
        defence before the attack, a leader's road to cut). Owner 2026-10-04, game 13."""
        if self._ringed_pick(img, texts, "place_knight", self.brain.choose_free_knight_spot, "place free knight"):
            self.note("free knight placed")

    def on_place_metropolis(self, img, ocr, texts, info):
        """'Select position for Metropolis' (game 13 17:01, read as unknown): the game rings our
        cities that can take it -- the city we would least like to lose (it can't be pillaged)."""
        picked = []

        def choose(gs, opts):
            mine = [v for v in opts if v in gs.me().cities and v not in gs.me().metropolises] or opts
            picked.append(max(mine, key=lambda v: (self.brain.city_keep_value(gs, v), v)))
            return picked[-1]
        if self._green_check(img) or self._vertex_rings(img):
            if self._ringed_pick(img, texts, "place_metropolis", choose, "place metropolis") and picked:
                self._record_metro(picked[-1])
            return
        # game 13 17:20: no ring is drawn (one eligible city) -> tap our own cities, best first
        gs = self.ensure_state(img, texts) if self.mode is not None else None
        cities = [v for v in (gs.me().cities if gs is not None else []) if v not in gs.me().metropolises]
        if not cities:
            self._forced_fallback(img, "place_metropolis")
            return
        tried = self.__dict__.setdefault("_metro_tapped", set())
        left = [v for v in cities if v not in tried] or cities
        if len(left) == len(cities):
            tried.clear()
        v = choose(gs, left)
        tried.add(v)
        self.click(self.br.lattice.vertex_px(v), f"place metropolis on our city {v} (no ring shown)")
        if self._confirm_if_check("place metropolis"):
            self._record_metro(v)

    def _forget_knight(self, t):
        """Game 0959 10:45-10:53: the popup at (-3,-1) offered only a knight build (red), so no
        knight of ours stands there -- yet the remembered knight was re-added each read and
        "activate" tapped there on five turns. A popup with build options only = the knight is gone."""
        if not self.cfg.get("knight_gone_memory", True):
            return
        known = t in self.__dict__.get("_my_knights", {})
        for name in ("_my_knights", "_knight_levels", "_activated_at"):
            self.__dict__.get(name, {}).pop(t, None)
        self.__dict__.get("_moved_knights", set()).discard(t)
        gs = getattr(self, "gs", None)
        if gs is not None:
            gs.me().knights.pop(t, None)
        self.note(f"knight at {t} is gone: forgotten" + ("" if known else " (it was only on the board read)"))
        self._save_game_flags()

    def _record_metro(self, v):
        self.__dict__.setdefault("_my_metros", set()).add(v)
        gs = getattr(self, "gs", None)
        if gs is not None:
            gs.me().metropolises.add(v)
        self._save_game_flags()

    def on_remove_city(self, img, ocr, texts, info):
        """'Select position to remove City' (stage 2026-10-08 22:42, read as unknown, 9 min freeze
        until a human clicked our city (-3, 1) and the check). It came with "Jean upgrades a city to
        a metropolis" and cost us 2 VP: a rival took our metropolis and the game asks which city
        gives it up. So a city carrying our metropolis goes first; with none known (or the game
        refuses it) it is a city loss -- the city we'd least like to keep, never a metropolis.
        Each tap is verified by the green check; a tap the game won't take is not tapped again."""
        if self._green_check(img):
            self.click(self._check_at(img), "confirm city to remove (green check)")
            self._after_remove_city(*(getattr(self, "_remove_city_pending", None) or (None, None)))
            return
        gs = self.ensure_state(img, texts) if self.mode is not None else None
        if gs is None or getattr(self, "br", None) is None:
            self.note("remove city: board not read yet; waiting")
            return
        me = gs.me()
        cities = sorted(set(me.cities) | set(getattr(self, "_my_cities", set())))
        rings = self._vertex_rings(img)
        ringed = [v for v, _ in rings if v in cities]
        if ringed:
            cities = ringed                          # the game marks the legal ones: only those
        metros = set(me.metropolises) | set(getattr(self, "_my_metros", set()))
        tried = self.__dict__.setdefault("_remove_city_tried", set())
        left = [v for v in cities if v not in tried]
        if not left:
            tried.clear()
            left = cities
        if not left:
            self.note("remove city: no city of ours known")
            self._probe("remove_city_none", img)
            return
        mine_m = [v for v in left if v in metros]
        if mine_m:
            v, kind = min(mine_m, key=lambda c: (self.brain.city_keep_value(gs, c), c)), "metropolis"
        elif find_text(ocr or [], r"metropol") and not metros:
            # a rival's metropolis in the log, ours not tracked (placed before a restart): the
            # metropolis went on our best city -- try the best first
            v, kind = max(left, key=lambda c: (self.brain.city_keep_value(gs, c), c)), "metropolis"
        else:
            plain = [c for c in left if c not in metros] or left
            v, kind = self.brain.choose_city_to_lose(gs, plain), "city"
        tried.add(v)
        self._remove_city_pending = (v, kind)      # a later green-check confirm knows the city
        xy = dict(rings).get(v) or self.br.lattice.vertex_px(v)
        self.click(xy, f"remove {kind} at our city {v} ({len(cities)} candidates)")
        if self._confirm_if_check(f"remove {kind}"):
            self._after_remove_city(v, kind)

    def _after_remove_city(self, v=None, kind=None, gs=None):
        self.__dict__.setdefault("_remove_city_tried", set()).clear()
        self._remove_city_pending = None
        gs = gs if gs is not None else getattr(self, "gs", None)
        me = gs.me() if gs is not None else None
        if v is None:
            self.note("remove city: confirmed (city unknown)")
            return
        if kind == "metropolis":
            # the city stays, only the metropolis goes
            self.__dict__.setdefault("_my_metros", set()).discard(v)
            if me is not None:
                me.metropolises.discard(v)
            self.note(f"remove city: our metropolis at {v} given up (the city stays)")
        else:
            # a city lost (pillage): it is a settlement now and loses its wall
            self._lost_cities = getattr(self, "_lost_cities", 0) + 1
            self.__dict__.setdefault("_my_cities", set()).discard(v)
            self.__dict__.setdefault("_my_walls", set()).discard(v)
            if me is not None:
                me.cities.discard(v)
                me.settlements.add(v)
                me.walls.discard(v)
            self.note(f"remove city: our city at {v} is a settlement now (lost cities {self._lost_cities})")
        self._save_game_flags()

    def on_place_wall(self, img, ocr, texts, info):
        """'Select position for City wall' (Engineer, game 13 17:45; was read as setup_city and tapped a
        city that already had its wall for 10 min): our cities without a wall, the one we'd least like to lose."""
        gs = self.ensure_state(img, texts) if self.mode is not None else None
        picked = []

        def choose(gs, opts):
            mine = [v for v in opts if v in gs.me().cities and v not in gs.me().walls] or opts
            picked.append(max(mine, key=lambda v: (self.brain.city_keep_value(gs, v), v)))
            self._wall_pending = picked[-1]          # a green check on a later frame knows the city
            return picked[-1]
        if gs is not None and (self._green_check(img) or self._vertex_rings(img)):
            if self._ringed_pick(img, texts, "place_wall", choose, "place wall"):
                # the prompt closed (or its check was pressed): the wall stands there
                # review 0.3.1: the check-only frame has no pick of its own -- use the pending city
                self._record_wall(picked[-1] if picked else getattr(self, "_wall_pending", None))
                self._wall_pending = None
            return
        cities = list(gs.me().cities) if gs is not None else []
        if not cities:
            self._forced_fallback(img, "place_wall")
            return
        tried = self.__dict__.setdefault("_wall_tapped", set())
        left = [v for v in cities if v not in tried] or cities
        if len(left) == len(cities):
            tried.clear()
        v = choose(gs, left)
        tried.add(v)
        self.click(self.br.lattice.vertex_px(v), f"place wall on our city {v}")
        if self._confirm_if_check("place wall"):
            self._record_wall(v)                     # review 0.3.1: this wall was never recorded
            self._wall_pending = None

    def _our_knights_ringed(self, img) -> list:
        """[(vertex, (x, y))] of our knights with a selection ring. Stage 2026-10-09 22:00 (game
        212844): Jean's Deserter on us rang both our knights (a white ring round the piece, no
        text); ring_spots finds white blobs and the piece inside fills it, so it found none, the
        turn marker over Jean made it other_turn and nobody moved for 15 min. Read per knight."""
        if not (getattr(self, "cfg", None) or {}).get("deserter_rings_by_corner", True):
            return []
        gs, br = getattr(self, "gs", None), getattr(self, "br", None)
        if img is None or gs is None or br is None or not any(p.is_me for p in gs.players):
            return []
        ks = list(gs.me().knights)
        if not ks:
            return []
        from .vision.rings import vertex_rings
        io = self.io
        native = lay = None
        if getattr(io, "_frame", None) is img:
            native, lay = getattr(io, "native", None), getattr(io, "lay", None)
        try:
            found = vertex_rings(img, br.lattice, br.board, native, lay, vertices=ks)
        except Exception:                             # noqa: BLE001 -- vision must never stop play
            return []
        return [(v, xy) for _, v, xy in found]

    def _rings_only_on_our_knights(self, img, texts) -> bool:
        gs = self.ensure_state(img, texts) if self.mode is not None else None
        if gs is None or not gs.me().knights:
            return False
        try:
            vr = self._vertex_rings(img)
        except Exception:                            # noqa: BLE001
            return False
        return bool(vr) and all(v in gs.me().knights for v, _ in vr)

    def on_lose_knight(self, img, ocr, texts, info):
        """A Deserter played on us: the game rings our knights -- give up the one worth least now
        (owner 2026-10-04: lowest quality, or one the canal no longer needs). Prompt text is
        UNCONFIRMED: every sighting is saved to data/probe."""
        if time.time() - self.__dict__.get("_lose_knight_saved", 0) > 60:
            self._lose_knight_saved = time.time()
            self._probe("lose_knight", img, note="deserter victim prompt (text unconfirmed)")

        def choose(gs, opts):
            mine = [v for v in opts if v in gs.me().knights]
            return self.brain.choose_knight_to_lose(gs, mine) if mine else opts[0]
        self._ringed_pick(img, texts, "lose_knight", choose, "give up knight")

    def _mark_ship_turn(self, me) -> None:
        """Seafarers ship move (owner 2026-10-09): one move a turn, never a ship built this turn.
        The board read can't tell either, so the runner's turn memory fills them in."""
        me.ships_new = set(self.__dict__.get("_ships_this_turn", set())) & set(me.ships)
        me.ship_moved = self.__dict__.get("_ship_moved_turn") == self.__dict__.get("_turn_n", 0)

    def _mark_acted(self, me) -> None:
        """Game 0834: a knight activated this turn was told to move the same turn (refused) --
        one action per knight per turn, so activated or moved this turn = acted."""
        t0 = self.__dict__.get("turn_started")
        if not self.cfg.get("knight_acted_memory", True) or t0 is None:
            return
        at, moved = self.__dict__.get("_activated_at", {}), self.__dict__.get("_moved_knights", set())
        for v, k in me.knights.items():
            if at.get(v, 0) >= t0 or v in moved:
                k.acted = True

    def _hand_settled(self, img) -> dict:
        """Hand read; a slot unreadable within 1 s of a screen change is read once more after
        hand_settle_s from a fresh frame (game 0834: 30 "unreadable, assuming" lines, each on the
        first read after a progress card or unknown->main screen)."""
        h = self.hand.read(img)
        wait = self.cfg.get("hand_settle_s", 0.3)
        since = self.__dict__.get("_screen_since")
        if not wait or since is None or time.time() - since >= 1.0 or all(n is not None for n in h.values()):
            return h
        if type(getattr(self, "io", None)) is ReplayIO:     # a --replay run must not eat its next saved frame
            return h
        time.sleep(wait)
        img2 = self.io.snapshot() if getattr(self, "io", None) is not None else None
        if img2 is None:
            return h
        try:
            return self.hand.read(img2)
        except Exception:                            # noqa: BLE001
            return h

    def _hand_now(self, img=None) -> dict:
        if img is None:
            img = self.io.snapshot()
        hand = getattr(self, "hand", None)
        if img is None or hand is None:
            return {}
        try:
            return {k: v for k, v in hand.read(img).items() if v is not None}
        except Exception:                            # noqa: BLE001
            return {}

    @staticmethod
    def _hand_moved(before: dict, after: dict, give: dict, get: dict):
        """True when every readable give count fell and every get count rose, False when a
        readable count did not move, None when nothing could be compared."""
        checks = [after[k] <= before[k] - n for k, n in give.items() if k in before and k in after]
        checks += [after[k] >= before[k] + n for k, n in get.items() if k in before and k in after]
        if not checks:
            return None
        return all(checks)

    LOOP_WINDOW_S, LOOP_MAX, LOOP_BLOCK_S = 180, 12, 90
    LOOP_ESCALATE = 3      # the same click looping this many times: blocking alone is no way out

    def _click_loop(self, xy, why) -> bool:
        """Audit 2026-10-03: the worst live losses were one click repeated with nothing changing
        (1,250 discards in 60 min, 500 end turns in 25 min, 26 ship cancels a minute). The same
        click 12 times in 3 min = a loop: block that click for 90 s and save the frame; the third
        loop in a game alerts the owner (no STOP: the game goes on with the other actions)."""
        if isinstance(self.io, ReplayIO) and not self.cfg.get("loop_guard_in_replay"):
            return False
        now = time.time()
        key = (re.sub(r"\d+", "#", why), int(xy[0]) // 20, int(xy[1]) // 20)
        blocked = self.__dict__.setdefault("_loop_blocked", {})
        if blocked.get(key, 0) > now:
            return True
        hist = self.__dict__.setdefault("_click_hist", {})
        ts = [t for t in hist.get(key, []) if now - t < self.LOOP_WINDOW_S] + [now]
        hist[key] = ts
        if len(ts) <= self.LOOP_MAX:
            return False
        blocked[key], hist[key] = now + self.LOOP_BLOCK_S, []
        self._loops = getattr(self, "_loops", 0) + 1
        self.note(f"LOOP: '{why}' clicked {len(ts)}x in {self.LOOP_WINDOW_S}s with no way out; "
                  f"blocking it {self.LOOP_BLOCK_S}s (loop {self._loops} this game)")
        try:
            img = getattr(self, "_last_img", None)
            if img is not None:
                d = ROOT / "data" / "loops"; d.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(d / time.strftime("%H%M%S.png")), img)
        except Exception:                            # noqa: BLE001
            pass
        if self._loops == 3:
            self.alert(f"3 click loops this game (last: {why}); see data/loops", halt=False)
        # stage 2026-10-10 05:21-14:58: 'open rules info' looped 159 times on a menu, each loop
        # blocked 90 s and then started again. The same click looping LOOP_ESCALATE times escalates.
        counts = self.__dict__.setdefault("_loop_counts", {})
        counts[key] = counts.get(key, 0) + 1
        if counts[key] % self.cfg.get("loop_escalate", self.LOOP_ESCALATE) == 0:
            self._loop_escalate(why, counts[key])
        return True

    def _loop_escalate(self, why, n):
        """Out of a game (or on a menu) the home button is the way out; in a game the app is
        relaunched (it rejoins the match), the same remedy as a 15-min stall on one screen."""
        if not getattr(self, "_in_play", False) or offline_menu(getattr(self, "_last_img", None)):
            self.note(f"LOOP x{n}: '{why}' -- escalating: pressing home")
            self._go_home(f"loop escalation ({why})", guard=False)
        else:
            self.note(f"LOOP x{n}: '{why}' in a game -- escalating: relaunching the app")
            self.restart_app(f"click loop x{n}: {why}")

    def drag(self, p0, p1, why: str):
        self._last_click_at = time.time()
        if self.cfg.get("dry_run"):
            self.note(f"[dry] drag {why} {tuple(int(a) for a in p0)} -> {tuple(int(a) for a in p1)}")
            if isinstance(self.io, ReplayIO):
                self.io.drag(p0, p1, why)
            return
        self.note(f"drag {why} {tuple(int(a) for a in p0)} -> {tuple(int(a) for a in p1)}")
        self.io.drag(p0, p1, why)
        self._tcount("clicks")
        time.sleep(self.cfg["click_delay_s"] if not isinstance(self.io, ReplayIO) else 0)

    def stopped(self) -> bool:
        return STOP.exists() or self.io.paused()

    ALERT = ROOT / "data" / "ALERT"

    def alert(self, msg: str, halt: bool = True):
        """Owner 2026-10-03: something only the owner can fix (a sign-in prompt, a game that won't
        come back). Log it, append it to data/ALERT (the API's /live shows it), post a macOS
        notification, and with halt write STOP so the bot stops instead of looping."""
        self.note(f"ALERT: {msg}")
        if isinstance(self.io, ReplayIO):
            return
        with open(self.ALERT, "a") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")
        try:
            import subprocess
            subprocess.run(["osascript", "-e", f'display notification "{msg[:180]}" with title "marlow"'],
                           capture_output=True, timeout=10)
        except Exception:                            # noqa: BLE001
            pass
        if halt:
            STOP.write_text(f"alert: {msg}\n")

    # -- rules first (owner rule: read the rules before any strategy)
    def read_rules(self, texts: list[str]) -> bool:
        lines = [t for t in texts if t.strip()]
        import re
        from .rules import mode_from_text, other_scenario
        blob = "\n".join(lines)
        if mode_from_text(blob) is None:
            other = other_scenario(blob)
            if other:                    # owner 2026-10-04: marlow plays only The Great Canal
                self.note(f"rules: this is {other}, not The Great Canal: not playing it")
                return False
            if not re.search(r"\d+\s*victory points", blob.lower()):
                return False             # no scenario/rules on screen yet
        try:
            mode, info = parse_rules(lines)
        except Exception as e:          # noqa: BLE001 -- rules unreadable: keep waiting
            self.note(f"rules not parsed yet: {e}")
            return False
        # keep the fullest read for this game: a runner restart reloads it instead of opening
        # the info panel, which an incoming offer covers (live 2026-10-02: offers declined for 80 s)
        n_unconf = len(info.get("unconfirmed", [])) if isinstance(info, dict) else 99
        if self.mode is not None and n_unconf > getattr(self, "_rules_unconf", 99):
            # solo 2026-10-03: a Great Canal read (2 unconfirmed) was replaced 1 min later by a
            # "base, 10 VP" read of some other panel (5 unconfirmed) -- a thinner read never wins
            return True
        self.mode = mode
        if n_unconf <= getattr(self, "_rules_unconf", 99):
            self._rules_texts, self._rules_unconf = lines, n_unconf
            self._save_game_flags()
        if getattr(self, "_asked_info", False) and getattr(self, "_info_open", True):
            self._info_open = False
            self.click(UI["info_button"], "close rules info")   # don't leave the panel over the board
        self.note(f"rules: {mode.key} target={getattr(mode, 'vp_target', '?')} {info}")
        return True

    FLAGS = ROOT / "data" / "game_flags.json"
    RULES_RETRY_S = 8

    def _save_game_flags(self):
        """Per-game facts that survive a runner restart (cleared in the lobby)."""
        try:
            self.FLAGS.write_text(json.dumps({"lost_cities": getattr(self, "_lost_cities", 0),
                                              "barbarians_attacked": bool(getattr(self, "barbarians_attacked", False)),
                                              "rules_texts": getattr(self, "_rules_texts", None),
                                              "improvements": self._improve_levels(),
                                              # solo 2026-10-06: two mid-game restarts lost our knights
                                              # (knights=0, Smith and "build knight" on our own strong ones)
                                              "my_knights": [[list(v), a, getattr(self, "_knight_levels", {}).get(v, 1)]
                                                             for v, a in getattr(self, "_my_knights", {}).items()],
                                              # owner 03:22: after a restart the dug canal was forgotten and
                                              # the knight that had already dug was activated again
                                              "canal_dug": sorted(map(list, getattr(self, "_canal_seen", set()))),
                                              # owner 03:22: after a restart the start-city guess put the city on
                                              # the settlement (2,-2) and every improvement tap missed
                                              "my_cities": sorted(map(list, getattr(self, "_my_cities", set()))),
                                              "my_blds": sorted(map(list, getattr(self, "_my_blds", set()))),
                                              "my_walls": sorted(map(list, getattr(self, "_my_walls", set()))),
                                              "my_metros": sorted(map(list, getattr(self, "_my_metros", set()))),
                                              "not_city": sorted(map(list, getattr(self, "_not_city", set()))),
                                              # owner 2026-10-08: the bank dialog's ratios hold all game
                                              "ratio_seen": dict(getattr(getattr(self, "brain", None), "ratio_seen", {}) or {}),
                                              "mode": getattr(self.mode, "key", None) or getattr(self, "_mode_saved", None)}))
        except Exception:                            # noqa: BLE001
            pass

    def _load_game_flags(self):
        try:
            d = json.loads(self.FLAGS.read_text())
        except Exception:                            # noqa: BLE001
            return
        self._lost_cities = int(d.get("lost_cities", 0))
        if d.get("barbarians_attacked"):
            self.barbarians_attacked = True
        self._mode_saved = d.get("mode")
        if isinstance(d.get("improvements"), dict):
            self._my_improve = {t: int(d["improvements"].get(t, 0)) for t in IMPROVEMENT_TRACKS}
        for v, a, lv in d.get("my_knights") or []:
            v = tuple(v)
            self.__dict__.setdefault("_my_knights", {})[v] = bool(a)
            if lv > 1:
                self.__dict__.setdefault("_knight_levels", {})[v] = int(lv)
        for k in ("my_cities", "my_blds", "my_walls", "my_metros", "not_city"):
            for v in d.get(k) or []:
                self.__dict__.setdefault("_" + k, set()).add(tuple(v))
        for h in d.get("canal_dug") or []:
            self.__dict__.setdefault("_canal_seen", set()).add(tuple(h))
        if isinstance(d.get("ratio_seen"), dict) and getattr(self, "brain", None) is not None:
            self.brain.ratio_seen.update({r: int(n) for r, n in d["ratio_seen"].items()})
        if d.get("rules_texts") and self.mode is None:
            self.read_rules(d["rules_texts"])        # mid-game restart: same game, same rules

    def _watch_robbed(self, img, texts):
        """Cards taken from us (robber, Spy, Wedding...) make walls worth more (brain.wall_value).
        UNCONFIRMED wording: no robbery popup captured yet -- each hit saves a frame to check."""
        t = " ".join(texts).lower()
        m = re.search(r"(stole|steals|robbed|took|takes)[^.]{0,40}(marlow|from you|your)", t)
        if m and re.search(r"offer|trade", t):
            m = None                                # "<player> took your offer" is a trade, not a loss
        if not m or time.time() - getattr(self, "_robbed_at", 0) < 20:
            return
        self._robbed_at = time.time()
        self._cards_lost = getattr(self, "_cards_lost", 0) + 1
        self.note(f"cards taken from us? '{m.group(0)}' (lost so far {self._cards_lost})")
        # hand ledger: the thief gains one card shaped like our last hand (thief = an opponent
        # named in the same text; UNCONFIRMED wording)
        names = [n for n, _ in (getattr(self, "game_colors", None) or [])
                 if n.lower() != self.cfg["bot_name"].lower()]
        thief = next((n for n in names if n.lower()[:5] in t), None)
        last = getattr(self, "_last_hand", None) or {}
        if thief and last:
            led = self._ledger()
            led.steal(thief, led.me or "me", victim_est=dict(last))
        try:
            d = Path("data/popups"); d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / f"{time.strftime('%H%M%S')}_robbed.png"), img)
        except Exception:                           # noqa: BLE001 -- a debug frame must never stop play
            pass

    def _read_canal(self, img, gs, br, hidden):
        """Great Canal: dug hexes show a dark trench (owner 2026-10-03, live game 3).
        Compare with the first clean board frame; once seen dug a hex stays dug."""
        if not (self.mode and self.mode.canal and gs.board.canal_hexes):
            return
        from .vision.canal_reader import read_canal_dug
        ref = getattr(self, "_canal_ref", None)
        seen = self.__dict__.setdefault("_canal_seen", set())
        if ref is None or ref[0] is not br.lattice:
            if not hidden:                       # a dialog in the reference would hide trenches later
                self._canal_ref = (br.lattice, img.copy())
        elif not hidden:
            new = read_canal_dug(img, ref[1], br.lattice, gs.board) - seen
            if new:
                seen |= new
                self._save_game_flags()
                self.note(f"canal dug (picture): {sorted(new)}, {len(seen)} of {gs.canal_total}")
        gs.canal_dug = set(seen)
        gs.canal_progress = gs.canal_total if len(seen) >= gs.canal_total - 1 else len(seen)

    def _reset_harbors(self, why: str = "new game") -> None:
        """A new game (or a new board) reads its harbours again: nothing of the last game's
        harbours, the template lock, the logged list or the bank dialog's denials carries over.
        Already clear (home screen, every second on stage 2026-10-09): nothing to do or log."""
        if not (getattr(self, "_harbors_best", None) or getattr(self, "_harbors_tpl", False)
                or getattr(self, "_harbors_seen", None) or getattr(self, "_harbors_denied", None)):
            return
        self._harbors_best, self._harbors_tpl, self._harbors_seen = None, False, None
        self._harbors_denied = set()
        self.note(f"harbours: cleared ({why}), reading them again")

    def _pick_harbors(self, board) -> None:
        """Harbours are fixed for a game: keep the fullest read (live 2026-10-02: trade and
        discard overlays cover the coast and dropped the 8 real ones for 2 fake grain ports).
        A harbour the bank dialog has proven we don't have is left out for the rest of the game."""
        best = getattr(self, "_harbors_best", None) or []
        if hasattr(board, "template_offset"):
            # live 2026-10-03 game 6: a half-covered read found 9 junk ports and, being "fuller",
            # replaced the 8 template ones. A template-snapped read always wins and is then kept.
            self._harbors_tpl = True
            self._harbors_best = list(board.harbors)
        elif getattr(self, "_harbors_tpl", False):
            board.harbors = list(best)
        elif len(board.harbors) >= len(best):
            self._harbors_best = list(board.harbors)
        else:
            board.harbors = list(best)
        denied = getattr(self, "_harbors_denied", None) or set()
        if denied:
            board.harbors = [h for h in board.harbors if (h.kind, h.edge) not in denied]
        hb = sorted((h.kind, h.edge) for h in board.harbors)
        if hb != getattr(self, "_harbors_seen", None):
            self._harbors_seen = hb
            self.note(f"harbours read: {hb}")

    def _check_harbors(self, seen: dict) -> None:
        """Owner 2026-10-08: the bank dialog is the truth. A harbour on one of our corners that
        the dialog contradicts (a 3:1 while some card shows 4:1, a 2:1 card above 2) is a misread:
        drop it for the rest of the game (live 21:45-21:54: settlement and city on the template's
        3:1 edge at (5,1), the dialog showed 4 for every card but wool)."""
        gs = getattr(self, "gs", None)
        if gs is None or not seen:
            return
        mine = {v for v, _ in gs.me().buildings()}
        denied = self.__dict__.setdefault("_harbors_denied", set())
        for h in list(gs.board.harbors):
            if not (set(h.edge) & mine):
                continue
            wrong = any(n >= 4 for n in seen.values()) if h.kind == "3:1" else seen.get(h.kind, 2) > 2
            if wrong and (h.kind, h.edge) not in denied:
                denied.add((h.kind, h.edge))
                gs.board.harbors.remove(h)
                self.note(f"harbour {h.kind} at {h.edge}: the bank dialog says no ({seen}); dropped for this game")

    OCR_DEAD_NOTE_S = 60

    def _ocr_watch(self, img, ocr) -> None:
        """Stage 2026-10-10 05:19: Apple Vision went silent in the long-running process (OCR 200 ms
        -> 22 ms, no lines) for 9.5 h on a menu full of text; the runner restart at 15:09 fixed it.
        A busy frame with no text for `ocr_dead_reexec_s` re-executes the runner in place: same pid,
        so start_bot.command's caffeinate and tee carry on (no supervisor would restart an exit)."""
        if isinstance(self.io, ReplayIO) or img is None:
            return
        if ocr or float(img.std()) < 25:              # text read, or a black/loading frame
            self._ocr_empty_since = None
            return
        now = time.time()
        since = self.__dict__.get("_ocr_empty_since") or now
        self._ocr_empty_since = since
        if now - since > self.OCR_DEAD_NOTE_S and not getattr(self, "_ocr_dead_noted", None) == since:
            self._ocr_dead_noted = since
            self.note(f"OCR read nothing on a busy frame for {int(now - since)} s (Apple Vision stuck?)")
        limit = self.cfg.get("ocr_dead_reexec_s", 600)
        if limit and now - since > limit:
            self.alert(f"OCR read nothing for {int(now - since)} s: restarting the runner process", halt=False)
            self._reexec()

    def _reexec(self):
        import os
        import sys
        sys.stdout.flush(); sys.stderr.flush()
        os.execv(sys.executable, [sys.executable, "-m", "catanbot.runner", *sys.argv[1:]])

    def ensure_state(self, img, texts):
        if self.mode is None and not self.read_rules(texts):
            if not getattr(self, "_asked_info", False) \
                    or time.time() - getattr(self, "_asked_info_t", 0) > self.RULES_RETRY_S:
                # the in-game info panel shows the scenario and its rules
                # (UNCONFIRMED: second click on the same button assumed to close it).
                # live 2026-10-08 19:20: after a relaunch into the match it asked once and then
                # waited 2.5 min, declining every offer -- ask again every few seconds
                self._asked_info, self._asked_info_t = True, time.time()
                self._info_open = True
                self.click(UI["info_button"], "open rules info")
            else:
                self.note("rules not known yet: waiting, not playing blind")
            return None
        if not self._scan_ok(img):
            return None
        self._board_ref = img                         # SPEED_PLAN 8: the frame a drag is checked against
        from .vision.board_reader import detect_pieces, read_board
        br = read_board(img, BOARD_ROI, lattice=self.lattice, canal=bool(self.mode and self.mode.canal),
                        scale=getattr(self, "tok_scale", None))
        self.lattice = br.lattice
        self._pick_harbors(br.board)
        gs = GameState(br.board, self.mode)
        gs.ratio_seen = self.brain.ratio_seen                      # the bank dialog: one truth for every caller
        gs.seen_gain = self.__dict__.setdefault("_seen_gain", {})   # what each player collected, all game
        gs.hands = self._ledger()                                  # owner 2026-10-05: everyone's hand
        players = self._players(texts, img, getattr(self, '_last_ocr', None))
        idx = {}
        for i, (n, c) in enumerate(players):
            p = PlayerState(n, c); p.hand = new_hand(); gs.players.append(p); idx[c] = i
            p.is_me = n.lower().replace(" ", "") in (self.cfg["bot_name"].lower(), "marlow.systems")
        if not any(p.is_me for p in gs.players):
            gs.players[0].is_me = True
        pc = self._read_pieces(img, br, [c for _, c in players])
        for v, c in pc.blds.items():
            gs.players[idx[c]].settlements.add(v)
        for e, c in pc.roads.items():
            gs.players[idx[c]].roads.add(e)
        for e, c in pc.ships.items():
            gs.players[idx[c]].ships.add(e)
        for v, (c, active) in pc.knights.items():
            if v not in pc.blds:
                gs.players[idx[c]].knights[v] = Knight(active=active)
        gs.sea_only_edges = set(getattr(self, "_sea_edges", set()))
        if getattr(self.mode, "start_with_city", False):
            # cities and settlements aren't told apart by colour yet (UNCONFIRMED which is
            # which); everyone starts with one city here, so count one
            for p in gs.players:
                if p.is_me and getattr(self, "_my_cities", None):
                    continue          # we know where we placed our cities (applied below)
                if p.is_me and getattr(self, "_lost_cities", 0):
                    continue          # live 2026-10-02: our city was pillaged -- don't pretend we still have it
                if p.settlements and not p.cities:
                    ok = sorted(p.settlements - (getattr(self, "_not_city", set()) if p.is_me else set()))
                    if not ok:
                        continue
                    v = ok[0]
                    p.settlements.discard(v); p.cities.add(v)
        self.read_vp(img, gs)
        h = self._hand_settled(img)
        me = gs.me()
        me.walls |= self.__dict__.setdefault("_my_walls", set())
        # stage 2026-10-08: our metropolises were never tracked live, so "Select position to remove
        # City" (a rival took one) could not tell which city carries it
        me.metropolises |= {v for v in self.__dict__.setdefault("_my_metros", set()) if v in me.cities}
        # game 17 (2026-10-05): levels were never set live, so every improvement looked like it cost
        # 1 commodity (114 tries the game refused) and politics never reached 3 for promotions
        me.improvements = self._improve_levels()
        me.cards_lost = getattr(self, "_cards_lost", 0)   # 7-discards only; robbing isn't read yet   # walls aren't read off the board yet
        hidden = any(re.search(r"player trade|create an offer|hand over to the robber|no player wants", t, re.I) for t in texts)
        hidden = hidden or self.bar_hidden(img)
        if hidden and getattr(self, "_last_hand", None):
            # live 2026-10-02: a dialog covers the hand bar and it reads all zeros -> the bot
            # planned nothing with 11 cards. Use the last full read instead.
            h = dict(new_hand(), **self._last_hand)
        for k, n in h.items():
            if n is None and not hidden:
                # bright digit we couldn't read is 5+ (templates cover 1-4); a low guess beats
                # an empty hand, which made the bot idle with 12 cards (live 2026-10-02)
                # final probe game 10:31: "assume 5" planned a 4-brick bank trade we didn't have;
                # the last full read of that card is the better guess when there is one
                lh = getattr(self, "_last_hand", None) or {}
                n = lh.get(k, 0) if lh else 5        # _last_hand keeps non-zero counts only
                self.note(f"hand: {k} unreadable, assuming {n}")
            if n is not None:
                me.hand[k] = n
        exp = getattr(self, "_trade_expect", None)
        if exp and time.time() - exp[0] < 12:
            for k in exp[2]:
                if me.hand.get(k, 0) < exp[1].get(k, 0):
                    self.note(f"hand: {k} read {me.hand.get(k, 0)} just after the trade, using {exp[1][k]}")
                    me.hand[k] = exp[1][k]
        # our own buildings don't vanish between reads (only a pillage takes a city): remember
        # every corner seen in our colour (live 2026-10-03: city missed -> nothing ever built)
        for v, active in getattr(self, "_my_knights", {}).items():
            # solo 2026-10-03: the vision missed our knight and the bot kept "building" on it
            if v not in me.knights and not any(v in p.knights for p in gs.players if not p.is_me):
                me.knights[v] = Knight(active=active)
        for v, lv in getattr(self, "_knight_levels", {}).items():
            # solo 2026-10-06: vision reads every knight as basic, so a strong knight was
            # "promoted" again 36 times (mighty needs politics 3): promotions we landed stick
            if v in me.knights:
                me.knights[v].level = max(me.knights[v].level, lv)
        self._mark_acted(me)
        self._mark_ship_turn(me)
        for v, who in getattr(self, "_taken_spots", {}).items():
            if not any(v in p.settlements or v in p.cities or v in p.knights for p in gs.players):
                p = next((p for p in gs.players if who.lower() and who.lower() in p.name.lower()),
                         next((p for p in gs.players if not p.is_me), None))
                if p is not None:
                    p.settlements.add(v)
        for attr, net in (("_my_roads", me.roads), ("_my_ships", me.ships)):
            # live 2026-10-04: builds we saw land stay ours even when a read misses them
            for e in getattr(self, attr, set()):
                if not any(e in p.roads or e in p.ships for p in gs.players if not p.is_me):
                    net.add(e)
        mine = self.__dict__.setdefault("_my_blds", set())
        mine |= set(me.settlements) | set(me.cities)
        taken = {v for p in gs.players if not p.is_me for v in (p.settlements | p.cities)}
        for v in mine - taken - set(me.settlements) - set(me.cities):
            me.settlements.add(v)
        # live 2026-10-03: the start-city guess (first corner in sort order) called the gold-coast
        # settlement the city -- use the corners where we really placed or upgraded a city
        for v in getattr(self, "_my_cities", set()) - taken:
            me.settlements.discard(v); me.cities.add(v)
        if sum(me.hand.values()) > 0 and not hidden:
            self._last_hand = {k: v for k, v in me.hand.items() if v}
        self._read_canal(img, gs, br, hidden)
        self._apply_banners(img, gs)
        self._ledger_observe(gs, hidden)
        if getattr(self.mode, "ck", False):
            # owner 2026-10-05 game 15: the barbarian track on the left was never read -> danger 0 all game
            from .vision import barbarians as bb
            tr = bb.read_track(img)
            self._barb_read(tr)
            gs.barbarian_pos = getattr(self, "_barb_pos", 0)
        self._update_outlook(gs)
        self.gs, self.br = gs, br
        return gs

    def _barb_tracker(self):
        from .vision import barbarians as bb
        t = self.__dict__.get("_barb_track")
        if t is None:
            t = self._barb_track = bb.Track()
        return t

    def _barb_read(self, tr) -> None:
        """Stage 2026-10-09: step 6 read as 0 mid-voyage ("barbarians far" at 6/7). A raw reading
        only moves the ship forward; a backward jump counts only after an attack (step 7, the
        attack popup/message, our city lost) or when the new voyage is seen moving (1 then 2)."""
        trk = self._barb_tracker()
        raw = tr[0] if tr is not None else None
        before = trk.pos
        pos, ev = trk.update(raw)
        self._barb_pos = pos
        if tr is not None and (raw, pos) != getattr(self, "_barb_logged", None):
            self._barb_logged = (raw, pos)
            extra = ("" if raw == pos else f" -- ignored, ship stays at {pos}/7" if ev == "ignored"
                     else f" -> {pos}/7")
            self.note(f"barbarians: ship at {tr[1]} -> step {raw}/7 (score {tr[2]:.2f}){extra}")
        if ev == "attack":
            self._barb_attack(f"ship read at step 7 (was {before}/7)")
        elif ev == "unseen_attack":
            # the attack itself was never on screen: our knights went inactive with it, except
            # ones we activated after the ship was last seen before the new voyage
            self.note(f"barbarians: new voyage seen ({before}/7 -> {pos}/7): an attack we did not see")
            self.barbarians_attacked = True
            held = getattr(trk, "held_at", 0.0)
            at = self.__dict__.get("_activated_at", {})
            for v in getattr(self, "_my_knights", {}):
                if at.get(v, 0.0) <= held:
                    self._my_knights[v] = False
            self._save_game_flags()               # review 0.3.1: a restart keeps the attack
        if ev != "ignored" and raw is not None:
            trk.held_at = time.time()

    def _barb_attack(self, why: str) -> None:
        """A confirmed attack: every knight goes inactive, the ship starts again."""
        trk = self._barb_tracker()
        if trk.after_attack and time.time() - self.__dict__.get("_barb_attack_at", 0) < 120:
            return                                    # the same attack seen again
        self._barb_attack_at = time.time()
        if not getattr(self, "barbarians_attacked", False):
            self.__dict__.setdefault("_feed_events", []).append("barbarians attacked for the first time")
        self.barbarians_attacked = True               # unlocks Bishop (progress cards)
        self.note(f"barbarians attacked ({why}): our knights go inactive")
        for v in getattr(self, "_my_knights", {}):
            self._my_knights[v] = False
        trk.attacked()
        self._barb_pos = 0
        self._save_game_flags()                   # review 0.3.1: a restart keeps the attack

    def _ledger(self):
        """Owner 2026-10-05: per-player hand ledger (catanbot/hands.py), one per game."""
        h = self.__dict__.get("_hands")
        if h is None:
            from .hands import HandLedger
            h = self._hands = HandLedger()
        return h

    def _ledger_observe(self, gs, hidden=False):
        """Board pieces -> builds paid; our hand -> cards we took made exact; fresh banner counts
        -> reconcile_all (opponents only); the ledger's count stands in for an unread banner."""
        try:
            led = gs.hands = self._ledger()
            led.me = gs.me().name
            built = led.observe_pieces(gs)
            if built:
                self.note(f"hands: builds seen {built}")
            if not hidden:
                got = led.my_hand(dict(gs.me().hand))
                if got:
                    self.note(f"hands: took {got} (exact)")
            cache = self.__dict__.get("_banner_counts", {})
            read_at = self.__dict__.get("_banner_read_at", {})
            opp = gs.opponents()
            led.reconcile_all({p.name: cache.get(_norm(p.name), (None, None))[0] for p in opp},
                              {p.name: read_at.get(_norm(p.name)) for p in opp})
            for p in opp:
                if led.knows(p.name):
                    p.hand_count = led.count(p.name)  # owner 2026-10-06: always know everyone's hand
        except Exception as e:                        # noqa: BLE001 -- the ledger is a bonus, never stop play
            self.note(f"hands: ledger update failed: {type(e).__name__}: {e}")

    def _read_pieces(self, img, br, colors):
        """Pieces on the board (detect_pieces, on the native capture when the live IO has one).
        Live 2026-10-04: with a build popup open its green/red rings and labels cover the spot
        and the road under the popup read as a rival's -- keep the last clean read then.
        Buildings, roads and ships never leave the board (a pillage is tracked apart), so a
        clean read is merged with the last one; knights move and are taken as read."""
        from .vision.board_reader import Pieces, detect_pieces
        last = getattr(self, "_last_pieces", None)
        if self._popup_options(img):
            if last is not None:
                return last
            return Pieces({}, {}, {}, {})
        io = self.io
        nat = getattr(io, "native", None) if getattr(io, "_frame", None) is img else None
        lay = getattr(io, "lay", None) if nat is not None else None
        pc = detect_pieces(img, br, colors, native=nat, lay=lay)
        if last is not None:
            for name in ("blds", "roads", "ships"):
                cur, old = getattr(pc, name), getattr(last, name)
                for k, c in old.items():
                    if k not in cur and c in colors and not (name != "blds" and (k in pc.roads or k in pc.ships)):
                        cur[k] = c
        self._last_pieces = pc
        return pc

    def _apply_banners(self, img, gs):
        """Owner 2026-10-03: know what the others likely hold. Card counts come off the player
        banners (OCR, slow: read after a roll or every 45 s, else the cached counts)."""
        now = time.time()
        cache = self.__dict__.setdefault("_banner_counts", {})
        if getattr(self, "_banner_stale", True) or now - getattr(self, "_banner_at", 0) > 45 \
                or self._ledger().needs_recount(getattr(self, "_banner_at", 0), now):
            try:
                from .vision.banner import read_banner_counts
                for name, (res, prog) in read_banner_counts(img, self.io.ocr).items():
                    k = _norm(name)
                    old = cache.get(k, (None, None))
                    cache[k] = (res if res is not None else old[0], prog if prog is not None else old[1])
                    if res is not None:
                        self.__dict__.setdefault("_banner_read_at", {})[k] = now   # hand ledger reconcile
            except Exception as e:                    # noqa: BLE001 -- counts are a bonus, never stop play
                self.note(f"banner read failed: {type(e).__name__}: {e}")
            self._banner_stale, self._banner_at = False, now
        for p in gs.players:
            res, prog = cache.get(_norm(p.name), (None, None))
            if p.is_me:
                continue                              # our own hand is read exactly off the hand bar
            if res is not None:
                p.hand_count = res
            if prog is not None:
                p.progress_cards = prog

    def _watch_dice(self, img):
        """Every roll (anyone's): add what each player collects to the income tracker."""
        try:
            from .vision.banner import read_dice
            d = read_dice(img)
        except Exception:                             # noqa: BLE001
            return
        if not d or d == getattr(self, "_last_dice", None):
            return                                    # (two identical rolls in a row are missed)
        self._last_dice = d
        self._banner_stale = True                     # cards moved: read the counts again
        gl = self._glog()
        if gl is not None:
            gl.roll(d, self._glog_vp())
        total = sum(d)
        self.__dict__.setdefault("_feed_dice", []).append(total)
        if total == 7:
            self.__dict__.setdefault("_feed_events", []).append("a 7 was rolled")
        self._feed("live")
        if self.gs is not None:
            try:
                led = self.gs.hands = self._ledger()
                led.on_roll(self.gs, total)
            except Exception as e:                    # noqa: BLE001
                self.note(f"hands: roll update failed: {type(e).__name__}: {e}")
        if total != 7 and self.gs is not None:
            self.gs.seen_gain = self.__dict__.setdefault("_seen_gain", {})
            self.gs.record_roll(total)
            self.note(f"dice {d[0]}+{d[1]}={total}: income tracker updated")

    def _after_drag_scan(self):
        """A drag that misses its card could pan the board. Old: drop the scan (a full SCAN, and
        an 8 s gate when it fails). owner 2026-10-09 "play every move way faster": keep it and
        compare 3 hexes with the last board frame first ("full" = the old reset)."""
        if self.cfg.get("scan_after_drag", "check") == "check" and getattr(self, "scan", None) is not None \
                and self.__dict__.get("_board_ref") is not None:
            self._check_pan = True
        else:
            self.lattice = self.scan = None

    def _board_moved(self, img) -> bool | None:
        """Mean grey difference of 3 land hex patches (robber hex left out) between the last board
        frame and img, over scan_check_max_diff on any patch = moved. None = can't tell."""
        ref, lat, gs = self.__dict__.get("_board_ref"), getattr(self, "lattice", None), getattr(self, "gs", None)
        if ref is None or img is None or lat is None or gs is None or ref.shape != img.shape:
            return None
        hexes = [h.coord for h in gs.board.hexes.values()
                 if h.terrain not in ("sea", "canal", "unknown") and h.coord != gs.board.robber][:3]
        if len(hexes) < 3:
            return None
        g0 = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY) if ref.ndim == 3 else ref
        g1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        H, W = g1.shape[:2]
        diffs = []
        for c in hexes:
            x, y = map(int, lat.hex_px(*c))
            x0, y0, x1, y1 = max(0, x - 12), max(0, y - 12), min(W, x + 12), min(H, y + 12)
            if x1 - x0 < 8 or y1 - y0 < 8:
                return None
            diffs.append(float(np.mean(cv2.absdiff(g0[y0:y1, x0:x1], g1[y0:y1, x0:x1]))))
        return max(diffs) > float(self.cfg.get("scan_check_max_diff", 0.5))

    def _scan_ok(self, img) -> bool:
        """Owner 2026-10-03: at game start read and annotate everything once (players, colours,
        turn, hexes, numbers) and play only when that read is clean. Saved to
        data/popups/scan_annotated.png + scan.json."""
        if self.__dict__.get("_check_pan"):
            self._check_pan = False
            moved = self._board_moved(img)
            if moved is False:
                self.note("board check after drag: board did not move, scan kept")
            else:
                self.note(f"board check after drag: {'moved' if moved else 'unclear'}: scanning again")
                self.lattice = self.scan = None
                self._scan_at = 0                    # the old reset scanned at once
        if getattr(self, "scan", None) is not None and self.scan.ok:
            return True
        if time.time() - getattr(self, "_scan_at", 0) < 8:
            return False
        self._scan_at = time.time()
        from .vision import layout as L
        try:
            s = L.scan(img, getattr(self, "_last_ocr", None) or [], self.banner_color,
                       canal=bool(self.mode and self.mode.canal),
                       roi=(80, 150, LAY.size[0] - 10, LAY.size[1] - 40))   # whole window: zoom moves the board
            L.save(img, s, UI)
        except Exception as e:                       # noqa: BLE001 -- a failed scan means: don't play yet
            self.note(f"SCAN failed: {e!r}")
            return False
        self.scan = s
        self.note("SCAN " + s.summary())
        if s.ok and not self._game_file("scan.json").exists():
            self._game_save("scan.json", s.to_json())    # first clean scan of the game, never overwritten
            cv2.imwrite(str(self._game_file("scan.png")), img)
        if s.ok:
            self.lattice, self.tok_scale = s.lattice, s.token_scale
        return s.ok

    def _rank_threat_before_turn(self):
        """Re-plan from the board as the previous player left it and pick the biggest threat."""
        gs = getattr(self, "gs", None)
        if gs is None:
            return
        try:
            ranked = self.brain.update_outlook(gs, getattr(self, "_lost_cities", 0), rank=True)
            if ranked:
                self.note("threat before our turn: " + " > ".join(f"{n} ({w})" for n, _, w in ranked))
        except Exception as e:                       # never block the turn on planning
            self.note(f"threat ranking failed: {e!r}")

    def _update_outlook(self, gs):
        """Owner 2026-10-02: plan 3 moves ahead and re-plan after every move anyone makes."""
        try:
            if self.brain.update_outlook(gs, getattr(self, "_lost_cities", 0)):
                self.note("outlook: " + self.brain.outlook.summary())
                self._glog_outlook()
        except Exception as e:                   # noqa: BLE001
            self.note(f"outlook failed: {e!r}")

    def _glog_outlook(self):
        """Owner 2026-10-09: the ledger keeps the latest goals and threats (status only)."""
        o = getattr(self.brain, "outlook", None)
        gl = self._glog(start=False) if o is not None else None
        if gl is not None:
            gl.outlook(list(o.goals), dict(o.threats))

    @staticmethod
    def _plan_item(a) -> str:
        if a.kind in ("bank_trade", "player_trade"):
            give = "+".join(f"{v} {k}" for k, v in (a.give or {}).items() if v)
            get = "+".join(f"{v} {k}" for k, v in (a.get or {}).items() if v)
            return f"{'bank' if a.kind == 'bank_trade' else 'trade'} {give} -> {get}".strip()
        return a.kind if a.target is None else f"{a.kind} {a.target}"

    def _plan_reason(self, gs, plan) -> str:
        """One line: why this plan (stance, the goal it serves, the cards it still trades for)."""
        b, parts = self.brain, []
        if getattr(b, "stance", None):
            parts.append(f"{b.stance}" + (f" ({b.focus})" if getattr(b, "focus", None) else ""))
        o = getattr(b, "outlook", None)
        if o is not None and o.goals:
            parts.append(f"next goal {o.goals[0]}")
        notes = [a.note for a in plan if getattr(a, "note", "")]
        if notes:
            parts.append(notes[0])
        try:
            want = b._trade_targets(gs)
        except Exception:                        # noqa: BLE001
            want = []
        if want:
            parts.append(f"trading for {', '.join(map(str, want))}")
        if not plan:
            parts.append("nothing affordable or worth it: hold")
        return "; ".join(parts)

    @staticmethod
    def bar_hidden(img) -> bool:
        """Hand bar covered by a dialog: mean brightness ~26 vs ~74 when visible (2026-10-02)."""
        y0, y1, x0, x1 = BAR_BOX
        return float(img[y0:y1, x0:x1].mean()) < 45

    def read_vp(self, img, gs):
        """VP badge left of each avatar (white digit on a coloured disc, ~73 px left of the
        banner name). Vision misses a lone digit, so binarise it and add a 'VP' label.
        Keeps the last value when a read fails (live 2026-10-02: 7/3/3/3 read right)."""
        vp = self.__dict__.setdefault("_vp", {})
        if time.time() - getattr(self, "_vp_at", 0) > 8:      # 4 extra OCRs: refresh every 8 s
            self._vp_at = time.time()
            self._read_vp_badges(img, vp)
            names = {p.name for p in gs.players}
            if names:
                # solo 2026-10-06: banner OCR added "Ponts": 16 and "mærlow.systems" -- only seats count
                for n in [n for n in vp if n not in names]:
                    vp.pop(n)
        for p in gs.players:
            if p.name in vp:
                p.vp_visible = vp[p.name]

    def _read_vp_badges(self, img, vp):
        for o in getattr(self, "_last_ocr", None) or []:
            if o["y"] > 130:
                continue
            name = o["text"].strip()
            if not re.fullmatch(r"[A-Za-z][\w.\-]{2,}", name) or not re.search(r"[a-z]", name):
                continue
            bx = int(o["x"] - (73 if LAY.is_reference else 80))   # full screen: 80 px (live 2026-10-03)
            if not LAY.is_reference:
                # full screen the digit is ~9 px in the frame: a binary mask turns it into blocks.
                # Soft whiteness, cubic upscale, "VP" label on the digit's line (live 2026-10-03: 7/3/3/6)
                for w, fx in ((11, 6), (14, 6), (11, 8), (14, 8), (11, 5), (12, 7), (13, 5), (16, 6)):
                    c = img[72:92, max(0, bx - w):bx + w]
                    if c.size == 0:
                        break
                    g = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
                    v = np.where(g[..., 1] < 90, g[..., 2], 0).astype(np.uint8)
                    v = cv2.resize(255 - v, None, fx=fx, fy=fx, interpolation=cv2.INTER_CUBIC)
                    m = cv2.copyMakeBorder(v, 40, 40, 160, 60, cv2.BORDER_CONSTANT, value=255)
                    cv2.putText(m, "VP", (10, 40 + v.shape[0] // 2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 2, 0, 5)
                    # the digit often comes back as its own box: join the boxes
                    txt = " ".join(t["text"] for t in self.io.ocr(cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)))
                    d = re.search(r"VP\s*(\d{1,2})\b", txt)
                    if d and int(d.group(1)) <= 25:
                        vp[name] = int(d.group(1))
                        break
                continue
            # live 2026-10-02: "11" (two thin ones) failed the single variant -> Wren read
            # as 0 VP. Try a few crops/thickenings, first readable one wins.
            for w, dil, fx in ((14, 1, 4), (11, 0, 4), (14, 0, 3), (11, 1, 3), (11, 0, 2), (14, 1, 2), (11, 1, 4), (14, 0, 4)):
                c = img[68:94, max(0, bx - w):bx + w]
                if c.size == 0:
                    break
                hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
                m = ((hsv[..., 1] < 60) & (hsv[..., 2] > 200)).astype(np.uint8) * 255
                if dil:
                    m = cv2.dilate(m, np.ones((2, 2), np.uint8))
                m = cv2.resize(255 - m, None, fx=fx, fy=fx, interpolation=cv2.INTER_NEAREST)
                m = cv2.copyMakeBorder(m, 60, 60, 200, 200, cv2.BORDER_CONSTANT, value=255)
                cv2.putText(m, "VP", (20, 60 + 13 * fx), cv2.FONT_HERSHEY_SIMPLEX, 3, 0, 6)
                d = next((d for t in self.io.ocr(cv2.cvtColor(m, cv2.COLOR_GRAY2BGR))
                          for d in [re.search(r"VP\s*(\d{1,2})\b", t["text"])] if d and int(d.group(1)) <= 25), None)
                if d:
                    vp[name] = int(d.group(1))
                    break

    @staticmethod
    def banner_color(h, s_, v):
        """Player banner hue -> piece colour name (canal game 3 calibration, 2026-10-02)."""
        if s_ < 60:
            return "grey" if v < 170 else "white"    # marlow's grey banner: s0 v79 (live_frame_now)
        if h >= 157 and h <= 172:
            return "magenta"
        if 130 <= h < 157:
            return "purple"
        if 85 <= h <= 115:
            return "blue"
        if 40 <= h < 85:
            return "green"
        if h >= 173 or h < 5:
            return "red" if v > 130 else "brown"
        if h < 12:
            return "brown" if v < 140 else "red"
        if h <= 30:
            return "orange"
        return "white" if s_ < 60 else "orange"

    @staticmethod
    def _banner_slots(img, yc):
        """Coloured name bars along the top as (x0, x1, hsv median), found by colour alone.
        live 2026-10-03 game 6: reading the colour next to each OCR'd name dropped players whose
        name OCR missed (Jean, Lin) and read grey banners as brown. A banner is a run of >=110
        columns of one steady colour (text masked); the wood around it is too patchy to form one."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(int)
        band = hsv[max(0, yc - 9):yc + 10]
        txt = (band[..., 1] < 70) & (band[..., 2] > 170)
        cols = []
        for x in range(band.shape[1]):
            c = band[:, x][~txt[:, x]]
            cols.append(np.median(c, 0) if len(c) >= 5 else None)

        def close(a, b):
            dh = abs(a[0] - b[0]); dh = min(dh, 180 - dh)
            return (dh <= 8 or (a[1] < 60 and b[1] < 60)) and abs(a[1] - b[1]) <= 45 and abs(a[2] - b[2]) <= 40
        out, x = [], 0
        while x < len(cols):
            if cols[x] is None:
                x += 1; continue
            s0, acc = x, [cols[x]]
            while x + 1 < len(cols) and cols[x + 1] is not None and close(np.median(acc[-15:], 0), cols[x + 1]):
                x += 1; acc.append(cols[x])
            m = np.median(acc, 0)
            if x - s0 >= 110 and m[2] >= 40:          # not the black margin beside the window
                out.append((s0, x, tuple(float(v) for v in m)))
            x += 1
        return out

    def _banner_colors(self, img, ocr):
        """Read every player's colour from the name banners along the top: one slot per banner,
        named from OCR when it reads the name, else the name last seen in that slot."""
        names = sorted((o for o in ocr if o["y"] < 130 and re.match(r"^[A-Za-z][\w.\-]{2,}$", o["text"].strip())
                        and re.search(r"[a-z]", o["text"])), key=lambda o: o["x"])
        me = next((o for o in names if self.chat_is_me(o["text"].strip())), None) if names else None
        ys = [o["y"] + o["h"] / 2 for o in ([me] if me else names)]
        yc = int(np.median(ys)) if ys else 98
        slots = self._banner_slots(img, yc)
        if len(slots) < 2:
            return None
        known = self.__dict__.setdefault("_slot_names", {})
        out = []
        for i, (x0, x1, (hh, ss, vv)) in enumerate(slots):
            o = next((o for o in names if x0 - 90 <= o["x"] <= x1), None)
            if o:
                known[i] = o["text"].strip()
            out.append((known.get(i, f"P{i + 1}"), self.banner_color(hh, ss, vv)))
        return out

    def chat_is_me(self, name):
        chat = getattr(self, "chat", None)
        if chat is not None:
            return chat.is_me(name)
        return name.lower() == str(getattr(self, "cfg", {}).get("bot_name", "marlow.systems")).lower()

    COLOR_REREAD_S = 90

    def _players(self, texts, img=None, ocr=None):
        # owner 2026-10-02: recognise our colour at game start and keep it for this game only.
        # live 2026-10-03 game 6: the colours changed after a game relaunch, so read them again
        # every COLOR_REREAD_S; a read with fewer players never replaces a fuller one.
        cached = getattr(self, "game_colors", None)
        now = time.time()
        if cached and now - getattr(self, "_colors_at", 0) < self.COLOR_REREAD_S:
            return cached
        if img is not None and ocr is not None:
            cols = self._banner_colors(img, ocr)
            self._colors_at = now
            # audit 2026-10-03: two players never share a colour; and a change must read the
            # same twice in a row (a dialog over a banner flipped our own colour once)
            if cols and len({c for _, c in cols}) < len(cols):
                self.note(f"colour read with a duplicate colour, ignored: {cols}")
                cols = None
            if cached and cols and cols != cached and cols != getattr(self, "_colors_pending", None):
                self._colors_pending, self._colors_at = cols, now - self.COLOR_REREAD_S + 10
                cols = None
            if cols and (not cached or len(cols) >= len(cached)) and cols != cached:
                self._colors_pending = None
                self._game_save("players.json", {"colours": cols, "at": time.strftime("%H:%M:%S")})
                self.game_colors = cols
                self.note(f"colours this game: {cols}" if not cached else f"colours changed: {cached} -> {cols}")
                self.chat.color = next((c for n, c in cols if self.chat.is_me(n)), None)
                if not cached:
                    self._intro_due = True            # owner 2026-10-05: say hi after setup, not during it
                return cols
        if cached:
            return cached
        if self.cfg.get("players"):                  # fallback: data/config.json
            return [tuple(x) for x in self.cfg["players"]]
        return getattr(self, "players", None) or [("marlow.systems", "red"), ("P2", "purple"), ("P3", "brown")]

    def _stall_progress(self, state):
        """Move the stall clock only on real progress: a new screen, a hand change or one of our
        pieces landing. Clicks alone (the rescue's own taps) never move it."""
        gs = getattr(self, "gs", None)
        pieces = None
        try:
            if gs is not None:
                me = gs.me()
                pieces = (len(me.roads) + len(me.ships), len(list(me.buildings())), len(me.walls),
                          len(me.knights), sum(me.improvements.values()), me.vp_visible)
        except Exception:                             # noqa: BLE001
            pieces = None
        self._prompt_clock_clear(state)
        sig = (state, tuple(sorted((getattr(self, "_last_hand", None) or {}).items())), pieces)
        if sig != getattr(self, "_stall_sig", None):
            self._stall_sig = sig
            self._progress_at = time.time()

    def _marker_seat(self, img) -> int | None:
        """Banner index under the gold turn triangle while in play (0 = us), else None."""
        if not getattr(self, "_in_play", False) or img is None:
            return None
        try:
            seat = active_seat(img)
        except Exception:                             # noqa: BLE001 -- a reading aid only
            return None
        # live 2026-10-09 19:26: our portrait was banner 1, so our own robber prompt read as
        # "rings on marlow.systems's turn: not ours" and the bot sat; a banner with our name is us
        name = getattr(self, "_slot_names", {}).get(seat) if seat else None
        if name and self.chat_is_me(name):
            return 0
        return seat

    def _other_turn_by_marker(self, img, info: dict, seat: int | None, state: str = "unknown") -> str:
        """stage 2026-10-09 00:36/00:37: rival turns show no turn text, so they read as unknown
        and the stall clock/probe saw one long 'unknown'. The gold turn triangle over a rival's
        portrait makes it other_turn -- unless the screen asks us something without text (rings
        on the board: Deserter on our knights; the waiting green check), which stays unknown so
        the 45 s textless fallbacks still fire."""
        if not seat:                                  # None, or 0 = our own portrait
            return state
        if self._rings(img) or self._green_check(img) or self._our_knights_ringed(img):
            return state
        info["player"] = getattr(self, "_slot_names", {}).get(seat, f"P{seat + 1}")
        info["seat"] = seat
        return "other_turn"

    # -- one look at the screen
    def _stall_lead(self):
        """(ours, best rival) when we lead or trail the best rival by <= 2 VP, else None."""
        gs = getattr(self, "gs", None)
        if gs is None or not any(p.is_me for p in gs.players):
            return None
        ours = gs.me().vp_visible or 0
        best = max([p.vp_visible or 0 for p in gs.players if not p.is_me] or [0])
        return (ours, best) if ours >= best - 2 else None

    def _rescue(self, mark):
        img0 = self.io.snapshot()
        if img0 is None:
            return
        o0 = self.io.ocr(img0)
        self._probe(f"rescue_{mark}", img0, texts=[o["text"] for o in o0])
        if self.cfg.get("rescue_in_game_exits", True):
            self._rescue_ladder(img0, o0, mark)
            return
        btn = find_text(o0, r"^(ok|confirm|continue|close|done|accept)$")
        if btn:
            self.probe.tried("rescue_button")
            self.click((btn["x"] + btn["w"] / 2, btn["y"] + btn["h"] / 2), f"rescue: {btn['text']}", o0)
        elif self._green_check(img0):
            self.probe.tried("rescue_check")
            self.click(self._check_at(img0), "rescue: green check")
        elif self._rings(img0):
            self._forced_fallback(img0, "rescue_rings")
        else:
            # game 12 16:09-16:19: Marianne's turn froze with nothing on screen for
            # us; a click on empty table is the only harmless nudge left
            self.probe.tried("rescue_table_click")
            self.click(UI.get("pointer_park", self.PARK_SPOT), "rescue: click empty table")

    def _rescue_ladder(self, img0, o0, mark):
        """stage 2026-10-09 game 1548: the rescue went button, check, blob rings, table click and
        never tapped the Diplomat's road rings or the X. In-game exits first, one rung per mark,
        the table click last. Never Esc (owner: unverified, may open the quit menu)."""
        def rung(name):
            self.probe.tried(f"rescue_{mark}_{name}")
        btn = find_text(o0, r"^(ok|confirm|continue|close|done|accept)$")
        if btn:
            rung("button")
            self.click((btn["x"] + btn["w"] / 2, btn["y"] + btn["h"] / 2), f"rescue: {btn['text']}", o0)
            return
        if self._green_check(img0):
            rung("check")
            self.click(self._check_at(img0), "rescue: green check")
            return
        if self._edge_rings(img0):
            rung("road")
            self.note("rescue: ringed road")
            if self._diplomat_pick(img=img0):
                return
        if self._fallback_rings(img0):
            rung("rings")
            self._forced_fallback(img0, "rescue_rings")
            return
        if self._bar_blob(img0, "red") is not None:
            rung("red_x")
            self.click(self._x_at(img0), "rescue: red X")
            return
        phase = turn_phase(img0)
        if phase == "main":
            rung("end_turn")
            self.click(UI["dice_or_end_turn"], "rescue: ending the turn")
            return
        if phase == "idle" and mark == 420:
            rung("end_turn_grey")
            self.click(UI["dice_or_end_turn"], "rescue: End turn grey, pressing once "
                                               "(UNCONFIRMED whether the game takes it)")
            return
        rung("table_click")
        self.click(UI.get("pointer_park", self.PARK_SPOT), "rescue: click empty table")

    def step(self) -> str:
        if not isinstance(self.io, ReplayIO) and self.LEAVE_FLAG.exists() and \
                time.time() >= getattr(self, "_leave_retry_at", 0):
            return self.leave_on_request()
        if self.stopped():
            self.note("stop switch on: not acting")
            return "stopped"
        if not isinstance(self.io, ReplayIO) and self.RESTART_FLAG.exists():
            self.RESTART_FLAG.unlink()                # owner: `touch data/restart.flag` = restart now
            self.restart_app("restart requested")
            return "restart"
        # solo 2026-10-03: three stalls (Commercial Harbor, an offer, a knight prompt) of 13-55 min
        # with the log moving but no click -- solo only: a solo game waits on nobody but us
        if not isinstance(self.io, ReplayIO) and self.cfg.get("solo"):
            now = time.time()
            # stage 2026-10-08: the clock was the last click, so the 240 s rescue's own click
            # reset it and the 420 s rescue / 600 s restart never came. It now moves only on
            # real progress (_stall_progress: a new screen, a hand change, our piece landing).
            last = max(getattr(self, "_progress_at", 0), getattr(self, "_restarted_at", 0))
            if not last:
                self._progress_at = last = now
            limit, extra = self.cfg.get("stall_restart_s", 600), self.cfg.get("stall_lead_extra_s", 300) or 0
            if now - last > limit:
                lead = self._stall_lead() if extra and self.__dict__.get("_stall_extra_for") != last else None
                if lead:
                    # stage 2026-10-09 game 1548: the rescue quit a match we led 14/14/13/7 at 10 min
                    self._stall_extra_for = last
                    self.note(f"no progress for 10 min but we lead {lead[0]} vs {lead[1]}: one more window")
                    self._rescue(600)
                elif self.__dict__.get("_stall_extra_for") != last or now - last > limit + extra:
                    self._probe("stall_no_click")
                    self.restart_app(f"no progress for {int((now - last) // 60)} min")
                    return "restart"
            # probe 2026-10-04: an unhandled dialog (Deserter's player pick) cost the whole game to
            # the 10-min restart. At 4 and 7 min: save the frame, press a plainly labelled button
            for mark in (240, 420):
                done = self.__dict__.setdefault("_rescued", set())
                if now - last > mark and (last, mark) not in done:
                    done.add((last, mark))
                    self._rescue(mark)
        t_0 = time.perf_counter()
        snap_at = time.time()
        img = self.io.snapshot()
        if img is None:
            return "end"
        self._snap_at = snap_at                      # 0.4.3: our turn clock starts at the roll frame's capture
        t_snap = time.perf_counter()
        self._last_img = img
        ocr = self.io.ocr(img)
        self._last_ocr = ocr
        t_ocr = time.perf_counter()
        texts = [o["text"] for o in ocr]
        state, info = classify(texts, self.cfg["bot_name"].lower())
        if state == "unknown" and offline_menu(img):
            state = "offline_menu"                    # stage 2026-10-10: read by its pixels when OCR fails
        self._step_state = state
        self._ocr_watch(img, ocr)
        self._tcount("reads")
        if self.cfg.get("log_timing", True):
            # SPEED_PLAN "measure after": capture vs OCR vs classify had no line of their own
            self.note(f"t: snap={(t_snap - t_0) * 1000:.0f} ocr={(t_ocr - t_snap) * 1000:.0f} "
                      f"classify={(time.perf_counter() - t_ocr) * 1000:.0f} ({state})")
        # SPEED_PLAN 7: an unknown frame right after our roll / trade OK / End turn is the dice or a
        # closing panel -- no seat read; the next readable frame shows the seat
        transition = state == "unknown" and self.cfg.get("fast_transitions", True) and self._key_click_recent(3.0)
        if state == "other_turn" and self._our_turn_open():
            self._turn_closed_at = time.time()       # the timer ended it (or a read missed our click)
            self._turn_done("other_turn: ended without our End turn")
        seat = self._marker_seat(img) if state == "unknown" and not transition else None
        self._stall_progress(f"other_turn:{seat}" if seat else state)   # a new player on turn is progress
        if self._relaunch_rejoined(img, texts, state):
            return "rejoined"
        if self._monopoly_title(ocr) is not None:
            pick = self._fleet_choice() if "fleet" in self._monopoly_title(ocr)["text"].lower() \
                else self._monopoly_choice(ocr)
            self.note(f"pick dialog up: naming {pick}")
            self._monopoly_pick(pick, ocr)
            return "monopoly"
        # live 2026-10-08 19:06:59: one frame right after "send offer" read as lobby mid-game and
        # wiped the game (half-board rescan, a lost turn, the intro again). In play, a lobby
        # reading must hold for LOBBY_CONFIRM frames before it counts.
        # review 0.3.1: "home" wiped the game on a single frame too -- same confirm streak
        if state in ("lobby", "home", "offline_menu") and getattr(self, "_in_play", False):
            self._lobby_streak = getattr(self, "_lobby_streak", 0) + 1
            if self._lobby_streak < self.LOBBY_CONFIRM:
                self.note(f"{state} read mid-game ({self._lobby_streak}/{self.LOBBY_CONFIRM}): waiting to confirm")
                return f"{state}_unconfirmed"
        else:
            self._lobby_streak = 0
        if state in ("lobby", "home", "offline_menu", "game_over"):
            self._lobby_streak = 0
        if state in ("lobby", "home", "offline_menu"):
            self._reset_game_memory(state)
        blob = " ".join(texts).lower()
        # solo 2026-10-03: the rules panel ("...upon or after the first barbarian attack") set the
        # flag at the start of game 2 -- rules text never counts as an attack
        if not re.search(r"scenario configuration|additional rules", blob) and \
                re.search(r"loses a city|(?<!first )barbarians? (attack|defeated|repelled|won|lost)|defender of catan|\bdefeat!|\bvictory!", blob):
            # stage 2026-10-09: knights go inactive only on a confirmed attack -- the text counts
            # while the ship was last seen near the coast (or the track was never read)
            if self._barb_tracker().near_attack():
                self._barb_attack("attack message")
            elif time.time() - self.__dict__.get("_barb_text_note", 0) > 30:
                self._barb_text_note = time.time()
                self.note(f"attack text with the ship at {self._barb_tracker().pos}/7: not an attack, knights stay")
        self._watch_robbed(img, texts)
        if getattr(self, "_in_play", False) and state not in ("lobby", "home", "offline_menu", "game_over"):
            self._watch_dice(img)
        if state in ("unknown", "my_turn", "other_turn", "trade_offer_in"):
            phase = turn_phase(img)
            if phase != "idle" and state != "trade_offer_in":
                state = phase
            elif phase == "main" and state == "trade_offer_in" and not find_text(ocr, r"offer from"):
                state = "trade_panel" if find_text(ocr, r"player trade") else state
        if state == "unknown":
            state = self._other_turn_by_marker(img, info, seat, state)
        if find_text(ocr, r"offer from"):
            self._offer_seen_t = time.time()
        elif time.time() - getattr(self, "_offer_seen_t", 0) > 4:
            self._answered = None                     # offer panel gone: the next offer is new
        if self.cfg.get("solo") and (state == "offline_menu" or re.search(r"offline games|scenario selection", " ".join(texts), re.I)):
            state = "home"                            # solo menus: start_solo continues from here
        if self._wildcard_panel(img, ocr):
            return state
        if re.search(r"maritime trade|trade to the supply", " ".join(texts), re.I) and \
                time.time() - getattr(self, "_bank_clicked_at", 0) > 8:   # our own trade still closing
            # solo 2026-10-03: a restart mid bank trade left the dialog half filled and every
            # board scan failed under it -- bank trades are clicked in one go, so one still open
            # here is a leftover: cancel it (red button, same spot as the trade panel's)
            self.click(UI["trade_cancel"], "close leftover bank trade")
            return state
        big = self._enlarged_card(ocr)
        if big is None:
            self._enl_n = 0
        elif time.time() - getattr(self, "_progress_click_at", 0) > 6:
            # an enlarged progress card outside our own play flow covers the board. 10:19 it was a
            # hover preview (pointer on the small card); 11:12 a selected card that stayed with the
            # pointer parked. Rotate: park the pointer, click the big card, click empty table.
            key, box = big
            n = self.__dict__.get("_enl_n", 0)
            self._enl_n = n + 1
            way = ("park pointer", "click big card", "click table")[n % 3]
            if n % 3 == 0 and n // 3 < 3:
                self._probe("enlarged_card", img, texts=texts, note={"try": n, "next": way})
            self.probe.tried(f"close_card_{way.replace(' ', '_')}")
            if way == "park pointer":
                self._park_pointer(f"enlarged card {key}")
            elif way == "click big card":
                self.click((box["x"] + box["w"] / 2, box["y"] + box["h"] / 2 + 80), f"close enlarged card {key} (big card)")
            else:
                self.click(UI.get("pointer_park", self.PARK_SPOT), f"close enlarged card {key} (table)")
            self._enl_last = way
            return state
        if getattr(self, "_enl_last", None) and big is None:
            self.probe.result(f"close_card_{self._enl_last.replace(' ', '_')}", True)
            self.note(f"enlarged card closed by: {self._enl_last}")
            self._enl_last = None
        if re.search(r"create an offer for all other players", " ".join(texts), re.I) and \
                time.time() - max(getattr(self, "_offer_sent_at", 0), getattr(self, "_bank_clicked_at", 0)) > 8:
            # offers are built and sent in one go: an empty offer panel on a fresh step is a leftover
            self.click(UI["trade_cancel"], "close leftover offer panel")
            return state
        if re.search(r"withdraw your offer", " ".join(texts), re.I) and not getattr(self, "_pending_offer", None) \
                and time.time() - getattr(self, "_offer_sent_at", 0) > 30:
            # probe game 3 2026-10-04: our offer panel outlived the runner (restart): nobody tracked
            # it, the board scan failed under it for 15 min. An offer we don't track is withdrawn.
            self.click(UI["trade_cancel"], "withdraw an untracked offer")
            return state
        if re.search(r"choose what the dice", " ".join(texts), re.I):
            # game 13 17:28: the Alchemist dice dialog sat as screen=unknown
            dice = getattr(self, "_last_alchemist", None)
            if dice is None:
                try:
                    gs = self.ensure_state(img, texts) if self.mode is not None else None
                    from . import progress_cards as pc
                    dice = (pc.choose_alchemist(gs, gs.me()) or {}).get("dice") if gs is not None else None
                except Exception:                     # noqa: BLE001
                    dice = None
            self._alchemist_pick(dice or (6, 2))
            return state
        if re.search(r"commodit(y|ies) you want to give", " ".join(texts), re.I):
            self._commercial_harbor()
            return state
        # live 2026-10-03: a misread prompt kept clicking for 45 min, so the no-click watchdog
        # never fired -- one prompt screen (solo) for 15 min is a stall too
        now = time.time()
        if state != getattr(self, "_screen_state", None):
            self._screen_state, self._screen_since = state, now
        # owner 2026-10-03: live games too -- a prompt screen never needs 15 min of us
        elif (state in STALL_SCREENS and not isinstance(self.io, ReplayIO)
              and now - self._screen_since > self.cfg.get("same_screen_restart_s", 900)):
            self._probe(f"stall_{state}", img)
            self.probe.result(state, False, f"stall restart after 15 min on {state}")
            self.restart_app(f"screen {state} for {int((now - self._screen_since) // 60)} min")
            self._screen_state = None
            return "restart"
        handler = getattr(self, f"on_{state}", None)
        self._track_pending(state)
        self.probe.maybe_dump()
        self.note(f"screen={state}")
        if state in ("unknown", "my_turn") and time.time() - getattr(self, "_dbg_t", 0) > 10 and not transition:
            self._dbg_t = time.time()                 # what an unread screen looks like, for calibration
            stamp, frame, lines = time.strftime('%H%M%S'), img, "\n".join(texts)

            def _unread(stamp=stamp, frame=frame, lines=lines, st=state):
                d = ROOT / "data" / "unread"; d.mkdir(exist_ok=True)
                cv2.imwrite(str(d / f"{stamp}_{st}.png"), frame)
                (d / f"{stamp}_{st}.txt").write_text(lines)
            try:
                if self.cfg.get("probe_async", True) and hasattr(self.probe, "call_async"):
                    self.probe.call_async(lambda f=img.copy(), u=_unread: u(frame=f))   # SPEED_PLAN 1: off the turn
                else:
                    _unread()
            except Exception:                         # noqa: BLE001
                pass
            # SPEED_PLAN 1: on our own turn (or 5 s after our roll) an unknown frame is the dice
            # or a panel closing -- 21 of 64 live screen_unknown probes; not saved any more
            dice = self._our_turn_open() or (self.__dict__.get("_key_click") or ("", 0))[0].startswith("roll") \
                and self._key_click_recent(5.0)
            if getattr(self, "_in_play", False) and state == "unknown" and \
                    (self.cfg.get("probe_unknown_on_turn", False) or not dice) and \
                    time.time() - getattr(self, "_probe_unknown_t", 0) > 30:
                self._probe_unknown_t = time.time()
                self._probe("screen_unknown", img, texts=texts)
        if getattr(self, "_intro_due", False) and getattr(self, "_in_play", False) \
                and state in ("other_turn", "unknown") and not isinstance(self.io, ReplayIO) and not self._rings(img):
            # owner 2026-10-05: the intro comes once all buildings are set and play runs, on a
            # turn that is not ours (game 14 03:20: three lines ate 40 s of our first setup turn)
            self._intro_due = False
            self.chat_intro()
        if state in ("other_turn", "unknown") and not isinstance(self.io, ReplayIO):
            try:
                self.setup_watch(img, texts)
            except Exception as e:                    # noqa: BLE001 -- watching must never stop play
                self.note(f"setup watch failed: {type(e).__name__}: {e}")
        clicked_at = getattr(self, "_last_click_at", 0)
        if state in ("unknown", "main", "other_turn", "roll", "my_turn", "promote_knight", "place_knight",
                     "place_robber", "place_ship", "place_pirate") and self.mode is not None \
                and not isinstance(self.io, ReplayIO) and self._recenter_board(img):
            return state
        if state == "unknown" and self.mode is not None and not getattr(self, "_in_play", False) \
                and not isinstance(self.io, ReplayIO) and len(self._rings(img)) >= 5:
            if self._setup_textless(img, texts):
                return state
        if state in ("unknown", "promote_knight") and self.mode is not None and not isinstance(self.io, ReplayIO) \
                and time.time() - clicked_at > 45 and self._green_check(img):
            self.probe.tried("waiting_check")
            self._probe("waiting_check", img)
            self.click(self._check_at(img), "press the waiting green check")
            return state
        if state == "unknown" and (getattr(self, "_in_play", False) or self.mode is not None) and not isinstance(self.io, ReplayIO) \
                and time.time() - clicked_at > self._ring_wait_s(texts) \
                and (self._rings(img) or (self._edge_rings_on() and self._edge_rings(img)) or self._our_knights_ringed(img)):
            # final probe game 2026-10-04 10:03: Diplomat rings the open roads with no prompt text;
            # the bot tapped one and sat. Rings on an unknown screen with no click for 45 s = a
            # forced choice: click the ringed targets in turn (confirm if it stays)
            if self._rings_only_on_our_knights(img, texts):
                # solo 2026-10-06 03:51: a Deserter on us shows no text, only rings on our knights;
                # the generic fallback gave up the first ring (a level-2 knight). Choose by value
                self.note("ringed prompt: only our knights ringed -> deserter, give up the cheapest")
                self.on_lose_knight(img, self.io.ocr(img), texts, {})
                return state
            seat = self._marker_seat(img) if self.cfg.get("ring_fallback_our_turn", True) else None
            if seat:
                # game 0959: 4 fallback clicks on a rival's turn (their prompt, not ours)
                if self.__dict__.get("_rival_ring_seat") != seat:
                    self._rival_ring_seat = seat
                    name = getattr(self, "_slot_names", {}).get(seat, f"P{seat + 1}")
                    self.note(f"rings on {name}'s turn: not ours")
            else:
                self._rival_ring_seat = None
                if time.time() - getattr(self, "_diplomat_played_t", 0) < 600 and self._edge_rings(img):
                    # road rings now reach this gate (r4): after our Diplomat pick by the brain's
                    # order (rivals' roads first), not the first ring found
                    self.note("textless prompt after a Diplomat: road rings shown -> pick one")
                    self._diplomat_played_t = 0.0
                    self._diplomat_pick(img=img)
                    return state
                self._forced_fallback(img, "ringed_prompt")
                return state
        if state != "unknown":
            self.__dict__.get("_prompt_since", {}).pop("unknown", None)   # a screen with text: that prompt left
        elif not isinstance(self.io, ReplayIO) and self._cap_s() and self._prompt_cap_unknown() \
                and self._looks_like_our_prompt(img):
            # game 1548: a textless prompt of ours gets the same 90 s as any other, then the exit ladder
            self._prompt_clock("unknown")
            if self._prompt_over_cap("unknown"):
                self._prompt_exit(img, "unknown")
                if getattr(self, "_last_click_at", 0) != clicked_at:
                    return state
        if state == "unknown" and getattr(self, "_in_play", False) and not isinstance(self.io, ReplayIO) \
                and time.time() - clicked_at > 45 and time.time() - getattr(self, "_diplomat_played_t", 0) < 600 \
                and getattr(self, "br", None) is not None and self._edge_rings(img):
            # stage 2026-10-08 22:33: the Diplomat's small road rings are split by the road, so
            # _rings saw none and the bot read 'unknown' for minutes: pick a ringed road
            self.note("textless prompt after a Diplomat: road rings shown -> pick one")
            self._diplomat_played_t = 0.0
            self._diplomat_pick(img=img)
            return state
        if handler and state in self.RING_FALLBACK and self._prompt_over_cap(state):
            self._prompt_exit(img, state)              # game 0959: no more ring taps past the cap
        elif handler:
            handler(img, ocr, texts, info)
        if state == "game_over":
            # review 0.3.1: the reset used to run before on_game_over, so result.json landed in a
            # new folder with vp None and the log ended with no VP -- record first, then forget
            self._reset_game_memory(state)
        self._after_handler(state, getattr(self, "_last_click_at", 0) != clicked_at, img)
        if state in ("intro", "invite"):
            self._saw_game_start = True
        self.chat_poll(img, ocr, state)
        self.probe_game_log(img, state)
        if state == "other_turn" and getattr(self, "_last_click_at", 0) == clicked_at:
            self._prewarm_turn()                      # 0.4.3 step 4: the quiet wait plans our turn
        return state

    # ---- bot 0.4.3 step 4: plan on rivals' turns (config preplan) ----
    PREPLAN_TOTALS = (6, 8, 5, 9, 4, 10, 3, 11, 2, 12)   # most likely first; a 7 is never pre-planned

    @staticmethod
    def _board_key(gs) -> tuple | None:
        """Everything the plan reads off the board and our pieces (not our hand): a rival build,
        a robber move or a knight change between the pre-plan and our roll = a key miss."""
        try:
            ps = []
            for p in gs.players:
                ps.append((p.name, tuple(sorted(map(repr, p.settlements))), tuple(sorted(map(repr, p.cities))),
                           tuple(sorted(map(repr, p.metropolises))), tuple(sorted(map(repr, p.walls))),
                           tuple(sorted(map(repr, p.roads))), tuple(sorted(map(repr, p.ships))),
                           tuple(sorted((repr(v), k.level, k.active, k.acted) for v, k in p.knights.items())),
                           tuple(sorted(p.improvements.items())), tuple(sorted(map(repr, p.ships_new))),
                           p.ship_moved, p.vp_visible, tuple(sorted(p.progress_hand))))
            return (repr(gs.board.robber), getattr(gs, "canal_progress", None),
                    tuple(sorted(map(repr, getattr(gs, "blocked", set()) or ()))), tuple(ps))
        except Exception:                            # noqa: BLE001 -- no key = no pre-plan
            return None

    def _prewarm_turn(self) -> None:
        """Owner 2026-10-09 "turn can be planned before the dice is rolled but has to be redone once
        the dice is rolled": on a rival's turn, rank the threats once, then plan our turn for one
        more dice total per loop step (hand + that total's income). on_main takes the plan for
        the rolled total only when the board key and our hand after the roll match it."""
        if not self.cfg.get("preplan", True):
            return
        gs = self.__dict__.get("gs")
        if gs is None or not getattr(self, "_in_play", False) or self._our_turn_open() \
                or self.__dict__.get("brain") is None:
            return
        gs.blocked = set(self.__dict__.get("_not_offered", set()))     # a fresh turn's blocked set
        key = self._board_key(gs)
        if key is None:
            return
        hand = tuple(sorted((k, v) for k, v in gs.me().hand.items() if v))
        pre = self.__dict__.get("_preplan")
        if pre is None or pre["key"] != key or pre["hand"] != hand:
            pre = self._preplan = {"key": key, "hand": hand, "plans": {}, "threat": False}
        if not pre["threat"]:
            pre["threat"] = True
            self._rank_threat_before_turn()
            self._threat_key = (key, hand)
            return
        todo = [t for t in self.PREPLAN_TOTALS if t not in pre["plans"]]
        if not todo:
            return
        total = todo[0]
        brain = self.brain
        quiet, brain.note = brain.note, (lambda *a, **k: None)
        try:
            g = brain.clone(gs)
            for name, cards in g.roll_yield(total).items():
                pl = next((p for p in g.players if p.name == name), None)
                if pl is not None:
                    for r, n in cards.items():
                        pl.hand[r] = pl.hand.get(r, 0) + n
            want = tuple(sorted((k, v) for k, v in g.me().hand.items() if v))
            pre["plans"][total] = (want, brain.plan_turn(g))
        except Exception as e:                       # noqa: BLE001 -- pre-planning never stops play
            pre["plans"][total] = (None, None)
            quiet(f"preplan {total} failed: {type(e).__name__}: {e}")
        finally:
            brain.note = quiet
        if len(pre["plans"]) == len(self.PREPLAN_TOTALS):
            self.note(f"preplan: {len(pre['plans'])} totals ready")

    def _preplan_for(self, gs):
        """The pre-plan for the rolled total when board, blocked set and hand all match, else None
        (first plan of the turn only; a plan with a player offer is planned fresh -- rival hands
        moved with the roll)."""
        if not self.cfg.get("preplan", True):
            return None
        pre, ts = self.__dict__.get("_preplan"), self.__dict__.get("turn_started")
        if pre is None or self.__dict__.get("_preplan_turn") == ts:
            return None
        self._preplan_turn = ts
        d = self.__dict__.get("_last_dice")
        if not d:
            return None
        entry = pre["plans"].get(sum(d))
        hand = tuple(sorted((k, v) for k, v in gs.me().hand.items() if v))
        if entry is None or entry[1] is None or entry[0] != hand or self._board_key(gs) != pre["key"]:
            self.note(f"preplan: no match for {sum(d)} (planned {sorted(pre['plans'])}): planning now")
            return None
        if any(a.kind == "player_trade" for a in entry[1]):
            return None
        self.note(f"preplan: total {sum(d)} matches: plan from the rival's turn")
        self._tcount("preplan hits")
        return list(entry[1])

    _PER_GAME_LAZY = (
        "_vp", "_vp_at", "_last_hand", "_progress_once", "_progress_tried", "_pick_tried",
        "_ring_pick_tried", "_wall_tapped", "_metro_tapped", "_remove_city_tried",
        "_remove_city_pending", "_fb_tried", "_spy_victim", "_player_tried", "_want_player",
        "_gain_round", "_answered", "_answered_at", "_answer_retries", "_pending_offer", "_offer_pending",
        "_offers_made", "_want_declined", "_prompt_since", "_prompt_exit_at", "_token_pairs", "_rival_ring_seat",
        "_progress_blocked", "_edge_cache", "_stall_extra_for",
        "_failed_taps", "_unaffordable", "_tapped_spots", "_build_rounds",
        "_build_turn", "_plan_hand", "_sold_turn", "_misclick_retry", "_promote_tried",
        "_robber_tried", "_pdiscard_tried", "_discarded", "_discard_prev", "_crane_track",
        "_harbor", "_harbor_give", "_harbor_try", "_take_round", "_sheds", "_last_alchemist",
        "_lose_knight_saved", "_ledger_steal_at", "_barb_attack_at", "_wall_pending",
        "_offer_booked", "_gain_at", "_progress_confirmed", "_progress_fail_shot", "_declined", "_decline_n", "_turn_n", "_pdiscard_retry", "_moved_knights",
        "_ships_this_turn", "_ship_moved_turn", "_ship_move_misses", "_bank_combined_off")

    def _reset_game_memory(self, state):
        """Everything one game knows, forgotten when we leave it (lobby, home, or after the game
        over screen was recorded). One runner plays many games: every per-game field resets here."""
        gl = self.__dict__.get("_glog_obj")
        if state != "game_over" and gl is not None and gl.active() and not isinstance(self.io, ReplayIO):
            from .chat import record_left       # owner 2026-10-09: a game left without its end screen
            # stage 2026-10-09 game 1548: the tally did not show the 14/14/13/7 we walked away from
            vp = ", ".join(f"{n} {v}" for n, v in self._glog_vp().items()) or "unread"
            if record_left(gl.game_id, f"left the game ({state}); vp {vp}"):
                self.note(f"record: left game {gl.game_id} ({state}); vp {vp}; failed unless we rejoin the same match")
        if self.probe.card and state != "game_over":
            p = self.probe.dump(final=True, extra={"ended": f"left the game ({state})"})
            self.note(f"scorecard (left game): {self.probe.summary()} -> {p}")
            self.probe.reset()
        self._pend = None
        self._progress_refused, self._improve_seen, self._bank_blocked = set(), False, set()
        self._not_offered, self._promote_disabled, self._promote_skip_turn = set(), False, None
        self.brain.ratio_seen = {}
        self.brain.merchant_res = None             # a new game: no Merchant yet
        self.game_colors = None                   # new game -> read colours again
        self._slot_names = {}
        self._loops, self._loop_blocked, self._click_hist = 0, {}, {}
        self._lobby_alerted = False
        self._game_dir, self._colors_pending = None, None
        self._glog_new = True                     # the next game opens a new turn log
        # audit 2026-10-03: setup memory leaked from one game into the next
        self._last_setup_v = self._last_setup_kind = self._road_at = None
        self._setup_bad, self._road_tried, self._ship_tried = set(), set(), set()
        self._setup_picks, self._ring_miss, self._watch_n = [], 0, -1
        self._barb_pos, self._barb_logged = 0, None    # barbarian track: a new game starts at 0
        self._barb_track, self._activated_at = None, {}
        self._login_restarts = 0                  # signed in: a later sign-in prompt gets one relaunch again
        self.scan, self.lattice = None, None      # and scan the new board
        self._intro_sent, self._saw_game_start, self._chat_primed = False, True, False
        self._intro_due = False
        self._in_play = False
        self.barbarians_attacked = False
        self._lost_cities = 0
        self._my_improve = {t: 0 for t in IMPROVEMENT_TRACKS}
        self._mode_saved = None
        if state != "game_over":
            self._reload_model_if_promoted()
            if not isinstance(self.io, ReplayIO):
                from . import live_learn
                live_learn.reset()               # a log left by a game whose end we never saw
        self._my_blds, self._my_cities, self._trade_expect = set(), set(), None
        self._my_walls, self._not_city, self._my_metros = set(), set(), set()
        self._taken_spots, self._my_knights, self._knight_levels = {}, {}, {}
        self._offered_spots = {}
        getattr(getattr(self, "brain", None), "__dict__", {}).pop("_vp_hist", None)   # threat rise: per game
        self._my_roads, self._my_ships, self._sea_edges, self._last_pieces = set(), set(), set(), None
        # new game, maybe new rules: read them again in the game
        self.mode, self._rules_texts, self._rules_unconf = None, None, 99
        self._asked_info, self._info_open = False, True
        self._reset_harbors(f"left the game ({state})")
        self._canal_ref, self._canal_seen = None, set()
        self._move_misses = 0
        self.__dict__.pop("_ship_move_misses", None)
        if getattr(self, "brain", None) is not None:     # a new game may move ships again
            self.brain.ship_moves = bool((getattr(self, "cfg", None) or {}).get("ship_move", True))
        self._cards_lost = 0
        self._seen_gain, self._banner_counts, self._last_dice = {}, {}, None
        self._hands, self._banner_read_at = None, {}
        self._feed_dice, self._feed_events = [], []
        # review 0.3.1: per-game fields set lazily (read through getattr/setdefault) leaked into
        # the next game -- _progress_once kept a card "played once" for the whole session, _vp and
        # _last_hand fed the old game's scores and hand to the new one. Dropping them restores
        # exactly the state the first game of the session starts from.
        for name in self._PER_GAME_LAZY:
            self.__dict__.pop(name, None)
        chat = getattr(self, "chat", None)
        if chat is not None:
            chat.seen = set()                     # the new game's chat is primed again (_chat_primed)
        self.gs = None                            # last game's board must not score new rolls
        self._save_game_flags()

    def _side_icons(self, img) -> list:
        """The orange icon strip on the left edge, top to bottom (chat, emotes, log?, stats, info,
        rules, settings, exit), found by colour: the strip moves between layouts."""
        s = img[:, :70]
        m = cv2.inRange(cv2.cvtColor(s, cv2.COLOR_BGR2HSV), (10, 150, 180), (30, 255, 255))
        n, _, st, cen = cv2.connectedComponentsWithStats(m)
        icons = [(int(c[0]), int(c[1])) for c, a in zip(cen[1:], st[1:]) if 250 < a[4] < 1200]
        return sorted(icons, key=lambda xy: xy[1])

    def probe_game_log(self, img, state):
        """Owner 2026-10-06: "there's a log when the trade happens, bot should know immediately once
        any action is performed". The 3rd icon on the left strip looks like the game's event log:
        once per game, off our turn, open it, save frame + text to data/probe, close it again.
        Calibration only -- nothing reads it yet."""
        if isinstance(self.io, ReplayIO) or not self.cfg.get("probe_game_log", True) or state != "other_turn" \
                or getattr(self, "_log_probed", None) == getattr(self, "game_colors", None) or img is None:
            return
        icons = self._side_icons(img)
        if len(icons) < 6:
            return
        self._log_probed = getattr(self, "game_colors", None)
        xy = icons[2]
        self.click(xy, "probe: open the game log icon")
        time.sleep(1.0)
        im2 = self.io.snapshot()
        if im2 is not None:
            self._probe("game_log_panel", im2, texts=[o["text"] for o in self.io.ocr(im2)])
        self.click(xy, "probe: close the game log icon")
        time.sleep(0.6)
        im3 = self.io.snapshot()
        if im3 is not None and im2 is not None and float(np.mean(cv2.absdiff(im3, img))) > 12:
            self._probe("game_log_still_open", im3)     # left for the next look to report

    # -- handlers
    STEAM_APP = "544730"                             # Catan Universe on Steam
    APP_ID = "unity.UnitedSoftMediaVerlagGmbH.CatanUniverse"

    def on_login(self, img, ocr, texts, info):
        # never type credentials: the app's saved login signs in by itself; just wait
        now = time.time()
        if now - getattr(self, "_login_last", 0) > 10:
            self._login_since = now                   # a fresh visit to the login screen
        self._login_last = now
        if now - getattr(self, "_login_seen", 0) > 60:
            self._login_seen = now
            self.note("login screen: waiting for the app's saved sign-in (not typing anything)")
        if now - self._login_since > 120 and now - getattr(self, "_restarted_at", 0) > 300 \
                and not isinstance(self.io, ReplayIO):
            # owner 2026-10-03: never type a password. One relaunch (the saved login normally
            # signs in by itself); still on the sign-in screen after that -> stop and tell the owner
            if getattr(self, "_login_restarts", 0) >= 1:
                self.alert("Catan Universe wants a sign-in; please log in on the server, then delete data/STOP")
                return
            self._login_restarts = getattr(self, "_login_restarts", 0) + 1
            self.restart_app("stuck on the login screen")

    @staticmethod
    def _trim_black(img, level: int = 12):
        """Stage 2026-10-08 23:16: the bot frame is the fixed 1372x888 layout with the game drawn
        at 1308x790 from y=54, so the shared ranking shot had black bars top, right and bottom.
        Crop to the rows / columns that hold any non-black pixel."""
        if img is None or img.ndim < 2 or img.size == 0:
            return img
        g = img.max(axis=2) if img.ndim == 3 else img
        rows = np.where(g.max(axis=1) > level)[0]
        cols = np.where(g.max(axis=0) > level)[0]
        if rows.size == 0 or cols.size == 0:
            return img
        return img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]

    def save_endgame_shot(self, img, kind: str) -> Path:
        """Owner 2026-10-04: keep a screenshot of every end-of-game screen (result banner, ranking,
        statistics) in data/endgame/ for the group."""
        d = ROOT / "data" / "endgame"
        d.mkdir(exist_ok=True)
        p = d / f"{time.strftime('%Y%m%d_%H%M%S')}_{kind}.png"
        from catanbot import group_share
        if group_share.mostly_black(img):
            # stage 2026-10-10 15:10: trimming a black frame left just the macOS toast, which was shared
            return p
        try:
            cv2.imwrite(str(p), self._trim_black(img))
        except Exception:                    # noqa: BLE001
            pass
        return p

    def on_game_result(self, img, ocr, texts, info):
        """'You've won!' / 'You've lost' banner with VIEW PROGRESS and CONTINUE (game 13 18:10):
        screenshot it, then CONTINUE on to the ranking / statistics screen (game_over)."""
        if isinstance(self.io, ReplayIO):
            return
        if time.time() - getattr(self, "_result_shot", 0) > 60:
            self._result_shot = time.time()
            self.note(f"game result screen: {' | '.join(texts[:6])} -> {self.save_endgame_shot(img, 'result')}")
        btn = find_text(ocr, r"^continue$")
        self.click((btn["x"] + btn["w"] / 2, btn["y"] + btn["h"] / 2) if btn else (1197, 749), "game result: CONTINUE", ocr)

    @classmethod
    def end_screen_kind(cls, img, texts) -> str | None:
        """'result' for the "You've won!/lost" banner with VIEW PROGRESS and CONTINUE, 'ranking' for
        the ranking screen (rank rows + QUIT), else None. Stage 2026-10-10 15:10: after a kill and
        relaunch, a black frame with the macOS "Game Mode: On ... Game Overlay" toast read as
        game_over -- recorded, learned and shared to the group as winner=None ranking=[]."""
        from catanbot import group_share
        if img is not None and group_share.mostly_black(img):
            return None
        low = [t.strip().lower() for t in texts]
        joined = " | ".join(low)
        if re.search(r"you.?ve (won|lost)|you (won|lost)", joined) and "view progress" in joined \
                and "continue" in low:
            return "result"
        if "quit" in low and cls.read_ranking(texts):
            return "ranking"
        return None

    def on_game_over(self, img, ocr, texts, info):
        """Owner 2026-10-02: after a game ends, restart the app via Steam -- otherwise it keeps
        saying we're still in the game. Once per result screen."""
        if isinstance(self.io, ReplayIO) or time.time() - getattr(self, "_restarted_at", 0) < 120:
            return
        if self.end_screen_kind(img, texts) is None:
            if time.time() - getattr(self, "_fake_end_noted", 0) > 30:
                self._fake_end_noted = time.time()
                self.note(f"game over: not a real end screen, ignored ({' | '.join(texts[:4]) or 'no text'})")
            return
        cv2.imwrite(str(ROOT / "data" / "popups" / time.strftime("%H%M%S_game_over.png")), img)
        self.save_endgame_shot(img, "game_over")
        if self._game_over_seen(texts):
            # solo 2026-10-05 23:27: a runner restarted on the end screen counted the game again,
            # re-learned it and re-posted the shots to the group; record once, then just move on
            self.note("game over already handled before a runner restart: no record/share again")
            self.probe.reset()
        else:
            self._handle_game_over(img, texts)
        if self.cfg.get("solo") and not isinstance(self.io, ReplayIO):
            # owner 2026-10-06 "now just started a new game by itself": a solo game is a one-off,
            # the next one is started on purpose; after it, back to invite mode
            self.cfg["solo"] = False
            try:
                saved = config.load()
                saved["solo"] = False
                config.save(saved)
            except Exception as e:                # noqa: BLE001
                self.note(f"solo off: config not saved ({e})")
            self.note("game over: solo turned OFF (no new solo game; waiting for an invite)")
        stop = self.STOP_AFTER_GAME
        if stop.exists():
            # owner 2026-10-03: live games only to test clicks and strategy; finish this one,
            # then hand the machine to simulations (STOP ends the run loop).
            # Owner 2026-10-06 "the solo game never exited": leave the end screen to the front
            # page first; not out yet -> keep the flag, the next look on the end screen retries
            if not self.leave_game(tries=8):
                self.note("game over: end screen not left yet; trying again")
                return
            stop.unlink()
            STOP.write_text("stop after game (stop_after_game.flag)\n")
            self.note("game over: left to the front page; stop_after_game.flag set, idle (STOP written)")
            return
        self.restart_app("game over")

    STOP_AFTER_GAME = ROOT / "data" / "stop_after_game.flag"
    GAME_OVER_MARK = ROOT / "data" / "last_game_over.json"

    def _game_over_seen(self, texts, window_s: float = 12 * 3600) -> bool:
        """The same result (same ranking) was already handled, by this or an earlier runner
        process, and no game was played since (our first roll clears the mark). Owner 2026-10-06:
        with stop_after_game.flag the end screen stays up, and a restart past the old 30 min
        window recorded, learned and posted the game again. window_s is only a backstop."""
        ranking = self.read_ranking(texts)
        try:
            m = json.loads(self.GAME_OVER_MARK.read_text())
        except Exception:                          # noqa: BLE001
            m = {}
        return bool(ranking) and m.get("ranking") == ranking and time.time() - m.get("at", 0) < window_s

    def _handle_game_over(self, img, texts):
        self._game_over_cleared = False               # the next game's first roll clears the mark again
        self._record_game_over(texts)
        try:
            self.GAME_OVER_MARK.write_text(json.dumps({"ranking": self.read_ranking(texts), "at": time.time()}))
        except Exception:                          # noqa: BLE001
            pass
        # owner 2026-10-04: end-of-game screenshots go to the Catan iMessage group
        try:
            from catanbot import group_share
            res = getattr(self, "_last_result", None) or {}
            shots = group_share.shots_since(ROOT / "data" / "endgame", getattr(self, "_result_shot", 0) or time.time() - 600)
            # owner 2026-10-06 "bot sent the message twice, again": the group shows each picture as
            # its own message; send one -- the ranking screen (game_over), else the newest shot
            shots = [p for p in shots if p.stem.endswith("game_over")][-1:] or shots[-1:]
            # owner 2026-10-06 duplicates: two posts of one game 6-8 s apart; a per-game key lets the
            # server drop the second one, whichever runner sent it
            ranking = res.get("ranking") or []
            vps = [next((p.vp_visible for p in getattr(self.gs, "players", []) if p.name == n), None) for n in ranking]
            group_share.send(group_share.caption_for(res.get("winner"), ranking),
                             shots, url=self.cfg.get("group_send_url", group_share.DEFAULT_URL), note=self.note, wait=True,
                             idem_key=group_share.game_key(ranking, vps) if ranking else None,
                             winner=res.get("winner"), ranking=ranking)
        except Exception as e:                    # noqa: BLE001
            self.note(f"group share skipped: {e}")
        # owner 2026-10-04: no pile of Terminal windows; close idle bot launchers after every game
        from catanbot.terminals import close_idle_bot_terminals
        n = close_idle_bot_terminals()
        if n:
            self.note(f"closed {n} idle bot Terminal window(s)")
        if self.probe.card:
            p = self.probe.dump(final=True, extra={"ended": "game over", "result_texts": texts[:40]})
            self.note(f"scorecard: {self.probe.summary()} -> {p}")
            self.probe.reset()

    RESTART_FLAG = ROOT / "data" / "restart.flag"
    LEAVE_FLAG = ROOT / "data" / "leave.flag"
    LEAVE_RETRY_S = 30

    def leave_on_request(self) -> str:
        """Owner 2026-10-05: `touch data/leave.flag` = stop playing and walk out of the game with the
        game's own buttons (gear -> QUIT GAME -> check, then close the end screens) to the front
        page. Never kill or close the app: a killed game rejoins the same match on relaunch and
        sticks on its READY screen. Once home: flag removed, the bot waits there for the next invite
        (STOP still pauses it). Not home yet: keep the flag, try again next look."""
        if self.leave_game(tries=8):
            self.LEAVE_FLAG.unlink(missing_ok=True)
            self._in_play, self._pend = False, None
            self.note("leave: at the front page; waiting for an invite")
            return "left"
        self._probe("leave_not_home")
        # not out: play on normally for a bit (a popup over the board may be what blocks the
        # gear), then walk out again -- never the same presses back to back
        self._leave_retry_at = time.time() + self.LEAVE_RETRY_S
        self.note(f"leave: not at the front page yet; trying again in {self.LEAVE_RETRY_S} s")
        return "leaving"
    LEAVE_WORDS = re.compile(r"^(leave( game| match)?|exit( game| match)?|quit( game| match)?|"
                             r"back to (the )?(main )?menu|main menu|continue)$", re.I)
    CONFIRM_WORDS = re.compile(r"^(yes|ok|confirm|leave)$", re.I)
    # live 19:47 2026-10-08 (Great Canal, our turn): gear -> bottom-left panel "QUIT GAME / Do you
    # want to go back to the main menu?" + a green check under the question
    QUIT_ASKED = re.compile(r"really want to (quit|leave)|want to (quit|leave)|(quit|leave) this (game|match)|"
                            r"(go )?back to the main menu\?", re.I)
    LEAVE_MAX_SAME = 2           # the same label/spot clicked this often without the screen moving on: stop
    LEAVE_MAX_GEAR = 3
    LEAVE_SETTLE_S = 20          # after YES / the check the app loads the front page (6 s live): look, don't click

    def _leave_action(self, img, ocr, texts, last: str | None):
        """One leave step, decided only from what is on screen (live dialogs seen so far):
          - mid game, gear -> "Quit game / Do you really want to quit the game? / NO / YES" -> YES
          - older/end variant: "Do you really want to quit this game?" + a green check -> the check
          - end screens: QUIT (ranking screen, (653, 720)) / CONTINUE (You've won/lost) / LEAVE / EXIT
          - an OK / CONFIRM popup over the board -> press it
          - a QUIT GAME menu item still there right after we pressed it -> the confirm is the check
          - matchmaking words on screen (front page still loading) -> no click
          - nothing readable -> the bottom-left gear
        Returns (kind, xy, label) with kind in yes/check/button/forbidden/gear."""
        joined = " ".join(texts)
        if self.QUIT_ASKED.search(joined):
            yes = find_text(ocr, r"^yes$")
            if yes:
                return "yes", (yes["x"] + yes["w"] / 2, yes["y"] + yes["h"] / 2), "YES"
            q = find_text(ocr, self.QUIT_ASKED.pattern)
            return "check", self._dialog_check(img, q) or self._green_button(img) or UI["quit_check"], "green check"
        btn = find_text(ocr, self.CONFIRM_WORDS.pattern) or find_text(ocr, self.LEAVE_WORDS.pattern)
        if btn is not None and (last or "").startswith("button:quit") and re.match(r"quit", btn["text"], re.I):
            return "check", self._dialog_check(img, btn) or self._green_button(img) or UI["quit_check"], "green check"
        if btn is not None:
            return "button", (btn["x"] + btn["w"] / 2, btn["y"] + btn["h"] / 2), btn["text"]
        if FORBIDDEN_BUTTONS.search(joined):
            return "forbidden", None, ""
        return "gear", UI["quit_gear"], "quit gear"

    def leave_game(self, tries: int = 8) -> bool:
        """Walk out of the match with the game's own buttons until the front page (home / lobby /
        login) is read. Owner 2026-10-08: "if you don't quit right the game will try to bring you
        back in the old game and freeze" -- so every press is chosen by its OCR'd label
        (_leave_action); one label/spot is pressed at most LEAVE_MAX_SAME times in a row and the
        gear at most LEAVE_MAX_GEAR times; after YES / the check it only looks (up to
        LEAVE_SETTLE_S) while the front page loads. At most `tries` presses. Every step's frame +
        OCR goes to data/leave/; a failed walk ends with a 'fail' frame."""
        if isinstance(self.io, ReplayIO) or not hasattr(self.io, "mac"):
            return False
        last, same, gear, presses, settle_until = None, 0, 0, 0, 0.0
        img, texts, why = None, [], "out of presses"
        while True:
            img = self.io.snapshot()
            if img is None:
                why = "no frame"
                break
            ocr = self.io.ocr(img)
            texts = [o["text"] for o in ocr]
            state, _ = classify(texts, self.cfg["bot_name"].lower())
            if state in ("home", "lobby", "login", "offline_menu") or state == "unknown" and offline_menu(img):
                self.note(f"leave game: out ({state})")
                self._save_leave_frame(img, texts, f"out_{state}")
                return True
            kind, xy, label = self._leave_action(img, ocr, texts, last)
            self._save_leave_frame(img, texts, kind)
            if kind in ("gear", "forbidden") and time.time() < settle_until:
                time.sleep(1.5)                       # confirmed: the front page is loading
                continue
            if presses >= tries:
                break
            presses += 1                               # a look at a matchmaking screen counts too
            if kind == "forbidden":
                # 19:24:02 live: the half-drawn front page carries matchmaking words; never click it
                self.note("leave game: matchmaking words on screen, not clicking; looking again")
                time.sleep(2)
                continue
            key = f"{kind}:{label.lower()}"
            same = same + 1 if key == last else 1
            gear += kind == "gear"
            if same > self.LEAVE_MAX_SAME or gear > self.LEAVE_MAX_GEAR:
                n = self.LEAVE_MAX_SAME if same > self.LEAVE_MAX_SAME else self.LEAVE_MAX_GEAR
                why = f"'{label}' pressed {n}x without a way out"
                break
            self.click(xy, f"leave game: {label}", ocr if kind in ("yes", "button") else None)
            last = key
            if kind in ("yes", "check"):
                settle_until = time.time() + self.LEAVE_SETTLE_S
            time.sleep(2)
        self._save_leave_frame(img, texts, "fail")
        self.note(f"leave game: NOT out ({why}); screen: {' | '.join(texts[:8])}")
        return False

    @staticmethod
    def _dialog_check(img, box, dx: int = 220, dy: int = 220):
        """The green check of the dialog whose text is `box`: the biggest green blob just below it
        (live 19:47: the screen-wide search took the green Crane card at (871, 730) instead of
        the panel's check at (198, 701) -- three wasted presses)."""
        if img is None or not box:
            return None
        h, w = img.shape[:2]
        x0, x1 = max(0, int(box["x"] - dx)), min(w, int(box["x"] + box["w"] + dx))
        y0, y1 = max(0, int(box["y"])), min(h, int(box["y"] + box["h"] + dy))
        if x1 - x0 < 10 or y1 - y0 < 10:
            return None
        hsv = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (40, 120, 110), (85, 255, 255))
        n, _, stats, cent = cv2.connectedComponentsWithStats(mask)
        best = max(range(1, n), key=lambda i: stats[i, cv2.CC_STAT_AREA], default=None)
        if best is None or stats[best, cv2.CC_STAT_AREA] < 150:
            return None
        cx, cy = cent[best]
        return (x0 + float(cx), y0 + float(cy))

    @staticmethod
    def _green_button(img):
        """Centre of the biggest saturated green blob in the middle-lower screen (the confirm
        check of a dialog), or None."""
        if img is None:
            return None
        h, w = img.shape[:2]
        x0, y0 = int(w * 0.25), int(h * 0.35)
        hsv = cv2.cvtColor(img[y0:, x0:int(w * 0.75)], cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (40, 120, 110), (85, 255, 255))
        n, _, stats, cent = cv2.connectedComponentsWithStats(mask)
        best = max(range(1, n), key=lambda i: stats[i, cv2.CC_STAT_AREA], default=None)
        if best is None or stats[best, cv2.CC_STAT_AREA] < 300:
            return None
        cx, cy = cent[best]
        return (x0 + float(cx), y0 + float(cy))

    GAME_PROC = "steamapps/common/Cat"               # the game's path under Steam (pgrep -f)

    def restart_app(self, reason: str):
        """Quit Catan Universe, relaunch it through Steam; the app signs in with its own saved
        login (we never type credentials). Then on_invite/on_lobby take it from there."""
        import subprocess
        now = time.time()
        recent = [t for t in getattr(self, "_restart_times", []) if now - t < 1800] + [now]
        self._restart_times = recent
        if len(recent) > 3 and not isinstance(self.io, ReplayIO):
            self.alert(f"4 restarts in 30 min (last: {reason}); stopping")
            return
        self._restarted_at = now
        p = getattr(self, "_pend", None)
        if p and not p[4] and hasattr(self, "probe"):
            self.probe.result(p[0], False, f"restart ({reason}) while pending")
        self._pend = None
        # owner 2026-10-08: "if you don't quit right the game will try to bring you back in the old
        # game and freeze" -- leave the match by its own buttons first (two walks); the app is
        # only quit once the front page is read, or when leaving can't work (hung/black window)
        left = self.leave_game(tries=8) or self.leave_game(tries=8)
        self._relaunch_watch = (time.time(), left)
        if not left:
            self.note(f"{reason}: could NOT leave the match by its buttons; relaunching anyway (it may rejoin)")
        self.note(f"{reason}: quitting Catan Universe and relaunching it through Steam")
        try:
            subprocess.run(["osascript", "-e", f'tell application id "{self.APP_ID}" to quit'],
                           capture_output=True, timeout=30)
        except subprocess.TimeoutExpired:            # 21:42 2026-10-03: a hung game ignores quit
            self.note("quit timed out: the game looks hung")
        for _ in range(30):
            if subprocess.run(["pgrep", "-f", self.GAME_PROC], capture_output=True).returncode:
                break
            time.sleep(1)
        else:
            subprocess.run(["pkill", "-x", "Catan Universe"], check=False)   # quit was ignored
            time.sleep(3)
            subprocess.run(["pkill", "-9", "-f", self.GAME_PROC], check=False)
            time.sleep(2)
        self.mode, self.lattice, self.gs = None, None, None
        subprocess.run(["open", f"steam://rungameid/{self.STEAM_APP}"], check=False)
        time.sleep(30)                               # Unity start + login before the next snapshot

    RELAUNCH_WATCH_S = 300

    def _relaunch_rejoined(self, img, texts, state) -> bool:
        """After restart_app: the first screens show whether the app pulled us back into the old
        match (live 19:19 2026-10-08: a relaunch without a proper quit did). Front page first =
        clean. A game screen first after a match we had left = leave it again (leave.flag);
        after a match we could not leave = play on (it may be the owner's game)."""
        w = getattr(self, "_relaunch_watch", None)
        if not w or isinstance(self.io, ReplayIO):
            return False
        t, left = w
        if time.time() - t > self.RELAUNCH_WATCH_S:
            self._relaunch_watch = None
            self.note("relaunch: no front page or game read in 5 min (watch ended)")
            return False
        if state in ("unknown", "login"):
            return False                              # still starting / signing in
        self._relaunch_watch = None
        if state in ("home", "lobby", "invite", "offline_menu"):
            self.note(f"relaunch: {state} -- the old match was not rejoined")
            return False
        self._save_leave_frame(img, texts, "rejoined")
        if left:
            self.note(f"relaunch: REJOINED the old match ({state}) although it was left; leaving it again")
            self.LEAVE_FLAG.touch()
            return True
        self.note(f"relaunch: back in the old match ({state}); it was not left before the relaunch: playing on")
        return False

    def on_intro(self, img, ocr, texts, info):
        self.read_rules(texts)

    def on_lobby(self, img, ocr, texts, info):
        """Custom-match lobby (joined via a friend's invite): press READY while we're not ready."""
        if FORBIDDEN_BUTTONS.search(" ".join(texts)) or isinstance(self.io, ReplayIO):
            return
        self._save_lobby_frame(img, texts, "lobby")
        now = time.time()
        if now - getattr(self, "_lobby_last", 0) > 60:
            self._lobby_since = now
        self._lobby_last = now
        if now - self._lobby_since > 900 and not getattr(self, "_lobby_alerted", False):
            self._lobby_alerted = True
            self.alert("in the lobby 15 min and the game hasn't started", halt=False)
        if not re.search(r"not ready", " ".join(texts), re.I):
            return
        if time.time() - getattr(self, "_ready_at", 0) < 15:
            return
        btn = find_text(ocr, r"^ready$")
        if btn:
            self._ready_at = time.time()
            self.click((btn["x"] + btn["w"] / 2, btn["y"] + btn["h"] / 2), "lobby READY", ocr)

    def _save_leave_frame(self, img, texts, kind):
        """Live 2026-10-08 19:18: the leave loop clicked the green check 13x with the board on
        screen. Keep each leave step's frame + OCR (data/leave/, newest 40) to see why."""
        if isinstance(self.io, ReplayIO) or img is None:
            return
        try:
            d = ROOT / "data" / "leave"; d.mkdir(parents=True, exist_ok=True)
            stem = d / time.strftime(f"%Y%m%d_%H%M%S_{kind}")
            cv2.imwrite(str(stem) + ".png", img)
            Path(str(stem) + ".txt").write_text("\n".join(texts))
            for old in sorted(d.glob("*.png"))[:-40]:
                old.unlink(missing_ok=True); old.with_suffix(".txt").unlink(missing_ok=True)
        except Exception:                            # noqa: BLE001 -- diagnostics never stop the leave
            pass

    def _save_lobby_frame(self, img, texts, kind):
        """Audit 2026-10-03: no invite/lobby frame was ever kept, so lobby wording and the host's
        options are unknown. Keep one frame + its OCR text every 30 s while on these screens."""
        if isinstance(self.io, ReplayIO) or time.time() - getattr(self, "_lobby_saved", 0) < 30:
            return
        self._lobby_saved = time.time()
        d = ROOT / "data" / "lobby"; d.mkdir(parents=True, exist_ok=True)
        stem = d / time.strftime(f"%Y%m%d_%H%M%S_{kind}")
        cv2.imwrite(str(stem) + ".png", img)
        (Path(str(stem) + ".txt")).write_text("\n".join(texts))

    # -- solo games vs the app's computer players (owner 2026-10-03: "play it single player and test").
    # Frame coordinates read live 2026-10-03 on the server (data/calib/solo_*.png).
    SOLO = {"solo_tab": (59, 745), "play": (1218, 750), "special": (768, 542),
            "canal": (800, 400), "start": (1186, 760), "back": (60, 76), "home": (1245, 77)}
    MENU_HOME_EVERY_S = 8

    def on_offline_menu(self, img, ocr, texts, info):
        """Stage 2026-10-10 03:10-14:58: with solo off the runner sat on SOLO: OFFLINE GAMES for
        12 h; invites only come on the front page. Press home (every 3rd try the back arrow)."""
        if self.cfg.get("solo"):
            return self.on_home(img, ocr, texts, info)
        if time.time() - getattr(self, "_menu_home_at", 0) < self.MENU_HOME_EVERY_S:
            return
        self._go_home("solo off: leave the offline menu, wait for invites")

    def _go_home(self, why, guard=True):
        self._menu_home_at = time.time()
        n = self._menu_home_n = getattr(self, "_menu_home_n", 0) + 1
        way = "back" if n % 3 == 0 else "home"
        if not guard:                                 # the loop guard's own way out is never loop-blocked
            self._loop_blocked = {k: t for k, t in self.__dict__.get("_loop_blocked", {}).items()
                                  if k[0] != f"menu {way} button"}
        self.click(self.SOLO[way], f"menu {way} button")
        self.note(f"{why}: pressed {way}")

    def on_home(self, img, ocr, texts, info):
        if self._leave_hosted_party(ocr, texts):
            return
        if self.cfg.get("solo") and not isinstance(self.io, ReplayIO):
            if time.time() - getattr(self, "_solo_tried", 0) > 90:
                self._solo_tried = time.time()
                self.start_solo(img)

    def _leave_hosted_party(self, ocr, texts) -> bool:
        """Live 2026-10-09 18:09: "Bramble" invited us to a custom match, then left; the party
        passed to us. The front page then shows our name with a crown and "Host", CUSTOM + LEAVE
        bottom left: it reads as home, so we sat there and friends saw us busy. marlow never hosts
        (it only joins invites): after `leave_hosted_party_s` on that page, press LEAVE."""
        wait = float(self.cfg.get("leave_hosted_party_s", 30))
        joined = " ".join(texts)
        if wait <= 0 or isinstance(self.io, ReplayIO) or not (
                re.search(r"\bhost\b", joined, re.I) and re.search(r"\bleave\b", joined, re.I)):
            self._hosted_since = None
            return False
        now = time.time()
        if getattr(self, "_hosted_since", None) is None:
            self._hosted_since = now
            self.note(f"home: we host a custom party (someone left it); LEAVE in {wait:.0f} s")
            return True
        if now - self._hosted_since < wait or now - getattr(self, "_party_leave_at", 0) < 15:
            return True
        btn = find_text(ocr, r"^leave$")
        if not btn:
            return False
        self._party_leave_at = now
        self.click((btn["x"] + btn["w"] / 2, btn["y"] + btn["h"] / 2), "home: LEAVE the hosted party", ocr)
        time.sleep(1.5)
        ocr2 = self.io.ocr(self.io.snapshot())
        yes = find_text(ocr2, r"^yes$")
        if yes and re.search(r"leave|party|group|lobby", " ".join(o["text"] for o in ocr2), re.I):
            self.click((yes["x"] + yes["w"] / 2, yes["y"] + yes["h"] / 2), "home: confirm LEAVE party", ocr2)
        return True

    def _solo_wait(self, pattern: str, secs: float = 12):
        """Snapshot until `pattern` shows; returns (img, ocr) or (None, None)."""
        end = time.time() + secs
        while time.time() < end:
            time.sleep(1.5)
            img = self.io.snapshot()
            ocr = self.io.ocr(img)
            joined = " ".join(o["text"] for o in ocr)
            # the offline-games cards say "Quickplay" (offline, vs the computer): not matchmaking
            if FORBIDDEN_BUTTONS.search(re.sub(r"(?i)quick ?play", "", joined)
                                        if re.search(r"offline games", joined, re.I) else joined):
                return None, None
            if re.search(pattern, joined, re.I):
                return img, ocr
        return None, None

    @staticmethod
    def _yellow(img, xy) -> bool:
        x, y = map(int, xy)
        hsv = cv2.cvtColor(img[y - 6:y + 6, x - 20:x + 20], cv2.COLOR_BGR2HSV)
        return float(((hsv[..., 0] > 15) & (hsv[..., 0] < 35) & (hsv[..., 1] > 120)).mean()) > 0.4

    def start_solo(self, img) -> bool:
        """Home -> SOLO -> Special Scenarios -> The Great Canal -> Start game. Every step checks
        the next screen's text; anything unexpected stops it (and an online screen turns solo off:
        never public matchmaking)."""
        S = self.SOLO
        img, ocr = self._solo_wait(r"offline games|scenario selection", 2)   # already in the solo menu?
        if img is None:
            self.click(S["solo_tab"], "solo: SOLO tab")
            time.sleep(1.0)
            img = self.io.snapshot()
            if not self._yellow(img, S["solo_tab"]):
                self.note("solo: SOLO tab not highlighted, not pressing PLAY")
                return False
            self.click(S["play"], "solo: PLAY")
            img, ocr = self._solo_wait(r"offline games|scenario selection")
        if img is None:
            self.cfg["solo"] = False
            self.note("solo: no 'offline games' screen after PLAY -- solo turned OFF")
            return False
        # the offline-games screen also says "SCENARIO SELECTION" (on every card): test its title
        if re.search(r"offline games", " ".join(o["text"] for o in ocr), re.I):
            # 2026-10-04: in a windowed game the fixed spot missed the card; click its title when read
            sp = find_text(ocr, r"special")
            xy = (sp["x"] + sp["w"] / 2, sp["y"] + sp["h"] / 2) if sp else S["special"]
            self.click(xy, "solo: Special Scenarios")   # not "Continue" (a saved game)
        img, ocr = self._solo_wait(r"victory points:? ?\d+")   # the options panel draws late
        if img is None:
            self.note("solo: scenario options panel not shown")
            return False
        el = find_text(ocr, r"enchanted land")
        if el and el["x"] > 400:                      # Enchanted Land still selected (left of Canal)
            self.click(S["canal"], "solo: The Great Canal")
            img, ocr = self._solo_wait(r"great canal")
            el = find_text(ocr or [], r"enchanted land")
            if img is None or (el and el["x"] > 400):
                self.note("solo: could not select The Great Canal")
                return False
        if not find_text(ocr, r"victory points:? ?18"):
            self.note("solo: Great Canal panel not confirmed (no 'Victory Points: 18')")
            return False
        img, ocr = self._bots_to_veteran(img, ocr)
        cv2.imwrite(str(ROOT / "data" / "popups" / time.strftime("%H%M%S_solo_setup.png")), img)
        self.click(S["start"], "solo: Start game")
        self._solo_games = getattr(self, "_solo_games", 0) + 1
        self.note(f"solo: game {self._solo_games} started (The Great Canal vs the computer)")
        return True

    # owner 2026-10-05: "when marlow starts a game against bots, ... get all the bots to be veterans".
    # The options panel shows one level button per computer player (seen: ROOKIE, VETERAN, MASTER);
    # a click moves that button to the next level, so click until it reads VETERAN.
    # Stage 2026-10-08 21:45: a ROOKIE seat was not in the list, so it was never clicked and the
    # log said "all veteran" -- any one-word caps label in the seat column now counts as a level,
    # and "all veteran" needs every computer seat read (3 in the 4-player Great Canal).
    BOT_LEVEL = (r"^(rookie|beginner|novice|easy|apprentice|normal|intermediate|medium|advanced|"
                 r"veteran|master|expert|hard|legend|pro)$")
    N_BOTS = 3

    def _bot_levels(self, ocr) -> list[dict]:
        ocr = ocr or []
        me = next((o for o in ocr if re.match(r"marlow", o["text"].strip(), re.I)), None)

        def seat_label(o):
            t = o["text"].strip()
            if re.fullmatch(self.BOT_LEVEL, t, re.I):
                return True
            if me is None or not re.fullmatch(r"[A-Z]{4,12}", t):
                return False
            cx, mx = o["x"] + o["w"] / 2, me["x"] + me["w"] / 2
            return abs(cx - mx) < 60 and 0 < o["y"] - me["y"] < 200
        return sorted((o for o in ocr if seat_label(o)), key=lambda o: o["y"])

    def _bots_to_veteran(self, img, ocr, max_clicks: int = 6):
        """Every computer player on VETERAN before Start game; a row that won't get there is
        logged and left (the game still starts)."""
        levels = self._bot_levels(ocr)
        self.note("solo: bot levels " + ", ".join(o["text"] for o in levels))
        for row in range(max(len(levels), self.N_BOTS)):
            for _ in range(max_clicks):
                levels = self._bot_levels(ocr)
                if row >= len(levels) or levels[row]["text"].strip().lower() == "veteran":
                    break
                b = levels[row]
                self.click((b["x"] + b["w"] / 2, b["y"] + b["h"] / 2), f"solo: bot {row + 1} level {b['text']} -> next")
                time.sleep(0.8)
                img2 = self.io.snapshot()
                if img2 is None:
                    break
                img, ocr = img2, self.io.ocr(img2)
        levels = self._bot_levels(ocr)
        ok = len(levels) >= self.N_BOTS and all(o["text"].strip().lower() == "veteran" for o in levels)
        why = "" if ok else (f" (not all veteran: {len(levels)} of {self.N_BOTS} seats read)"
                             if len(levels) < self.N_BOTS else " (not all veteran)")
        self.note("solo: bot levels now " + ", ".join(o["text"] for o in levels) + why)
        return img, ocr

    def on_invite(self, img, ocr, texts, info):
        if FORBIDDEN_BUTTONS.search(" ".join(texts)):
            self.note("invite screen mentions matchmaking: ignoring")
            return
        self._save_lobby_frame(img, texts, "invite")
        if not self.cfg.get("accept_any_friend", True):
            return
        joined = " ".join(texts)
        # audit 2026-10-03: only a friend's private invitation -- never a bare JOIN on some list
        if not re.search(r"invited you|custom match|invitation", joined, re.I):
            self.note("invite screen without 'invited you / custom match': not joining")
            return
        allowed = [a.lower() for a in self.cfg.get("allowed_inviters", [])]
        if allowed and not any(a in joined.lower() for a in allowed):
            self.note(f"invite not from {allowed}: ignoring")
            return
        btn = find_text(ocr, r"^(accept|join)$")
        if btn is None and re.search(r"invited you to a custom match", " ".join(texts), re.I):
            # live 2026-10-02: "GAME INVITATION -- X has invited you to a Custom Match" NO / YES
            btn = find_text(ocr, r"^yes$") or self._yellow_yes(img, ocr)
        if btn:
            self.click((btn["x"] + btn["w"] / 2, btn["y"] + btn["h"] / 2), "accept friend invite", ocr)

    @staticmethod
    def _yellow_yes(img, ocr):
        """Full screen 2026-10-03: OCR read the invitation text but not YES. YES is the bright
        yellow button under "Do you want to join the game?" (NO beside it is dark)."""
        q = find_text(ocr, r"join the game|invited you to a custom match")
        if q is None:
            return None
        y0, x0 = int(q["y"] + q["h"]), max(0, int(q["x"] - 150))
        roi = img[y0:y0 + 110, x0:int(q["x"] + q["w"] + 250)]
        if roi.size == 0:
            return None
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        m = cv2.inRange(hsv, (18, 150, 180), (35, 255, 255))
        n, _, st, cen = cv2.connectedComponentsWithStats(m)
        if n < 2:
            return None
        i = 1 + int(np.argmax(st[1:, 4]))
        if st[i][4] < 1500:
            return None
        x, y, w, h = st[i][:4]
        return {"text": "YES (yellow button)", "x": x0 + x, "y": y0 + y, "w": w, "h": h}

    def _reconcile_setup_pieces(self, gs, kind):
        """Owner 2026-10-05 game 15: "picked the city where it should have been a settlement and
        the other way around". The scan saw a phantom piece of ours at (-6,-8) before our first
        pick, so the brain (which reads the pick number from our building count) scored the first
        settlement as the city and the city as the last settlement. The prompt says which piece
        this is: make our pieces match it (drop scanned ones we never placed, add the ones we did)."""
        if not getattr(gs.mode, "start_with_city", False):
            return
        picks = getattr(self, "_setup_picks", [])
        city_seen = any(p["kind"] == "city" for p in picks) or (
            not picks and any(k == "city" for _, k in gs.me().buildings()))   # rejoined at the last pick
        want = 1 if kind == "city" else (2 if city_seen else 0)
        landed = []                                   # the last try of each earlier piece landed
        for i, p in enumerate(picks):
            if i + 1 == len(picks) or picks[i + 1]["kind"] != p["kind"]:
                landed.append((p["v"], p["kind"]))
        if kind == "city":
            landed = [x for x in landed if x[1] == "settlement"][-1:]
        landed = landed[:want]
        me = gs.me()
        have = list(me.buildings())
        if len(have) == want and (len(landed) < want or all(v in {w for w, _ in have} for v, _ in landed)):
            return
        if len(landed) < want and len(have) >= want:
            return                                    # rejoined mid-setup: no memory, trust the scan
        known = {v for v, _ in landed}
        keep = [(v, k) for v, k in have if v in known]
        extra = [(v, k) for v, k in have if v not in known]
        for v, k in landed:
            if v not in {w for w, _ in keep}:
                keep.append((v, k))
        while len(keep) < want and extra:
            keep.append(extra.pop(0))
        me.settlements = {v for v, k in keep if k == "settlement"}
        me.cities = {v for v, k in keep if k == "city"}
        self.note(f"setup {kind}: our pieces {sorted(w for w, _ in have)} -> {sorted(v for v, _ in keep)} "
                  f"(pick {want + 1} of 3)")

    def _setup(self, img, texts, kind):
        gs = self.ensure_state(img, texts)
        if gs is None:
            return
        from .vision.board_reader import setup_options
        if kind == "ship":
            # owner 2026-10-02: the coastal/outer 3rd settlement must start with a ship, not a
            # road ("bot has failed that every time"); sea edges at the spot we just placed
            b, me = gs.board, gs.me()
            mine = [v for v, _ in me.buildings()]
            if getattr(self, "_in_play", False):
                # live 2026-10-03: a road tap on a coastal edge opened this prompt and nothing
                # answered it until the 10-min watchdog abandoned the game -- cancel it (red X)
                self.click(UI.get("place_cancel", (654, 646)), "cancel ship placement outside setup")
                return
            bare = [v for v in mine if b.is_coastal_vertex(v)
                    and not any(x in me.roads or x in me.ships for x in b.vertex_edges(v))]
            at = getattr(self, "_last_setup_v", None) or (bare[-1] if bare else mine[-1] if mine else None)
            if at is None:
                self.note("setup ship: own settlement not found")
                return
            tried = getattr(self, "_ship_tried", set())
            opts = [e for e in b.vertex_edges(at) if e not in tried and not gs.edge_owner(e)
                    and (b.is_sea_edge(e) or len(b.edge_hexes(e)) < 2)]
            if not opts:                      # all refused: widen to any edge at the spot
                tried.clear()
                opts = [e for e in b.vertex_edges(at) if not gs.edge_owner(e)]
            e = self.brain.choose_setup_road(gs, at, opts)
            tried.add(e)
            self._ship_tried = tried
            self.click(self.br.lattice.edge_px(e), f"setup ship {e}")
        elif kind == "road":
            me = gs.me()
            mine = [v for v, _ in me.buildings()]
            at = next((v for v in mine if not any(e for e in me.roads if v in e)), mine[-1] if mine else None)
            if at is None or (getattr(self, "_last_setup_v", None) is not None and self._last_setup_v not in
                              {v for e in me.roads for v in e}):
                # piece colour not seen: build from the settlement we just placed
                at = getattr(self, "_last_setup_v", at)
            if at is None:
                self.note("setup road: own settlement not found")
                return
            self._last_setup_kind = "road"
            # live 2026-10-03 game 6: the same refused edge was clicked for 15 min (our settlement
            # was not where we thought). Never repeat a refused edge: rotate through the edges at
            # the spot, then at every corner we own, then at every corner the game ringed.
            b = gs.board
            tried = self.__dict__.setdefault("_road_tried", set())
            if getattr(self, "_road_at", None) != at:
                tried.clear()
                self._road_at = at
            # game 13 18:07: a mid-game road prompt (Road Building card) stalled on "no free edge" at the
            # last settlement -- extend from our road network too
            net = sorted({v for e in me.roads for v in e} - set(mine))
            starts = [at] + [v for v in mine if v != at] + net + [v for v in self.br.highlights if v not in mine]
            src, opts = at, []
            for src in starts:
                opts = [e for e in b.vertex_edges(src) if e not in tried and not gs.edge_owner(e)
                        and b.is_land_edge(e)]
                if opts:
                    break
            if not opts:
                tried.clear()
                for src in starts:
                    opts = [e for e in b.vertex_edges(src) if not gs.edge_owner(e) and b.is_land_edge(e)]
                    if opts:
                        break
                if not opts:
                    self.note(f"setup road: no free edge at {at}")
                    return
            # game 14 03:24: "Select position for Road, Ship" at an outer-coast settlement -- the
            # legal edges were sea edges and 16 land edges were refused. The game rings the legal
            # edges: take those first (a sea edge there is the ship).
            lat = self.br.lattice
            rings = self._rings(img)
            if rings:
                near = lambda e: min(math.hypot(lat.edge_px(e)[0] - x, lat.edge_px(e)[1] - y) for x, y in rings)
                edges = {e for v in b.vertices for e in b.vertex_edges(v)}
                ringed = sorted(e for e in edges if not gs.edge_owner(e) and near(e) < 0.3 * lat.size)
                pick = [e for e in ringed if e not in tried] or ringed
                if pick:
                    src = next((v for v in [at] + mine if any(v in e for e in pick)), pick[0][0])
                    opts = [e for e in pick if src in e] or pick
            e = self.brain.choose_setup_road(gs, src, opts)
            if len(tried) >= 3:
                self.note(f"setup road: {len(tried)} edges refused; trying {e}")
            tried.add(e)
            self.click(self.br.lattice.edge_px(e), f"setup road {e}")
        else:
            # review 0.3.1: drop phantom pieces before the 3rd-pick rules read our building count
            # (a phantom at the city pick limited it to coastal corners with a 1-ring floor)
            self._reconcile_setup_pieces(gs, kind)
            opts = setup_options(gs, self.br)
            if gs.mode.canal and len(gs.me().settlements) + len(gs.me().cities) >= 2:
                # live 2026-10-02 game 4: the game rings only harbour / outer-coast corners; a
                # corner on an inland lake is "coastal" but refused -> trust the rings first
                legal = set(gs.legal_settlement_spots(gs.me(), setup=True))
                rings = [v for v in self.br.highlights if v in legal]
                opts = rings or [v for v in (opts or legal) if gs.board.is_coastal_vertex(v)]
            # live 2026-10-03: we didn't see an opponent's piece and clicked a taken corner
            # until the timer ran out. The game rings every legal corner: pick only from those,
            # and never repeat a corner whose click left us on the same prompt.
            lat = self.br.lattice
            # audit 2026-10-03: the 3rd (coastal) settlement gets only a few rings, so 1 is enough
            # there; first picks need 6. Wait until the ring count stops growing (they fade in).
            third = gs.mode.canal and len(gs.me().settlements) + len(gs.me().cities) >= 2
            floor = 1 if third else 6
            rings = self._rings(img)
            prev = -1
            for _ in range(6 if isinstance(self.io, LiveIO) else 0):
                if len(rings) >= floor and len(rings) == prev:
                    break
                prev = len(rings)
                time.sleep(0.6)
                nxt = self.io.snapshot()
                if nxt is None or not getattr(nxt, "size", 0):
                    break                                # replay ran out of frames
                img = nxt
                rings = self._rings(img)
            ringed = [v for v in gs.board.vertices if gs.board.is_land_vertex(v) and rings and
                      min(math.hypot(lat.vertex_px(v)[0] - x, lat.vertex_px(v)[1] - y) for x, y in rings) < 0.35 * lat.size]
            bad = getattr(self, "_setup_bad", set())
            if getattr(self, "_last_setup_v", None) is not None and getattr(self, "_last_setup_kind", None) == kind:
                bad.add(self._last_setup_v)              # still asked for the same piece: that click failed
            self._setup_bad = bad
            if len(ringed) < floor and opts:
                # audit 2026-10-03: clicking our own guess at a non-ringed corner was refused 5x
                # in a row (20:05). Look again twice before falling back to our legal spots.
                miss = self._ring_miss = getattr(self, "_ring_miss", 0) + 1
                if miss < 3 and isinstance(self.io, LiveIO):
                    self.note(f"setup {kind}: only {len(ringed)} ringed corners (need {floor}); looking again")
                    return
                ringed = []                              # still too few to trust: use our legal spots
            self._ring_miss = 0
            if ringed:
                pool = [v for v in (opts or ringed) if v in set(ringed)] or ringed
                opts = [v for v in pool if v not in bad] or [v for v in ringed if v not in bad] or pool
            elif opts:
                opts = [v for v in opts if v not in bad] or opts
            self.note(f"setup {kind}: {len(rings)} rings, {len(ringed)} ringed corners, skip {sorted(bad)}")
            v = self.brain.choose_setup_settlement(gs, opts or None)   # lookahead + 3-roll estimate
            self._last_setup_kind = kind
            self._note_setup_plan(gs, f"placing {kind}")
            self._last_setup_v = v
            picks = self.__dict__.setdefault("_setup_picks", [])
            picks.append({"kind": kind, "v": v, "seat": self.brain.setup_seat(gs), "players": len(gs.players),
                          "score": round(self.brain.setup_vertex_score(gs, v, gs.me()), 2),
                          "options": len(opts or []), "rings": len(rings), "retry": v in bad})
            self._game_save("setup.json", picks)         # learning: setup picks joined with result.json
            if kind == "city":
                self.__dict__.setdefault("_my_cities", set()).add(v)
                self._save_game_flags()
            self.click(self.br.lattice.vertex_px(v), f"setup {kind} {v}")
        self.click(UI["setup_confirm"], "confirm placement")

    def _note_setup_plan(self, gs, why: str):
        """Log our top spots and what the others are predicted to take before our next pick."""
        b, me = self.brain, gs.me()
        opts = gs.legal_settlement_spots(me, setup=True)
        if not opts:
            return
        top = sorted(opts, key=lambda v: b.setup_vertex_score(gs, v, me), reverse=True)[:3]
        k = b.opponent_picks_before_next(gs)
        gone = b.predict_opponent_takes(gs, k, set())
        placed = {p.name: sorted(v for v, _ in p.buildings()) for p in gs.players if p.buildings()}
        self.note(f"setup plan ({why}) seat {b.setup_seat(gs)}/{len(gs.players)}: placed={placed} our top={[(v, round(b.setup_vertex_score(gs, v, me), 1)) for v in top]} "
                  f"others likely take {sorted(gone)} ({k} picks)")

    def setup_watch(self, img, texts):
        """Owner 2026-10-02: during setup, re-read the board after every opponent placement and
        re-plan, so our pick is ready when our turn comes."""
        if getattr(self, "_in_play", False) or self.mode is None or self.lattice is None:
            return
        if time.time() - getattr(self, "_watch_t", 0) < 4:
            return
        self._watch_t = time.time()
        gs = self.ensure_state(img, texts)
        if gs is None:
            return
        n = sum(len(p.buildings()) for p in gs.players)
        if n != getattr(self, "_watch_n", -1) and n < 3 * len(gs.players):
            self._watch_n = n
            self._note_setup_plan(gs, f"{n} pieces down")

    def on_setup_settlement(self, img, ocr, texts, info):
        me = self.gs.me() if self.gs is not None else None
        # game 14 03:24: a phantom 3rd building (read off a ring) made the real 3rd setup settlement
        # look like a barbarian city loss -- after a game start we saw, only play makes it a loss
        if getattr(self, "_in_play", False) or (me is not None and len(me.buildings()) >= 3
                                                 and not getattr(self, "_saw_game_start", False)):
            # same prompt mid-game = barbarians took a city: pick which one becomes a settlement
            return self.lose_city(img, texts)
        self._setup(img, texts, "settlement")

    @staticmethod
    def ring_spots(img, native=None, lay=None):
        """White selection rings the game draws on legal spots (live 2026-10-02), in frame px.
        Game 7 2026-10-03: in the 1372x888 frame of a full-screen 1920x1080 window the thin
        rings fade below v215 (1 ring found, ~34 on screen): read them on the native capture."""
        k = 1.0
        if native is not None and lay is not None:
            img, k = native, lay.s
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        m = ((hsv[:, :, 1] < 50) & (hsv[:, :, 2] > 215)).astype(np.uint8) * 255
        top, bot = (lay.to_native(0, 190)[1], lay.to_native(0, 740)[1]) if k != 1.0 else (190, 740)
        m[:max(0, int(top))] = 0
        m[int(bot):] = 0
        n, _, st, _ = cv2.connectedComponentsWithStats(m)
        out = []
        for i in range(1, n):
            x, y, w, h, a = st[i]
            if 12 * k <= w <= 30 * k and 12 * k <= h <= 30 * k and abs(w - h) <= 5 * k and a < 0.6 * w * h:
                cx, cy = x + w / 2, y + h / 2
                if k != 1.0:
                    cx, cy = lay.to_frame(cx, cy)
                out.append((int(cx), int(cy)))
        return out

    def _rings(self, img):
        """ring_spots on the native capture behind this frame when the live IO has one. Game 8
        13:01: an "o" of the white "Select position for Settlement" prompt was taken for a ring
        and clicked 13x as the city to lose -- rings inside the prompt's text box are dropped."""
        io = self.io
        if getattr(io, "_frame", None) is img and getattr(io, "native", None) is not None and getattr(io, "lay", None) is not None:
            rings = self.ring_spots(img, io.native, io.lay)
        else:
            rings = self.ring_spots(img)
        if img is getattr(self, "_last_img", None):
            boxes = [o for o in (getattr(self, "_last_ocr", None) or []) if len(o["text"]) >= 8]
            rings = [r for r in rings if not any(o["x"] - 8 <= r[0] <= o["x"] + o["w"] + 8 and
                                                 o["y"] - 8 <= r[1] <= o["y"] + o["h"] + 8 for o in boxes)]
        return rings

    def lose_city(self, img, texts):
        """Mid-game "Select position for Settlement" = the barbarians took one of our cities.
        review 0.3.1: like on_remove_city, the loss is recorded once, when the game takes it (the
        prompt is gone, or its green check was pressed) -- it used to count a loss and drop a
        city on every frame of the prompt, so a tap that did not take cost a second city."""
        gs = self.ensure_state(img, texts)
        self.barbarians_attacked = True               # a city loss means the barbarians attacked
        self._barb_attack("our city is lost")
        if gs is not None:
            gs.barbarians_attacked = True
        pend = getattr(self, "_remove_city_pending", None)
        if pend and self._green_check(img):
            self.click(self._check_at(img), "confirm city loss (green check)")
            self._city_loss_taken(*pend, gs=gs)
            return
        rings = self._rings(img)
        xy, why, v = None, "", None
        if gs is not None and self.br is not None:
            lat = self.br.lattice
            opts = list(gs.me().cities)
            if rings and opts:
                # only cities with a ring are legal
                near = [v for v in opts if min(np.hypot(lat.vertex_px(v)[0] - rx, lat.vertex_px(v)[1] - ry)
                                               for rx, ry in rings) < 25]
                opts = near or opts
            tried = self.__dict__.setdefault("_remove_city_tried", set())
            left = [c for c in opts if c not in tried]
            if not left:
                tried.clear()
                left = opts
            v = self.brain.choose_city_to_lose(gs, left) if left else None
            if v is not None:
                tried.add(v)
                xy, why = lat.vertex_px(v), f"lose weakest city {v}"
        if xy is None and rings:
            xy, why = rings[0], "lose city at ring (no city known)"
        if xy is None:
            self.note("lose city: no city or ring found")
            return
        self._remove_city_pending = (v, "city")
        self.click(xy, why)
        time.sleep(0.8)
        im2 = self.io.snapshot()
        if im2 is not None and find_text(self.io.ocr(im2), r"select position"):
            self.click(UI["setup_confirm"], "confirm city loss")
            time.sleep(0.8)
            im2 = self.io.snapshot()
            if im2 is not None and find_text(self.io.ocr(im2), r"select position"):
                self.note(f"lose city: the prompt is still up after {why}: next frame tries again")
                return
        self._city_loss_taken(v, "city", gs=gs)

    def _city_loss_taken(self, v, kind="city", gs=None):
        """The game took the city loss: record it once (the city at v, when known)."""
        if v is None:
            self._lost_cities = getattr(self, "_lost_cities", 0) + 1
            self.__dict__.setdefault("_remove_city_tried", set()).clear()
            self._remove_city_pending = None
            self.note(f"lose city: confirmed (city unknown, lost cities {self._lost_cities})")
            self._save_game_flags()
            return
        self._after_remove_city(v, kind, gs=gs)

    def _setup_textless(self, img, texts) -> bool:
        """Game 14 2026-10-05 03:20: our first setup turn showed the selection rings but no
        "Select position for ..." text; 40 s later the blind ring fallback placed our first
        settlement. In setup, rings on board corners are our placement (settlement, city, then the
        canal's coastal settlement); rings only on edges are the road. True = handled."""
        gs = self.ensure_state(img, texts)
        if gs is None:
            return False
        vr = self._vertex_rings(img)
        n = len(gs.me().buildings())
        # review 0.3.1: the order is the rules' (brain._setup_kind) -- with a starting city the
        # second piece is the city and the Great Canal's third is a settlement again; the old
        # "settlement for the first two" put the city last and the canal start cards on it
        build = self.brain._setup_kind(gs, gs.me())
        kind = build if len(vr) >= 5 else ("road" if n else None)
        if kind is None:
            return False
        self.note(f"setup prompt without text: {len(vr)} ringed corners, {n} buildings -> {kind}")
        self.probe.tried(f"setup_textless_{kind}")
        self._setup(img, texts, kind)
        return True

    def on_setup_city(self, img, ocr, texts, info):
        self._setup(img, texts, "city")

    def on_setup_road(self, img, ocr, texts, info):
        self._setup(img, texts, "road")

    def on_place_ship(self, img, ocr, texts, info):
        """'Select position for Ship': in setup this is the ship off the coastal 3rd settlement.
        Mid-game ship builds click their edge from the build loop and never wait here."""
        self._setup(img, texts, "ship")

    def on_promote_knight(self, img, ocr, texts, info):
        """'Select position for Strong/Mighty knight': tap our knights one at a time until the
        prompt goes away (the game only accepts a knight that can be promoted)."""
        # audit 2026-10-04: 211 misses guessing knight spots; the game rings the legal knights
        if self._green_check(img):
            self.click(self._check_at(img), "confirm strong knight (green check)")
            return
        # game 13: a free Strong/Mighty knight (Deserter takes the victim's level) rings EMPTY
        # corners, not our knights -- that is a placement, not a promotion
        gs0 = getattr(self, "gs", None)
        vr = self._vertex_rings(img)
        if vr and gs0 is not None and not any(v in gs0.me().knights or v in getattr(self, "_my_knights", {})
                                              for v, _ in vr):
            self.on_place_knight(img, ocr, texts, info)
            return
        if self._rings(img):
            self._forced_fallback(img, "promote_knight")
            return
        gs = self.ensure_state(img, texts)
        mine = list(gs.me().knights) if gs is not None else []
        mine += [v for v in getattr(self, "_my_knights", {}) if v not in mine]
        if not mine and gs is not None:
            # solo 2026-10-03: right after a restart the knights aren't read (solo vision misses
            # them) -- knights stand on free corners of our road network, so try those
            me = gs.me()
            mine = sorted({v for e in me.roads for v in e
                           if gs.board.is_land_vertex(v) and v not in me.settlements and v not in me.cities})
        tried = self.__dict__.setdefault("_promote_tried", set())
        todo = [v for v in mine if v not in tried]
        if not todo:
            tried.clear()
            self.note(f"promote knight: none of our knights {mine} took it; waiting")
            return
        v = todo[0]
        tried.add(v)
        self.click(self.br.lattice.vertex_px(v), f"promote knight at {v}")
        if self._confirm_if_check("promote knight"):
            tried.clear()
            return
        time.sleep(0.2)
        after = self.io.snapshot()
        if after is not None and not find_text(self.io.ocr(after), r"(strong|mighty) ?knight"):
            tried.clear()
            if find_text(self.io.ocr(after), r"confirm"):
                self.click(self._check_at(after), "confirm promote knight")

    def on_place_robber(self, img, ocr, texts, info):
        """Robber on the most-likely winner's best hex (owner 2026-10-02: block the leader).
        Tap the hex, then the confirm check (same spot as setup_confirm); if no check shows
        up the hex was illegal, so the next loop tries the next-best hex."""
        gs = self.ensure_state(img, texts)
        if gs is None:
            return
        tried = getattr(self, "_robber_tried", set())
        b = gs.board
        lat = self.br.lattice
        # live 2026-10-02: the game rings the legal hexes (friendly robber etc.); the ring
        # sits inside its hex, so map each ring to the nearest hex centre and tap the ring
        ring_at = {}
        rings = self._rings(img)
        land = sum(1 for h in b.hexes.values() if h.is_land)
        io = getattr(self, "io", None)
        if getattr(self, "cfg", {}).get("robber_ring_reread", True) and len(rings) < min(8, land // 2) \
                and io is not None and not isinstance(io, ReplayIO):
            # live 2026-10-09 19:44: read right after the discard closed, 4 of ~29 rings showed and
            # the robber went on an 8 with nobody on it. The rings fade in: look again.
            for _ in range(2):
                time.sleep(0.6)
                again = self._rings(io.snapshot())
                if len(again) > len(rings):
                    rings = again
                if len(rings) >= min(8, land // 2):
                    break
        for rx, ry in rings:
            c = min((c for c, h in b.hexes.items() if h.is_land),
                    key=lambda c: np.hypot(lat.hex_px(*c)[0] - rx, lat.hex_px(*c)[1] - ry), default=None)
            if c is not None:
                ring_at.setdefault(c, (rx, ry))
        if not ring_at:
            # stage 2026-10-09 01:07:06/10: a place_robber read with no ring on Jean's turn; the
            # bot tapped two hexes and the confirm spot twice. The gold turn marker over a rival's
            # portrait means the robber is theirs to move: watch, don't tap.
            seat = self._marker_seat(img)
            if seat:
                who = getattr(self, "_slot_names", {}).get(seat, f"P{seat + 1}")
                self.note(f"robber prompt with no ring while {who} is on turn: theirs to move, not tapping")
                return
        pool = list(ring_at) or [c for c, h in b.hexes.items() if h.is_land and c != b.robber]
        opts = [c for c in pool if c not in tried] or pool
        hx, victim = self.brain.choose_robber(gs, opts)
        tried.add(hx)
        self._robber_tried = tried if len(tried) < 6 else set()
        xy = ring_at.get(hx) or lat.hex_px(*hx)
        self.note(f"robber rings: {len(ring_at)} legal hexes")
        self.click(xy, f"robber on {hx} (victim {getattr(victim, 'name', victim)})")
        # live 2026-10-09 19:29: the fixed spot sat between check and X, the robber on the leader
        # (Rhea, 10 VP) never confirmed and the next pass moved it to a weaker rival
        if self.cfg.get("robber_fast", True):
            # 0.4.3 step 5: the confirm button is read as soon as the screen settles (<= 0.8 s)
            im2 = self._settled_snapshot(img, 0.8) if not isinstance(self.io, ReplayIO) else None
        else:
            time.sleep(0.8)
            im2 = self.io.snapshot() if not isinstance(self.io, ReplayIO) else None
        self.click(self._check_at(im2) if im2 is not None else UI["setup_confirm"], "confirm robber")

    def on_choose_resource(self, img, ocr, texts, info):
        """Gold field / aqueduct (science 3) / free pick. Owner 2026-10-02: these are the only
        grain in the Great Canal. Layout UNCONFIRMED: save the screen, click the wanted card's
        label if OCR finds one, else leave the pick to the game."""
        d = Path(getattr(self, "popup_dir", None) or ROOT / "data" / "popups")
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / f"{time.strftime('%H%M%S')}_choose_resource.png"), img)
        gs = getattr(self, "gs", None)
        try:
            want = self.brain.choose_resource(gs, 1)[0] if gs is not None else "grain"
        except Exception:                            # noqa: BLE001
            want = "grain"
        if re.search(r"that you'?.?ll gain|science bonus|from the supply", " ".join(texts), re.I):
            # game 13 16:30: the aqueduct pick is the trade dialog's top row + the check; the
            # bot read it as main and clicked End turn behind it 13x. Wanted card first, then
            # grain/ore/wool... if the dialog stays (a card the supply can't give)
            n = self.__dict__.get("_gain_round", 0)
            if time.time() - self.__dict__.get("_gain_at", 0) > 20:
                n = 0                                # review 0.3.1: a new pick starts on the wanted card
            self._gain_at = time.time()
            order = [want] + [r for r in ("grain", "ore", "wool", "brick", "lumber") if r != want]
            res = order[n % len(order)]
            self._gain_round = n + 1
            self._probe("gain_pick", img, texts=texts, note={"pick": res})
            self.click((UI["trade_want_row"]["x"][CARD_ORDER.index(res)], UI["trade_want_row"]["y"]), f"gain {res}")
            self.click(UI["trade_ok"], f"gain {res}: confirm")
            time.sleep(1.0)
            im2 = self.io.snapshot()
            if im2 is not None and not re.search(r"that you'?.?ll gain|science bonus",
                                                 " ".join(o["text"] for o in self.io.ocr(im2)), re.I):
                self._gain_round = 0
            return
        names = {"grain": r"grain|wheat|hay", "ore": r"ore|stone", "wool": r"wool|sheep",
                 "lumber": r"lumber|wood", "brick": r"brick|clay"}
        hit = next((o for o in ocr if re.fullmatch(names.get(want, want), o["text"].strip().lower())), None)
        if hit is None:
            self.note(f"choose resource: want {want}, no label found; screen saved (layout UNCONFIRMED)")
            return
        self.click((hit["x"] + hit["w"] // 2, hit["y"] + hit["h"] // 2), f"choose resource {want}")
        time.sleep(0.6)
        self._confirm_accept()

    def on_steal(self, img, ocr, texts, info):
        """'Steal resource cards -- From which player do you want to steal?' with a coloured
        check button under each name (live 2026-10-02: no handler, the bot sat on it).
        Steal from the opponent most likely to win (same threat model as blocking)."""
        names = [n for n, _ in (getattr(self, "game_colors", None) or [])
                 if n.lower() != self.cfg["bot_name"].lower()]
        hdr = find_text(ocr, r"want to steal|steal from|choose a player")
        y0 = (hdr["y"] + hdr["h"]) if hdr else 380
        cands = []
        for o in ocr:
            t = o["text"].strip()
            if o["y"] <= y0 or o["y"] > y0 + 150:
                continue
            m = next((n for n in names if n.lower().startswith(t.lower()[:5]) and len(t) >= 3), None)
            if m:
                cands.append((m, o["x"] + o["w"] / 2, o["y"] + o["h"] / 2 + 60))
        if not cands:
            self.note(f"steal: no names read in {texts}")
            return
        gs = self.ensure_state(img, texts) or self.gs
        probs = self.brain.probs(gs) if gs is not None else {}
        vp = getattr(self, "_vp", {}) or {}
        pick = max(cands, key=lambda c: (probs.get(c[0], 0.0), vp.get(c[0], 0)))
        if gs is not None and getattr(gs, "hands", None) is not None:
            # owner 2026-10-06: steal where the ledger says the useful cards are
            v = self.brain.choose_victim(gs, [c[0] for c in cands])
            pick = next((c for c in cands if c[0] == v), pick)
        self.note(f"steal: from {pick[0]} (choices {[c[0] for c in cands]}, win probs "
                  f"{ {c[0]: round(probs.get(c[0], 0), 2) for c in cands} })")
        self.click((pick[1], pick[2]), f"steal from {pick[0]}")
        if time.time() - getattr(self, "_ledger_steal_at", 0) > 20:   # one steal per robber, not per click
            self._ledger_steal_at = time.time()
            self._ledger().expect_take([pick[0]], 1, why="robber steal")  # exact once our hand shows it
        # 0.4.3 step 5: the loop's next read (0.25 s after a click) sees the card arrive
        time.sleep(0.25 if self.cfg.get("robber_fast", True) else 1.0)

    def on_choose_player(self, img, ocr, texts, info):
        """Progress-card player pick (Deserter, Spy, Wedding, Master Merchant, ...): a dialog with
        a coloured check button ~60 px under each legal name. The card's target first, else the
        opponent most likely to win; a pick that leaves the dialog up moves to the next name."""
        names = [n for n, _ in (getattr(self, "game_colors", None) or [])
                 if n.lower() != self.cfg["bot_name"].lower()]
        hdr = find_text(ocr, r"choose (a player|an opponent)|which player|select a player")
        y0 = (hdr["y"] + hdr["h"]) if hdr else 340
        cands = []
        for o in ocr:
            t = o["text"].strip()
            if o["y"] <= y0 or o["y"] > y0 + 150 or len(t) < 2:
                continue
            m = next((n for n in names if n.lower().startswith(t.lower()[:5])), None) if names else t
            if m and not any(c[0] == m for c in cands):
                cands.append((m, o["x"] + o["w"] / 2, o["y"] + o["h"] / 2 + 60))
        if not cands:
            if time.time() - getattr(self, "_cp_probe_t", 0) > 60:
                self._cp_probe_t = time.time()
                self._probe("choose_player_unread", img, texts=texts)
            self.note(f"choose player: no names read in {texts}")
            return
        tried = self.__dict__.setdefault("_player_tried", set())
        todo = [c for c in cands if c[0] not in tried] or cands
        want = getattr(self, "_want_player", None)
        gs = getattr(self, "gs", None)
        probs = self.brain.probs(gs) if gs is not None else {}
        pick = next((c for c in todo if want and c[0].lower() == want.lower()), None) or \
            max(todo, key=lambda c: probs.get(c[0], 0.0))
        tried.add(pick[0])
        self.note(f"choose player: {pick[0]} (choices {[c[0] for c in cands]}, wanted {want})")
        self.click((pick[1], pick[2]), f"choose player {pick[0]}")
        if self.cfg.get("robber_fast", True):
            im2 = self._settled_snapshot(img, 1.0)    # 0.4.3 step 5: read when the dialog has moved
        else:
            time.sleep(1.0)
            im2 = self.io.snapshot()
        if im2 is not None and classify([o["text"] for o in self.io.ocr(im2)], self.cfg["bot_name"].lower())[0] != "choose_player":
            self._want_player, self._player_tried = None, set()

    # Commercial Harbor (ours), stage 2026-10-09 00:40:50: "Choose a player with whom you want to
    # trade." with a check under each opponent's name; the screens after a pick were never seen.
    HARBOR_MAX_ROUNDS = 4          # dialog sightings per card before we close it with its X
    HARBOR_MAX_S = 120

    def _dialog_names(self, ocr, hdr_rx):
        """(name, check x, check y) for each opponent name under a dialog header (the coloured
        check sits ~60 px under the name)."""
        names = [n for n, _ in (getattr(self, "game_colors", None) or [])
                 if n.lower() != self.cfg["bot_name"].lower()]
        hdr = find_text(ocr, hdr_rx)
        y0 = (hdr["y"] + hdr["h"]) if hdr else 340
        cands = []
        for o in ocr:
            t = o["text"].strip()
            if o["y"] <= y0 or o["y"] > y0 + 150 or len(t) < 2 or re.search(r"hide overlay", t, re.I):
                continue
            m = next((n for n in names if n.lower().startswith(t.lower()[:5])), None) if names else t
            if m and not any(c[0] == m for c in cands):
                cands.append((m, o["x"] + o["w"] / 2, o["y"] + o["h"] / 2 + 60))
        return cands

    def _close_card_dialog(self, img, ocr, title_rx, why):
        from .vision.screens import dialog_close_x
        boxes = [o for o in ocr if re.search(title_rx, o["text"], re.I) and o["y"] < 600]
        title = max(boxes, key=lambda o: o["h"]) if boxes else None
        xy = dialog_close_x(img, title) if title is not None and img is not None else None
        if xy is None and title is not None:          # the X sits ~165 px right of the title centre
            xy = (title["x"] + title["w"] / 2 + 165, title["y"] + title["h"] / 2 - 12)
        if xy is None:
            self.note(f"{why}: no dialog X found")
            return False
        self.click(xy, f"{why}: close the dialog (X)")
        return True

    def on_harbor_player(self, img, ocr, texts, info):
        """Our Commercial Harbor: offer each opponent in turn one resource for a commodity. Pick
        the next opponent's check; the screen after the pick is UNCONFIRMED -- it is captured,
        and when it is not a known prompt (the robber-style card row reads as discard) we give
        the planned resource on the trade give row + check (the confirmed robber-dialog layout).
        Owner rule: an unknown UI is tried once with capture, never held -- after
        HARBOR_MAX_ROUNDS sightings or HARBOR_MAX_S the dialog is closed with its X."""
        now = time.time()
        st = getattr(self, "_harbor", None)
        if st is None or now - st["t0"] > self.HARBOR_MAX_S + 60:
            st = self._harbor = {"t0": now, "tried": set(), "rounds": 0}
        st["rounds"] += 1
        cands = self._dialog_names(ocr, r"with whom you want")
        todo = [c for c in cands if c[0] not in st["tried"]]
        if st["rounds"] > self.HARBOR_MAX_ROUNDS or now - st["t0"] > self.HARBOR_MAX_S or not todo:
            self._probe("harbor_close", img, texts=texts, note={"rounds": st["rounds"], "tried": sorted(st["tried"])})
            self.probe.tried("harbor_close_x")
            self.note(f"commercial harbor: closing the dialog (round {st['rounds']}, tried {sorted(st['tried'])}, "
                      f"names {[c[0] for c in cands]})")
            self._close_card_dialog(img, ocr, r"commercial harbo", "commercial harbor")
            return
        gs = getattr(self, "gs", None)
        probs = self.brain.probs(gs) if gs is not None else {}
        pick = max(todo, key=lambda c: probs.get(c[0], 0.0))   # the likeliest winner gives first
        st["tried"].add(pick[0])
        self.probe.tried("harbor_pick_player")
        self.note(f"commercial harbor: offer to {pick[0]} (choices {[c[0] for c in cands]})")
        self.click((pick[1], pick[2]), f"commercial harbor: choose {pick[0]}")
        time.sleep(1.5)
        im2 = self.io.snapshot()
        if im2 is None:
            return
        o2 = self.io.ocr(im2)
        t2 = [o["text"] for o in o2]
        st2, _ = classify(t2, self.cfg["bot_name"].lower())
        self._probe("harbor_after_pick", im2, texts=t2, note={"player": pick[0], "screen": st2})
        if st2 in ("harbor_player", "discard"):
            return                                   # next player / the step loop's discard answers it
        give = self._harbor_resource(gs)
        if give is None:
            self.note("commercial harbor: no resource to give")
            return
        self.probe.tried("harbor_give")
        self.click((UI["trade_give_row"]["x"][CARD_ORDER.index(give)], UI["trade_give_row"]["y"]),
                   f"commercial harbor: give {give} to {pick[0]} (UNCONFIRMED layout)")
        time.sleep(0.5)
        self.click(UI["trade_ok"], "commercial harbor: confirm give (check mark)")
        time.sleep(1.5)
        im3 = self.io.snapshot()
        if im3 is not None:
            self._probe("harbor_after_give", im3, texts=[o["text"] for o in self.io.ocr(im3)],
                        note={"player": pick[0], "give": give})

    def _harbor_resource(self, gs):
        """Next planned give (choose_commercial_harbor), else our least valued resource; never grain
        (owner: grain is never given to a player) -- None when only grain is left."""
        plan = getattr(self, "_harbor_give", None) or []
        hand = dict(gs.me().hand) if gs is not None else dict(getattr(self, "_last_hand", None) or {})
        while plan:
            r = plan.pop(0)
            if hand.get(r, 0) > 0:
                return r
        have = [r for r in ("brick", "lumber", "wool", "ore") if hand.get(r, 0) > 0]
        if not have:
            return None
        if gs is not None:
            from .progress_cards import card_value
            have.sort(key=lambda r: card_value(gs, r))
        return have[0]

    def on_discard_progress(self, img, ocr, texts, info):
        """'Discard progress card' (5th card): drag the card we least want to keep onto the
        board -- a click only enlarges it (live 2026-10-03, stalled 45 min as a resource discard)."""
        from .progress_cards import keep_value
        held = self._read_progress_hand(ocr, img)
        if not held:
            # game 11 15:41: no title read -> 10 min stall. Drag a card from the hand area anyway,
            # a different spot each time (left to right)
            x0, y0, x1, y1 = UI["progress_hand"]
            n = self.__dict__.get("_pdiscard_blind", 0)
            self._pdiscard_blind = n + 1
            xy = (x0 + 70 + (n % 5) * 75, y0 + 105)
            self._probe("discard_progress_blind", img)
            self.note(f"discard progress: no card read in the hand; dragging the card at {xy}")
            self.drag(xy, tuple(UI.get("progress_discard_drop", (700, 400))), "discard progress (blind)")
            self._after_drag_scan()
            self._park_pointer("after discard drag")
            return
        tried = self.__dict__.setdefault("_pdiscard_tried", set())
        todo = [h for h in held if h[0] not in tried] or held
        if len(todo) == len(held):
            tried.clear()
        gs = getattr(self, "gs", None)
        me = gs.me() if gs is not None and any(p.is_me for p in gs.players) else None
        key, box = min(todo, key=lambda h: keep_value(gs, me, h[0], self.brain.wedding_dead_when_leading))
        tried.add(key)
        xy = (box["x"] + box["w"] / 2, box["y"] + box["h"] / 2 + 25)
        self.drag(xy, tuple(UI.get("progress_discard_drop", (700, 400))), f"discard progress {key}")
        # audit 2026-10-03: a drag that misses the card would pan the board (if the 2D camera lock
        # ever lets go): scan the board again before the next board click (SPEED_PLAN 8: only if it moved)
        self._after_drag_scan()
        if self.cfg.get("discard_reread", True) and not self.cfg.get("dry_run"):
            # stage 2026-10-09 07:20:15: planned saboteur, but saboteur was still held at 07:20:39
            # and the next pass dragged another card -- re-read and retry the planned card once
            time.sleep(0.5)
            im2 = self.io.snapshot()
            left = self._read_progress_hand(self.io.ocr(im2), im2) if im2 is not None else []
            n_before = sum(1 for k, _ in held if k == key)
            if left and sum(1 for k, _ in left if k == key) >= n_before:
                again = self.__dict__.setdefault("_pdiscard_retry", set())
                kept = [k for k, _ in held if k not in [k2 for k2, _ in left]]
                self.note(f"discard progress: kept {key}, planned {key}"
                          + (f" (gone instead: {kept})" if kept else "")
                          + ("" if key in again else " -- retrying it once"))
                if key not in again:
                    again.add(key)
                    tried.discard(key)

    def on_discard(self, img, ocr, texts, info):
        """Robber 7 over the hand limit, or a progress card (Wedding, Master Merchant...) that
        makes us give cards: give the least-needed ones (brain.choose_discard).
        Card click = icon right of the hand-bar digit, then the confirm button by label
        (UNCONFIRMED flow; frames saved to data/discard/ to calibrate)."""
        key = int(time.time() // 15)
        d = ROOT / "data" / "discard"; d.mkdir(exist_ok=True)
        cv2.imwrite(str(d / f"{time.strftime('%H%M%S')}_in.png"), img)
        if getattr(self, "_discarded", None) == key:
            return
        m0 = re.search(r"(\d+)\s*(?:more\s*)?(?:resource|card|commodit)", " ".join(texts).lower())
        n0 = int(m0.group(1)) if m0 else None
        prev = getattr(self, "_discard_prev", None)
        self._discard_prev = (n0, time.time())
        if prev and n0 is not None and prev[0] == n0 and time.time() - prev[1] < 60:
            # final probe game 10:30: the hand read said 5 wool (we had 1); wool clicks did
            # nothing and the same "Choose 2 more" came back 13x. The count didn't drop after a
            # full round: the hand read is wrong, click cards in order watching the count
            self.note(f"discard: still {n0} more after our clicks -> blind discard")
            self._blind_discard(texts, d)
            self._discarded = key
            return
        unread = False
        if self.cfg.get("discard_reread", True) and self._hand_unread(img):
            img, unread = self._discard_reread(img)
        gs = self.ensure_state(img, texts)
        if gs is None:
            # solo 2026-10-03: a restart under the robber dialog -- no board, no hand read
            self._blind_discard(texts, d)
            self._discarded = key
            return
        if unread and getattr(self, "_last_hand", None):
            # stage 2026-10-09 07:20:23: every slot unreadable -> plan from the tracked last hand
            gs.me().hand.update(dict(new_hand(), **self._last_hand))
            self.note(f"discard: hand still unreadable after re-reads, planning from the last hand {self._last_hand}")
        hand = {k: v for k, v in gs.me().hand.items() if v}
        if not hand and getattr(self, "_last_hand", None):
            # the dialog hides the hand bar: use the last full read
            hand = dict(self._last_hand)
            gs.me().hand.update(hand)
        m = re.search(r"(\d+)\s*(?:more\s*)?(?:resource|card|commodit)", " ".join(texts).lower())
        n = int(m.group(1)) if m else sum(hand.values()) // 2
        n = min(n, sum(hand.values()))
        if n <= 0:
            self.note(f"discard screen but hand unknown {hand}: leaving it to the game")
            return
        pick = self.brain.choose_discard(gs, n)
        self._cards_lost = getattr(self, "_cards_lost", 0) + n      # walls get worth more (brain.wall_value)
        self.note(f"discard {n} of {hand}: giving least needed {pick}")
        # live 2026-10-02: "Robber" dialog, our card row = the trade panel's give row,
        # check mark = trade_ok ("Choose 5 more resources to hand over to the robber")
        for res, k in pick.items():
            x = UI["trade_give_row"]["x"][CARD_ORDER.index(res)]
            for _ in range(k):
                self.click((x, UI["trade_give_row"]["y"]), f"discard {res}")
                time.sleep(0.3)
        time.sleep(0.5)
        after = self.io.snapshot()
        try:                                         # review 0.3.1: a None frame raised before the confirm
            if after is not None:
                cv2.imwrite(str(d / f"{time.strftime('%H%M%S')}_after.png"), after)
        except Exception:                            # noqa: BLE001
            pass
        self.click(UI["trade_ok"], "confirm discard (check mark)")
        self._discarded = key

    def _hand_unread(self, img) -> bool:
        """Every hand slot unreadable: some read None and none read a count."""
        try:
            vals = list(self.hand.read(img).values())
        except Exception:                            # noqa: BLE001
            return False
        return any(v is None for v in vals) and not any(v for v in vals if v is not None)

    def _discard_reread(self, img):
        """stage 2026-10-09 07:20:23 and 07:20:35: 'unreadable, assuming' on every slot dropped
        the paper science 5 needed. Re-read up to 3 times, 1 s apart (3 s cap, under the game's
        discard timer). Returns (frame, still_unreadable)."""
        for i in range(3):
            time.sleep(1.0)
            im = self.io.snapshot()
            if im is None:
                continue
            img = im
            if not self._hand_unread(img):
                self.note(f"discard: hand readable on re-read {i + 1}")
                return img, False
        return img, True

    # owner 2026-10-08: grain is the last card ever thrown away
    BLIND_DISCARD = ("brick", "lumber", "wool", "cloth", "coin", "ore", "paper", "grain")

    def _blind_discard(self, texts, d):
        """Hand unknown: click one card at a time in least-needed order and watch the dialog's
        'Choose N more' count; a card whose click doesn't lower it is empty, so move on."""
        def left(tx):
            m = re.search(r"(\d+)\s*more", " ".join(tx).lower())
            return int(m.group(1)) if m else 0
        n = left(texts)
        self.note(f"discard {n} with the hand unknown: clicking cards in order {self.BLIND_DISCARD}")
        for res in self.BLIND_DISCARD:
            x = UI["trade_give_row"]["x"][CARD_ORDER.index(res)]
            while n > 0:
                self.click((x, UI["trade_give_row"]["y"]), f"discard {res} (blind)")
                time.sleep(0.4)
                img = self.io.snapshot()
                if img is None:
                    return
                m = left([o["text"] for o in self.io.ocr(img)])
                if m >= n:
                    break
                n = m
            if n <= 0:
                break
        self.click(UI["trade_ok"], "confirm discard (check mark)")

    def on_roll(self, img, ocr, texts, info):
        self._in_play = True                          # setup is over once dice are rolled
        if not getattr(self, "_game_over_cleared", False) and not isinstance(getattr(self, "io", None), ReplayIO):
            self._game_over_cleared = True            # a game is being played: the last end screen is history
            try:
                self.GAME_OVER_MARK.unlink(missing_ok=True)
            except Exception:                          # noqa: BLE001
                pass
        self._live_snapshot()
        gl = self._glog()
        if gl is not None and self.gs is not None:
            gl.our_turn(self._glog_vp(), {k: v for k, v in self.gs.me().hand.items() if v})
        self._promote_fb_n = 0
        self._moved_knights = set()
        self._ships_this_turn = set()                 # Seafarers: ships built this turn can't move
        self._prompt_since = {}                        # every turn starts with a fresh prompt clock
        self._robber_tried = set()
        self.offers_this_turn = 0
        self._sheds = 0
        self._offers_made = set()
        if self._grain_asks_capped():
            self._offers_made.add(("want", "grain"))   # game 0959: nobody gives grain -- don't ask
        self._turn_n = self.__dict__.get("_turn_n", 0) + 1
        # 0.4.3 step 1: the turn clock runs from the first roll frame we captured (the game's timer
        # was already running); a second roll frame in the same turn (Alchemist) keeps it
        t0 = self.__dict__.get("_clock_t0")
        if not (t0 and self._our_turn_open() and time.time() - t0 < (self._turn_timer_s() or 100)):
            snap = self.__dict__.get("_snap_at") or 0
            self._clock_t0 = snap if 0 < time.time() - snap < 10 else time.time()
        self.turn_started = time.time()
        self._clock_turn = self.turn_started
        self._turn_begin()
        gs0 = self.__dict__.get("gs")
        tk = self.__dict__.get("_threat_key")
        if not (self.cfg.get("preplan", True) and gs0 is not None and tk is not None
                and tk == (self._board_key(gs0), tuple(sorted((k, v) for k, v in gs0.me().hand.items() if v)))):
            self._rank_threat_before_turn()          # 0.4.3: ranked on the rival's turn when nothing moved since
        if "alchemist" in self.play_progress("pre_roll", img, ocr):
            self.note("alchemist played: dice set by the card; re-reading before rolling (flow UNCONFIRMED)")
            return
        self.click(UI["dice_or_end_turn"], "roll dice")

    def _reload_model_if_promoted(self):
        """Between games: a gate (live_learn / train_loop) may have promoted a better model."""
        from .learning import MODEL
        try:
            mt = MODEL.stat().st_mtime
        except OSError:
            return
        if getattr(self, "_model_mtime", None) is None:
            self._model_mtime = mt
            return
        if mt != self._model_mtime:
            old = self.brain
            try:
                model = Model()
            except (OSError, ValueError) as e:
                # review 0.3.1: a torn/corrupt file keeps the old model; the mtime stays unseen so
                # the next poll tries again
                self.note(f"model reload failed: {type(e).__name__}: {e}")
                return
            self._model_mtime = mt
            self.model = model
            self.brain = Brain(self.model)
            self.brain.note = self.note
            self._brain_cfg()
            # owner 2026-10-08: a mid-game model swap dropped the bank ratios seen in the dialog,
            # so a board-read 3:1 came back -- the game's trade facts carry over to the new brain
            self.brain.ratio_seen = old.ratio_seen
            self.brain.merchant_res = old.merchant_res
            self.brain.fleet_resource = old.fleet_resource
            self.note(f"model reloaded: promoted model, {self.model.games} games")

    def _live_snapshot(self):
        """Owner 2026-10-06: live games train the model too (catanbot/live_learn.py)."""
        if isinstance(getattr(self, "io", None), ReplayIO) or getattr(self, "gs", None) is None:
            return
        try:
            from . import live_learn
            live_learn.snapshot(self.gs, self.brain, getattr(self.mode, "key", None), self.cfg.get("bot_name", ""))
        except Exception as e:                       # noqa: BLE001 -- learning never stops play
            self.note(f"live snapshot: {type(e).__name__}: {e}")

    def on_progress_card(self, img, ocr, texts, info):
        """game 6 2026-10-03: 10 min of "progress card" frames after our roll, no handler, then
        a forced restart. Keep the frame for later and play the turn as on the main screen."""
        self._save_lobby_frame(img, texts, "progress_card")
        # probe game 2 2026-10-04: a "receives 1 progress card" banner in our roll phase sent us to
        # on_main, which bank-traded with the trade button grey (failed twice): follow the buttons
        phase = turn_phase(img)
        if phase == "main":
            self.on_main(img, ocr, texts, info)
        elif phase == "roll":
            self.on_roll(img, ocr, texts, info)

    def on_main(self, img, ocr, texts, info):
        # owner can ask for the intro again: touch data/say_intro
        f = ROOT / "data" / "say_intro"
        if not isinstance(self.io, ReplayIO) and f.exists():
            f.unlink()
            self._intro_sent, self._saw_game_start = False, True
            self.chat_intro()
        self._in_play = True
        gs = self.ensure_state(img, texts)
        if gs is None:
            return
        waited = time.time() - getattr(self, "offer_sent", 0)
        if waited < self.offer_wait_s() + 20 and self._handle_responses(img):
            return
        if waited < self.offer_wait_s() and getattr(self, "offer_sent", 0) and find_text(ocr, r"no player wants to trade"):
            # owner 2026-10-06: the game itself says nobody wants it -> move on now, don't wait
            self.note("offer: no player wants to trade, moving on")
            self.click(UI["trade_cancel"], "close declined trade")
            self.trade_refused = getattr(self, "last_want", None)
            self.offer_sent = 0
            return
        if waited < self.offer_wait_s():
            # owner 2026-10-02: wait for the humans to answer our offer (>= 10 s)
            if int(waited) % 3 == 0:
                try:
                    d = ROOT / "data" / "trade_wait"; d.mkdir(exist_ok=True)
                    cv2.imwrite(str(d / f"{time.strftime('%H%M%S')}.png"), img)
                except Exception:                    # noqa: BLE001
                    pass
            return
        if self._confirm_accept(img, ocr):
            return
        if find_text(ocr, r"create an offer|no player wants to trade|^player trade$"):
            # the trade panel is still up (screen classed as main): close it so the hand bar shows
            self.click(UI["trade_cancel"], "close leftover trade panel")
            if getattr(self, "last_want", None) and not getattr(self, "trade_refused", None):
                self.trade_refused = self.last_want    # nobody took it: bank next (owner rule)
                self.last_want = None
            return
        if self._counter_offer(img, ocr):
            return
        self._ring_check(img)                        # 0.4.3: the timer border moves our clock forward
        if self.bar_hidden(img) and not self._turn_late():
            # live 2026-10-02: ended turns on hand={} while a dialog covered the bar -- wait for it
            if time.time() - getattr(self, "_hidden_note", 0) > 5:
                self._hidden_note = time.time()
                self.note("hand bar hidden by a dialog: waiting before planning")
            return
        if self._turn_past("hard") and time.time() - (self.__dict__.get("offer_sent") or 0) > self.offer_wait_s():
            # 0.4.3 step 1: seconds left on the game's timer -- End turn now; a turn the timer ends
            # loses whatever the last click was doing (live 3: 6 of 22 turns)
            self.note(f"turn clock: {self._turn_used_s():.0f} of {self._turn_timer_s():.0f} s used: end turn now")
            self._end_turn_click("end turn (clock)")
            return
        ending = self._turn_past("end")             # 0.4.3: builds only, then End turn
        if not ending and not self._turn_past("chat") and self._chat_poll_turn(img, ocr):
            return                                   # chat opened and closed: re-read the screen first
        steal_first = self.cfg.get("steal_first", True)
        self._phase("cards")
        if steal_first and not ending and self.tap_chase_robber(gs):
            # owner 2026-10-09 "it uses the cards that can steal more resources, the plan should be
            # made once all resources are in hand": the knight's robber chase steals first too
            return                              # the robber prompt comes next: on_place_robber
        # owner 2026-10-06: cards first -- every useful progress card is played before the turn
        # is planned (resources and cheaper builds widen the plan); each on_main pass offers the
        # held cards again, so ones that only get a use after a trade/build still play this turn
        if not ending and self.play_progress("main", img, ocr):
            return                              # cards changed hand/board: re-read, then plan
        if not steal_first and self.tap_chase_robber(gs):
            return                              # the robber prompt comes next: on_place_robber
        late = self._turn_late() or ending
        # live 2026-10-02 game 4: two player offers ate the turn budget, then plan=[] and the
        # road/improve it could afford never happened -- builds always run, trades only in time
        if self.turn_started and getattr(self, "_build_turn", None) != self.turn_started:
            self._build_turn, self._build_rounds = self.turn_started, 0
            self._failed_taps = set()
            self._tapped_spots, self._unaffordable = {}, set()
            self._misclick_retry = set()
            self._sold_turn = set()
            self._bank_blocked = set()
        hand_key = tuple(sorted((k, v) for k, v in gs.me().hand.items() if v))
        if getattr(self, "_plan_hand", None) not in (None, hand_key) and \
                (getattr(self, "_unaffordable", None) or getattr(self, "_tapped_spots", None)):
            # live 2026-10-08 19:11: activate was red (no grain), we banked the grain, then
            # "spot already tapped this turn" skipped the activate 4x and the grain sat unused --
            # new cards make a refused build worth one more tap
            # review 0.3.1: the red taps also leave _failed_taps (it feeds gs.blocked) -- grain from
            # a progress card (Wedding, Monopoly, ...) left that activate blocked all turn
            self._clear_red_taps()
            self._unaffordable, self._tapped_spots = set(), {}
        self._plan_hand = hand_key
        # solo 2026-10-06: builds the game refused this turn (or never offers) leave the plan
        # before it trades for them; live 2026-10-08 18:56: a red knight was planned 3x more
        gs.blocked = set(getattr(self, "_failed_taps", set())) | set(getattr(self, "_not_offered", set())) \
            | set(getattr(self, "_unaffordable", set()))
        if self._promote_skipped():
            # game 0834 09:09-09:10: knight builds planned behind a skipped promote (no basic piece
            # left) were refused 3 times -- a promote the game won't take this turn leaves the plan
            gs.blocked.add("promote")
        self._phase("plan")
        plan = self._preplan_for(gs)
        if plan is None:
            plan = self.brain.plan_turn(gs)
        danger = self.brain.city_in_danger(gs)
        if late:
            n0 = len(plan)
            plan = self._late_plan(gs, plan)
            # stage 2026-10-09: 17 of 24 turns ran late; log how much each late turn dropped
            self.note(f"late turn: {time.time() - getattr(self, 'turn_started', time.time()):.0f}s, "
                      f"dropped {n0 - len(plan)} actions")
        if getattr(self, "_build_rounds", 0) >= 3:
            plan = [a for a in plan if a.kind in ("bank_trade", "player_trade")]   # builds not landing
        me = gs.me()
        self.note(f"me: hand={ {k: v for k, v in me.hand.items() if v} } cities={len(me.cities)} "
                  f"knights={len(me.knights)} targets={self.brain._trade_targets(gs)} plan={[a.kind for a in plan]}")
        gl = self._glog(start=False)
        if gl is not None:
            # owner 2026-10-09: the plan and its reason into the ledger (hand: event file only)
            try:
                o = getattr(self.brain, "outlook", None)
                gl.plan([self._plan_item(a) for a in plan], self._plan_reason(gs, plan),
                        list(o.goals) if o is not None else None, dict(o.threats) if o is not None else None,
                        hand={k: v for k, v in me.hand.items() if v},
                        stance=getattr(self.brain, "stance", None))
            except Exception as e:                   # noqa: BLE001 -- the ledger never stops play
                self.note(f"game log: plan not logged: {type(e).__name__}")
        if ending:
            # 0.4.3 drop order: past turn_end_frac only the builds the hand pays for (and the bank
            # trades feeding them) -- no offers, no extra trades
            plan = [a for a in plan if a.kind != "player_trade"]
        if (not late or danger) and self.brain.grain_rush(gs):
            bt = self._bank_trade_for(gs, {"grain": 1})
            if bt:
                # solo 2026-10-06: barbarians close, knight inactive -> a bank set becomes grain now
                self.note(f"barbarians close, knight inactive: bank {bt.give} -> grain")
                self.bank_offer(bt.give, bt.get)
                return
        if danger and not ending and self._offer_allowed(urgent=True) \
                and not any(a.kind in ("knight", "activate") for a in plan):
            # stage 2026-10-09: "routes: player trade" was only logged. The city is in danger and
            # the bank can't make the knight: ask the table now, late or not, before any build
            if self._save_city_offer(gs, late):
                return
        trade = next((a for a in plan if a.kind == "player_trade"), None)
        if trade is not None and self._was_declined(trade.give, trade.get):
            self.note(f"plan offer {trade.give} -> {trade.get} was declined by all earlier: skip")
            trade = None
        if trade is None and not late and self._offer_allowed():
            trade = self._want_trade(gs)
        builds = [a for a in plan if a.kind not in ("bank_trade", "player_trade")]
        from .rules import COSTS, missing
        need = missing(dict(gs.me().hand), COSTS["city"])
        if not late and not builds and sum(need.values()) <= 2 and getattr(self, "_lost_cities", 0) \
                and gs.me().settlements and getattr(self, "_sheds", 0) < 4:
            # owner 2026-10-02: rebuild the city -- 3 spare wool at the harbor = 1 grain. Only when
            # nothing else is buildable and the city is 1-2 cards away (game 1: banked away the
            # lumber+wool of a planned ship, ship for 2 grain with 0 ore)
            for res in sorted(need, key=lambda r: r != "grain"):
                bt = self._bank_trade_for(gs, {res: 1})
                if bt and not set(bt.give) & set(COSTS["city"]):
                    self._sheds = getattr(self, "_sheds", 0) + 1
                    self.note(f"city lost: bank {bt.give} -> {bt.get} toward rebuilding it")
                    self.bank_offer(bt.give, bt.get)
                    return
        refused = getattr(self, "trade_refused", None)
        if refused and not late and not builds:
            # owner 2026-10-02: players refused -> use the bank/harbor (3:1) for what we need --
            # but only if a build is still short of it, and never paying with the build's own cards
            # (game 1: banked 3 ore for a 3rd grain while holding the full city, 3 ore + 2 grain)
            self.trade_refused = None
            from .rules import COSTS, missing
            need = {r for k in self.brain._trade_targets(gs) for r in missing(dict(gs.me().hand), COSTS.get(k, {}))}
            bt = self._bank_trade_for(gs, refused) if set(refused) & need else None
            if bt and set(bt.give) & {r for k in self.brain._trade_targets(gs) for r in COSTS.get(k, {})
                                      if set(missing(dict(gs.me().hand), COSTS[k])) <= set(refused)}:
                self.note(f"refused want {refused}: bank {bt.give} would spend the goal's own cards: skip")
                bt = None
            if bt:
                self.bank_offer(bt.give, bt.get)
                return
        first = next((a for a in plan if a.kind != "player_trade"), None)
        if first is not None and first.kind == "bank_trade" and getattr(self, "_sheds", 0) < 4:
            # the plan banks (harbor 3:1) toward a build: do it, then re-read and build
            batch = self._bank_batch(gs, plan) if self.cfg.get("bank_batch", True) else []
            if len(batch) >= 2:
                # owner 2026-10-09 21:09 "took too much time to trade": every planned bank trade
                # the hand pays for now, in one go -- no re-plan between them
                self._sheds = getattr(self, "_sheds", 0) + len(batch)
                self.note(f"plan: {len(batch)} bank trades at once: {[(a.give, a.get) for a in batch]}")
                self._phase("trade")
                self.bank_offer_batch([(a.give, a.get) for a in batch])
                return
            self._sheds = getattr(self, "_sheds", 0) + 1
            self.note(f"plan: bank {first.give} -> {first.get}")
            self._phase("trade")
            self.bank_offer(first.give, first.get)
            return
        if ending:
            trade = None
        if builds and trade is not None and self._offer_allowed() \
                and (self._trade_spares_builds(gs, trade, builds) or self._trade_beats_builds(gs, trade, builds)):
            # owner 2026-10-06 "it built a ship and then offered trade ... it should first do all
            # trades and then build": an offer that leaves the plan's cards alone goes first, so
            # the builds are planned again on the hand after it
            self.note(f"trade before building: offer {trade.give} -> {trade.get}, then {[a.kind for a in builds]}")
            self.last_want = trade.get
            self.__dict__.setdefault("_offers_made", set()).update(("want", r) for r in trade.get)
            self._offers_made.update(("ask", r, g) for r in trade.get for g in trade.give)
            self._phase("offer")
            self.offer(trade.give, trade.get)
            return
        if builds:
            # build first: what we can afford now beats waiting 20 s on a player offer
            self._build_rounds = getattr(self, "_build_rounds", 0) + 1
            self._phase("build")
            self._run_builds(gs, plan)
            return                                   # re-read the hand, then plan the rest
        if trade is not None and self._offer_allowed():
            self.last_want = trade.get
            self.__dict__.setdefault("_offers_made", set()).update(("want", r) for r in trade.get)
            self._offers_made.update(("ask", r, g) for r in trade.get for g in trade.give)
            self._phase("offer")
            self.offer(trade.give, trade.get)
            return
        if self._build_offered(gs):
            return
        if not late and self._bank_into_build(gs):
            return
        if self.cfg.get("pause_before_end"):          # calibration: leave the turn open
            self.note("pause_before_end: stopping before end turn")
            STOP.touch()
            return
        self._phase("end")
        if self._shed_hand(gs):                      # never end on 7+ cards, late or not
            return
        if not ending and self.play_progress("end_turn"):
            # owner 2026-10-06 "use all cards first and then make the plan": a card played (or
            # discarded) before End turn may bring cards or open a build -- re-read and plan again
            return
        self._end_turn_click("end turn")

    def _end_turn_click(self, why: str) -> None:
        self.click(UI["dice_or_end_turn"], why)
        self._turn_done(why)
        p = self._pend
        if p and p[0] == "end_turn":
            self._pend = (p[0], p[1], p[2], p[3] + 1, p[4])
            if p[3] + 1 == 4:
                self.probe.result("end_turn", False, "end turn clicked 4 times, still our main screen")
                self._probe("stuck_end_turn")
                self._pend = (p[0], p[1], p[2], p[3] + 1, True)
        else:
            self.probe.tried("end_turn")
            self._pend = ("end_turn", "main", time.time(), 1, False)
        self._end_turn_at = time.time()

    def _promote_skipped(self) -> bool:
        """The game refused a promotion (red X alone) this turn, or (cfg off) this game."""
        if getattr(self, "_promote_disabled", False):
            return True
        t = self.__dict__.get("_promote_skip_turn")
        return t is not None and t == self.__dict__.get("_turn_n", 0)

    def _still_legal(self, gs, before, skipped, a) -> bool:
        br = self.__dict__.get("brain")
        if br is None or not hasattr(gs, "players"):
            return True
        try:
            return br.still_legal(gs, before, skipped, a)
        except Exception as e:                       # noqa: BLE001 -- never block a build on a check
            self.note(f"plan check after a skip failed ({type(e).__name__}: {e})")
            return True

    def _run_builds(self, gs, plan):
        done = set()
        landed: list = []                            # 0.4.3 build_batch: builds placed in this pass
        expect = {k: v for k, v in gs.me().hand.items() if v} if gs is not None else {}
        for i, a in enumerate(plan):
            if a.kind in ("bank_trade", "player_trade"):
                continue
            # a landed build returns (re-read, re-plan) unless the batch goes on -- every build
            # before this one that did not land was skipped
            skipped = [b for b in plan[:i] if b.kind not in ("bank_trade", "player_trade")
                       and not any(b is x for x in landed)]
            if skipped and not self._still_legal(gs, plan[:i], skipped, a):
                # game 0834: a knight planned behind a promote that never happened (no basic piece left)
                self.note(f"{a.kind} at {a.target}: no longer legal after skipping {[s.kind for s in skipped]}; dropped")
                continue
            if a.kind in getattr(self, "_unaffordable", set()):
                # live 2026-10-04: the popup showed this option red (can't pay) this turn
                self.note(f"{a.kind}: popup said we can't afford it this turn; skipping")
                continue
            if a.target is not None and a.target in done:
                # live 2026-10-04: knight then activate at one corner tapped twice in one pass --
                # the second tap hit the first popup. Re-read and re-plan first.
                self.note(f"{a.kind} at {a.target}: spot already used this pass; re-plan first")
                continue
            xy = UI.get(f"build_{a.kind}")
            if xy is not None:
                self.click(xy, f"build {a.kind}")
            if a.kind == "improve":
                # solo 2026-10-03: the chart click didn't land and was repeated every 6 s all turn
                if ("improve", a.target) in getattr(self, "_failed_taps", set()):
                    self.note(f"improve {a.target} already tried this turn; skipping")
                    continue
                self.__dict__.setdefault("_failed_taps", set()).add(("improve", a.target))
                lv0 = dict(self.__dict__.get("_my_improve", {}))
                self.tap_improve(gs, a.target)
                if self.__dict__.get("_my_improve", {}) != lv0:
                    # live 2026-10-08: science 4 opened "Select position for Metropolis" and the
                    # next two taps hit that prompt (~6 s lost) -- re-read the screen first
                    self._build_rounds = 0           # review 0.3.1: a landed improvement is progress
                    return
                continue
            if a.target is None:
                self.note(f"wanted {a.kind}: no board spot, popup flow unknown (UNCALIBRATED)")
                continue
            if a.kind == "promote" and self._promote_skipped():
                continue
            if (a.kind, a.target) in getattr(self, "_failed_taps", set()) or \
                    (a.kind, a.target) in getattr(self, "_not_offered", set()):
                self.note(f"{a.kind} at {a.target} already failed this turn; skipping")
                continue
            done.add(a.target)
            if a.kind == "move_knight":
                if self.tap_move_knight(a):
                    self._build_rounds = 0           # review 0.3.1: a landed move is progress
                    return
                continue
            if a.kind == "move_ship":
                if self.tap_move_ship(a):
                    self._build_rounds = 0
                    return
                continue
            # owner 2026-10-02: tap the spot, see the popup, accept the right build
            if self.tap_build(a):
                self._build_rounds = 0               # builds are landing: keep building this turn
                landed.append(a)
                if self._batch_next(gs, plan, i, a, expect):
                    continue                         # hand read = hand - cost: next build, no re-plan
                return                               # board and hand changed: re-read, re-plan

    def _batch_next(self, gs, plan, i, a, expect: dict) -> bool:
        """0.4.3 step 3 (owner 2026-10-09 "the plan should be made once all resources are in hand,
        and then all resources used quickly"): after a landed build, go on to the plan's next build
        without a re-plan only when the screen is still our main screen with no prompt, the hand
        read is exactly the expected hand (hand before - cost, every readable count), and the
        expected hand pays the next build. Anything else = the old re-read and re-plan."""
        if not (self.__dict__.get("cfg") or {}).get("build_batch", True) or gs is None:
            return False
        from .moves import build_cost
        from .rules import missing
        try:
            cost = build_cost(gs, a.kind, a.target)
        except Exception:                            # noqa: BLE001
            return False
        for k, n in cost.items():
            expect[k] = expect.get(k, 0) - n
        nxt = next((b for b in plan[i + 1:] if b.kind not in ("bank_trade", "player_trade")), None)
        if nxt is None or nxt.kind in ("improve", "move_knight", "move_ship"):
            return False                             # nothing left, or a flow that re-reads anyway
        try:
            if missing({k: v for k, v in expect.items() if v > 0}, build_cost(gs, nxt.kind, nxt.target)):
                return False                         # a trade in between feeds it: re-plan after the read
        except Exception:                            # noqa: BLE001
            return False
        if self._turn_past("hard") or not self.__dict__.pop("_last_build_ok", False):
            return False                             # out of time, or the build was not seen landing
        img = self.io.snapshot()
        if img is None or turn_phase(img) != "main" or self.bar_hidden(img):
            self.note(f"batch: after {a.kind} the screen is not our main screen: re-read")
            return False
        ocr = self.io.ocr(img)
        if self._prompt_pending(ocr):
            self.note(f"batch: after {a.kind} a prompt is up: re-read")
            return False
        seen = self._hand_now(img)
        want = {k: v for k, v in expect.items() if v > 0}
        if not seen or any(seen.get(k, 0) != want.get(k, 0) for k in set(seen) | set(want) if k in seen):
            self.note(f"batch: after {a.kind} hand {seen} != expected {want}: re-plan")
            return False
        self._tcount("batched builds")
        self.note(f"batch: {a.kind} landed, hand as expected: {nxt.kind} next without a re-plan")
        gs.me().hand = dict(want)
        return True

    def _commercial_harbor(self):
        """Solo 2026-10-03: an opponent's Commercial Harbor asks us for one commodity; the bot
        read it as our main screen and sat there 55 min. Give the one we value least (owner:
        paper > coin > cloth); if the dialog is still up a few seconds later, try the next one."""
        gs = getattr(self, "gs", None)
        hand = gs.me().hand if gs is not None else {}
        # stage 2026-10-09 07:25:15: cloth went here with the trade metropolis at 4 -- a held
        # commodity of a level-4 metropolis track is given only when it is the last one held
        locked = self.brain.reserved_commodities(gs) if gs is not None else set()
        order = sorted(("cloth", "coin", "paper"), key=lambda c: (hand.get(c, 0) == 0, c in locked,
                                                                   ("cloth", "coin", "paper").index(c)))
        n = getattr(self, "_harbor_try", (0, 0))
        k = n[1] + 1 if time.time() - n[0] < 30 else 0
        self._harbor_try = (time.time(), k)
        c = order[k % 3]
        self.click((UI["trade_give_row"]["x"][CARD_ORDER.index(c)], UI["trade_give_row"]["y"]),
                   f"commercial harbor: give {c}")
        time.sleep(0.5)
        self.click(UI["trade_ok"], "commercial harbor: ok")
        time.sleep(1.5)

    POPUP_WORDS = {"settlement": r"settle", "city": r"city|upgrade", "road": r"road", "ship": r"ship",
                   "knight": r"[bp][uoa]i?l?d\s*k|knight|recruit",   # OCR: "Build knighit", solo "Baild Khiai", "puildkaias"
                   "activate": r"(?<!in)activ|[a-z]{1,3}ve?te\s*k[mnh]",   # not "(inactive)"; OCR "Olvete Kmighi"
                   "promote": r"promot|upgrade\s*kn",
                   "wall": r"wall", "move_knight": r"move", "move_ship": r"move"}
    # improvement tracks: the colour/commodity/level-3 building names a menu may show
    IMPROVE_WORDS = {"science": r"science|paper|aqueduct|green", "politics": r"politic|coin|fortress|blue",
                     "trade": r"trade|cloth|trading\s*house|yellow"}

    def tap_improve(self, gs, track: str):
        """City improvement. UNCONFIRMED flow (no capture yet): tap one of our cities, read the
        popup, and take the option naming the track (or 'improve'/'develop' first, then the
        track). Never clicks anything it can't read; screenshots go to data/popups."""
        me = gs.me()
        if not me.cities:
            return
        self.probe.tried(f"improve_{track}")
        res = self._improve_via_city(gs, track)
        if res == "refused":
            self._refused(f"improve_{track}")
            return
        if res is not None or getattr(self, "_improve_seen", False):
            self.probe.result(f"improve_{track}", res, "city popup option did not take")
            if res is True:
                self._improved(track)
            return
        chart = UI.get(f"improve_chart_{track}")
        if chart is not None and self._tap_improve_chart(track, chart):
            self._improved(track)
            return
        lat = self.br.lattice
        v = sorted(me.cities)[0]
        self.click(lat.vertex_px(v), f"tap city {v} for improve {track}")
        for step in range(2):
            time.sleep(0.8)
            img = self.io.snapshot()
            if img is None:
                return
            ocr = self.io.ocr(img)
            try:
                cv2.imwrite(str(ROOT / "data" / "popups" / f"{time.strftime('%H%M%S')}_improve{step}.png"), img)
            except Exception:                            # noqa: BLE001
                pass
            self.note(f"improve popup texts: {[o['text'] for o in ocr][:20]}")
            opt = find_text(ocr, self.IMPROVE_WORDS[track])
            if opt:
                self.click((opt["x"] + opt["w"] / 2, opt["y"] + opt["h"] / 2 - (45 if step == 0 else 0)),
                           f"improve {track}", ocr)
                time.sleep(0.8)
                img = self.io.snapshot()
                if img is not None and find_text(self.io.ocr(img), r"confirm"):
                    self.click(UI["setup_confirm"], f"confirm improve {track}")
                return
            gate = find_text(ocr, r"improv|develop")
            if not gate or step:
                break
            self.click((gate["x"] + gate["w"] / 2, gate["y"] + gate["h"] / 2 - 45), "open improvements", ocr)
        self.note(f"improve {track}: no option found (UNCONFIRMED flow); leaving it")
        self._close_popup()

    IMPROVE_COMMODITY = {"trade": "cloth", "politics": "coin", "science": "paper"}

    def _improve_ok(self, img, ocr=None, why="improve (OK)"):
        """'City improvements / Do you want to improve your city?' CANCEL / OK: press OK."""
        if img is None:
            return False
        ocr = self.io.ocr(img) if ocr is None else ocr
        xy = self._dialog_ok_xy(ocr)
        if xy is None:
            return False
        self.click(xy, why, ocr)
        time.sleep(0.8)
        return True

    @staticmethod
    def _dialog_ok_xy(ocr):
        """Where OK sits on the 'Do you want to improve your city?' dialog, or None when it isn't
        open. Solo 2026-10-06 02:32: OCR read OK as Cyrillic 'ОK' (or not at all), the dialog
        stayed open and ate the next bank trades and knight taps. OK mirrors CANCEL across the
        dialog's title ('City improvements')."""
        if not find_text(ocr, r"want to improve"):
            return None
        ok = next((o for o in ocr if o["text"].strip().upper().translate(str.maketrans("ОКО", "OKO")) == "OK"), None)
        if ok:
            return (ok["x"] + ok["w"] / 2, ok["y"] + ok["h"] / 2)
        cancel = find_text(ocr, r"^cancel$")
        title = find_text(ocr, r"city improvements")
        if cancel is None:
            return None
        cx = title["x"] + title["w"] / 2 if title else 654
        return (2 * cx - (cancel["x"] + cancel["w"] / 2), cancel["y"] + cancel["h"] / 2)

    def _improve_levels(self) -> dict:
        """Our city improvement levels this game, counted from the improvements that landed."""
        lv = self.__dict__.setdefault("_my_improve", {t: 0 for t in IMPROVEMENT_TRACKS})
        return {t: lv.get(t, 0) for t in IMPROVEMENT_TRACKS}

    def _improved(self, track: str):
        lv = self._improve_levels()
        self._my_improve = dict(lv, **{track: min(5, lv[track] + 1)})
        self.note(f"improve {track}: level {self._my_improve[track]}")
        # owner 2026-10-06: 3 paper bought science 1 but not 2 -- a landed level frees the track
        # for the next level this turn (the once-per-turn mark is only for taps that didn't land)
        getattr(self, "_failed_taps", set()).discard(("improve", track))
        self._save_game_flags()

    def _improve_via_city(self, gs, track: str):
        """Audit 2026-10-04 gap 5: a tap on our city offers wall/trade/politics/science rings.
        Click the track's ring (green only), answer OK/confirm, verify by the commodity leaving
        the hand. True/False = landed or not, None = no city popup offered the track."""
        lat = self.br.lattice
        cities = sorted(set(gs.me().cities) | set(getattr(self, "_my_cities", set())))
        todo, reset = list(cities[:2]), False
        while todo:
            v = todo.pop(0)
            self._close_leftover()
            self._wait_announcement()
            xy = lat.vertex_px(v)
            before = self._hand_now()
            self.click(xy, f"tap city {v} for improve {track}")
            time.sleep(0.7)
            img = self.io.snapshot()
            if img is None:
                return None
            rings = self._popup_options(img, near=xy)
            ring = next((r for r in rings if r["kind"] == track), None)
            if any(r["kind"] in self.WALL_OR_IMPROVE for r in rings):
                self._improve_seen = True             # the city popup works: no chart fallback
            if ring is None and any(r["kind"] == "city" for r in rings):
                # live 2026-10-08: after the city was pillaged a reset re-read it as a city; 8 improve
                # taps and 3 walls failed on the settlement. "Build city" there = it is a settlement
                self._settlement_seen(gs, v)
                self._close_leftover(img)
                continue
            if ring is None:
                self.note(f"improve {track}: city {v} popup options {[(r['kind'], r['colour']) for r in rings]}")
                self._close_leftover(img)
                if not reset and not any(r["kind"] in self.WALL_OR_IMPROVE for r in rings):
                    # owner 2026-10-06 "clicks on something else by mistake -> reset and try
                    # again": no city popup means the tap missed; drag a panned board back, retry
                    reset, self._recenter_t = True, 0
                    if self._recenter_board(self.io.snapshot()):
                        self.note(f"improve {track}: tap on city {v} missed; board reset, tapping again")
                        time.sleep(0.8)
                        todo.insert(0, v)
                continue
            if ring["colour"] == "red":
                self.note(f"improve {track}: red option (cannot afford / max level)")
                lvl, com = self._improve_levels()[track], self.IMPROVE_COMMODITY[track]
                if lvl < 5 and (before or {}).get(com, 0) >= lvl + 1:
                    # we hold the price we think it costs: the track is a level further than we knew
                    self._my_improve[track] = lvl + 1
                    self.note(f"improve {track}: affordable at level {lvl} but refused -> level {lvl + 1}")
                    self._save_game_flags()
                self.__dict__.setdefault("_unaffordable", set()).add(f"improve_{track}")
                self._close_leftover(img)
                return "refused"
            self.click((ring["x"], ring["y"]), f"improve {track} (city ring)")
            time.sleep(0.8)
            img = self.io.snapshot()
            if not self._improve_ok(img) and img is not None and find_text(self.io.ocr(img), r"confirm"):
                self.click(UI["setup_confirm"], f"confirm improve {track}")
                time.sleep(0.8)
            img = self.io.snapshot()
            com = self.IMPROVE_COMMODITY[track]
            cost = self._improve_levels()[track] + 1 - (getattr(self, "_crane_track", None) == track)
            ok = self._hand_moved(before, self._hand_now(img), {com: cost}, {}) if cost else True
            if ok is False:
                self._probe(f"improve_failed_{track}", img, note={"city": str(v), "before": before})
            self._close_leftover(img)
            self.note(f"improve {track}: " + ("landed" if ok else "not seen" if ok is False else "UNVERIFIED"))
            return ok
        return None

    def _settlement_seen(self, gs, v):
        """The game offered "build city" at our spot v: it is a settlement, not a city."""
        was = v in getattr(self, "_my_cities", set()) or (gs is not None and v in gs.me().cities)
        self.__dict__.setdefault("_not_city", set()).add(v)
        self.__dict__.setdefault("_my_cities", set()).discard(v)
        if gs is not None:
            me = gs.me()
            if v in me.cities:
                me.cities.discard(v)
                me.settlements.add(v)
            if was and not me.cities and getattr(self.mode, "start_with_city", False):
                self._lost_cities = max(1, getattr(self, "_lost_cities", 0))
        if was:
            self.note(f"spot {v} offers 'build city': it is our settlement, not a city")
            self._save_game_flags()

    def _refused(self, kind):
        """The game refused (red option): a planning miss, not a click miss. Not a failure."""
        row = self.probe._row(kind)
        row["refused"] = row.get("refused", 0) + 1
        row["tries"] -= 1

    def _tap_improve_chart(self, track: str, xy) -> bool:
        """Our three improvement charts sit above the hand bar (yellow trade, blue politics,
        green science; seen in game 2/4 frames). UNCONFIRMED: tapping the chart offers the
        next level. Clicks only an option it can read; True when it clicked one."""
        self.click(xy, f"tap {track} improvement chart")
        time.sleep(0.8)
        img = self.io.snapshot()
        if img is None:
            return False
        ocr = self.io.ocr(img)
        try:
            cv2.imwrite(str(ROOT / "data" / "popups" / f"{time.strftime('%H%M%S')}_improve_chart.png"), img)
        except Exception:                                # noqa: BLE001
            pass
        self.note(f"improve chart texts: {[o['text'] for o in ocr][:20]}")
        # short labels only: the chart tooltips ("...third level of city improvements...") match too
        # solo 2026-10-03: the chart tap opens "City improvements / Do you want to improve your
        # city?" with CANCEL and OK -- the title matched "improve" and was clicked instead of OK
        if find_text(ocr, r"want to improve"):
            before = self._hand_now(img)
            if not self._improve_ok(img, ocr, f"improve {track} (OK)"):
                self._close_popup()
                return False
            com = self.IMPROVE_COMMODITY[track]
            cost = self._improve_levels()[track] + 1 - (getattr(self, "_crane_track", None) == track)
            ok = self._hand_moved(before, self._hand_now(), {com: cost}, {}) if cost else True
            self.note(f"improve {track} (chart OK): " + ("landed" if ok else "not seen" if ok is False else "UNVERIFIED"))
            return ok is not False
        # solo 2026-10-03: "BUILD COSTS" (the cost card, bottom left) matched "build" and got clicked
        opt = find_text([o for o in ocr if len(o["text"].strip()) <= 20 and not re.search(r"build\s*cost", o["text"], re.I)],
                        r"improve|upgrade|build|buy|purchase")
        if not opt:
            self._close_popup()
            return False
        before = self._hand_now(img)
        self.click((opt["x"] + opt["w"] / 2, opt["y"] + opt["h"] / 2), f"improve {track} (chart)", ocr)
        time.sleep(0.8)
        img = self.io.snapshot()
        if img is not None and find_text(self.io.ocr(img), r"confirm"):
            self.click(UI["setup_confirm"], f"confirm improve {track}")
        # review 0.3.1: this path counted a level it never checked -- check the hand like the OK path
        com = self.IMPROVE_COMMODITY[track]
        cost = self._improve_levels()[track] + 1 - (getattr(self, "_crane_track", None) == track)
        ok = self._hand_moved(before, self._hand_now(), {com: cost}, {}) if cost else True
        self.note(f"improve {track} (chart label): " + ("landed" if ok else "not seen" if ok is False else "UNVERIFIED"))
        return ok is not False

    def _close_popup(self):
        # Escape does nothing in Catan Universe (live 2026-10-02): click a readable close/cancel
        img = self.io.snapshot()
        hit = find_text(self.io.ocr(img), r"^(cancel|close|back|no)$") if img is not None else None
        if hit:
            self.click((hit["x"] + hit["w"] / 2, hit["y"] + hit["h"] / 2), "close popup")

    CLOSE_SPOT = (500, 560)          # doc: open sea west of the board -- a click there closes a build popup
    WALL_OR_IMPROVE = ("wall", "trade", "politics", "science")

    def _close_spot(self):
        """Open sea nearest the doc's close spot (500,560): a tap there selects nothing."""
        br = getattr(self, "br", None)
        try:
            sea = [br.lattice.hex_px(*c) for c, h in br.board.hexes.items() if h.terrain == "sea"]
        except Exception:                            # noqa: BLE001
            sea = []
        if not sea:
            return self.CLOSE_SPOT
        x, y = min(sea, key=lambda p: math.hypot(p[0] - self.CLOSE_SPOT[0], p[1] - self.CLOSE_SPOT[1]))
        return (int(x), int(y))

    def _popup_options(self, img, near=None) -> list[dict]:
        """Build-popup option icons (ring + icon template); only rings whose icon is known
        unless near= is given (the row of the spot we just tapped)."""
        from .vision.screens import icon_kind, option_rings
        rings = option_rings(img, near=near)
        for r in rings:
            r["kind"], r["icon"] = icon_kind(img, r["x"], r["y"])
        self._improve_by_order(rings)
        return [r for r in rings if r["kind"] or near is not None]

    def _improve_by_order(self, rings) -> None:
        """Game 0834 08:34:44: the city popup read [wall, trade, science, science] and politics was
        never tapped. 4 rings, every read kind wall/improve, a kind repeated or unread -> it is the
        city popup: label by position, left to right wall / trade / politics / science."""
        if len(rings) != 4 or not (self.__dict__.get("cfg") or {}).get("improve_by_order", True):
            return
        kinds = [r["kind"] for r in rings]
        known = [k for k in kinds if k]
        if len(known) < 2 or not all(k in self.WALL_OR_IMPROVE for k in known):
            return
        if len(set(known)) == len(known) and None not in kinds:
            return
        for r, k in zip(sorted(rings, key=lambda r: r["x"]), self.WALL_OR_IMPROVE):
            r["kind"] = k
        if time.time() - self.__dict__.get("_relabel_noted", 0) > 5:
            self._relabel_noted = time.time()
            self.note(f"popup: improve icons relabelled by order (read {kinds})")

    def _close_leftover(self, img=None) -> bool:
        """Live 2026-10-04: a popup left open took the next tap (04:57:20 pressed "Build city"
        16 px from the ring instead of the planned spot). Close any open build popup first."""
        if img is None:
            img = self.io.snapshot()
        if img is None:
            return False
        # solo 2026-10-06 02:32: an improve dialog left open took two bank trades and three knight
        # taps; it has no rings, only CANCEL / OK
        ocr = self.io.ocr(img)
        cancel = find_text(ocr, r"^cancel$")
        if cancel is not None and self._dialog_ok_xy(ocr) is not None:
            self.click((cancel["x"] + cancel["w"] / 2, cancel["y"] + cancel["h"] / 2), "close leftover improve dialog")
            time.sleep(0.5)
            return True
        if not self._popup_options(img):
            return False
        self.click(self._close_spot(), "close leftover popup")
        time.sleep(0.5)
        img = self.io.snapshot()
        if img is not None and self._popup_options(img):
            self.note("popup still open after the close click")
        return True

    def _verify_build(self, kind, t, before: dict, img):
        """True when the build is seen (cost left the hand, or our piece is on the spot), False when
        neither moved, None when nothing could be read."""
        from .rules import COSTS
        cost = COSTS.get(kind, {})
        hand = getattr(self, "hand", None)
        seen = []
        if img is not None and hand is not None and before:
            try:
                after = hand.read(img)
            except Exception:                            # noqa: BLE001
                after = {}
            d = [(after.get(k), before.get(k), n) for k, n in cost.items()]
            d = [(a, b, n) for a, b, n in d if a is not None and b is not None]
            if d:
                seen.append(all(a <= b - n for a, b, n in d))
        gs = getattr(self, "gs", None)
        if img is not None and kind in ("road", "ship", "settlement", "knight", "activate") \
                and gs is not None and hasattr(gs, "players") and not self._popup_options(img):
            try:
                from .vision.board_reader import detect_pieces
                me = gs.me()
                pc = detect_pieces(img, self.br, [p.color for p in gs.players])
                if kind in ("road", "ship"):
                    seen.append((pc.roads.get(t) or pc.ships.get(t)) == me.color)
                elif kind == "settlement":
                    seen.append(pc.blds.get(t) == me.color)
                else:
                    k = pc.knights.get(t)
                    seen.append(bool(k) and k[0] == me.color and (kind == "knight" or k[1]))
            except Exception as e:                       # noqa: BLE001
                self.note(f"verify: piece read failed ({e})")
        if not seen:
            return None
        return any(seen)

    @staticmethod
    def _late_plan(gs, plan):
        """Turn over budget: player offers are dropped (they ate the time, live 2026-10-02).
        Stage 2026-10-09 06:35: dropping the bank trades too left "4 ore -> brick" out and the road
        it paid for in -- red 3 times, and every settlement waiting on its banked grain died the
        same way. A bank trade (one quick dialog) stays when a later build uses it; trailing
        trades go, and so does any build the hand can no longer pay for."""
        from .moves import build_cost
        from .rules import missing
        last_build = max((i for i, a in enumerate(plan) if a.kind not in ("bank_trade", "player_trade")),
                         default=-1)
        hand = dict(gs.me().hand)
        out = []
        for i, a in enumerate(plan):
            if a.kind == "player_trade":
                continue
            if a.kind == "bank_trade":
                if i > last_build or missing(hand, a.give):
                    continue
                for r, n in a.give.items():
                    hand[r] = hand.get(r, 0) - n
                for r, n in a.get.items():
                    hand[r] = hand.get(r, 0) + n
                out.append(a)
                continue
            try:
                cost = build_cost(gs, a.kind, a.target)
            except Exception:                        # noqa: BLE001
                cost = {}
            if missing(hand, cost):
                continue
            for r, n in cost.items():
                hand[r] = hand.get(r, 0) - n
            out.append(a)
        return out

    def _record_build(self, kind, t):
        exp = getattr(self, "_trade_expect", None)
        if exp:
            # stage 2026-10-09 04:59: cloth->grain landed, the activation spent the grain, and the
            # post-trade hand put it back ("grain read 0 just after the trade, using 1"): the next
            # activation and the settlement both came up red. A build spends the expected cards.
            from .rules import COSTS
            cost = COSTS.get(kind)
            if cost is None:
                # improvements spend commodities by level: stop correcting those, keep the rest
                from .board import COMMODITIES
                self._trade_expect = (exp[0], exp[1], {k for k in exp[2] if k not in COMMODITIES})
            else:
                h = dict(exp[1])
                for r, n in cost.items():
                    h[r] = max(0, h.get(r, 0) - n)
                self._trade_expect = (exp[0], h, exp[2])
        if kind == "settlement" and self.brain.ratio_seen:
            # owner 2026-10-08: harbours don't change once owned except when we build a new one.
            # Only a new settlement on a harbour corner can lower a ratio; a city upgrades a corner
            # we already hold (live 21:53:34: the city at (5,1) wiped the dialog's all-4 ratios
            # and 21:54:12 planned a 3:1 from the board read)
            gs = getattr(self, "gs", None)
            hb = gs.board.harbor_at(tuple(t)) if gs is not None and t is not None else None
            seen = self.brain.ratio_seen
            if hb == "3:1":
                for r in [r for r, n in seen.items() if n >= 4]:
                    seen.pop(r)
            elif hb:
                seen.pop(hb, None)
            self._save_game_flags()
        if kind == "city":
            self.__dict__.setdefault("_my_cities", set()).add(t)
            if getattr(self, "_lost_cities", 0):
                self._lost_cities -= 1                  # rebuilt: the start-city guess holds again
                self._save_game_flags()
        elif kind == "settlement":
            self.__dict__.setdefault("_my_blds", set()).add(t)
        elif kind == "wall":
            self._record_wall(t)
        elif kind in ("knight", "activate"):
            self.__dict__.setdefault("_my_knights", {})[t] = kind == "activate"
            if kind == "activate":
                self.__dict__.setdefault("_activated_at", {})[t] = time.time()
            if kind == "knight":
                self.__dict__.setdefault("_knight_levels", {}).pop(t, None)
        elif kind == "promote":
            lv = self.__dict__.setdefault("_knight_levels", {})
            lv[t] = min(3, lv.get(t, 1) + 1)
        if kind in ("knight", "activate", "promote"):
            self._save_game_flags()
        elif kind == "road":
            self.__dict__.setdefault("_my_roads", set()).add(t)
        elif kind == "ship":
            self.__dict__.setdefault("_my_ships", set()).add(t)
            self.__dict__.setdefault("_ships_this_turn", set()).add(t)

    def _record_wall(self, v):
        """A wall we built (wall build or Engineer): kept for the game so the live discard limit
        counts it (+2 per wall; was never recorded live before 2026-10-06). Max 3."""
        walls = self.__dict__.setdefault("_my_walls", set())
        if v is None or v in walls or len(walls) >= 3:
            return
        walls.add(tuple(v))
        gs = getattr(self, "gs", None)
        if gs is not None:
            gs.me().walls.add(tuple(v))
        self._save_game_flags()

    EDGE_KINDS = {"road", "ship"}

    def _misclick(self, a, t, kinds, img) -> bool:
        """Owner 2026-10-06: "if it wants to do something and clicks on something else by mistake,
        it should reset and try again". Solo 02:05: activate at (2,-2) opened a road popup -- the
        tap hit an edge, not the knight's corner. A popup of the other spot type (edge options for
        a corner action, or the reverse) is a missed tap: close it, drag the board back if it moved,
        and leave the spot open so the next plan taps it again (once per spot per turn)."""
        kinds = {k for k in kinds if k}
        if not kinds:
            return False
        want_edge = a.kind in self.EDGE_KINDS
        if want_edge == bool(kinds & self.EDGE_KINDS) and (not want_edge or kinds <= self.EDGE_KINDS):
            return False
        retried = self.__dict__.setdefault("_misclick_retry", set())
        if (a.kind, t) in retried:
            return False
        retried.add((a.kind, t))
        self.note(f"missed tap: {a.kind} at {t} opened {sorted(kinds)}; resetting to try again")
        self.probe.result(f"build_{a.kind}", False, f"missed tap at {t} ({sorted(kinds)})")
        self._close_leftover(img)
        self._recenter_t = 0                          # a missed tap is the sign of a pan: check now
        self._recenter_board(self.io.snapshot())
        self.lattice = self.scan = None               # re-read the board before the retry
        self.__dict__.get("_tapped_spots", {}).pop(t, None)
        return True

    def tap_move_knight(self, a) -> bool:
        """Owner 2026-10-06: push a rival knight off the canal / step onto a canal corner.
        UNCONFIRMED flow: tap our knight, take the "Move knight" ring, tap the destination corner
        (a displaced knight's owner then picks where it goes). Every frame goes to data/popups;
        after 2 misses in a game the move is not tried again."""
        src, dst = a.target
        failed = self.__dict__.setdefault("_failed_taps", set())
        failed.add(("move_knight", a.target))
        if getattr(self, "_move_misses", 0) >= 2:
            return False
        lat = self.br.lattice
        self._close_leftover()
        self.probe.tried("move_knight")
        xy = lat.vertex_px(src)
        self.click(xy, f"tap knight {src} to move it to {dst}")
        time.sleep(0.7)
        img = self.io.snapshot()
        if img is None:
            return False
        d = ROOT / "data" / "popups"
        stamp = time.strftime('%H%M%S')
        try:
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / f"{stamp}_move_knight.png"), img)
        except Exception:                            # noqa: BLE001
            pass
        from .vision.screens import label_under
        ocr = self.io.ocr(img)
        try:
            ocr = ocr + ocr_near(self.io.ocr, img, xy)
        except Exception:                            # noqa: BLE001
            pass
        rings = self._popup_options(img, near=xy)
        for r in rings:
            lab, _ = label_under(ocr, r)
            r["kind"] = lab if lab == "move" else (r["kind"] or lab)
        self.note(f"move knight popup: {[(r['kind'], r['colour']) for r in rings]}")
        ring = next((r for r in rings if r["kind"] in ("move", "move_knight")), None)
        if ring is None:
            hit = find_popup_option(ocr, "move_knight", self.POPUP_WORDS["move_knight"])
            if hit:
                ring = {"x": hit["x"] + hit["w"] / 2, "y": hit["y"] + hit["h"] / 2 - 45, "colour": "green"}
        if ring is None or ring.get("colour") == "red":
            self.note(f"move knight: no usable move option at {src}")
            self._move_misses = getattr(self, "_move_misses", 0) + 1
            self._probe("nooption_move_knight", img, note={"target": str(a.target)})
            self._close_leftover(img)
            return False
        self.click((ring["x"], ring["y"]), "popup move knight")
        time.sleep(0.8)
        img = self.io.snapshot()
        try:
            cv2.imwrite(str(d / f"{stamp}_move_knight_pick.png"), img)
        except Exception:                            # noqa: BLE001
            pass
        self.click(lat.vertex_px(dst), f"move knight to {dst}")
        time.sleep(0.9)
        img = self.io.snapshot()
        if img is None:
            return False
        try:
            cv2.imwrite(str(d / f"{stamp}_move_knight_after.png"), img)
        except Exception:                            # noqa: BLE001
            pass
        if find_text(self.io.ocr(img), r"confirm"):
            self.click(UI["setup_confirm"], "confirm move knight")
            time.sleep(0.8)
        # the knight acted: it stands at dst, inactive, same level
        self.__dict__.setdefault("_moved_knights", set()).add(dst)
        kn = self.__dict__.setdefault("_my_knights", {})
        kn.pop(src, None)
        kn[dst] = False
        lv = self.__dict__.setdefault("_knight_levels", {})
        if src in lv:
            lv[dst] = lv.pop(src)
        self._save_game_flags()
        self.probe.result("move_knight", None, f"{src}->{dst} (UNVERIFIED)")
        self.note(f"moved knight {src} -> {dst} (UNVERIFIED: frames in data/popups/{stamp}_move_knight*)")
        self._close_leftover()
        return True

    def tap_move_ship(self, a) -> bool:
        """Owner 2026-10-09 "build into the game the way to move ships". UNVERIFIED flow (no
        capture of the live UI yet; modelled on tap_move_knight): tap the open-end ship, take the
        "Move" option of its popup, tap the free sea edge it goes to, confirm if asked. Every
        frame goes to data/popups/<time>_move_ship*.png and the probe scorecard; after 2 misses
        in a game the move is not tried again. Off with config "ship_move": False."""
        src, dst = a.target
        failed = self.__dict__.setdefault("_failed_taps", set())
        failed.add(("move_ship", a.target))
        if not self.cfg.get("ship_move", True) or self._ship_move_blocked():
            return False
        if self.__dict__.get("_ship_moved_turn") == self.__dict__.get("_turn_n", 0):
            return False                             # one ship move a turn
        lat = self.br.lattice
        self._close_leftover()
        self.probe.tried("move_ship")
        xy = lat.edge_px(src)
        self.click(xy, f"tap ship {src} to move it to {dst}")
        time.sleep(0.7)
        img = self.io.snapshot()
        if img is None:
            return False
        d = ROOT / "data" / "popups"
        stamp = time.strftime('%H%M%S')

        def shot(name, im):
            try:
                d.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(d / f"{stamp}_{name}.png"), im)
            except Exception:                        # noqa: BLE001
                pass
        shot("move_ship", img)
        from .vision.screens import label_under
        ocr = self.io.ocr(img)
        if find_text(ocr, self.SHIP_PLACE_RX):
            # solo 2026-10-09 23:04:00 (probe 230404_009): the tap lifts the open-end ship straight
            # into "Select position for Ship" -- there is no popup; tap the free sea edge
            self.note("move ship: the tap went straight to 'Select position for Ship' (no popup)")
            return self._ship_move_place(a, d, stamp, shot)
        try:
            ocr = ocr + ocr_near(self.io.ocr, img, xy)
        except Exception:                            # noqa: BLE001
            pass
        rings = self._popup_options(img, near=xy)
        for r in rings:
            lab, _ = label_under(ocr, r)
            r["kind"] = lab if lab == "move" else (r["kind"] or lab)
        self.note(f"move ship popup: {[(r['kind'], r['colour']) for r in rings]}")
        ring = next((r for r in rings if r["kind"] in ("move", "move_ship")), None)
        if ring is None:
            hit = find_popup_option(ocr, "move_ship", self.POPUP_WORDS["move_ship"])
            if hit:
                ring = {"x": hit["x"] + hit["w"] / 2, "y": hit["y"] + hit["h"] / 2 - 45, "colour": "green"}
        if ring is None or ring.get("colour") == "red":
            self.note(f"move ship: no usable move option at {src}")
            self._ship_move_miss(f"no move option at {src}", "nooption_move_ship", img, a)
            self._close_leftover(img)
            return False
        self.click((ring["x"], ring["y"]), "popup move ship")
        time.sleep(0.8)
        img = self.io.snapshot()
        if img is not None:
            shot("move_ship_pick", img)
        return self._ship_move_place(a, d, stamp, shot)

    SHIP_PLACE_RX = r"select ?position ?for ?ship"

    def _ship_move_blocked(self) -> bool:
        """Config ship_move_max_misses (owner 2026-10-09): after that many failed ship moves in
        a game, no more tries -- and the brain stops planning them (0 = never blocked)."""
        cap = int(self.cfg.get("ship_move_max_misses", 2) or 0)
        return bool(cap) and getattr(self, "_ship_move_misses", 0) >= cap

    def _ship_move_miss(self, why, probe_kind, img, a):
        """One failed ship move: count it, and at the cap take ship moves out of the plan for the
        rest of the game (solo 2026-10-09: move_ship sat in the plan 23:03-23:41 after 0/2 and
        spun "already failed this turn; skipping" several times a turn)."""
        self._ship_move_misses = getattr(self, "_ship_move_misses", 0) + 1
        self.probe.result("move_ship", False, why)
        if img is not None:
            self._probe(probe_kind, img, note={"target": str(a.target)})
        if self._ship_move_blocked():
            self.brain.ship_moves = False
            self.note(f"move_ship: {self._ship_move_misses} misses this game; blocked for the rest "
                      f"of the game (ship_move_max_misses)")

    def _ship_move_place(self, a, d, stamp, shot) -> bool:
        """Ship lifted ("Select position for Ship"): tap the destination sea edge, confirm if
        asked. Still on the placement prompt after the tap = a miss; cancel it."""
        src, dst = a.target
        lat = self.br.lattice
        self.click(lat.edge_px(dst), f"move ship to {dst}")
        time.sleep(0.9)
        img = self.io.snapshot()
        if img is None:
            return False
        shot("move_ship_after", img)
        ocr = self.io.ocr(img)
        if find_text(ocr, self.SHIP_PLACE_RX):
            self.note(f"move ship: {dst} not taken, still placing; cancel")
            self._ship_move_miss(f"destination {dst} not taken", "noplace_move_ship", img, a)
            self.click(UI.get("place_cancel", (654, 646)), "cancel ship move placement")
            time.sleep(0.8)
            return False
        if find_text(ocr, r"confirm"):
            self.click(UI["setup_confirm"], "confirm move ship")
            time.sleep(0.8)
        # the ship stands at dst now: the "seen ours" memory must not put it back at src
        mine = self.__dict__.setdefault("_my_ships", set())
        mine.discard(src)
        mine.add(dst)
        self._ship_moved_turn = self.__dict__.get("_turn_n", 0)
        self.probe.result("move_ship", None, f"{src}->{dst} (UNVERIFIED)")
        self.note(f"moved ship {src} -> {dst} (UNVERIFIED: frames in data/popups/{stamp}_move_ship*)")
        self._close_leftover()
        return True


    def _brain_cfg(self):
        """Owner 2026-10-09 knight switches (config) onto the brain."""
        cfg = getattr(self, "cfg", None) or {}
        self.brain.knight_move_ships = bool(cfg.get("knight_move_ships", True))
        self.brain.knight_chase = bool(cfg.get("knight_chase_robber", True))
        self.brain.ship_moves = bool(cfg.get("ship_move", True)) and not (   # owner 2026-10-09
            cfg and self._ship_move_blocked())
        self.brain.generous_accept = bool(cfg.get("generous_offer_accept", True))
        self.brain.settle_big_hand = bool(cfg.get("settle_big_hand", True))   # owner 2026-10-09 21:08
        self.brain.road_settle_lookahead = bool(cfg.get("road_settle_lookahead", True))   # owner 2026-10-09
        # owner 2026-10-09: rival knight targets (Deserter, displacement) by the spot they hold
        self.brain.knight_target_by_spot = bool(cfg.get("knight_target_by_spot", True))
        self.brain.keep_grain_set = bool(cfg.get("grain_set_keep", True))   # solo 2026-10-09 22:58
        self.brain.city_always = bool(cfg.get("city_always_goal", True))          # owner 2026-10-10
        self.brain.city_rebuild_w = 0.05 if cfg.get("city_rebuild_first", True) else 0.0
        self.brain.opening_set_first = bool(cfg.get("opening_set_first", True))
        from . import knight_targets
        knight_targets.ENABLED = self.brain.knight_target_by_spot
        from .vision import screens
        screens.PICK_PROGRESS_FIRST = bool(cfg.get("pick_progress_first", True))   # stage 2026-10-09 22:46

    CHASE_WORDS = r"chase|robber|expel|drive|banish|pirate"

    def tap_chase_robber(self, gs) -> bool:
        """Owner 2026-10-09: "knights can move the robber if active and get deactivated".
        After the first attack, an active knight on a corner of the robber's hex (Brain.chase_pick:
        it frees our blocked hex, or steals with knight_bounce) is tapped and its chase option
        taken; the game then asks for the robber's new hex, which on_place_robber answers with
        choose_robber (steal included). UNVERIFIED flow (no capture yet): every frame goes to
        data/popups/*_chase*, at most one chase per turn, and after 2 misses in a game it is not
        tried again. The knight is marked inactive and acted."""
        if not (getattr(self, "cfg", None) or {}).get("knight_chase_robber", True):
            return False
        t0 = getattr(self, "turn_started", None)
        if t0 is None or getattr(self, "_chased_turn", None) == t0 or getattr(self, "_chase_misses", 0) >= 2:
            return False
        gs.barbarians_attacked = bool(getattr(self, "barbarians_attacked", False)) or gs.barbarians_attacked
        try:
            pick = self.brain.chase_pick(gs)
        except Exception as e:                       # noqa: BLE001 -- a bad pick never stops the turn
            self.note(f"chase: pick failed: {type(e).__name__}: {e}")
            return False
        if pick is None:
            return False
        a, (hx, victim) = pick
        v, at, piece = a.target
        if (("chase", a.target) in getattr(self, "_failed_taps", set())):
            return False
        self._chased_turn = t0
        self.__dict__.setdefault("_failed_taps", set()).add(("chase", a.target))
        lat = self.br.lattice
        self._close_leftover()
        self.probe.tried("chase")
        xy = lat.vertex_px(v)
        self.click(xy, f"tap knight {v} to chase the {piece} off {at}")
        time.sleep(0.7)
        img = self.io.snapshot()
        if img is None:
            return False
        d = ROOT / "data" / "popups"
        stamp = time.strftime('%H%M%S')
        try:
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / f"{stamp}_chase.png"), img)
        except Exception:                            # noqa: BLE001
            pass
        from .vision.screens import label_under
        ocr = self.io.ocr(img)
        try:
            ocr = ocr + ocr_near(self.io.ocr, img, xy)
        except Exception:                            # noqa: BLE001
            pass
        rings = self._popup_options(img, near=xy)
        for r in rings:
            lab, _ = label_under(ocr, r)
            if lab and re.search(self.CHASE_WORDS, str(lab), re.I):
                r["kind"] = "chase"
        self.note(f"chase popup: {[(r['kind'], r['colour']) for r in rings]}")
        ring = next((r for r in rings if r["kind"] == "chase"), None)
        if ring is None:
            hit = find_popup_option(ocr, "chase", self.CHASE_WORDS)
            if hit:
                ring = {"x": hit["x"] + hit["w"] / 2, "y": hit["y"] + hit["h"] / 2 - 45, "colour": "green"}
        if ring is None or ring.get("colour") == "red":
            self.note(f"chase: no usable chase option at knight {v} (UNVERIFIED flow)")
            self._chase_misses = getattr(self, "_chase_misses", 0) + 1
            self._probe("nooption_chase", img, note={"target": str(a.target)})
            self.probe.result("chase", False, f"no option at {v}")
            self._close_leftover(img)
            return False
        self.click((ring["x"], ring["y"]), f"popup chase {piece}")
        time.sleep(0.8)
        img = self.io.snapshot()
        try:
            cv2.imwrite(str(d / f"{stamp}_chase_after.png"), img)
        except Exception:                            # noqa: BLE001
            pass
        # the knight acted: inactive until activated again (grain)
        kn = self.__dict__.setdefault("_my_knights", {})
        kn[v] = False
        self.__dict__.setdefault("_moved_knights", set()).add(v)
        self._save_game_flags()
        self.probe.result("chase", None, f"{v} chased {piece} off {at} (UNVERIFIED)")
        self.note(f"knight {v} chased the {piece} off {at}; robber wanted on {hx} "
                  f"(victim {getattr(victim, 'name', victim)}) (UNVERIFIED: frames data/popups/{stamp}_chase*)")
        gl = self._glog(start=False)
        if gl is not None:
            try:
                gl.event("knight_chase", knight=str(v), hex=str(at), piece=piece, to=str(hx),
                         victim=getattr(victim, "name", victim), verified=False)
            except Exception:                        # noqa: BLE001
                pass
        return True

    def _settled_snapshot(self, ref, max_s: float, min_s: float = 0.2, step: float = 0.1):
        """Owner 2026-10-08 "some of the moves take a long time": instead of a fixed sleep after a
        click, poll the window until it differs from `ref` (the frame before the click) and two
        frames in a row agree, never waiting longer than the old fixed `max_s`."""
        if not isinstance(self.io, LiveIO):
            if not isinstance(self.io, ReplayIO) and not getattr(self.io, "settle_poll", False):
                time.sleep(max_s)
            if not getattr(self.io, "settle_poll", False):
                return self.io.snapshot()
        t0 = time.time()
        time.sleep(min_s)
        prev = None
        while True:
            img = self.io.snapshot()
            if img is None or time.time() - t0 >= max_s:
                return img
            small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (160, 104))
            changed = ref is None or ref.shape != img.shape or float(np.mean(cv2.absdiff(
                small, cv2.resize(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), (160, 104))))) > 0.5
            if changed and prev is not None and float(np.mean(cv2.absdiff(small, prev))) < 0.3:
                return img
            prev = small
            time.sleep(step)

    # a rival's progress card is announced over the board ("Bramble plays the progress card
    # Wedding." + the card text); a tap under it opens nothing (live 2026-10-08 19:16:29)
    ANNOUNCE_RE = r"plays the progress card|each player who has more|must give you|of his choice"
    ANNOUNCE_WAIT_S = 6.0

    def _wait_announcement(self) -> bool:
        """Wait (up to ANNOUNCE_WAIT_S) for a progress-card announcement to leave the board.
        True when one was seen."""
        if not isinstance(self.io, LiveIO):
            return False
        seen, end = False, time.time() + self.ANNOUNCE_WAIT_S
        while time.time() < end:
            img = self.io.snapshot()
            if img is None or not find_text(self.io.ocr(img), self.ANNOUNCE_RE):
                break
            if not seen:
                self.note("a progress card is announced over the board: waiting before the tap")
            seen = True
            time.sleep(1.0)
        return seen

    def tap_build(self, a) -> bool:
        """Tap the target spot, read the popup (option rings + icon templates, label OCR as the
        cross-check), click the option for this build, then verify it landed.
        True when the build was seen (or could not be checked); the caller re-plans then."""
        lat = self.br.lattice
        t = a.target
        xy = lat.edge_px(t) if a.kind in ("road", "ship") else lat.vertex_px(t)
        self._wait_announcement()
        tapped = self.__dict__.setdefault("_tapped_spots", {})
        if tapped.get(t) is False and t not in self.__dict__.get("_offered_spots", {}).get(a.kind, ()):
            # live 2026-10-04: the same spot was tapped 2-4 times a turn, each tap reopening the popup
            self.note(f"{a.kind} at {t}: spot already tapped this turn; skipping")
            return False
        self._close_leftover()
        tapped[t] = False
        self.probe.tried(f"build_{a.kind}")
        ref = self.io.snapshot() if isinstance(self.io, LiveIO) else None    # the frame before the tap
        self.click(xy, f"tap spot for {a.kind} at {t}")
        img = self._settled_snapshot(ref, 0.7)
        if img is None:
            return False
        hand = getattr(self, "hand", None)
        try:
            before = hand.read(img) if hand is not None else {}
        except Exception:                                # noqa: BLE001
            before = {}
        ocr = self.io.ocr(img)
        try:
            d = ROOT / "data" / "popups"; d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / f"{time.strftime('%H%M%S')}_{a.kind}.png"), img)
        except Exception:
            pass
        self.note(f"popup texts: {[o['text'] for o in ocr if abs(o['y'] - xy[1]) < 260][:14]}")
        # solo 2026-10-03: "settle" matched the BUILD COSTS panel's "Settleme" label at the
        # bottom left and the bot clicked the cost card -- options sit next to the tapped spot
        ocr = [o for o in ocr if abs(o["x"] + o["w"] / 2 - xy[0]) < 320 and abs(o["y"] - xy[1]) < 200]
        # live 2026-10-04: 0 of 61 builds landed -- whole-frame OCR read the small labels as
        # "Build foad", "Promotekmigit"; add a x3 crop around the spot and match labels fuzzily
        try:
            near = ocr_near(self.io.ocr, img, xy)
        except Exception as e:                       # noqa: BLE001
            near = []
            self.note(f"popup: near OCR failed ({e})")
        if near:
            self.note(f"popup texts (x3 crop): {[o['text'] for o in near][:14]}")
        ocr = ocr + [o for o in near if abs(o["x"] + o["w"] / 2 - xy[0]) < 320 and abs(o["y"] - xy[1]) < 200]
        own = find_text(ocr, r"\(owned by ")
        not_ours = own and "marlow" not in own["text"].lower() and a.kind not in ("road", "ship")
        # live 2026-10-04: the options are 57 px rings on the spot's row (green = affordable,
        # red = not) at fixed offsets; the icon names the option, the label OCR cross-checks it
        from .vision.screens import label_under
        rings = [] if not_ours else self._popup_options(img, near=xy)
        for r in rings:
            lab, _ = label_under(ocr, r)
            if lab and r["kind"] and lab != r["kind"] and not (lab == "improve" and r["kind"] in self.WALL_OR_IMPROVE):
                self.note(f"popup: icon says {r['kind']}, label says {lab} at ({r['x']},{r['y']}); using the icon")
            r["kind"] = r["kind"] or lab
        if rings:
            self.note(f"popup options: {[(r['kind'], r['colour']) for r in rings]}")
        kinds = {r["kind"] for r in rings}
        offered = self.__dict__.setdefault("_offered_spots", {})
        for k in ("settlement", "city"):
            # owner 2026-10-06: it traded down to the limit and ended the turn instead of building
            # the settlement the game had offered at (2,2) and (3,5) -- remember what the game offers
            (offered.setdefault(k, set()).add if k in kinds else offered.setdefault(k, set()).discard)(t)
        if "city" in kinds:
            # owner 03:22: our building offering "city" is a settlement -- the start-city guess was wrong
            self._settlement_seen(getattr(self, "gs", None), t)
        if kinds & set(self.WALL_OR_IMPROVE) and t not in getattr(self, "_my_cities", set()):
            # only our city offers walls and improvements: it stands here
            self.note(f"popup: walls/improvements offered at {t} -- our city is there")
            self.__dict__.setdefault("_my_cities", set()).add(t)
            if getattr(self, "_lost_cities", 0):
                self._lost_cities = 0
                self._save_game_flags()
        if a.kind == "road" and kinds == {"ship"}:
            self.__dict__.setdefault("_sea_edges", set()).add(t)
            self.note(f"edge {t} takes only a ship: no road planned there again")
        opt = None
        if rings:
            ring = next((r for r in rings if r["kind"] == a.kind), None)
            if ring is None and a.kind in ("knight", "promote") and "activate" in kinds:
                act = next(r for r in rings if r["kind"] == "activate")
                self.__dict__.setdefault("_my_knights", {})[t] = False
                gs = getattr(self, "gs", None)
                if a.kind == "knight" and act["colour"] == "green" and gs is not None \
                        and gs.me().hand.get("grain", 0) >= 1:
                    self.note(f"knight at {t} is already ours (inactive): activating it")
                    ring, a = act, dataclasses.replace(a, kind="activate")
                elif a.kind == "knight":
                    self.note(f"knight at {t} is already ours (inactive): no grain to activate")
            if ring is not None and ring["colour"] == "red":
                self.note(f"cannot afford {a.kind} (red option at {t})")
                self.probe._row(f"build_{a.kind}").setdefault("refused", 0)
                self.probe._row(f"build_{a.kind}")["refused"] += 1
                self.probe._row(f"build_{a.kind}")["tries"] -= 1     # the game said no: a plan miss, not a click miss
                self.__dict__.setdefault("_unaffordable", set()).add(a.kind)
                self.__dict__.setdefault("_failed_taps", set()).add((a.kind, t))
                self._close_leftover()
                return False
            if ring is not None:
                opt = (ring["x"], ring["y"])
        else:
            hit = None if not_ours else find_popup_option(ocr, a.kind, self.POPUP_WORDS.get(a.kind, a.kind))
            if not hit and a.kind in ("knight", "promote") and not own:
                act = find_popup_option(ocr, "activate", self.POPUP_WORDS["activate"])
                if act:
                    # solo 2026-10-03: "Activate knight" on the spot -- it is already our knight
                    self.__dict__.setdefault("_my_knights", {})[t] = False
                    gs = getattr(self, "gs", None)
                    if gs is not None and gs.me().hand.get("grain", 0) >= 1:
                        self.note(f"knight at {t} is already ours (inactive): activating it")
                        hit, a = act, dataclasses.replace(a, kind="activate")
                    else:
                        self.note(f"knight at {t} is already ours (inactive): no grain to activate")
                        self._close_leftover()
                        return False
            if hit:
                # live 2026-10-02: the option is a round icon ~45 px ABOVE its label; the label
                # itself is not clickable
                opt = (hit["x"] + hit["w"] / 2, hit["y"] + hit["h"] / 2 - 45)
        if opt is None and self._misclick(a, t, kinds, img):
            return False
        if opt is None:
            self.note(f"popup: no option for {a.kind}; leaving it")
            if rings and a.kind in ("activate", "promote", "move_knight") and kinds \
                    and kinds <= {"knight", "settlement", "road", "ship"}:
                self._forget_knight(t)
            from .vision.screens import popup_label_kind
            if not rings and not not_ours and not any(popup_label_kind(o["text"])[0] for o in ocr):
                # probe 2026-10-04: an inactive knight with no grain / an illegal knight corner
                # opens no popup at all -- the game offers nothing there (frame kept to check)
                self._refused(f"build_{a.kind}")
                self._probe(f"nopopup_{a.kind}", img, note={"target": str(t)})
            elif rings and all(r["kind"] for r in rings) or not_ours:
                # the popup opened and every option was read: the game doesn't offer this build
                # here (probe 2026-10-04: promote on a knight that only offered "Move knight") --
                # a plan miss, not a click miss
                self._refused(f"build_{a.kind}")
                if a.kind == "promote":
                    self.__dict__.setdefault("_not_offered", set()).add((a.kind, t))
            else:
                self.probe.result(f"build_{a.kind}", False, f"no option at {t}")
                self._probe(f"nooption_{a.kind}", img, note={"target": str(t), "rings": [(r["kind"], r["colour"]) for r in rings]})
            # solo 2026-10-03: promote + knight retried at the same spots every 10 s all turn
            self.__dict__.setdefault("_failed_taps", set()).add((a.kind, t))
            if not_ours:
                # live 2026-10-03: knight aimed twice at Vale's settlement the vision missed --
                # the popup names the owner, so keep that corner taken for the rest of the game
                who = re.sub(r".*owned by\s*|\)\s*$", "", own["text"]).strip()
                self.__dict__.setdefault("_taken_spots", {})[t] = who
                self.note(f"spot {t} belongs to {who}: not planning there again")
            self._close_leftover()
            return False
        ref = img
        self.click(opt, f"popup {a.kind}", ocr)
        img = self._settled_snapshot(ref, 0.8)
        if img is None:
            return False
        try:
            cv2.imwrite(str(ROOT / "data" / "popups" / f"{time.strftime('%H%M%S')}_{a.kind}_after.png"), img)
        except Exception:                            # noqa: BLE001
            pass
        if find_text(self.io.ocr(img), r"confirm"):
            ref = img
            self.click(UI["setup_confirm"], f"confirm {a.kind}")
            img = self._settled_snapshot(ref, 0.8)
        ok = self._verify_build(a.kind, t, before, img)
        if ok is False:
            self.note(f"build {a.kind} at {t} not seen (hand unchanged, spot not ours)")
            self.probe.result(f"build_{a.kind}", False, f"not seen at {t}")
            self._probe(f"notseen_{a.kind}", img, note={"target": str(t), "before": before})
            self.__dict__.setdefault("_failed_taps", set()).add((a.kind, t))
            self._close_leftover(img)
            return False
        self._record_build(a.kind, t)
        self._last_build_ok = bool(ok)               # 0.4.3 build_batch goes on only after a seen build
        self.probe.result(f"build_{a.kind}", True if ok else None)
        tapped[t] = True                             # the spot changed: a later action there may tap it again
        self.note(f"built {a.kind} at {t}" + ("" if ok else " (UNVERIFIED: hand and spot unreadable)"))
        self._close_leftover(img)
        return True

    # -- progress cards (C&K). UI UNCALIBRATED: see ui.json "_progress_UNCONFIRMED".
    PROGRESS_MAX_PER_CALL = 4
    # cards whose play opens a prompt the runner has never seen; ui.json "progress_play_ok" lifts one
    PLAY_HELD = {"alchemist", "inventor", "resource_monopoly", "merchant_fleet", "commercial_harbor"} \
        - set(UI.get("progress_play_ok", []))

    def _read_progress_hand(self, ocr, img=None) -> list[tuple[str, dict]]:
        """(card_key, ocr box) for every progress-card name OCR'd inside UI["progress_hand"]."""
        from . import progress_cards as pc
        x0, y0, x1, y1 = UI["progress_hand"]
        ocr = list(ocr or [])
        if img is not None and getattr(img, "ndim", 0) == 3 and img.shape[0] >= y0 + 110:
            # live 2026-10-03: full-frame OCR missed the small titles ("Bishop", "Resource
            # Monop"); a 2x crop of the title strip reads them all. Boxes mapped back to frame.
            try:
                # game 11 15:41: with 5 cards the titles sat at y~714, below the old 50 px strip
                crop = cv2.resize(img[y0:y0 + 110, x0:x1], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                for o in self.io.ocr(crop) or []:
                    o = dict(o, x=x0 + o["x"] / 2, y=y0 + o["y"] / 2, w=o["w"] / 2, h=o["h"] / 2)
                    if not any(abs(o["x"] - q["x"]) < 20 and abs(o["y"] - q["y"]) < 12 for q in ocr):
                        ocr.append(o)
            except Exception:                        # noqa: BLE001
                pass
        out = []
        for o in ocr:
            cx, cy = o["x"] + o["w"] / 2, o["y"] + o["h"] / 2
            if not (x0 <= cx <= x1 and y0 <= cy <= y1):
                continue
            try:
                # solo 2026-10-06: Medicine covered Crane down to "Cra"
                out.append((pc.card_key(o["text"], min_prefix=3), o))
            except KeyError:
                continue                        # not a card name (button label, count, ...)
        return out

    def _save_progress_step(self, img, tag):
        if img is None or self.cfg.get("dry_run"):
            return
        try:
            d = ROOT / "data" / "popups"; d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / f"{time.strftime('%H%M%S')}_progress_{tag}.png"), img)
        except Exception:                            # noqa: BLE001
            pass

    def _progress_choices(self, key, args, ocr):
        """Answer the card's on-screen choices. Flows UNCONFIRMED: best effort, logged."""
        lat = getattr(getattr(self, "br", None), "lattice", None)
        if "player" in args:
            # probe 2026-10-04: the player pick is a dialog (name + check button under it) that the
            # step loop answers (on_choose_player); a click on the name hit the top banner instead
            self._want_player = str(args["player"])
            self._player_tried = set()               # review 0.3.1: a new card starts its own tries
            if key == "spy":
                self._spy_victim = str(args["player"])     # on_pick_progress: leader denial
            if key == "deserter" and args.get("knight") is not None:
                # owner 2026-10-09: the victim picks the knight; this is the one we were after
                self.note(f"deserter on {args['player']}: after the knight at {args['knight']} (their choice)")
        if key in ("resource_monopoly", "trade_monopoly", "merchant_fleet") and (args.get("resource") or args.get("commodity")):
            # solo 2026-10-06 06:47: the Merchant Fleet dialog is the Monopoly tile row (icons, no
            # names) under its own title; OCR found no 'lumber' and the dialog stalled 17 min
            self._monopoly_pick(str(args.get("resource") or args.get("commodity")), ocr)
            args = {k: v for k, v in args.items() if k not in ("resource", "commodity")}
        for k in ("resource", "commodity"):
            if k in args:
                box = find_text(ocr, rf"^{re.escape(str(args[k]))}$")
                if box is None:
                    self.note(f"progress {key}: no '{args[k]}' option on screen (UNCALIBRATED)")
                else:
                    self.click((box["x"] + box["w"] / 2, box["y"] + box["h"] / 2), f"progress {key} {k}={args[k]}", ocr)
        if lat is not None:
            if "hex" in args:
                # game 13 16:56: the Bishop's hex click used the lattice from before the drag and
                # hit 20 px off the ringed hex (robber prompt left up, Warlord then ignored)
                self.click(self._snap_ring(lat.hex_px(*args["hex"])), f"progress {key} hex {args['hex']}")
            # game 6 11:34: Smith's knight was selected and the green check waited 10 min
            if "hex" in args:
                self._confirm_if_check(f"progress {key} hex")
            for k in ("city", "settlement", "knight_at"):
                if k in args:
                    self.click(lat.vertex_px(args[k]), f"progress {key} {k} {args[k]}")
                    self._confirm_if_check(f"progress {key} {k}")
            for v in args.get("knights", []):
                self.click(lat.vertex_px(v), f"progress {key} knight {v}")
                self._confirm_if_check(f"progress {key} knight")
            if args.get("own"):                 # Diplomat moving our own road: UI UNCONFIRMED
                self._probe(f"progress_{key}_own_road", note={"args": str(args), "ui": "UNCONFIRMED"})
            edges = ([args["edge"]] if "edge" in args else []) + list(args.get("edges", []))
            if key == "diplomat" and edges:
                # both taps go to a shown ring: the removal, then (own move) the new spot
                self._diplomat_played_t = time.time()   # the step loop's textless recovery window
                ok = self._diplomat_pick(edges[0], own=bool(args.get("own")))
                edges = edges[1:] if ok else []
            if key == "diplomat" and edges and args.get("own"):
                self._diplomat_place(edges[0])
                edges = edges[1:]
            for e in edges:
                self.click(lat.edge_px(e), f"progress {key} edge {e}")
                self._confirm_if_check(f"progress {key} edge")
        if lat is not None and "swap" in args and self.cfg.get("inventor_ringed_only", True):
            # game 0959 10:34: "Select position for Number Token" rings only the chips the game
            # allows; the planned barren hex had no ring. Choose again among the ringed ones
            self._inventor_swap(key, args["swap"])
        elif lat is not None and "swap" in args:
            # Inventor: tap the two number chips' hexes (UI UNCONFIRMED -- probe-captured)
            for h in args["swap"]:
                h = getattr(h, "coord", h)
                self.click(self._snap_ring(lat.hex_px(*h)), f"progress {key} swap hex {h}")
                time.sleep(0.5)
            self._confirm_if_check(f"progress {key} swap")
        if "dice" in args:
            self._last_alchemist = tuple(args["dice"])
            if self._alchemist_pick(args["dice"]):
                args = {k: v for k, v in args.items() if k != "dice"}
        if key == "commercial_harbor" and "give" in args:
            # our harbor: the player dialog (harbor_player) is answered by on_harbor_player
            self._harbor_give = list(args["give"])
            self._harbor = None
            self.note(f"progress {key}: planned gives {self._harbor_give} (the player dialog follows)")
            args = {k: v for k, v in args.items() if k != "give"}
        for k in ("dice", "swap", "give"):
            if k in args:
                if key not in self.__dict__.get("_progress_confirmed", ()):
                    self._probe(f"progress_{key}_{k}", note={"args": str(args[k]), "ui": "UNCONFIRMED"})
                if k != "swap":
                    self.note(f"progress {key}: choice {k}={args[k]} has no UI flow yet (UNCALIBRATED, captured)")
                    self._confirm_if_check(f"progress {key} {k} (default)")

    @staticmethod
    def alchemist_dice_boxes(img):
        """Alchemist "Choose what the dice will show." (game 13 17:28): a row of 6 white dice over
        a row of 6 red dice, a green check below. Returns (white xs, y, red xs, y, check xy) or None."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        def blobs(m, lo=1500):
            n, _, st, c = cv2.connectedComponentsWithStats(m)
            return [(float(c[i][0]), float(c[i][1]), int(st[i][4])) for i in range(1, n) if st[i][4] > lo]
        white = blobs(cv2.inRange(hsv, (0, 0, 235), (180, 25, 255)))
        red = blobs(cv2.inRange(hsv, (0, 150, 170), (8, 255, 255)) | cv2.inRange(hsv, (172, 150, 170), (180, 255, 255)))

        def row(bs):
            for y in sorted({round(b[1] / 10) for b in bs}):
                r = sorted([b for b in bs if abs(b[1] / 10 - y) <= 1], key=lambda b: b[0])
                if len(r) == 6 and max(b[2] for b in r) < 4 * min(b[2] for b in r):
                    return r
            return None
        w, rd = row(white), row(red)
        if not w or not rd or rd[0][1] <= w[0][1]:
            return None
        green = [b for b in blobs(cv2.inRange(hsv, (45, 120, 120), (80, 255, 255)), 1000)
                 if b[1] > rd[0][1] + 40 and w[0][0] < b[0] < w[-1][0]]
        chk = max(green, key=lambda b: b[2]) if green else None
        return ([b[0] for b in w], w[0][1], [b[0] for b in rd], rd[0][1], (chk[0], chk[1]) if chk else None)

    def _alchemist_pick(self, dice) -> bool:
        """Click the white die value, the red die value, then the green check. True when the
        dialog is gone afterwards."""
        boxes = None
        for _ in range(6):
            img = self.io.snapshot()
            boxes = self.alchemist_dice_boxes(img) if img is not None else None
            if boxes:
                break
            time.sleep(0.5)
        if not boxes:
            self.note("alchemist: dice dialog not found")
            return False
        wx, wy, rx, ry, chk = boxes
        a, b = int(dice[0]), int(dice[1])
        self.click((wx[a - 1], wy), f"alchemist white die {a}")
        time.sleep(0.3)
        self.click((rx[b - 1], ry), f"alchemist red die {b}")
        time.sleep(0.3)
        if chk:
            self.click(chk, "alchemist confirm (green check)")
        time.sleep(1.0)
        im2 = self.io.snapshot()
        ok = im2 is not None and self.alchemist_dice_boxes(im2) is None
        self.probe.result("alchemist_dice", ok, f"alchemist dice {dice}: dialog still up")
        return ok

    # Monopoly picker (solo 2026-10-05 06:12): "Name a resource. Each opponent must give you two
    # of that type" -- a row of 8 icon tiles (lumber, brick, wool, grain, ore, cloth, coin, paper;
    # the other kind greyed out) and a green check; no text labels, so the OCR pick never clicked.
    # The dialog then sat 13 min; a rescue click on the table hid it (the glowing trade button
    # brings it back) and End turn stayed grey. Offsets from the big title's centre in NATIVE px of
    # the 1920x1080 window (062048_033_rescue_240_native.png; re-checked on 22:37 live screen).
    # Solo 22:37: offsets kept in the plain-resize 1372x888 frame missed in the live 1308x790
    # virtual frame (UI x1.468) -- ore/check clicks 33/50 px short, loop-blocked.
    MONO_ORDER = ("lumber", "brick", "wool", "grain", "ore", "cloth", "coin", "paper")
    MONO_REF = (1372, 888)
    MONO_NATIVE_H = 1080
    MONO_TILE0, MONO_TILE_DX, MONO_TILE_DY = -389.0, 115.2, 193.4
    MONO_CHECK = (410.0, 517.0)
    MONO_TEXT = r"name a (resource|commodity)|confirm your choice or change it"

    def _monopoly_title(self, ocr):
        """The dialog's big title box ('Resource Monopoly' / 'Trade Monopoly' / 'Merchant Fleet'), else None."""
        # the big title alone (w~186 in the frame); the small card label under the hand is w~76
        tt = [o for o in ocr or [] if re.fullmatch(r"\s*((resource|trade)\s+monopoly|merchant\s+fleet)\s*", o["text"], re.I)
              and o["w"] > 120 * FRAME[0] / self.MONO_REF[0]]
        return max(tt, key=lambda o: o["w"] * o["h"]) if tt else None

    def _monopoly_pick(self, choice: str, ocr=None) -> bool:
        """Click the tile for `choice`, then the green check. True when the dialog is gone."""
        # always a fresh screen: the play flow's OCR can be stale (22:37 one click went to 1390,1142)
        img = self.io.snapshot()
        if img is not None:
            ocr = self.io.ocr(img)
        t = self._monopoly_title(ocr)
        if t is None or choice not in self.MONO_ORDER:
            self.note(f"monopoly: dialog not found for {choice!r}")
            return False
        sx, sy = self._native_ui_scale()
        cx, cy = t["x"] + t["w"] / 2, t["y"] + t["h"] / 2
        i = self.MONO_ORDER.index(choice)
        tile = (cx + (self.MONO_TILE0 + i * self.MONO_TILE_DX) * sx, cy + self.MONO_TILE_DY * sy)
        chk = (cx + self.MONO_CHECK[0] * sx, cy + self.MONO_CHECK[1] * sy)
        fw, fh = (img.shape[1], img.shape[0]) if img is not None else FRAME
        if not all(0 <= p[0] < fw and 0 <= p[1] < fh for p in (tile, chk)):
            self.note(f"monopoly: targets {tile} / {chk} outside the {fw}x{fh} frame; not clicking")
            return False
        self.click(tile, f"monopoly pick {choice}")
        time.sleep(0.6)
        self.click(chk, "monopoly confirm (green check)")
        time.sleep(1.2)
        im2 = self.io.snapshot()
        ok = im2 is not None and self._monopoly_title(self.io.ocr(im2)) is None
        self.probe.result("monopoly_pick", ok, f"monopoly {choice}: dialog still up")
        fleet = "fleet" in t["text"].lower()
        if fleet:
            if ok:
                self.brain.start_fleet(choice)
            self.note(f"merchant fleet: 2:1 on {choice} " + ("set" if ok else "-- dialog still up"))
            return ok
        if ok and self.gs is not None:              # hand ledger (deduped with the play flow's call)
            self._ledger().monopoly(self._ledger().me or "", choice, 2 if self.MONO_ORDER.index(choice) < 5 else 1,
                                    [p.name for p in self.gs.players], gs=self.gs)
        return ok

    def _native_ui_scale(self):
        """Frame px per native-1080p px: the live virtual frame is uniform (native / lay.s, UI
        drawn in proportion to the window height); a replay frame is the plain to_frame resize."""
        lay = getattr(self.io, "lay", None)
        if lay is not None:
            k = lay.h / self.MONO_NATIVE_H / lay.s
            return k, k
        return FRAME[0] / 1920, FRAME[1] / self.MONO_NATIVE_H

    def _fleet_choice(self) -> str:
        """Merchant Fleet dialog up outside the play flow: the planned resource, else our biggest pile."""
        if self.brain.fleet_resource:
            return self.brain.fleet_resource
        try:
            from . import progress_cards as pc
            if self.gs is not None:
                me = self.gs.me()
                c = pc.choose_merchant_fleet(self.gs, me, force=True)
                if c and c.get("resource") in self.MONO_ORDER:
                    return c["resource"]
                return max(self.MONO_ORDER, key=lambda r: me.hand.get(r, 0))
        except Exception:                            # noqa: BLE001
            pass
        return "lumber"

    def _monopoly_choice(self, ocr) -> str:
        """What to name when the dialog is up outside the play flow (rescue / runner restart)."""
        commodity = bool(find_text(ocr or [], r"name a commodity|trade monopoly"))
        try:
            from . import progress_cards as pc
            if self.gs is not None:
                fn = pc.choose_trade_monopoly if commodity else pc.choose_resource_monopoly
                a = fn(self.gs, self.gs.me()) or {}
                if a.get("commodity") or a.get("resource"):
                    return a.get("commodity") or a.get("resource")
        except Exception:                   # noqa: BLE001
            pass
        return "coin" if commodity else "grain"

    def _snap_ring(self, xy, r=30):
        """The ringed target nearest xy (within r px) on a fresh screen, else xy."""
        im = self.io.snapshot()
        if im is None:
            return xy
        try:
            rings = self.ring_spots(im, getattr(self.io, "native", None), getattr(self.io, "lay", None)) \
                if getattr(self.io, "_frame", None) is im else self.ring_spots(im)
        except Exception:                            # noqa: BLE001
            return xy
        best = min(rings, key=lambda q: math.hypot(q[0] - xy[0], q[1] - xy[1]), default=None)
        return best if best is not None and math.hypot(best[0] - xy[0], best[1] - xy[1]) <= r else xy

    def _token_rings(self, img):
        """Ringed number chips as {hex: ring xy} (Inventor's "Select position for Number Token")."""
        lat = getattr(getattr(self, "br", None), "lattice", None)
        if img is None or lat is None:
            return {}
        # stage 2026-10-09 game 1632: the swapped pair differed from the planned one twice; a ring
        # between two chips snapped at 0.6 hex sizes could name the wrong chip (UNCONFIRMED)
        snap = float((self.__dict__.get("cfg") or {}).get("token_ring_snap", 0.35) or 0.6)
        out, dropped = {}, []
        for x, y in self._rings(img):
            try:
                h = tuple(lat.nearest_hex(x, y))
                hx, hy = lat.hex_px(*h)
            except Exception:                        # noqa: BLE001
                continue
            d = math.hypot(hx - x, hy - y)
            if d <= snap * lat.size:                 # a chip sits on the hex centre
                out.setdefault(h, (x, y))
            elif d <= 0.6 * lat.size:
                dropped.append(f"{h} at {d / lat.size:.2f}")
        if dropped:
            self.note(f"inventor: rings off the chip centre dropped (snap {snap}): {', '.join(dropped)}")
        return out

    def _inventor_swap(self, key="inventor", planned=None, img=None) -> bool:
        """Tap two ringed number chips, best swap among them, 0.5 s apart; no confirm unless a
        green check shows (the 0959 frame has none). False = fewer than two usable rings."""
        if img is None:
            time.sleep(0.5)
            img = self.io.snapshot()
        rings = self._token_rings(img)
        gs = getattr(self, "gs", None)
        pair = None
        want = tuple(getattr(h, "coord", h) for h in planned) if planned else None
        if want:
            # game 1632: the executed pair differed from the plan with no line saying why
            self.note(f"inventor: planned {want[0]}<->{want[1]}; ringed {sorted(rings)}; planned ringed: "
                      + " ".join(f"{k}={'yes' if h in rings else 'no'}" for k, h in zip("ab", want)))
        if want and all(h in rings for h in want):
            pair = want
        elif gs is not None and len(rings) >= 2:
            from . import progress_cards as pc
            got = pc.choose_inventor(gs, gs.me(), allowed=set(rings), force=True)
            pair = got["swap"] if got else None
        if not pair:
            self.note(f"progress {key}: {len(rings)} ringed number chips, no swap to tap")
            return False
        if want and tuple(pair) != want:
            miss = next(h for h in want if h not in rings)
            self.note(f"inventor: re-planned {pair[0]}<->{pair[1]} (planned hex {miss} had no ring)")
            self._probe("inventor_pairs", img)
        self.note(f"progress {key}: swapping {pair[0]} <-> {pair[1]} ({len(rings)} chips ringed)")
        self._token_pairs = self.__dict__.get("_token_pairs", 0) + 1
        for h in pair:
            self.click(rings[h], f"progress {key} swap chip {h}")
            time.sleep(0.5)
        if self._check_xy(self.io.snapshot()) is not None:
            self._confirm_if_check(f"progress {key} swap")
        return True

    def on_select_token(self, img, ocr, texts, info):
        """'Select position for Number Token' (Inventor) from the step loop: the swap among the
        ringed chips; after 2 pairs that did not land, the prompt cap's exit ladder."""
        self._prompt_clock("select_token")
        n = self.__dict__.get("_token_pairs", 0)
        if n >= 2:
            self._prompt_exit(img, "select_token")
            return
        self._inventor_swap("inventor", img=img)

    def _ring_wait_s(self, texts=()) -> float:
        """stage 2026-10-09 07:18:41: Merchant, Inventor and Deserter left textless ring prompts
        that each waited the 45 s ring gate (~145 s a game). Within 90 s of our own card play
        the gate is textless_ring_wait_s; elsewhere it stays 45 s."""
        if time.time() - self.__dict__.get("_progress_click_at", 0) < 90:
            return self.cfg.get("textless_ring_wait_s", 5)
        return 45

    PENDING_PROMPTS = {"place_robber", "place_pirate", "place_knight", "promote_knight", "place_metropolis", "place_wall",
                       "remove_city", "lose_knight", "choose_player", "choose_resource", "pick_progress", "take_cards", "steal",
                       "discard", "discard_progress", "place_ship", "choose_deck", "select_token"}

    def _prompt_pending(self, ocr) -> str | None:
        """A card's follow-up prompt is still up (game 13 16:56: Warlord dragged while the
        Bishop's robber prompt waited -- the game ignored it and it was marked refused)."""
        texts = [o["text"] for o in ocr or []]
        st, _ = classify(texts, self.cfg["bot_name"].lower())
        if st in self.PENDING_PROMPTS:
            return st
        return "select_position" if find_text(ocr or [], r"select ?position") else None

    def _smith_check(self, card, held, knights):
        """Owner 2026-10-06: "keeps trying to use the smith to upgrade knights but it already has
        two level 2 knights and has not upgraded politics". Smith still in hand after its knight
        tap = the game would not promote them: they are at the cap (strong without politics 3)."""
        time.sleep(0.8)
        img = self.io.snapshot()
        if img is None:
            return
        left = [k for k, _ in self._read_progress_hand(self.io.ocr(img), img)]
        if left.count(card) < sum(1 for k, _ in held if k == card):
            return
        gs = getattr(self, "gs", None)
        cap = 3 if gs is not None and gs.me().improvements.get("politics", 0) >= 3 else 2
        lv = self.__dict__.setdefault("_knight_levels", {})
        for v in knights:
            lv[v] = max(lv.get(v, 1), cap)
            self.__dict__.setdefault("_my_knights", {}).setdefault(v, True)
        self.note(f"smith refused on {knights}: knights at level {cap} (politics {'3+' if cap == 3 else '<3'})")
        self._save_game_flags()

    SLOW_CARDS = frozenset({"spy", "intrigue", "diplomat", "deserter", "bishop", "saboteur"})

    def play_progress(self, phase: str, img=None, ocr=None) -> list[str]:
        """Read our progress cards from the screen, ask the brain, play one card, re-read the
        screen, repeat. Returns the card keys played (or discarded). With ui.json
        progress_live=false (and not dry_run) it only logs and saves a screenshot."""
        if self.mode is None or not getattr(self.mode, "ck", False):
            return []
        tried = self.__dict__.setdefault("_progress_tried", set())
        last = getattr(self, "_progress_last_phase", None)
        if phase == "pre_roll" and last in ("main", "end_turn"):
            tried.clear()                       # a new own turn began
            self.brain.end_fleet()
        self._progress_last_phase = phase
        live = bool(UI.get("progress_live")) or bool(self.cfg.get("dry_run"))
        played: list[str] = []
        fresh = img is not None
        # owner 2026-10-04 (game 13): one card that can't go now must not block the rest of the
        # hand -- skip it and ask again for the next card; cards with an unseen UI are tried once
        # per game (probe-captured) instead of held all game
        skip: set[str] = set()
        once = self.__dict__.setdefault("_progress_once", set())
        refused = self.__dict__.setdefault("_progress_refused", set())
        # stage 2026-10-09 07:21: once.discard() after a play re-armed "trying it once" (9x in game
        # 0738): a flow that played this game is confirmed -- no preview, no probe captures
        sure = self.__dict__.setdefault("_progress_confirmed", set()) \
            if self.cfg.get("progress_skip_confirmed", True) else set()
        for _ in range(self.PROGRESS_MAX_PER_CALL + 6):
            if img is None:
                img = self.io.snapshot()
                if img is None:
                    self.note(f"progress {phase}: no screenshot")
                    break
                ocr = None
            if ocr is None:
                ocr = self.io.ocr(img)
            held = self._read_progress_hand(ocr, img)
            self.note(f"progress {phase}: held={[k for k, _ in held]}")
            if not held:
                break
            pend = self._prompt_pending(ocr)
            if pend:
                self.note(f"progress {phase}: {pend} prompt is up -- answering it first")
                break
            blk = self.__dict__.get("_progress_blocked")
            if blk and time.time() - blk[1] < 600:
                # stage 2026-10-09 16:21:10: a textless Diplomat prompt we could not answer
                self.note(f"progress {phase}: {blk[0]} prompt still open -- no more cards this turn")
                break
            if fresh and phase == "main" and self.gs is not None:
                gs = self.gs                    # on_main just read this screen
            else:
                gs = self.ensure_state(img, [o["text"] for o in ocr])
            if gs is None:
                self.note(f"progress {phase}: no game state")
                break
            counts = getattr(self, "_banner_counts", {})
            for p in gs.players:
                if not p.is_me and not p.total_cards() and counts.get(_norm(p.name), (None,))[0] is None:
                    p.hand_count = 4         # banner count unread for this player: guess
            # the plan counts held cards as resources (Warlord = activation): never one the game
            # refused this game
            if phase == "main" and self.cfg.get("spend_before_slow_cards", True):
                # owner 2026-10-09 21:00 "marlow timed out with 15 resources in its hand": Spy and
                # Intrigue ate a minute and the turn timer ended the turn before the city -- over the
                # discard limit, builds go first and cards that bring no resources wait for end_turn
                from . import progress_cards as pc
                if sum(gs.me().hand.values()) > pc.discard_limit(gs, gs.me()):
                    slow = {c for c, _ in held if c in self.SLOW_CARDS} - skip
                    if slow:
                        self.note(f"progress main: {sorted(slow)} wait -- {sum(gs.me().hand.values())} cards, spend first")
                        skip |= slow
            gs.me().progress_hand = [k for k, _ in held if k not in refused]
            gs.me().progress_cards = len(held)
            gs.barbarians_attacked = bool(getattr(self, "barbarians_attacked", False))
            choice = self.brain.progress_play(gs, phase, hand_cards=[k for k, _ in held if k not in skip])
            if choice is None:
                self.note(f"progress {phase}: nothing to play")
                break
            key, args = choice
            card = args["card"] if key == "discard" else key
            if key != "discard":
                # owner 2026-10-06: never drag a card the game can't play now (pc.can_play)
                from . import progress_cards as pc
                ok, why_not = pc.can_play(gs, gs.me(), key, phase)
                if not ok:
                    self.note(f"progress {phase}: holding {key}: can't play — {why_not}")
                    skip.add(card)
                    continue
            if key in self.PLAY_HELD and not self.cfg.get("dry_run"):
                if key in once:
                    self.note(f"progress {phase}: {key} UI flow unconfirmed and tried once this game; skipping it")
                    skip.add(card)
                    continue
            if card in refused and key != "discard":
                self.note(f"progress {phase}: {card} was refused earlier this game; skipping")
                skip.add(card)
                continue
            if (key, card) in tried:
                self.note(f"progress {phase}: {key} {card} already tried this turn; not retrying")
                skip.add(card)
                continue
            tried.add((key, card))
            try:
                d = Path(getattr(self, "popup_dir", None) or ROOT / "data" / "popups")
                d.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(d / f"{time.strftime('%H%M%S')}_progress_{phase}_{key}_{card}.png"), img)
            except Exception:                   # noqa: BLE001
                pass
            box = next((o for k, o in held if k == card), None)
            if not live:
                self.note(f"progress {phase}: would {key} {card} {args} (progress_live=false, UNCALIBRATED)")
                break
            if box is None:
                self.note(f"progress {phase}: {card} not on screen")
                break
            if key in sure:
                self.note(f"progress {phase}: {key} flow confirmed this game -- playing it")
            elif key in self.PLAY_HELD and not self.cfg.get("dry_run"):
                # review 0.3.1: the one try is spent only by a real drag -- a refused, already
                # tried or unseen card used to burn it before any click
                once.add(key)                   # try once per game, every step probe-captured
                self._probe(f"progress_once_{key}", img, note={"args": str(args), "ui": "UNCONFIRMED"})
                self.note(f"progress {phase}: {key} UI flow unconfirmed -- trying it once this game")
            self.note(f"progress {phase}: {key} {card} {args}")
            self._progress_click_at = time.time()
            self.click((box["x"] + box["w"] / 2, box["y"] + box["h"] / 2), f"progress select {card}", ocr)
            time.sleep(0 if self.cfg.get("dry_run") else 0.7)
            if key == "discard":
                self.click(UI["progress_discard"], f"progress discard {card}", ocr)
            else:
                # live 2026-10-03: a click on the small card enlarges it; the Play button is
                # found by its label (never seen yet -- each step is saved for calibration)
                img = self.io.snapshot()
                ocr = self.io.ocr(img) if img is not None else []
                if key not in sure:
                    self._save_progress_step(img, f"{card}_open")
                btn = find_text(ocr, r"^(play|use)\b") if ocr else None
                self.probe.tried("progress_play")
                if btn is None and not self.cfg.get("dry_run"):
                    # audit 2026-10-04: the enlarged card has no Play button (58 closes, 0 plays);
                    # the 5th-card discard is a drag onto the board, so play = the same drag here.
                    # Verified by the card leaving the hand; a refused card is not retried this game.
                    if key not in sure:
                        self._probe(f"progress_open_{card}", img, texts=[o["text"] for o in ocr])
                    self.click((box["x"] + box["w"] / 2, box["y"] + box["h"] / 2), f"progress close {card}", ocr)
                    time.sleep(0.6)
                    self.drag((box["x"] + box["w"] / 2, box["y"] + box["h"] / 2 + 25),
                              tuple(UI.get("progress_play_drop", (700, 400))), f"progress play {card} (drag)")
                    self._after_drag_scan()            # a missed drag could pan the board
                    self._park_pointer("after progress drag")
                else:
                    self.click((btn["x"] + btn["w"] / 2, btn["y"] + btn["h"] / 2) if btn else UI["progress_play"],
                               f"progress play {card}", ocr)
                time.sleep(0 if self.cfg.get("dry_run") else 0.8)
                img = self.io.snapshot()
                ocr = self.io.ocr(img) if img is not None else []
                if not self.cfg.get("dry_run"):
                    left = [k for k, _ in self._read_progress_hand(ocr, img)]
                    n_before = sum(1 for k, _ in held if k == card)
                    gone = left.count(card) < n_before
                    if not gone and self.probe.card.get("progress_play", {}).get("landed", 0):
                        # the drag already played a card this game: the game refused this one
                        # (game 7: Smith on knights it would not promote) -- a plan miss
                        self._refused("progress_play")
                    else:
                        self.probe.result("progress_play", gone, f"{card} still in hand after the drag")
                    fails = self.__dict__.setdefault("_progress_fail_shot", set())
                    if key not in sure or (not gone and key not in fails):
                        if key in sure:
                            fails.add(key)         # a confirmed flow that failed: one frame per game
                        self._probe(f"progress_after_{card}", img, texts=[o["text"] for o in ocr],
                                    note={"held_before": [k for k, _ in held], "held_after": left, "played": gone})
                    if not gone:
                        pend = self._prompt_pending(ocr)
                        if pend:                   # the game was waiting on another prompt: not a refusal
                            self.note(f"progress {phase}: {card} not played -- {pend} prompt is up; answering it first")
                            tried.discard((key, card))
                            break
                        self.note(f"progress {phase}: {card} not played (still in hand); not trying it again this game")
                        refused.add(card)
                        skip.add(card)
                        img, ocr, fresh = None, None, False
                        continue
                if key not in sure:
                    self._save_progress_step(img, f"{card}_played")
                if key in once and not self.cfg.get("dry_run"):
                    # owner 2026-10-06 "should never hold cards it can use": the once-only try left
                    # the hand, so the flow works -- the next copy plays normally
                    once.discard(key)
                    self.note(f"progress {phase}: {key} left the hand -- flow works, next copy plays normally")
                self._progress_choices(key, args, ocr)
                if self.cfg.get("progress_skip_confirmed", True) and not self.cfg.get("dry_run"):
                    sure.add(key)                  # it left the hand: the flow works this game
                if self.gs is not None and not self.cfg.get("dry_run"):
                    self._ledger().on_my_card(key, args, self.gs)   # cards taken: exact via our hand
                if key == "smith" and args.get("knights") and not self.cfg.get("dry_run"):
                    self._smith_check(card, held, args["knights"])
                if find_text(ocr, r"confirm"):
                    self.click(UI["setup_confirm"], f"confirm progress {card}")
                if key in ("resource_monopoly", "trade_monopoly") and (args.get("resource") or args.get("commodity")) \
                        and self.gs is not None and not self.cfg.get("dry_run"):
                    # hand ledger: C&K resource monopoly takes up to 2 from each, trade monopoly 1
                    self._ledger().monopoly(self._ledger().me or "", args.get("resource") or args["commodity"],
                                            2 if key == "resource_monopoly" else 1,
                                            [p.name for p in self.gs.players])
                if key == "merchant" and "hex" in args and self.gs is not None and not self.cfg.get("dry_run"):
                    # owner 2026-10-06 "let's see if it knows it can change 2:1": the hex's resource
                    # trades 2:1 from now on; forget the old dialog ratio so planning sees it
                    from .board import TERRAIN_RES
                    h = self.gs.board.hexes.get(tuple(args["hex"]))
                    res = TERRAIN_RES.get(h.terrain) if h is not None else None
                    if res:
                        self.brain.merchant_res = res
                        self.brain.ratio_seen.pop(res, None)
                        self.gs.me().merchant_hex = tuple(args["hex"])
                        self.note(f"progress: merchant on {args['hex']} -> {res} 2:1 with the bank, +1 VP")
                # review 0.3.1: Merchant Fleet's 2:1 is set by _monopoly_pick only when the pick
                # dialog closed (here, or on a later frame via step) -- a failed pick has no 2:1
                if not self.cfg.get("dry_run"):
                    im_p = self.io.snapshot()
                    pend = self._prompt_pending(self.io.ocr(im_p) if im_p is not None else [])
                    if pend:                       # the card's own prompt: the step loop answers it
                        played.append(card)
                        self.note(f"progress {phase}: {card} left a {pend} prompt -- answering it before the next card")
                        break
            played.append(card)
            if key == "crane" and args.get("track") and not self.cfg.get("dry_run"):
                # owner 2026-10-06: Crane held all game; the plan can't see the discount, so take
                # the improvement it was played for right now (1 commodity less, level 1 free)
                self._crane_track = args["track"]
                lv0 = self._improve_levels().get(args["track"], 0)
                try:
                    self.tap_improve(gs, args["track"])
                finally:
                    self._crane_track = None
                ok = self._improve_levels().get(args["track"], 0) > lv0
                if not ok:
                    # live 2026-10-09 19:56: Crane played, the turn timer ran out before the
                    # improvement; the card was spent for nothing and no log said so
                    im_c = self.io.snapshot()
                    seat = self._marker_seat(im_c) if im_c is not None else 0
                    who = getattr(self, "_slot_names", {}).get(seat, f"P{seat + 1}") if seat else None
                    self.note(f"progress failed: crane played, improve {args['track']} not completed (level {lv0} unchanged)"
                              + (f"; turn already passed to {who} (timer?)" if who else ""))
            time.sleep(0 if self.cfg.get("dry_run") else 0.8)
            img, ocr, fresh = None, None, False    # re-read the screen before the next card
        if phase == "end_turn":
            tried.clear()
            self.brain.end_fleet()
        if played or self.__dict__.get("_progress_click_at", 0) > time.time() - 30:
            self._park_pointer("after progress cards")
        return played

    def _trade_spares_builds(self, gs, trade, builds) -> bool:
        """The offer's give leaves every card the planned builds pay with (unknown cost = no)."""
        from .rules import COSTS
        hand = dict(gs.me().hand)
        for r, n in (trade.give or {}).items():
            hand[r] = hand.get(r, 0) - n
            if hand[r] < 0:
                return False
        for a in builds:
            cost = COSTS.get(a.kind)
            if cost is None:
                return False
            for r, n in cost.items():
                hand[r] = hand.get(r, 0) - n
                if hand[r] < 0:
                    return False
        return True

    @staticmethod
    def _offer_key(give, get):
        return tuple(sorted((give or {}).items())), tuple(sorted((get or {}).items()))

    def _wanted_held(self, get) -> float:
        """The ledger's estimate of the wanted cards held by all opponents."""
        led, gs = self.__dict__.get("_hands"), self.__dict__.get("gs")
        if led is None or gs is None:
            return 0.0
        try:
            return sum(led.holds(p.name, r) for p in gs.opponents() for r in get)
        except Exception:                            # noqa: BLE001 -- an estimate only
            return 0.0

    def _save_city_offer(self, gs, late=False) -> bool:
        """The city is in danger and the bank can't make the knight: ask the table. Game 0959:
        28 offers all declined, the same save-city offer sent again and again -- an offer every
        player declined earlier goes the bank route instead (True = an offer was sent)."""
        made = self.__dict__.setdefault("_offers_made", set())
        sc = self.brain.save_city_offer(gs, made, CARD_ORDER)
        if sc is None:
            return False
        if self.cfg.get("grain_ask_cap", True) and self._was_declined(sc.give, sc.get):
            self.note(f"save the city: offer {sc.give} -> {sc.get} declined earlier, bank route only")
            made.update(("ask", r, g) for r in sc.get for g in sc.give)
            return False
        self.note(f"save the city: offer {sc.give} -> {sc.get}" + (" (late turn, sent anyway)" if late else ""))
        self.last_want = sc.get
        made.update(("want", r) for r in sc.get)
        made.update(("ask", r, g) for r in sc.get for g in sc.give)
        self.offer(sc.give, sc.get)
        return True

    def _grain_asks_capped(self) -> bool:
        """Game 0959: grain asked 8 more times after 2 refusals. Grain declined by all
        offer_max_declines times: no grain asks for the rest of the game (bank routes only)."""
        cap = self.cfg.get("offer_max_declines", 2)
        return bool(self.cfg.get("grain_ask_cap", True) and cap
                    and self.__dict__.get("_want_declined", {}).get("grain", 0) >= cap)

    def _remember_decline(self, give, get):
        # stage 2026-10-09: ore->lumber offered 4 times, ore->coin 3 times, all declined
        # (~85 s a game) -- remember a decline for the game
        wd = self.__dict__.setdefault("_want_declined", {})
        for r in get or {}:
            wd[r] = wd.get(r, 0) + 1
        if self._grain_asks_capped():
            self.__dict__.setdefault("_offers_made", set()).add(("want", "grain"))
        if not self.cfg.get("offer_repeat_after_turns", 6):
            return
        key = self._offer_key(give, get)
        n = self.__dict__.setdefault("_decline_n", {})
        n[key] = n.get(key, 0) + 1
        self.__dict__.setdefault("_declined", {})[key] = \
            (self.__dict__.get("_turn_n", 0), self._wanted_held(get), n[key])

    def _was_declined(self, give, get) -> bool:
        """Declined by all earlier this game, no opponent gained a whole wanted card since, and
        fewer than offer_repeat_after_turns own turns ago (0 = off). After offer_max_declines
        declines of the same offer it stays blocked for the game (0 = no cap)."""
        cfg = self.__dict__.get("cfg") or {}
        n = cfg.get("offer_repeat_after_turns", 6)
        dec = self.__dict__.get("_declined", {})
        key = self._offer_key(give, get)
        if not n or key not in dec:
            return False
        turn, held = dec[key][:2]
        cap = cfg.get("offer_max_declines", 2)
        if cap and self.__dict__.get("_decline_n", {}).get(key, 0) >= cap:
            return True                              # game 0834: wool->ore offered 3 times, all declined
        # game 0834: +0.5 was crossed by income estimates alone in 1-2 rounds; a real gain is a card
        if self.__dict__.get("_turn_n", 0) - turn >= n or self._wanted_held(get) >= held + 1.0:
            del dec[key]
            return False
        return True

    def _want_trade(self, gs):
        """_want_trade1, never an offer every player declined earlier this game (change: F6)."""
        made = self.__dict__.setdefault("_offers_made", set())
        for g, w in list(self.__dict__.get("_declined", {})):
            if self._was_declined(dict(g), dict(w)):
                made.update(("ask", r, c) for r, _ in w for c, _ in g)
        trade = self._want_trade1(gs)
        if trade is not None and self._was_declined(trade.give, trade.get):
            self.note(f"offer {trade.give} -> {trade.get} was declined by all earlier: not sending it again")
            return None
        return trade

    def _want_trade1(self, gs):
        """One-for-one offer toward the brain's top trade target, if any."""
        from .brain import Action
        from .board import COMMODITIES
        from .rules import COSTS
        from .rules import missing
        me = gs.me()
        kinds = self.brain._trade_targets(gs)
        if getattr(self, "_lost_cities", 0) and me.settlements:
            # owner 2026-10-02: after the barbarians take our city, getting it back comes first
            kinds = ["city"] + [k for k in kinds if k != "city"]
        made = self.__dict__.setdefault("_offers_made", set())
        if self.brain.grain_rush(gs) and ("want", "grain") not in made:
            # solo 2026-10-06: barbarians at 5/7, knight inactive, offers asked for lumber/ore ->
            # the only city lost. Grain first, whoever the ledger thinks holds it, 2 cards for 1.
            val = self.brain.resource_value(gs, me)
            # never grain; commodities only when nothing else is in hand (they go to improvements)
            # review 0.3.1: no commodity in a 2-for-1 give either (a commodity only for 2 cards)
            spare = sorted((r for r in CARD_ORDER if me.hand.get(r, 0) and r != "grain"
                            and r not in COMMODITIES), key=lambda r: val.get(r, 1.0))
            give: dict = {}
            for r in spare:
                while sum(give.values()) < 2 and me.hand.get(r, 0) > give.get(r, 0):
                    give[r] = give.get(r, 0) + 1
            if give:
                self.note(f"barbarians close, knight inactive: asking grain for {give}")
                return Action("player_trade", give=give, get={"grain": 1})
        if self.brain.grain_rush(gs):
            # owner 2026-10-06: nobody gives grain -> trade for the cards that complete a bank
            # set, then bank it for grain; 2 cards for 2 when those 2 finish the set
            val = self.brain.resource_value(gs, me)
            for r in sorted((r for r in CARD_ORDER if r != "grain"), key=lambda r: -me.hand.get(r, 0)):
                ratio = min(self.brain.bank_ratio(gs, r), 3) if self._cfg_3to1() else self.brain.bank_ratio(gs, r)
                short = ratio - me.hand.get(r, 0)
                if not 1 <= short <= 2 or ("want", r) in made:
                    continue
                pool = [x for x in sorted(CARD_ORDER, key=lambda x: val.get(x, 1.0))
                        if x not in (r, "grain") and x not in COMMODITIES
                        for _ in range(me.hand.get(x, 0))]
                if len(pool) < short:
                    continue
                give: dict = {}
                for x in pool[:short]:
                    give[x] = give.get(x, 0) + 1
                self.note(f"grain refused: asking {short} {r} for {give} to bank {ratio} {r} -> grain")
                return Action("player_trade", give=give, get={r: short})
        retry = self._retry_top_ask(gs, kinds, made)
        if retry is not None:
            return retry                              # the top target before side targets
        for i, kind in enumerate(kinds):
            need = missing(dict(me.hand), COSTS.get(kind, {}))
            # the top target may be 2 short (knight = wool + ore); side targets only 1
            if not need or sum(need.values()) > (3 if kind == "city" else 2 if i == 0 else 1):
                continue
            if any(("want", r) in made for r in need) and kind != "city" and i > 0:
                continue                              # refused once this turn: don't re-ask
            # owner 2026-10-05: ask only for a card someone we'd trade with plausibly holds
            # (hand ledger); nobody has it -> the bank route or the next plan
            want = next((r for r in need if ("want", r) not in made and self.brain.table_has(gs, r)), None)
            if want is None:
                continue
            val = self.brain.resource_value(gs, me)
            # owner 2026-10-02: never give away the card that keeps a 3:1 (harbor) trade alive
            # (3 wool -> 1 grain was the way to the city; offering 1 wool broke it)
            ratio = {r: (min(self.brain.bank_ratio(gs, r), 3) if self._cfg_3to1() else self.brain.bank_ratio(gs, r))
                     for r in CARD_ORDER}
            # owner 2026-10-02: "gave the hay away for a sheep, no logic in that" -- never offer a
            # card our top goal (or a lost city) needs, whatever this side target is
            protect = set(self.brain.keep) | ({"grain"} if gs.mode.canal else set())   # canal: no grain fields
            protect |= self._improve_commodities(gs) | self._grain_set_keep(gs)
            if getattr(self, "_lost_cities", 0):
                protect |= {"grain", "ore"}
            spare = [r for r in CARD_ORDER if me.hand.get(r, 0) - COSTS[kind].get(r, 0) > 0 and r != want
                     and me.hand.get(r, 0) != ratio[r] and r not in protect]
            if spare:
                give = min(spare, key=lambda r: val.get(r, 1.0))
                return Action("player_trade", give={give: 1}, get={want: 1}, note=f"for:{kind}")
        return self._improve_want(gs) or self._complete_bank_set(gs, kinds)

    def _grain_set_keep(self, gs) -> set:
        """Config grain_set_keep (solo 2026-10-09 22:58:51): while an idle knight waits for grain,
        the cards of a 3:1 set or pair are never offered to a player (brain.grain_set_cards)."""
        try:
            keep = self.brain.grain_set_cards(gs)
        except Exception:                            # noqa: BLE001 -- an offer is never blocked by this
            return set()
        key = (self.__dict__.get("_turn_n", 0), tuple(sorted(keep)))
        if keep and self.__dict__.get("_grain_set_noted") != key:
            self._grain_set_noted = key
            self.note(f"grain set: idle knight, no grain -- {sorted(keep)} stay for the 3:1 grain set, "
                      f"never offered")
        return keep

    def _retry_top_ask(self, gs, kinds, made):
        """owner 2026-10-06 "it could trade out the wood it has for the stone it needs, it never
        tried that": the top target is 1 card short and the table refused our first card -> offer
        each other card the target itself doesn't use (cards held for a lesser plan too), once each."""
        from .brain import Action
        from .rules import COSTS, missing
        me = gs.me()
        kind = kinds[0] if kinds else None
        if kind not in COSTS:
            return None
        need = missing(dict(me.hand), COSTS[kind])
        if sum(need.values()) != 1:
            return None
        want = next(iter(need))
        if ("want", want) not in made or not self.brain.table_has(gs, want):
            return None                                # first ask goes the normal way
        hard = ({"grain"} if gs.mode.canal else set()) | self._improve_commodities(gs) | self._grain_set_keep(gs)
        if getattr(self, "_lost_cities", 0):
            hard |= {"grain", "ore"}
        val = self.brain.resource_value(gs, me)
        tried = {t[2] for t in made if t[0] == "ask" and t[1] == want}
        if not tried:
            return None                                # no record of what we offered: normal path
        gives = [r for r in CARD_ORDER if me.hand.get(r, 0) - COSTS[kind].get(r, 0) > 0
                 and r != want and r not in hard and r not in tried]
        if not gives:
            return None
        give = min(gives, key=lambda r: val.get(r, 1.0))
        self.note(f"{want} refused for {sorted(tried)}: offering 1 {give} instead toward {kind}")
        return Action("player_trade", give={give: 1}, get={want: 1}, note=f"for:{kind}")

    def _trade_beats_builds(self, gs, trade, builds) -> bool:
        """The offer completes a target ranked above every planned build (knight before ship)."""
        from .rules import COSTS, missing
        kind = (trade.note or "")[4:] if (trade.note or "").startswith("for:") else None
        if kind not in COSTS:
            return False
        hand = dict(gs.me().hand)
        for r, n in trade.give.items():
            hand[r] = hand.get(r, 0) - n
        for r, n in trade.get.items():
            hand[r] = hand.get(r, 0) + n
        if any(v < 0 for v in hand.values()) or missing(hand, COSTS[kind]):
            return False
        kinds = self.brain._trade_targets(gs)
        rank = {k: i for i, k in enumerate(kinds)}
        return all(rank.get(kind, 99) < rank.get(a.kind, 99) for a in builds)

    def _improve_want(self, gs):
        """Owner 2026-10-06: "if it does not get a development from the game, it should ... exchange
        in the game to get that development". The dice didn't bring the commodity for the next
        improvement level: ask a player for it (1 short; 2 short only for the first two levels),
        ladder order -- every track to 3 first, then 4, 5 -- giving a spare resource 1 for 1."""
        from .brain import Action
        from .board import COMMODITIES
        from .rules import improvement_cost, missing
        me = gs.me()
        if not gs.mode.ck or not me.cities:
            return None
        made = self.__dict__.setdefault("_offers_made", set())
        val = self.brain.resource_value(gs, me)
        protect = set(self.brain.keep) | ({"grain"} if gs.mode.canal else set()) | self._improve_commodities(gs)
        protect |= self._grain_set_keep(gs)
        if getattr(self, "_lost_cities", 0):
            protect |= {"grain", "ore"}
        tracks = sorted((t for t in IMPROVEMENT_TRACKS if me.improvements.get(t, 0) < 5),
                        key=lambda t: (me.improvements.get(t, 0) >= 3, me.improvements.get(t, 0)))
        for t in tracks:
            lvl = me.improvements.get(t, 0)
            cost = improvement_cost(t, lvl)
            need = missing(dict(me.hand), cost)
            if not need or sum(need.values()) > (2 if lvl < 2 else 1):
                continue
            want = next((r for r in need if r in COMMODITIES and ("want", r) not in made
                         and self.brain.table_has(gs, r)), None)
            if want is None:
                continue
            spare = [r for r in CARD_ORDER if r not in COMMODITIES and r not in protect and me.hand.get(r, 0) > 0]
            if spare:
                give = min(spare, key=lambda r: val.get(r, 1.0))
                self.note(f"improvement: asking 1 {want} for 1 {give} toward {t} {lvl + 1}")
                return Action("player_trade", give={give: 1}, get={want: 1})
        return None

    @staticmethod
    def _improve_commodities(gs) -> set:
        """Owner 2026-10-06: "giving away paper for sheep that it does not need" -- a commodity
        still buys a city improvement (a city, track below 5) is never a spare card.
        Review 0.3.1: owner-commodity-trades -- "a commodity only for 2 cards": every caller is a
        1-for-1 spare list, so no commodity is spare there, city or not (a lost city left them
        free for 1-for-1 gives)."""
        from .board import COMMODITIES
        return set(COMMODITIES) if gs.mode.ck else set()

    def _complete_bank_set(self, gs, kinds):
        """Owner 2026-10-02: "bot should see it has 2 clay and could ask for a third to be able
        to get the resource it needs". One card short of a bank/harbor set of a card the goal
        doesn't use -> ask a player for that card, then bank the set for what's missing."""
        from .brain import Action
        from .rules import COSTS, missing
        me = gs.me()
        made = self.__dict__.setdefault("_offers_made", set())
        val = self.brain.resource_value(gs, me)
        protect = set(self.brain.keep) | ({"grain", "ore"} if getattr(self, "_lost_cities", 0) else set()) \
            | ({"grain"} if gs.mode.canal else set()) | self._improve_commodities(gs) | self._grain_set_keep(gs)
        goals = [(kind, COSTS.get(kind, {})) for kind in kinds]
        if gs.mode.ck and me.cities:
            # owner 03:23: 3 paper in hand -- ask for the 3rd wood, bank it for the 4th paper (science 4)
            from .rules import improvement_cost
            for track in IMPROVEMENT_TRACKS:
                lvl = me.improvements.get(track, 0)
                if lvl < 5:
                    goals.append((f"{track} {lvl + 1}", improvement_cost(track, lvl)))
        for kind, cost in goals:
            need = missing(dict(me.hand), cost)
            if not need or sum(need.values()) > self.brain.set_need_cap(gs, kind):
                continue
            for r in sorted(CARD_ORDER, key=lambda r: val.get(r, 1.0)):
                ratio = self.brain.bank_ratio(gs, r)
                if self._cfg_3to1():
                    ratio = min(ratio, 3)
                have = me.hand.get(r, 0)
                if r in need or ("want", r) in made or have < 1 or have + 1 - cost.get(r, 0) != ratio:
                    continue
                if not self.brain.table_has(gs, r):
                    continue                          # owner 2026-10-05: nobody holds the 3rd card
                spare = [x for x in CARD_ORDER if me.hand.get(x, 0) - cost.get(x, 0) > 0 and x != r
                         and x not in need and x not in protect and me.hand.get(x, 0) != ratio]
                if spare:
                    give = min(spare, key=lambda x: val.get(x, 1.0))
                    self.note(f"bank set: {have} {r}, asking for 1 more to bank {ratio} {r} -> "
                              f"{next(iter(need))} for the {kind}")
                    return Action("player_trade", give={give: 1}, get={r: 1})
        return None

    def _wanted_resource(self, gs, giving: dict):
        """The one card that best completes a build after giving `giving` away."""
        t = self._want_trade(gs)
        if t is not None:
            return next(iter(t.get))
        best, bv = None, -1.0
        g = self.brain.clone(gs)
        from .rules import pay
        pay(g.me().hand, giving)
        for r in CARD_ORDER:
            g2 = self.brain.clone(g)
            g2.me().hand[r] = g2.me().hand.get(r, 0) + 1
            v = self.brain._best_followup(g2)
            if v > bv:
                best, bv = r, v
        return best

    def _bank_trade_for(self, gs, want: dict):
        """Best bank/harbor trade giving our least valuable spare card for `want`."""
        from .brain import Action
        res = next(iter(want))
        me = gs.me()
        val = self.brain.resource_value(gs, me)
        best = None
        locked = self.brain.reserved_commodities(gs)
        for r in CARD_ORDER:
            if r == res or r in locked:
                continue
            ratio = self.brain.bank_ratio(gs, r)
            if self._cfg_3to1():        # owner-confirmed 3:1 (board read may miss it)
                ratio = min(ratio, 3)
            if me.hand.get(r, 0) >= ratio:
                cost = ratio * val.get(r, 1.0)
                if best is None or cost < best[0]:
                    best = (cost, Action("bank_trade", give={r: ratio}, get={res: 1}))
        return best[1] if best else None

    def _build_offered(self, gs) -> bool:
        """Owner 2026-10-06: "instead of building a city or a settlement, it kept trading resources
        until it didn't have too many and then gave out its turn". The plan was empty while the
        hand paid for a settlement the game had offered (red) at two corners earlier that turn.
        Affordable city/settlement + a corner the game offered it at -> build there."""
        from .rules import COSTS, missing
        me = gs.me()
        offered = getattr(self, "_offered_spots", {})
        failed = getattr(self, "_failed_taps", set())
        for kind in ("city", "settlement"):
            if missing(dict(me.hand), COSTS[kind]) or kind in getattr(self, "_unaffordable", set()):
                continue
            # a city goes on one of our settlements (the plan may be cut after 3 failed build rounds)
            spots = set(offered.get(kind, ())) | (set(me.settlements) if kind == "city" else set())
            spots = [v for v in spots if (kind, v) not in failed]
            if not spots:
                continue
            from .brain import Action
            v = max(spots, key=gs.board.vertex_pips)
            self.note(f"plan has no build but {kind} is paid for: building at {v}, where the game offered it")
            self.__dict__.setdefault("_failed_taps", set()).add((kind, v))   # one try per spot per turn
            self.tap_build(Action(kind, v))
            return True
        return False

    def _affordable_keep(self, gs) -> dict:
        """Cards of every target build we can pay for right now: a bank trade never spends them
        (solo 2026-10-06 02:33: 3 brick -> ore for a knight broke the settlement it had just paid for)."""
        from .rules import COSTS, missing
        me = gs.me()
        keep = {}
        for kind in ["city", "settlement"] + list(self.brain._trade_targets(gs)):
            cost = COSTS.get(kind, {})
            if cost and not missing(dict(me.hand), cost):
                for r, n in cost.items():
                    keep[r] = max(keep.get(r, 0), n)
        return keep

    def _bank_into_build(self, gs) -> bool:
        """Owner 2026-10-03: players had no wool -> bank 3 brick for the wool, then build and
        activate the knight. A build one card short gets that card from the bank/harbor, paid
        only with cards the build itself doesn't use."""
        from .rules import COSTS, missing
        if getattr(self, "_sheds", 0) >= 4:
            return False
        me = gs.me()
        keep = self._affordable_keep(gs)
        # owner 2026-10-06: one grain short of the second city, it banked 3 ore for the settlement's
        # lumber, then 3 brick back into ore for a knight. A city comes before a settlement.
        targets = list(self.brain._trade_targets(gs))
        if "city" in targets and "settlement" in targets and targets.index("city") > targets.index("settlement"):
            targets.remove("city")
            targets.insert(targets.index("settlement"), "city")
        for kind in targets:
            cost = dict(COSTS.get(kind, {}))
            if not cost:
                continue
            if kind == "knight":
                cost["grain"] = cost.get("grain", 0) + 1      # owner: built AND activated
            need = missing(dict(me.hand), cost)
            if sum(need.values()) != 1:
                continue
            res = next(iter(need))
            sold = self.__dict__.setdefault("_sold_turn", set())
            if res in sold:                               # owner: 3 ore -> lumber, then 3 brick -> ore
                continue
            val = self.brain.resource_value(gs, me)
            best = None
            for r in CARD_ORDER:
                if r == res:
                    continue
                ratio = self.brain.bank_ratio(gs, r)
                if self._cfg_3to1():
                    ratio = min(ratio, 3)
                if me.hand.get(r, 0) - max(cost.get(r, 0), keep.get(r, 0)) >= ratio:
                    c = ratio * val.get(r, 1.0)
                    if best is None or c < best[0]:
                        best = (c, r, ratio)
            if best:
                self._sheds = getattr(self, "_sheds", 0) + 1
                self.note(f"{kind} one card short: bank {best[2]} {best[1]} -> {res}")
                sold.add(best[1])
                self.bank_offer({best[1]: best[2]}, {res: 1})
                return True
        return False

    def _shed_hand(self, gs) -> bool:
        """Owner 2026-10-02: never end a turn over the hand limit -- bank-trade (harbor 3:1)
        the surplus into what we need (grain to rebuild a lost city, wool/ore for knights)
        until we are at or under the limit, so the next 7 takes nothing."""
        from .rules import COSTS
        from .brain import missing
        me = gs.me()
        limit = discard_limit(gs.mode, me)
        total = sum(me.hand.values())
        if total <= limit or getattr(self, "_sheds", 0) >= 4:
            return False
        if self._bank_into_build(gs):                # a trade that lets us build beats a dump
            return True
        want = None
        kinds = (["city"] if me.settlements and getattr(self, "_lost_cities", 0) else []) + self.brain._trade_targets(gs)
        for kind in kinds:
            need = missing(dict(me.hand), COSTS.get(kind, {}))
            if need:
                want = next(iter(need))
                break
        if want is None:
            want = min(("grain", "ore", "wool", "brick", "lumber"), key=lambda r: me.hand.get(r, 0))
        bt = self._bank_trade_for(gs, {want: 1})
        if bt is None:
            self.note(f"hand {total} > limit {limit}: no bank trade possible")
            return False
        self._sheds = getattr(self, "_sheds", 0) + 1
        self.note(f"hand {total} > limit {limit}: bank {bt.give} -> {bt.get} before ending the turn")
        self.bank_offer(bt.give, bt.get)
        return True

    def _clear_red_taps(self):
        """New cards: every build the popup showed red (can't pay) this turn gets another look."""
        red = getattr(self, "_unaffordable", set())
        if red:
            failed = getattr(self, "_failed_taps", set())
            for kind, t in [x for x in failed if x[0] in red]:
                failed.discard((kind, t))
                self.__dict__.get("_tapped_spots", {}).pop(t, None)
            self._unaffordable = set()

    def _expect_trade(self, give: dict, get: dict):
        """Live 2026-10-03: the hand read 5 s after a trade still lacked the ore just received,
        so the bot banked away the knight's wool. Keep the post-trade hand for a few reads."""
        # owner 2026-10-06 02:59: activate was red before the grain came in, the ore->grain trade
        # landed, and the "can't afford" mark ended the turn with the knight idle: city lost.
        # New cards -> every build the popup showed red gets another look.
        self._clear_red_taps()
        gs = self.gs
        if gs is None:
            return
        h = {k: v for k, v in gs.me().hand.items()}
        for k, v in give.items():
            h[k] = max(0, h.get(k, 0) - v)
        for k, v in get.items():
            h[k] = h.get(k, 0) + v
        self._trade_expect = (time.time(), h, set(get))

    def _cfg_3to1(self) -> bool:
        """Config "harbor_3to1" (owner-confirmed in one game) counts only until the bank dialog has
        shown the real ratios: probe game 2026-10-04 offered 3 wool at a 4:1 every turn."""
        return bool(self.cfg.get("harbor_3to1")) and not self.brain.ratio_seen

    def _read_bank_ratios(self, img=None) -> dict:
        """'4:1' / '3:1' / '2:1' under each card of the bank dialog's give row (resource -> ratio).
        Read on the native capture (x2 and x3 crops): the frame is too small for the labels."""
        if img is None:
            img = self.io.snapshot()
        if img is None:
            return {}
        row = UI["trade_give_row"]
        xs, yc = row["x"], row["y"] + 59
        x0, x1, y0, y1 = xs[0] - 50, xs[-1] + 50, yc - 18, yc + 19
        io = self.io
        native, lay = getattr(io, "native", None), getattr(io, "lay", None)
        if getattr(io, "_frame", None) is img and native is not None and lay is not None:
            (nx0, ny0), (nx1, ny1) = lay.to_native(x0, y0), lay.to_native(x1, y1)
            src, back = native[int(ny0):int(ny1), int(nx0):int(nx1)], lambda x: lay.to_frame(nx0 + x, 0)[0]
        else:
            src, back = img[y0:y1, x0:x1], lambda x: x0 + x
        out = {}
        for sc in (2, 3):
            try:
                big = cv2.resize(src, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
                boxes = self.io.ocr(big)
            except Exception:                        # noqa: BLE001
                continue
            for o in boxes:
                m = re.fullmatch(r"\s*([234])\s*[:;.]\s*1\s*", o["text"])
                if not m:
                    continue
                cx = back((o["x"] + o["w"] / 2) / sc)
                i = min(range(len(xs)), key=lambda k: abs(xs[k] - cx))
                if abs(xs[i] - cx) < 30:
                    out.setdefault(CARD_ORDER[i], int(m.group(1)))
        if len(out) >= 3:
            # probe 2026-10-04: ore's label went unread and 3 ore was offered at a 4:1 twice. A 3:1
            # harbour shows on every card, so an unread card has the common rate -- unless the
            # board says it has its own 2:1 (resource harbour, Trade 3 for commodities)
            common = 3 if 3 in out.values() else 4
            gs = getattr(self, "gs", None)
            for r in CARD_ORDER:
                if r not in out:
                    own = None
                    if gs is not None:
                        try:
                            own = __import__("catanbot.moves", fromlist=["bank_ratio"]).bank_ratio(gs, r)
                        except Exception:            # noqa: BLE001
                            own = None
                    out[r] = 2 if own == 2 else common
        return out

    def _read_dialog_counts(self, img=None) -> dict:
        """Our card counts as the trade dialog shows them (bottom-left of each card in the give
        row). Game 9 14:08: the hand bar said 5 lumber, the dialog 2 -- a 4-lumber trade failed
        twice. Native crop per card, read like the hand bar ("VP n" trick)."""
        if img is None:
            img = self.io.snapshot()
        io = self.io
        native, lay = getattr(io, "native", None), getattr(io, "lay", None)
        ocr_fn = getattr(self.hand, "ocr_fn", None) or getattr(io, "ocr", None)
        if img is None or native is None or lay is None or getattr(io, "_frame", None) is not img or ocr_fn is None:
            return {}
        row = UI["trade_give_row"]
        out = {}
        for res, x in zip(CARD_ORDER, row["x"]):
            (a, b), (c, d) = lay.to_native(x - 36, row["y"] + 14), lay.to_native(x - 12, row["y"] + 37)
            crop = native[int(b):int(d), int(a):int(c)]
            if crop.size == 0:
                continue
            m = (cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) > 170).astype(np.uint8) * 255
            r = cv2.resize(255 - m, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
            mm = cv2.copyMakeBorder(r, 40, 40, 140, 60, cv2.BORDER_CONSTANT, value=255)
            cv2.putText(mm, "VP", (10, 40 + r.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 2, 0, 5)
            try:
                boxes = ocr_fn(cv2.cvtColor(mm, cv2.COLOR_GRAY2BGR))
            except Exception:                        # noqa: BLE001
                continue
            for t in boxes:
                q = re.search(r"VP\s*\[?(\d{1,2})\b", t["text"])
                if q and int(q.group(1)) <= 30:
                    out[res] = int(q.group(1))
                    break
        return out

    def _bank_batch(self, gs, plan) -> list:
        """Owner 2026-10-09 21:09: "took too much time to trade, if it can do multiple trades at
        once, it should try that". The plan's bank trades (before or between its builds) that the
        hand pays for together right now: every give comes out of the current hand, and walking
        the plan with those trades done first never leaves a build before them short."""
        from .moves import build_cost
        from .rules import missing
        room = 4 - getattr(self, "_sheds", 0)
        blocked = self.__dict__.get("_bank_blocked", set())
        hand0 = dict(gs.me().hand)
        last_build = max((i for i, a in enumerate(plan) if a.kind not in ("bank_trade", "player_trade")),
                         default=-1)
        picked, gives = [], {}
        for i, a in enumerate(plan):
            if a.kind != "bank_trade" or not a.give or not a.get or len(picked) >= room:
                continue
            if i > last_build and picked:
                break                                # trailing trades wait for the re-plan
            key = (tuple(sorted(a.give.items())), tuple(sorted(a.get.items())))
            if key in blocked:
                continue
            g2 = dict(gives)
            for r, n in a.give.items():
                g2[r] = g2.get(r, 0) + n
            if missing(hand0, g2):
                continue                             # the hand can't pay this one alongside the others
            hand = dict(hand0)
            for t in picked + [a]:
                for r, n in t.give.items():
                    hand[r] = hand.get(r, 0) - n
                for r, n in t.get.items():
                    hand[r] = hand.get(r, 0) + n
            ok = True
            for b in plan[:i]:                       # builds planned before this trade still paid
                if b.kind in ("bank_trade", "player_trade"):
                    continue
                try:
                    cost = build_cost(gs, b.kind, b.target)
                except Exception:                    # noqa: BLE001
                    cost = {}
                for r, n in cost.items():
                    hand[r] = hand.get(r, 0) - n
                if any(v < 0 for v in hand.values()):
                    ok = False
                    break
            if ok:
                picked.append(a)
                gives = g2
        return picked

    def bank_offer_batch(self, trades: list) -> int:
        """Several bank trades for one plan. (a) cfg bank_combined: every want and every give in
        one bank dialog, one trade press (UNVERIFIED live: frames saved before and after the press;
        a miss turns it off for the rest of the game). (b) back to back single trades, each checked
        as it lands, without a re-plan between them. Returns how many landed."""
        trades = [(dict(g), dict(w)) for g, w in trades]
        if len(trades) == 1:
            return int(bool(self.bank_offer(*trades[0])))
        # owner 2026-10-09 23:20 "in solo games the bots don't know to do combined trades, but bot
        # should still try in games with my friends": combined only in games against people
        if self.cfg.get("bank_combined", True) and not getattr(self, "_bank_combined_off", False) \
                and not (self.cfg.get("solo") and self.cfg.get("bank_combined_people_only", True)):
            got = self._bank_combined(trades)
            if got is True:
                return len(trades)
            if got is None:                          # something moved or was refused: re-read first
                return 0
        n = 0
        for give, get in trades:
            if not self.bank_offer(give, get):
                break
            n += 1
            gs = getattr(self, "gs", None)
            if gs is not None:                       # the next trade starts from this one's hand
                h = gs.me().hand
                for r, k in give.items():
                    h[r] = max(0, h.get(r, 0) - k)
                for r, k in get.items():
                    h[r] = h.get(r, 0) + k
        self.note(f"bank batch: {n}/{len(trades)} trades landed back to back")
        return n

    def _bank_combined(self, trades: list):
        """Every trade in one dialog, one press. True = landed; False = nothing moved (or not
        tried): single trades now; None = refused or the dialog closed on a partial move: re-plan."""
        give, get = {}, {}
        for g, w in trades:
            for r, n in g.items():
                give[r] = give.get(r, 0) + n
            for r, n in w.items():
                get[r] = get.get(r, 0) + n
        if set(give) & set(get):
            self.note(f"bank combined: a card is both given and wanted ({give}->{get}); one at a time")
            return False
        self.probe.tried("bank_combined")
        gs0 = getattr(self, "gs", None)
        ref = self.io.snapshot()
        before = self._hand_now(ref)
        if not any(before.values()) and gs0 is not None:
            before = {k: v for k, v in gs0.me().hand.items()}
        self.click(UI["trade_open"], "open trade")
        self.click(UI["trade_bank"], "bank tab")
        verify = not isinstance(self.io, ReplayIO) or self.cfg.get("verify_in_replay")
        poll = self.cfg.get("bank_settle_poll", True)
        dlg = None
        if verify:
            if poll:                                 # SPEED_PLAN 4: poll, not a fixed 0.5 s
                dlg = self._settled_snapshot(ref, 0.5)
                seen = self._read_bank_ratios(dlg) if any(self._ratios_unknown(g, w) for g, w in trades) else {}
            else:
                time.sleep(0.5)
                seen = self._read_bank_ratios()
            if seen:
                self.brain.ratio_seen.update(seen)
                self._check_harbors(self.brain.ratio_seen)
                self._save_game_flags()
                self.note(f"bank ratios seen: {seen}")
                bad = [(g, w) for g, w in trades
                       if any(r in seen and n != seen[r] * sum(w.values()) for r, n in g.items())]
                if bad:
                    self.note(f"bank combined: {bad} do not fit the ratios {seen}: cancelled, re-planning")
                    self.click(UI["trade_cancel"], "cancel bank trade (ratio)")
                    self._refused("bank_trade")
                    return None
            held = self._read_dialog_counts(dlg) if poll else self._read_dialog_counts()
            short = {r: (n, held[r]) for r, n in give.items() if r in held and held[r] < n}
            if short:
                self.note(f"bank combined {give}->{get}: the dialog shows {held}, short {short}: cancelled")
                if gs0 is not None:
                    gs0.me().hand.update(held)
                self._last_hand = {k: v for k, v in held.items() if v}
                self.click(UI["trade_cancel"], "cancel bank trade (cards not held)")
                self._refused("bank_trade")
                return None
        for res, n in get.items():
            for _ in range(n):
                self.click((UI["trade_want_row"]["x"][CARD_ORDER.index(res)], UI["trade_want_row"]["y"]), f"bank want {res}")
        for res, n in give.items():
            for _ in range(n):
                self.click((UI["trade_give_row"]["x"][CARD_ORDER.index(res)], UI["trade_give_row"]["y"]), f"bank give {res}")
        if verify:                                   # UNVERIFIED flow: keep the dialog as filled in
            img0 = self.io.snapshot()
            self._probe("bank_combined_before", img0, texts=[o["text"] for o in self.io.ocr(img0)] if img0 is not None else [],
                        note={"give": give, "get": get, "before": before})
        self.click(UI["trade_ok"], f"bank trade {give}->{get} (combined)")
        self._bank_clicked_at = time.time()
        if not verify:
            self.note(f"bank trade landed (combined, {len(trades)} in one, UNVERIFIED replay): {give}->{get}")
            self._expect_trade(give, get)
            return True
        img, after = self._bank_after(dlg, poll)
        img_ocr = self.io.ocr(img) if img is not None else []         # one OCR for the probe and the check
        if poll and any(after.values()) and self._hand_moved(before, after, give, get) is False:
            img, after, _, img_ocr = self._bank_recheck(img, after, False, img_ocr, before, give, get)
        texts = [o["text"] for o in img_ocr]
        self._probe("bank_combined_after", img, texts=texts,
                    note={"give": give, "get": get, "before": before, "after": after})
        ok = self._hand_moved(before, after, give, get) if any(after.values()) else None
        still_open = bool(img is not None and find_text(img_ocr, r"maritime trade|trade to the supply"))
        if ok or (ok is None and not still_open):
            if still_open:
                self.click(UI["trade_cancel"], "close bank dialog (trade already landed)")
            self.probe.result("bank_combined", ok, f"{give}->{get}")
            self.note(f"bank trade landed (combined, {len(trades)} in one): {give}->{get}"
                      + ("" if ok else " (UNVERIFIED: hand unreadable)"))
            self._expect_trade(give, get)
            return True
        self._bank_combined_off = True               # one trade at a time for the rest of the game
        self.probe.result("bank_combined", False, f"{give}->{get}")
        if still_open or before == after:
            self.note(f"bank combined {give}->{get} did not land (hand {before} -> {after}): "
                      f"one trade at a time from now on")
            if still_open:
                self.click(UI["trade_cancel"], "cancel failed bank trade (combined)")
                time.sleep(0.8)
            return False
        # the dialog closed and the hand moved only in part: never trade again on a guess
        self.note(f"bank combined {give}->{get}: dialog closed, hand {before} -> {after} only partly moved; "
                  f"one trade at a time from now on, re-reading first")
        return None

    def _ratios_unknown(self, give: dict, get: dict) -> bool:
        """SPEED_PLAN 4: the dialog's ratio labels are read once a game -- again only when a give
        card has no ratio seen yet or the trade does not fit the ratios seen (a fleet, a new harbour)."""
        if not self.cfg.get("bank_settle_poll", True):
            return True
        seen = getattr(self.brain, "ratio_seen", None) or {}
        k = sum(get.values()) or 1
        return any(r not in seen or n != seen[r] * k for r, n in give.items())

    def _bank_after(self, dlg, poll: bool):
        """The frame and hand after the bank trade press. poll: wait for the screen to settle (at
        most the old 1.2 + 1.0 s); an all-zero hand (bar under the closing panel) gets one more look."""
        if poll:
            img = self._settled_snapshot(dlg, 2.2)
            after = self._hand_now(img)
            if not any(after.values()):
                if self._live_waits():
                    time.sleep(0.5)                  # probe game 3 09:28: zeros under the closing panel
                img = self.io.snapshot()
                after = self._hand_now(img)
            return img, after
        time.sleep(1.2)
        img = self.io.snapshot()
        after = self._hand_now(img)
        if not any(after.values()):
            time.sleep(1.0)                          # bar still under the panel: one more look
            img = self.io.snapshot()
            after = self._hand_now(img)
        return img, after

    def _bank_recheck(self, img, after, ok, img_ocr, before, give, get):
        """A settled frame can come before the hand bar's counts move: a 'not moved' read with the
        dialog closed is looked at once more at the old 1.2 s mark before it counts as a miss (a
        miss trades again -- never on an early frame)."""
        if img is None or find_text(img_ocr, r"maritime trade|trade to the supply"):
            return img, after, ok, img_ocr
        left = 1.2 - (time.time() - self.__dict__.get("_bank_clicked_at", 0))
        if left > 0 and self._live_waits():
            time.sleep(left)
        img2 = self.io.snapshot()
        if img2 is None:
            return img, after, ok, img_ocr
        after2 = self._hand_now(img2)
        ok2 = self._hand_moved(before, after2, give, get) if any(after2.values()) else None
        self.note(f"bank trade: hand read again at 1.2 s: {after} -> {after2}")
        return img2, after2, ok2, self.io.ocr(img2)

    def bank_offer(self, give: dict, get: dict) -> bool:
        """Bank/harbor trade through the trade panel's bank tab. Audit 2026-10-04: about half
        landed and failures repeated 4x in a row: verify by the hand change, cancel a dialog
        still open, retry once, then block that trade for the rest of the turn."""
        key = (tuple(sorted(give.items())), tuple(sorted(get.items())))
        blocked = self.__dict__.setdefault("_bank_blocked", set())
        if key in blocked:
            self.note(f"bank trade {give}->{get} failed twice this turn; not repeating")
            return False
        self.probe.tried("bank_trade")
        gs0 = getattr(self, "gs", None)
        planned = {k: v for k, v in gs0.me().hand.items()} if gs0 is not None else {}
        for attempt in range(2):
            # probe game 3 09:28: a fresh read under the closing panel was all zeros and a trade
            # counted as failed -- start from the hand we planned with unless a clean read differs
            ref = self.io.snapshot()
            before = self._hand_now(ref)
            if not any(before.values()) or (attempt and planned):
                before = dict(planned) or before
            self.click(UI["trade_open"], "open trade")
            self.click(UI["trade_bank"], "bank tab")
            poll = self.cfg.get("bank_settle_poll", True)
            dlg = None
            if not isinstance(self.io, ReplayIO) or self.cfg.get("verify_in_replay"):
                if poll:
                    # owner 2026-10-09 "play every move way faster": poll until the dialog is up
                    # (was a fixed 0.5 s); one frame for the ratio and the card-count reads
                    dlg = self._settled_snapshot(ref, 0.5)
                    seen = self._read_bank_ratios(dlg) if self._ratios_unknown(give, get) else {}
                else:
                    time.sleep(0.5)
                    seen = self._read_bank_ratios()
                if seen:
                    self.brain.ratio_seen.update(seen)
                    self._check_harbors(self.brain.ratio_seen)
                    self._save_game_flags()
                    self.note(f"bank ratios seen: {seen}")
                    k = sum(get.values())
                    bad = {r: (n, seen[r]) for r, n in give.items() if r in seen and n != seen[r] * k}
                    if bad:
                        self.note(f"bank trade {give}->{get} does not fit the ratios {bad}: cancelled, re-planning")
                        self.click(UI["trade_cancel"], "cancel bank trade (ratio)")
                        self._refused("bank_trade")
                        return False
                held = self._read_dialog_counts(dlg) if poll else self._read_dialog_counts()
                short = {r: (n, held[r]) for r, n in give.items() if r in held and held[r] < n}
                if short:
                    self.note(f"bank trade {give}->{get}: the dialog shows {held}, short {short}: cancelled")
                    gs1 = getattr(self, "gs", None)
                    if gs1 is not None:
                        gs1.me().hand.update(held)
                    self._last_hand = {k: v for k, v in held.items() if v}
                    self.click(UI["trade_cancel"], "cancel bank trade (cards not held)")
                    self._refused("bank_trade")
                    blocked.add(key)
                    return False
            for res, n in get.items():
                self.click((UI["trade_want_row"]["x"][CARD_ORDER.index(res)], UI["trade_want_row"]["y"]), f"bank want {res}")
            for res, n in give.items():
                for _ in range(n):
                    self.click((UI["trade_give_row"]["x"][CARD_ORDER.index(res)], UI["trade_give_row"]["y"]), f"bank give {res}")
            self.click(UI["trade_ok"], f"bank trade {give}->{get}")
            self._bank_clicked_at = time.time()
            if isinstance(self.io, ReplayIO) and not self.cfg.get("verify_in_replay"):
                self._expect_trade(give, get)
                return True
            img, after = self._bank_after(dlg, poll)
            ok = self._hand_moved(before, after, give, get) if any(after.values()) else None
            img_ocr = self.io.ocr(img) if img is not None else []     # one OCR for both checks below
            if poll and ok is False:
                img, after, ok, img_ocr = self._bank_recheck(img, after, ok, img_ocr, before, give, get)
            if img is not None and find_text(img_ocr, r"maritime trade|trade to the supply"):
                if ok:
                    # solo 2026-10-06 03:35: the trade landed but the dialog stayed open; read as "not
                    # through", the re-pick traded again (6 ore -> 2 grain, the city's ore gone).
                    # The hand moved: it landed -- close the dialog, never trade a second time
                    self.click(UI["trade_cancel"], "close bank dialog (trade already landed)")
                    self.probe.result("bank_trade", True, f"{give}->{get} dialog left open")
                    self.note(f"bank trade landed (dialog stayed open, closed it): {give}->{get}")
                    self._expect_trade(give, get)
                    return True
                ok = False                            # the dialog is still open: the trade did not go through
            if ok is not False:
                self.probe.result("bank_trade", ok)
                self.note(f"bank trade landed: {give}->{get}" + ("" if ok else " (UNVERIFIED: hand unreadable)"))
                self._expect_trade(give, get)
                return True
            texts = [o["text"] for o in img_ocr]
            if re.search(r"more resources? that you would like to receive", " ".join(texts), re.I):
                # final game 14:42: at a 2:1 the "want" click (made before the gives) was dropped and
                # the dialog asked for 1 more to receive: pick it again now and confirm
                for res in get:
                    self.click((UI["trade_want_row"]["x"][CARD_ORDER.index(res)], UI["trade_want_row"]["y"]), f"bank want {res} (again)")
                self.click(UI["trade_ok"], f"bank trade {give}->{get} (after re-picking the want)")
                time.sleep(1.2)
                img = self.io.snapshot()
                after = self._hand_now(img)
                if img is not None and not find_text(self.io.ocr(img), r"maritime trade|trade to the supply"):
                    ok = self._hand_moved(before, after, give, get) if any(after.values()) else None
                    self.probe.result("bank_trade", True if ok is None else ok, f"{give}->{get} after re-picking")
                    self.note(f"bank trade landed after re-picking the want: {give}->{get}")
                    self._expect_trade(give, get)
                    return True
                texts = [o["text"] for o in self.io.ocr(img)] if img is not None else []
            self._probe("bank_failed", img, texts=texts,
                        note={"give": give, "get": get, "before": before, "after": after, "attempt": attempt})
            self.note(f"bank trade {give}->{get} did not land (hand {before} -> {after})")
            if re.search(r"maritime trade|trade to the supply|bank", " ".join(texts), re.I):
                self.click(UI["trade_cancel"], "cancel failed bank trade")
                time.sleep(0.8)
        self.probe.result("bank_trade", False, f"{give}->{get} twice")
        blocked.add(key)
        return False

    def offer(self, give: dict, get: dict):
        """Build the offer in the trade panel and send it (calibrated in gc_trade_*)."""
        self.offers_this_turn += 1
        self._pending_offer = (dict(give), dict(get))
        self._offer_sent_at = time.time()
        self.click(UI["trade_open"], "open trade")
        for res, n in get.items():
            for _ in range(n):
                self.click((UI["trade_want_row"]["x"][CARD_ORDER.index(res)], UI["trade_want_row"]["y"]), f"want {res}")
        for res, n in give.items():
            for _ in range(n):
                self.click((UI["trade_give_row"]["x"][CARD_ORDER.index(res)], UI["trade_give_row"]["y"]), f"give {res}")
        self.click(UI["trade_ok"], f"send offer {give}->{get}")
        self.offer_sent = time.time()

    RESP_XS = _RESP_XS0 = (316, 364, 412)                  # answer circles under our offer, one per opponent
    _resp_t = None

    def offer_responses(self, img) -> list[tuple[int, str]]:
        """Each opponent's answer to our offer: accept (check), decline (x) or thinking (dots).
        Templates from live frames 2026-10-02 (data/calib/offer_response.npz)."""
        if Runner._resp_t is None:
            d = np.load(ROOT / "data" / "calib" / "offer_response.npz")
            Runner._resp_t = {k: d[k] for k in d.files}
        out = []
        for x in self.RESP_XS:
            # search +-16 px around the laptop spot: full screen the circles sat ~10 px lower and an
            # exact 18x18 probe missed Rhea's accept (live 2026-10-03)
            ry = int(LAY.y(801, "b"))
            c = cv2.cvtColor(img[ry - 16:ry + 34, x - 25:x + 25], cv2.COLOR_BGR2HSV)
            m = ((c[..., 1] < 70) & (c[..., 2] > 190)).astype(np.float32)
            if m.std() == 0:
                out.append((x, "none"))
                continue
            sc = {k: float(cv2.matchTemplate(m, t, cv2.TM_CCOEFF_NORMED).max()) for k, t in Runner._resp_t.items()}
            k = max(sc, key=sc.get)
            out.append((x, k if sc[k] > 0.6 else "thinking"))
        return out

    def _handle_responses(self, img) -> bool:
        """Owner 2026-10-02: a player accepted (check under their colour) and the bot never
        took it. Click the accepting player's circle; if everyone declined, stop waiting."""
        if not getattr(self, "offer_sent", 0):
            return False
        resp = self.offer_responses(img)
        n_opp = sum(1 for p in (self.gs.players if getattr(self, "gs", None) else []) if not p.is_me)
        if n_opp:
            resp = resp[:n_opp]                       # 3-player game: the 3rd circle is always empty
        states = [k for _, k in resp]
        if time.time() - getattr(self, "_resp_log_t", 0) > 3:
            # live 2026-10-03 full screen: an accept was missed; keep what we saw for calibration
            self._resp_log_t = time.time()
            self.note(f"offer answers so far {resp}")
            try:
                d = ROOT / "data" / "trade_wait"; d.mkdir(exist_ok=True)
                cv2.imwrite(str(d / f"{time.strftime('%H%M%S')}_wait.png"), img)
            except Exception:                        # noqa: BLE001
                pass
        if "accept" in states:
            # owner 2026-10-03: several yeses -> trade with the one furthest behind, and never
            # with a player holding more than 1.5x our points
            opp = [p for p in (self.gs.players if self.gs else []) if not p.is_me]
            my_vp = max(1, self.gs.me().vp_visible) if self.gs else 1
            cands = []
            for i, (x, k) in enumerate(resp):
                if k != "accept":
                    continue
                p = opp[i] if i < len(opp) else None
                vp = p.vp_visible if p else 0
                if p is not None and vp > 1.5 * my_vp:
                    self.note(f"offer: {p.name} said yes but has {vp} VP vs our {my_vp}: not trading")
                    continue
                po = getattr(self, "_pending_offer", None)
                why = self.brain.accepter_refusal(self.gs, p.name, *po) if p is not None and po else None
                if why:
                    # review 0.3.1: an accepter still has to pass the partner rules
                    self.note(f"offer: {p.name} said yes but {why}: not trading")
                    continue
                cands.append((vp, x, p.name if p else "?"))
            if not cands:
                if self._offer_answered(states) or time.time() - self.offer_sent >= self.offer_wait_s():
                    self.click(UI["trade_cancel"], "close trade: only leaders accepted")
                    self.trade_refused = getattr(self, "last_want", None)
                    self.offer_sent = 0
                    return True
                return False
            vp, x, who = min(cands)
            self.note(f"offer answers {states}: taking the trade with {who} ({vp} VP)")
            self.click((x, int(LAY.y(810, "b"))), "accept trade with player who said yes")
            time.sleep(1.0)
            snap = self.io.snapshot()
            try:
                d = ROOT / "data" / "trade_wait"; d.mkdir(exist_ok=True)
                cv2.imwrite(str(d / f"{time.strftime('%H%M%S')}_took.png"), snap)
            except Exception:                        # noqa: BLE001
                pass
            # live 2026-10-02: clicking their check opens "Do you want to accept the offer of
            # player X, or decline the trade?" -- the green check confirms. The bot used to hit
            # the red X next ("close leftover panel") and refused trades it had accepted.
            self._confirm_accept(snap)
            if getattr(self, "_pending_offer", None):
                self._expect_trade(*self._pending_offer)
                g, w = self._pending_offer            # we give g, get w: `who` gets g, gives w
                if who != "?":
                    self._ledger().trade(who, dict(w), self._ledger().me or "", dict(g))
            self.offer_sent = 0
            self.last_want = None
            return True
        waited = time.time() - self.offer_sent
        # owner 2026-10-06: everyone said no -> move on at once (no waiting for a change of mind).
        # Wait up to 15 s only while at least one player has not answered ("thinking"/"none").
        if states and (self._offer_answered(states) or waited >= self.offer_wait_s()):
            self.note(f"offer answers {states}: " + ("all declined" if self._offer_answered(states)
                                                      else f"no yes after {waited:.0f} s"))
            if self._offer_answered(states) and getattr(self, "_pending_offer", None):
                self._remember_decline(*self._pending_offer)
            self.click(UI["trade_cancel"], "close declined trade")
            self.trade_refused = getattr(self, "last_want", None)
            self.offer_sent = 0
            return True
        return False

    @staticmethod
    def _offer_answered(states) -> bool:
        """Every opponent has answered (yes or no); nobody is still thinking."""
        return bool(states) and all(k in ("accept", "decline") for k in states)

    def offer_wait_s(self) -> float:
        # owner 2026-10-09 "play every move way faster": the 15 s floor hid trade_wait_s (people
        # answered in 3-16 s); trade_wait_floor_s 15 = the old wait
        return max(float(self.cfg.get("trade_wait_floor_s", 15)), float(self.cfg.get("trade_wait_s", 15)))

    def _offer_allowed(self, urgent: bool = False) -> bool:
        """One more player offer this turn? max_trade_offers, then (SPEED_PLAN 3) offers_per_turn
        and no new offer offer_cutoff_s into the turn -- the city-save ask (urgent) skips the cutoff."""
        cap = int(self.cfg.get("max_trade_offers", 3))
        per = int(self.cfg.get("offers_per_turn", 0) or 0)
        if per:
            cap = min(cap, per)
        if self.offers_this_turn >= cap:
            return False
        cut = float(self.cfg.get("offer_cutoff_s", 0) or 0)
        ts = self.__dict__.get("turn_started")
        used = time.time() - ts if ts else 0.0
        clock = self._turn_cut("offer")
        if clock is not None:
            # 0.4.3: the cutoff scales with the game's timer (100 s -> 45 s) and runs on our clock
            cut, used = clock, self._turn_used_s()
            if urgent and self._turn_past("end"):
                return False                         # past turn_end_frac not even the city-save ask fits
        if cut and not urgent and ts and used > cut:
            if self.__dict__.get("_cutoff_noted") != ts:
                self._cutoff_noted = ts
                self.note(f"offer cutoff: {used:.0f} s into the turn, no new player offer")
            return False
        return True

    def _confirm_accept(self, img=None, ocr=None) -> bool:
        ocr = ocr if ocr is not None else self.io.ocr(img if img is not None else self.io.snapshot())
        if find_text(ocr, r"accept the offer of"):
            self.click(UI["trade_ok"], "confirm trade (green check)")
            return True
        return False

    def on_trade_panel(self, img, ocr, texts, info):
        if self._confirm_accept(img, ocr):
            return
        if self._handle_responses(img):
            return
        # owner 2026-10-02: give the humans at least 10 s to answer before closing the offer
        waited = time.time() - getattr(self, "offer_sent", 0)
        if waited < self.offer_wait_s():
            return                                   # owner: at least 15 s, then close (no need to time out)
        self.click(UI["trade_cancel"], f"close unanswered trade after {waited:.0f}s")
        # review 0.3.1: with no offer open (offer_sent 0: already closed) this re-armed a want the
        # turn had used up -> a second bank trade for the same card. Re-arm only a live offer's want.
        if getattr(self, "offer_sent", 0) and getattr(self, "last_want", None):
            self.trade_refused = self.last_want
        self.last_want, self.offer_sent = None, 0

    # card background colour (HSV median of saturated pixels) on the offer panel.
    # wool/grain measured from gc_trade_offer_in.jpg, lumber live 2026-10-02 (34,110,121); the rest are first guesses, refined
    # from the crops saved in data/offers/ (UNCONFIRMED until labelled).
    OFFER_HSV = {"wool": (35, 129, 159), "grain": (21, 135, 188), "brick": (14, 138, 169),
                 "lumber": (34, 110, 121), "ore": (105, 50, 140),
                 "paper": (29, 121, 127)}       # paper: live 2026-10-03 full screen (Rhea wool-for-paper)

    def _offer_card(self, img, y, dx=0):
        # laptop x 1150..1182; full screen the panel is right-anchored (live 2026-10-03: the
        # fixed column read the table (12,180,103) and every offer was declined); dx = how far
        # left of the rightmost column this card's count sits (multi-card offers)
        x0, x1 = int(LAY.x(1150, "r")), int(LAY.x(1182, "r"))
        pitch = LAY.x(1165, "r") - LAY.x(1065, "r")      # card columns sit 100 px apart
        k = round(dx / pitch) if pitch else 0
        if k > 0:
            # live 2026-10-09 20:40: the left-biased window read wool's dark edge as lumber --
            # an inner column is sampled across the card's middle
            x0, x1 = int(LAY.x(1162, "r") - k * pitch), int(LAY.x(1188, "r") - k * pitch)
        c = cv2.cvtColor(img[int(y) - 22:int(y) + 22, x0:x1], cv2.COLOR_BGR2HSV).reshape(-1, 3)
        m = c[:, 1] > 60
        grey = c[:, 1] < 60
        if grey.mean() > 0.6 and np.median(c[grey][:, 2]) > 120:
            # ore is a grey card: the saturated-pixel median only sees the frame (live
            # 2026-10-02, Vale's wool-for-ore read (21,117,115) and was declined)
            return "ore", tuple(int(x) for x in np.median(c[grey], 0))
        if m.sum() < 50:
            return None, None
        hsv = np.median(c[m], 0)
        if hsv[2] < 110:                 # table/pieces behind a panel still sliding in
            return None, tuple(int(x) for x in hsv)
        d = {k: abs(hsv[0] - v[0]) * 3 + abs(hsv[1] - v[1]) * 0.3 + abs(hsv[2] - v[2]) * 0.4
             for k, v in self.OFFER_HSV.items()}
        best = sorted(d, key=d.get)
        # solo 2026-10-03: lumber (35,126,124) was 5.9 nearer than wool and read as unknown
        ok = d[best[0]] < 30 and d[best[1]] - d[best[0]] > 4
        return (best[0] if ok else None), tuple(int(x) for x in hsv)

    def _counter_offer(self, img, ocr) -> bool:
        """'Counter offer -- Confirm your counteroffer or decline the trade.' (live 2026-10-03:
        Rhea countered our offer and the bot sat on it). The partner row is on top with a green
        marker under the card they give; our row below with a red marker over the card we give.
        Counts assumed 1 each (UNCONFIRMED for multiples)."""
        hdr = find_text(ocr, r"confirm your counter ?offer")
        if not hdr:
            return False
        title = find_text(ocr, r"^counter ?offer$")
        cx = (title["x"] + title["w"] / 2) if title else hdr["x"] + hdr["w"] / 2
        hy, sc = hdr["y"], LAY.s
        sp = 115 / sc                                   # card spacing (full screen 115 px native)
        y0, y1 = int(hy + 150 / sc), int(hy + 250 / sc)
        x0, x1 = int(cx - 4.5 * sp), int(cx + 4.5 * sp)
        hsv = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        marks = {}
        for side, (lo, hi) in {"get": ((45, 150, 100), (75, 255, 255)),
                               "give": ((0, 170, 120), (6, 255, 255))}.items():
            m = cv2.inRange(hsv, np.array(lo), np.array(hi))
            n, _, st, cen = cv2.connectedComponentsWithStats(m)
            got = {}
            for i in range(1, n):
                a = st[i, cv2.CC_STAT_AREA]; w = st[i, cv2.CC_STAT_WIDTH]; h = st[i, cv2.CC_STAT_HEIGHT]
                if 15 <= a <= 600 and w > h:            # flat triangle, not a card or badge
                    k = round((x0 + cen[i][0] - cx - 12 / sc) / sp + 3.5)
                    if 0 <= k < len(CARD_ORDER):
                        got[CARD_ORDER[k]] = 1
            marks[side] = got
        get, give = marks["get"], marks["give"]
        key = ("counter", tuple(sorted(get.items())), tuple(sorted(give.items())))
        if getattr(self, "_answered", None) == key:
            return True
        names = [n for n, _ in (getattr(self, "game_colors", None) or [])]
        partner = next((o["text"].strip() for o in ocr if o["text"].strip() in names
                        and o["y"] > hy and o["x"] < x0), "")
        gs = self.gs
        ok = self._counter_ok(gs, partner, give, get)
        try:
            d = ROOT / "data" / "offers"; d.mkdir(exist_ok=True)
            cv2.imwrite(str(d / f"{time.strftime('%H%M%S')}_counter_in.png"), img)
        except Exception:                            # noqa: BLE001
            pass
        self.note(f"counter offer from {partner or '?'}: give {give} get {get}: {'ACCEPT' if ok else 'decline'} "
                  f"hand={ {k: v for k, v in gs.me().hand.items() if v} if gs else '?'}")
        # buttons: red X then green check, bottom right of the panel (~+300 px native below the hint)
        bx0, bx1 = int(cx + 250 / sc), int(cx + 480 / sc)
        by0, by1 = int(hy + 390 / sc), int(hy + 500 / sc)
        bh = cv2.cvtColor(img[by0:by1, bx0:bx1], cv2.COLOR_BGR2HSV)
        lo, hi = (((45, 80, 60), (75, 255, 255)) if ok else ((0, 150, 120), (6, 255, 255)))
        m = cv2.inRange(bh, np.array(lo), np.array(hi))
        n, _, st, cen = cv2.connectedComponentsWithStats(m)
        best = max(range(1, n), key=lambda i: st[i, cv2.CC_STAT_AREA], default=None)
        if best is None or st[best, cv2.CC_STAT_AREA] < 100:
            self.note("counter offer: button not found by colour")
            self._answered = key
            return True
        self.click((int(bx0 + cen[best][0]), int(by0 + cen[best][1])),
                   "accept counter offer" if ok else "decline counter offer")
        self._answered = key
        return True

    def _counter_ok(self, gs, partner, give, get) -> bool:
        """Accept a counter offer? Review 0.3.1: when the partner's name was not read (the old
        `last_partner` fallback was never set, so "" skipped the 14-VP lock and every VP guard)
        the trade must pass with every opponent."""
        if not get or not give or gs is None:
            return False
        if partner:
            return self.brain.evaluate_offer(gs, partner, give, get)
        opps = gs.opponents()
        return bool(opps) and all(self.brain.evaluate_offer(gs, q.name, give, get) for q in opps)

    @staticmethod
    def wildcard_decline_button(img, hdr):
        """Red X of the 'Offer -- Complete the wildcard trade, change it, or decline it.' panel:
        bottom right, about (+0.77, +1.0) header widths from the hint's centre (live 18:35)."""
        w = max(150, hdr["w"])
        cx = hdr["x"] + hdr["w"] / 2
        x0, x1 = int(max(0, cx + 0.45 * w)), int(min(img.shape[1], cx + 1.05 * w))
        y0, y1 = int(max(0, hdr["y"] + 0.8 * w)), int(min(img.shape[0], hdr["y"] + 1.25 * w))
        if x1 <= x0 or y1 <= y0:
            return None
        hsv = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        m = cv2.inRange(hsv, np.array((0, 150, 120)), np.array((6, 255, 255))) | \
            cv2.inRange(hsv, np.array((174, 150, 120)), np.array((180, 255, 255)))
        n, _, st, cen = cv2.connectedComponentsWithStats(m)
        best = max(range(1, n), key=lambda i: st[i, cv2.CC_STAT_AREA], default=None)
        if best is None or st[best, cv2.CC_STAT_AREA] < 150:
            return None
        return int(x0 + cen[best][0]), int(y0 + cen[best][1])

    def _wildcard_panel(self, img, ocr) -> bool:
        """Live 2026-10-08 18:34: after our counter was countered back (Bramble, 15 VP, asked
        for a coin we didn't hold) this panel stayed over our turn and every end-turn click
        missed. Nothing in it is worth taking from there: decline, every 3 s while it stays."""
        hdr = find_text(ocr, r"complete the wildcard trade|change it,? or decline")
        if not hdr:
            return False
        if time.time() - getattr(self, "_wild_decl_t", 0) < 3:
            return True
        self._wild_decl_t = time.time()
        b = self.wildcard_decline_button(img, hdr)
        if b is None:
            cx = hdr["x"] + hdr["w"] / 2
            b = (int(cx + 0.77 * hdr["w"]), int(hdr["y"] + 1.0 * hdr["w"]))
            self.note("wildcard panel: red X not seen, clicking its usual spot")
        try:
            d = ROOT / "data" / "offers"; d.mkdir(exist_ok=True)
            cv2.imwrite(str(d / f"{time.strftime('%H%M%S')}_wildcard_panel.png"), img)
        except Exception:                            # noqa: BLE001
            pass
        self.click(b, "decline wildcard panel")
        return True

    @staticmethod
    def _offer_button(img, which):
        """Centre of the red X / yellow bag / green check under the offer cards, found by colour
        (live 2026-10-03 full screen: the fixed y hit the buttons' bottom edge); None if not seen."""
        x0, x1 = int(LAY.x(1010, "r")), int(LAY.x(1225, "r"))
        y0, y1 = int(LAY.y(520, "c")), int(LAY.y(640, "c"))
        hsv = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        lo, hi = {"decline": ((0, 150, 150), (6, 255, 255)), "counter": ((15, 150, 180), (30, 255, 255)),
                  "accept": ((45, 100, 100), (75, 255, 255))}[which]
        m = cv2.inRange(hsv, np.array(lo), np.array(hi))
        n, _, st, cen = cv2.connectedComponentsWithStats(m)
        best = max(range(1, n), key=lambda i: st[i, cv2.CC_STAT_AREA], default=None)
        if best is None or st[best, cv2.CC_STAT_AREA] < 300:
            return None
        return int(x0 + cen[best][0]), int(y0 + cen[best][1])

    def _click_offer(self, which, why, img=None):
        img = self.io.snapshot() if img is None else img
        pt = self._offer_button(img, which)
        if pt is None:
            self.note(f"offer {which} button not found by colour, using calibrated spot")
            pt = UI[f"offer_in_{which}"]
        self.click(pt, why)

    def on_trade_offer_in(self, img, ocr, texts, info):
        hdr = find_text(ocr, r"offer from")
        if not hdr:
            return
        partner = hdr["text"].split("from", 1)[-1].strip()
        # OCR adds noise like "Rhea 1" (live 2026-10-02: same offer answered twice)
        partner = re.sub(r"\s+\S{1,2}$", "", partner)
        names = [n for n, _ in (getattr(self, "game_colors", None) or [])]
        partner = next((n for n in names if n.lower().startswith(partner.lower()[:5])), partner)
        # answer each offer ONCE: a second click on the same button undoes the answer
        # (live 2026-10-02: a 20 s time bucket re-clicked every 20 s, flipping accept/decline
        # back off, so trades the bot accepted never happened and declines never showed)
        # live 2026-10-09 20:39: a multi-card offer lays its cards out in columns to the LEFT
        # (+brick +2 wool +1 ore for -1 lumber) and only the rightmost column was read, so a 4:1
        # was declined as 1:1 -- every column is read; a row's card sits at its count's offset
        x_lo = LAY.x(700, "r") if (getattr(self, "cfg", None) or {}).get("offer_columns", True) else LAY.x(1080, "r")
        toks = [o for o in ocr if x_lo < o["x"] < LAY.x(1165, "r") and LAY.y(430, "c") < o["y"] < LAY.y(590, "c")
                and re.fullmatch(r"[+-]\d", o["text"].strip())]
        ref_x = max((o["x"] for o in toks), default=0)
        rows = [(o["y"] + o["h"] / 2, o["text"].strip(), ref_x - o["x"]) for o in toks]
        if not rows:
            # 1-for-1 layout; full screen the rows sit 25 px above the laptop calibration (473/535)
            rows = [(LAY.y(448, "c"), "+1", 0), (LAY.y(508, "c"), "-1", 0)]
        elif len({t[0] for _, t, _ in rows}) == 1:
            # live 2026-10-03: OCR caught "+1" but not "-1" and the wool-for-brick offer was declined
            y, t, _ = max(rows, key=lambda r: -r[2])
            rows = rows + [(y + 60, "-1", 0)] if t[0] == "+" else [(y - 60, "+1", 0)] + rows
        if self._offer_card(img, rows[0][0], rows[0][2])[0] is None:
            # panel may still be animating: re-capture once before deciding
            time.sleep(0.7)
            img = self.io.snapshot()
        get, give, seen = {}, {}, []
        for y, t, dx in rows:
            res, hsv = self._offer_card(img, y, dx)
            seen.append((t, res, hsv))
            if res is None and hsv is not None and hsv[2] < 110:
                continue                             # no card on this row (e.g. wildcard "get")
            if res is None:
                get = give = None
                break
            side = get if t[0] == "+" else give
            side[res] = side.get(res, 0) + int(t[1:])
        # key on partner AND content: a player who changes the offer while the panel stays open
        # must get a fresh answer (live 2026-10-02 game 4: Rhea's re-made offer sat unanswered)
        key = (partner, tuple(sorted((get or {}).items())), tuple(sorted((give or {}).items())))
        if getattr(self, "_answered", None) == key:
            # solo 2026-10-03: the decline click did not register and the game waited on us for
            # 8 min -- an answered offer still up 20 s later gets one more answer
            # solo 2026-10-03: both answers fell while the panel slid in and the bot waited 13 min
            # -- keep answering every 20 s, up to 6 times
            t, retried = getattr(self, "_answered_at", (time.time(), None))
            n = getattr(self, "_answer_retries", 0) if retried == key else 0
            if n >= 6 or time.time() - t < 20:
                return
            self.note(f"offer from {partner} still open 20 s after our answer: answering again ({n + 1}/6)")
            self._answered_at, self._answer_retries = (time.time(), key), n + 1
            self._offer_pending = key
        elif getattr(self, "_answered_at", (0, None))[1] != key:
            self._answered_at = (time.time(), None)
        if get is None:
            # unreadable card: the panel is often still fading in -- look again for up to 6 s
            # before declining (live 2026-10-03 a fair offer was declined on the first frame)
            t0 = getattr(self, "_offer_unread_t", None)
            if t0 is None or time.time() - t0 > 30:
                self._offer_unread_t = time.time()
                return
            if time.time() - t0 < 6:
                return
        self._offer_unread_t = None
        if get and getattr(self, "_offer_pending", None) == key and getattr(self, "_answered", None) != key:
            # owner 2026-10-05: an offer shows the partner holds the cards it gives us
            self._ledger().saw_holding(partner, dict(get))
        if getattr(self, "_offer_pending", None) != key:
            self._offer_pending = key                # read it twice before clicking: an animating
            return                                   # panel misread would toggle a given answer
        try:
            d = ROOT / "data" / "offers"; d.mkdir(exist_ok=True)
            cv2.imwrite(str(d / f"{time.strftime('%H%M%S')}_{partner}.png"), img[int(LAY.y(420, "c")):int(LAY.y(620, "c")), int(LAY.x(1000, "r")):int(LAY.x(1230, "r"))])
        except Exception:                            # noqa: BLE001
            pass
        gs = self.ensure_state(img, texts) or self.gs     # fresh hand: a stale one declined a fair 1:1
        if gs is None:
            # owner 2026-10-02: the offer sat unanswered for 10 s after a restart -- if we can't
            # judge it yet, decline (only great trades are worth taking anyway)
            self.note(f"offer from {partner}: state not ready (rules being read) -- declining")
            self._click_offer("decline", "answer incoming offer", img)
            self._answered = key
            return
        wild = "wildcard" in " ".join(texts).lower()
        if wild and gs is not None and give and not get:
            # owner 2026-10-02: a wildcard offer asks for cards and lets us name the return:
            # counter with what we need most (counter = bag button; panel flow UNCONFIRMED)
            want = self._wanted_resource(gs, give)
            if give.get("grain"):
                # owner 2026-10-08: countered Bramble's wildcard with our grain -- grain never
                # goes to a player; fall through to the decline below
                self.note(f"wildcard from {partner}: asks for grain {give} -- never given")
                want = None
            # owner 2026-10-08: a commodity goes only for 2 cards, never to a player ahead of us
            # or one at level 3 of its track
            n = 2 if any(k in ("paper", "cloth", "coin") for k in give) else 1
            if want and not self.brain.commodity_ok(gs, partner, give, {want: n}):
                self.note(f"wildcard from {partner}: {give} is a commodity they must not get")
                want = None
            if want and self.brain.trade_locked(gs, partner, give, {want: n}):
                # owner 2026-10-08: nobody at 14+ VP gets a trade unless it wins us the game
                self.note(f"wildcard from {partner}: 14+ VP, {give}->{want} is no sure win")
                want = None
            p = next((q for q in gs.opponents() if q.name == partner), None)
            near_win = p is not None and p.vp_visible >= gs.target - 2 \
                and not (want and self.brain.sure_win_after(gs, give, {want: n}))
            if want and self.brain.past_half(gs, partner) and not self.brain.evaluate_offer(gs, partner, give, {want: n}):
                # owner 2026-10-02: a player past halfway only gets a trade that's extremely good for us
                self.note(f"wildcard from {partner}: past halfway, {give}->{want} not worth it")
                want = None
            if want and not near_win:
                # a 1-for-1 that names our card is fair; only refuse a player about to win
                self.note(f"wildcard from {partner}: give {give}, counter asking {n}x{want}")
                self._click_offer("counter", "counter wildcard", img)
                time.sleep(0.8)
                try:
                    cv2.imwrite(str(ROOT / "data" / "offers" / f"{time.strftime('%H%M%S')}_counter.png"), self.io.snapshot())
                except Exception:                    # noqa: BLE001
                    pass
                for _ in range(n):           # 2 clicks = 2 cards (UNCONFIRMED in the panel)
                    self.click((UI["trade_want_row"]["x"][CARD_ORDER.index(want)], UI["trade_want_row"]["y"]), f"want {want}")
                    time.sleep(0.6)
                self.click(UI["trade_ok"], f"send counter {give}->{n}x{want}")
                self._answered = key
                return
        ok = bool(get) and bool(give) and gs is not None and self.brain.evaluate_offer(gs, partner, give, get)
        self.note(f"offer from {partner}: {seen} -> give {give} get {get}: {'ACCEPT' if ok else 'decline'} hand={ {k: v for k, v in gs.me().hand.items() if v} if gs else '?'} vp={getattr(self, '_vp', {})}")
        self._click_offer("accept" if ok else "decline", "answer incoming offer", img)
        self._answered = key
        booked = self.__dict__.setdefault("_offer_booked", set())
        if ok and key not in booked:
            # review 0.3.1: a re-answer (every 20 s, up to 6) booked the same trade again each time
            booked.add(key)
            self._ledger().trade(partner, dict(get), self._ledger().me or "", dict(give))
        # owner 2026-10-02: no unprompted chat (only the intro); no counter-offers in chat

    @staticmethod
    def read_ranking(texts) -> list[str]:
        """Names in finishing order from the result screen ("1 Lin", "2 Marianne", ...).
        Game 7 2026-10-03: that screen has no "won" text, so the winner was always None."""
        # solo 2026-10-06: a tie ("2 marlow.systems", "2 Jean") kept one name per rank and Jean was
        # lost from the caption; keep every name, ordered by rank then reading order
        rows, seen = [], set()
        for i, t in enumerate(texts):
            m = re.match(r"^\s*([1-6])[.)]?\s+([A-Za-z][\w.\- ]{1,30})$", t.strip())
            if m and m.group(2).strip() not in seen:
                seen.add(m.group(2).strip())
                rows.append((int(m.group(1)), i, m.group(2).strip()))
        return [n for _, _, n in sorted(rows)] if any(r == 1 for r, _, _ in rows) else []

    def _record_game_over(self, texts):
        ranking = self.read_ranking(texts)
        winner = ranking[0] if ranking else next(
            (t.split(" ")[0] for t in texts if "won" in t.lower() or "victory" in t.lower()), None)
        self._ranking = ranking
        self._last_result = {"winner": winner, "ranking": ranking}
        self.note(f"game over, winner={winner} ranking={ranking}")
        # stage 2026-10-09 01:07:15: the game_over screen counts as leaving the game, which set
        # _glog_new first, so _glog() returned None and the log never ended (status read
        # "Turn 116: Jean to play" after Jean had won) -- end the game that is still open
        gl = self._glog(start=False, ending=True)
        if gl is not None:
            gl.end(winner, ranking, self._glog_vp())
        if not isinstance(self.io, ReplayIO):
            from .chat import record_result
            me = self.cfg["bot_name"].lower()
            solo = bool(self.cfg.get("solo"))
            won = bool(winner) and (winner.lower().startswith(me[:6]) or me.startswith(winner.lower()))
            glo = self.__dict__.get("_glog_obj")
            seats = list(getattr(glo, "seats", None) or [])
            place = next((i + 1 for i, n in enumerate(ranking)
                          if n.lower().startswith(me[:6]) or me.startswith(n.lower())), None)
            vp = next((v for n, v in self._glog_vp().items() if n.lower().startswith(me[:6])), None)
            if won and vp is not None and getattr(getattr(self, "gs", None), "target", None):
                vp = max(vp, self.gs.target)   # stage 073814: the last point lands after our final read
            r = record_result(won, solo=solo, game_id=getattr(glo, "game_id", None), place=place,
                              players=max(len(ranking), len(seats)) or None, vp=vp)
            self.note(f"record{' (solo)' if solo else ''}{' (same game again, not counted)' if r.get('duplicate') else ''}: "
                      f"{r['games']} games, {r['wins']} wins, streak {r['streak']}")
        self._feed("finished", {"winner": winner, "ranking": [
            {"name": n, "vp": next((p.vp_visible for p in getattr(self.gs, "players", []) if p.name == n), None)}
            for n in ranking]})
        self.save_game(winner)
        if not isinstance(self.io, ReplayIO):
            try:                                     # live games train the model, gated (live_learn.py)
                from . import live_learn
                f = live_learn.finish(winner, getattr(self.mode, "key", None))
                if f is None:
                    self.note("live learn: no snapshots or winner not matched, nothing to learn")
                else:
                    self.note(f"live learn: game saved for training -> {f.name}")
                    if self.cfg.get("live_learn_gate", True):
                        live_learn.spawn_gate(self.note)
            except Exception as e:                   # noqa: BLE001
                self.note(f"live learn failed: {type(e).__name__}: {e}")
        self.mode, self.lattice, self.gs = None, None, None

    def _game_file(self, name) -> Path:
        """Audit 2026-10-03: nothing about a game was kept. data/games/<start time>/ holds the
        first clean scan, the players + colours, the setup picks and the result."""
        d = getattr(self, "_game_dir", None)
        if d is None:
            d = self._game_dir = ROOT / "data" / "games" / time.strftime("%Y%m%d_%H%M%S")
        return d / name

    def _feed(self, status, result=None):
        """Owner 2026-10-04: public snapshot per roll + one at the end for Marlow chat (off by
        default, see live_feed.py). Never sends the hand, plan or estimates."""
        if isinstance(self.io, ReplayIO) or getattr(self, "gs", None) is None:
            return
        try:
            from . import live_feed
            if "_feed_conf" not in self.__dict__:
                self._feed_conf = live_feed.config()
            if self._feed_conf is None:
                return
            gid = self._game_file("x").parent.name
            live_feed.post(live_feed.snapshot(self.gs, gid, status, self.__dict__.get("_feed_dice", []),
                                               self.__dict__.get("_feed_events", []), result),
                           self._feed_conf, self.note)
        except Exception as e:                       # noqa: BLE001 -- the feed never stops play
            self.note(f"feed: {type(e).__name__}")

    def _game_save(self, name, obj):
        if isinstance(self.io, ReplayIO):
            return
        try:
            f = self._game_file(name)
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(obj, indent=1, default=str))
        except Exception as e:                       # noqa: BLE001 -- saving never stops play
            self.note(f"game save {name} failed: {e!r}")

    def _mode_key(self):
        """game JSONs had mode null: a runner restarted on the end screen has no rules read.
        Fall back to the mode saved in the game flags, then to the live training log."""
        k = getattr(self.mode, "key", None) or getattr(self, "_mode_saved", None)
        if k is None:
            try:
                from . import live_learn
                k = (live_learn._read(live_learn.GLOG) or {}).get("mode")
            except Exception:                        # noqa: BLE001
                k = None
        return k

    def save_game(self, winner):
        gs = getattr(self, "gs", None)
        self._game_save("result.json", {
            "winner": winner, "mode": self._mode_key(),
            "colours": getattr(self, "game_colors", None),
            "vp": {p.name: p.vp_visible for p in gs.players} if gs is not None and hasattr(gs, "players") else None,
            "my_buildings": sorted(map(str, getattr(self, "_my_blds", set()) or [])),
            "ranking": getattr(self, "_ranking", None),
            "loops": getattr(self, "_loops", 0)})
        d = ROOT / "data" / "games"
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"live_{time.strftime('%Y%m%d_%H%M%S')}.json"
        path.write_text(
            json.dumps({"winner": winner, "mode": self._mode_key(), "me": self.cfg.get("bot_name"),
                        "vp": {p.name: p.vp_visible for p in gs.players} if gs is not None and hasattr(gs, "players") else None,
                        "improvements": self._improve_levels(), "log": self.log[-500:]}, indent=1, default=str))
        try:                                         # chat RAG learns from every finished game
            from . import rag
            st = rag.ingest_game(path)
            self.note(f"rag: learned from game, {st['games']} games / {st['chunks']} chunks")
        except Exception as e:                       # noqa: BLE001 -- never let learning break the runner
            self.note(f"rag ingest failed: {type(e).__name__}: {e}")
        return path

    # -- in-game chat (catanbot/chat.py). Coordinates in data/calib/ui.json, confirmed 2026-10-02.
    def _chat_on(self) -> bool:
        if not self.cfg.get("chat", not isinstance(self.io, ReplayIO)):
            return False
        if self.cfg.get("solo"):                     # solo 2026-10-03: no chat box; typed text panned the board
            return False
        return all(UI.get(k) for k in ("chat_open", "chat_close", "chat_input", "chat_lines_region"))

    def _chat_say(self, text: str):
        """Chat panel must already be open."""
        self.click(UI["chat_input"], "chat input")
        if self.cfg.get("dry_run"):
            self.note(f"[dry] chat say {text!r}")
            if isinstance(self.io, ReplayIO):
                self.io.type_text(text)
            return
        self.note(f"chat say {text!r}")
        self.io.type_text(text)
        time.sleep(0.3)

    def chat_intro(self):
        """One short introduction per game (not after a mid-game restart)."""
        if self._intro_sent:
            return
        self._intro_sent = True
        if not self._saw_game_start:
            self.note("chat intro skipped: runner joined mid-game")
            return
        if not self._chat_on():
            return
        self.click(UI["chat_open"], "chat open")
        try:
            time.sleep(0.5 if not isinstance(self.io, ReplayIO) else 0)
            self._chat_say(self.chat.intro())
            for line in self.chat.intro_more():
                time.sleep(0.5 if not isinstance(self.io, ReplayIO) else 0)
                self._chat_say(line)
        finally:
            self.click(UI["chat_close"], "chat close")

    def _chat_texts(self, img, ocr) -> list[str]:
        x0, y0, x1, y1 = UI["chat_lines_region"]
        rows = [o for o in ocr if x0 <= o["x"] + o["w"] / 2 <= x1 and y0 <= o["y"] + o["h"] / 2 <= y1]
        return [o["text"] for o in sorted(rows, key=lambda o: (o["y"], o["x"]))]

    def _chat_poll_turn(self, img, ocr) -> bool:
        """Owner 2026-10-09: marlow answered only after End turn. On our own turn chat is read at
        the quiet point before planning (no offer pending, hand bar up). True = chat was opened."""
        if not self.cfg.get("chat_on_turn", True):
            return False
        turn = self.__dict__.get("turn_started")
        once = self.cfg.get("chat_once_per_turn", True)
        if once and turn is not None and self.__dict__.get("_chat_turn_done") == turn:
            # owner 2026-10-09 "play every move way faster": one look per own turn (2.6 polls of
            # ~2 s each live); a message sent later is answered by the poll right after End turn
            return False
        before = self._chat_last_poll
        self.chat_poll(img, ocr, "own_turn")
        opened = self._chat_last_poll != before
        if opened and turn is not None:
            self._chat_turn_done = turn
        return opened

    def chat_poll(self, img, ocr, state=None):
        """Open chat (at most every ~20 s: off-turn, our turn's quiet point, right after end turn), answer, close."""
        if not self._chat_on() or not getattr(self, "game_colors", None):
            return
        now = time.time()
        just_ended = now - self._end_turn_at < 8
        if state not in ("other_turn", "idle", "own_turn") and not just_ended:
            return
        if now - self._chat_last_poll < self.cfg.get("chat_poll_s", 20):
            return
        self._chat_last_poll = now
        self._tcount("chat polls")
        self.click(UI["chat_open"], "chat open")
        try:
            if self._live_waits():
                time.sleep(0.6)
            snap = self.io.snapshot()
            if snap is None:
                return
            texts = self._chat_texts(snap, self.io.ocr(snap))
            prime = not self._chat_primed
            self._chat_primed = True
            from .chat import parse_lines
            if prime and any(self.chat.is_me(sn) for sn, _ in parse_lines(texts)):
                self._intro_sent = True              # we already spoke this game (restart)
            for sender, msg in self.chat.new_messages(texts, prime=prime)[-3:]:
                reply = self.chat.respond(sender, msg, self.gs)
                if reply:
                    self.note(f"chat from {sender}: {msg!r}")
                    self._chat_say(reply)
                    break
        finally:
            self.click(UI["chat_close"], "chat close")

    def _window_missing(self, missing: bool, why: str = ""):
        """Owner 2026-10-03: no game window (crash, or Steam stuck on its own sign-in, which we
        can't see or type into): relaunch once after 60 s, stop and alert after 5 min.
        A running game whose window stays black (21:30, hung after a relaunch) is restarted."""
        if not missing:
            self._no_window_since = None
            return
        now = time.time()
        since = getattr(self, "_no_window_since", None) or now
        self._no_window_since = since
        if "black" in why and now - since > 60 and now - getattr(self, "_relaunch_at", 0) > 600:
            self._relaunch_at = now
            self.restart_app("game window black for 60 s")
            return
        if now - since > 300:
            self.alert("no Catan Universe window for 5 min (Steam sign-in or Steam Guard?); please check the server")
            self._no_window_since = None
        elif now - since > 60 and now - getattr(self, "_relaunch_at", 0) > 600:
            import subprocess
            self._relaunch_at = now
            # owner 2026-10-06: a launch with the game open gives "game is already running"
            if not subprocess.run(["pgrep", "-f", self.GAME_PROC], capture_output=True).returncode:
                self.note("no game window for 60 s but the game is running: bringing it to the front")
                subprocess.run(["osascript", "-e", f'tell application id "{self.APP_ID}" to activate'],
                               capture_output=True, timeout=30)
                return
            self.note("no game window for 60 s: launching it through Steam")
            # stage 2026-10-08 21:44: the crash relaunch set no watch, so a rejoin of the old match
            # would go unseen. Watch it like restart_app; the match was not left (crash), so a
            # rejoin plays on.
            self._relaunch_watch = (time.time(), False)
            subprocess.run(["open", f"steam://rungameid/{self.STEAM_APP}"], check=False)

    def run(self, max_steps: int | None = None):
        n = 0
        if not isinstance(self.io, ReplayIO):
            try:                                     # owner 2026-10-02: fixes -> docs -> RAG, every start
                from . import rag
                self.note(f"rag: re-ingested docs, {rag.ingest_docs()} doc chunks")
            except Exception as e:                   # noqa: BLE001
                self.note(f"rag ingest failed: {type(e).__name__}: {e}")
            if self.cfg.get("owner_commands", True):  # owner 2026-10-05: "marlow, exit the game" by iMessage
                from . import commands
                commands.start(self.cfg.get("owner_commands_url", commands.DEFAULT_URL), note=self.note)
        while max_steps is None or n < max_steps:
            try:
                s = self.step()
            except Exception as e:                   # noqa: BLE001 -- game restarting / window gone
                if isinstance(self.io, ReplayIO):
                    raise
                self.note(f"step failed ({type(e).__name__}: {e}); retrying in 3 s")
                if type(e).__name__ != "WindowNotFound":
                    seen = self.__dict__.setdefault("_errors_seen", set())
                    key = f"{type(e).__name__}: {e}"[:200]
                    if key not in seen:              # a code bug: traceback once, counted as a failure
                        seen.add(key)
                        import traceback
                        self.note("TRACEBACK " + traceback.format_exc().replace("\n", " | ")[-1500:])
                        self.probe.result("step_error", False, key)
                self.lattice = self.scan = None      # window may have moved: scan the board again
                self._window_missing(type(e).__name__ == "WindowNotFound", str(e))
                time.sleep(3)
                continue
            self._window_missing(False)
            if s == "end":
                break
            n += 1
            if not isinstance(self.io, ReplayIO):
                # owner 2026-10-08 "moves take a long time": right after our own click the next
                # screen is already up -- poll again quickly; waiting on others keeps poll_s
                # owner 2026-10-09 "play every move way faster": poll_s_turn on our own turn,
                # 0.25 s on transition frames after our roll / trade OK (SPEED_PLAN 6, 7)
                time.sleep(self._loop_sleep_s())


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--replay", nargs="*")
    a = ap.parse_args()
    cfg = config.load()
    if a.dry_run:
        cfg["dry_run"] = True
    io = ReplayIO(a.replay) if a.replay else LiveIO()
    # game 7 2026-10-03 22:21: the runner vanished without a traceback; leave a line for
    # fatal errors and for signals (a closed Terminal sends SIGHUP)
    import faulthandler, signal, atexit
    faulthandler.enable()
    def _sig(n, _f):
        print(f"{time.strftime('%H:%M:%S')} RUNNER EXIT: signal {signal.Signals(n).name}", flush=True)
        raise SystemExit(128 + n)
    for sg in (signal.SIGHUP, signal.SIGTERM):
        signal.signal(sg, _sig)
    atexit.register(lambda: print(f"{time.strftime('%H:%M:%S')} RUNNER EXIT", flush=True))
    if not a.replay:
        # final probe game 2026-10-04 10:18: two runners clicked the same game (interleaved log).
        # One runner per machine: a live pid in data/runner.pid running catanbot.runner wins.
        import os, subprocess
        pidf = ROOT / "data" / "runner.pid"
        try:
            old = int(pidf.read_text().strip())
        except Exception:                            # noqa: BLE001
            old = None
        if old and old != os.getpid():
            cmd = subprocess.run(["ps", "-o", "command=", "-p", str(old)], capture_output=True, text=True).stdout
            if "catanbot.runner" in cmd:
                print(f"{time.strftime('%H:%M:%S')} another runner (pid {old}) is running: exiting", flush=True)
                return
        pidf.write_text(str(os.getpid()))
        # owner 2026-10-03: launching from Terminal leaves the Mac on Terminal's desktop;
        # go back to the full-screen game's Space so the screen shows the game
        from . import mac
        mac.activate()
    Runner(io, cfg).run()


if __name__ == "__main__":
    main()
