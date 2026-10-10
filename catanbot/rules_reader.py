"""Read the actual rules of the game being played, before any strategy is made.

House rules change from game to game (VP target, friendly robber, dice mode, turn
timer, discard limit, player count, scenario). The runner OCRs the lobby settings,
the game-intro screen and the in-game info panel, and feeds the text here. Anything
we read overrides the mode defaults; anything we couldn't read is listed in
`unconfirmed` so the log shows what the bot assumed.
"""
from __future__ import annotations

import re
from dataclasses import replace

from .rules import MODES, Mode, mode_from_text

_NUM = r"(\d{1,2})"


def parse_rules(texts: list[str], base: Mode | None = None) -> tuple[Mode, dict]:
    """texts: OCR lines from any rules-bearing screen. Returns (mode, report)."""
    blob = " \n".join(texts)
    low = blob.lower()
    mode = base or mode_from_text(blob) or MODES["great_canal"]   # marlow only plays this game
    found: dict = {"scenario": mode.key}
    ch: dict = {}

    # in-game "SCENARIO CONFIGURATION" table: labels column, then values column in the same
    # order (Round Timer, Victory Points are the numeric ones). The table beats prose like
    # "21 victory points (in a 3-player game)" (live 2026-10-02: table said 19).
    labels = [t.strip().lower() for t in texts]
    nums = [int(t) for t in texts if t.strip().isdigit()]
    if "victory points" in labels and "round timer" in labels and len(nums) >= 2:
        ch["turn_timer_s"], ch["vp_target"] = nums[0], nums[1]
    elif "victory points" in labels and len(nums) == 1:
        ch["vp_target"] = nums[0]
    m = None if "vp_target" in ch else re.search(_NUM + r"\s*victory\s*points?", low) or re.search(r"victory\s*points?\D{0,12}" + _NUM, low)
    if m:
        ch["vp_target"] = int(m.group(1))
    # live 2026-10-02: a half-drawn table read "130" VP and a 666 s timer; drop implausible values
    if not 5 <= ch.get("vp_target", 10) <= 25:
        ch.pop("vp_target")
    if not 30 <= ch.get("turn_timer_s", 60) <= 600:
        ch.pop("turn_timer_s")
    m = re.search(r"(?:discard|hand)\D{0,20}" + _NUM, low)
    if m and 5 <= int(m.group(1)) <= 15:
        ch["discard_limit"] = int(m.group(1))
    if "friendly robber" in low:
        # the lobby shows a checkbox; the reader passes "[x] friendly robber" / "[ ] ..."
        ch["friendly_robber"] = not re.search(r"\[ \]\s*friendly robber|friendly robber\W{0,3}(off|no)", low)
    if "balanced" in low:
        ch["balanced_dice"] = True
    elif re.search(r"dice\W{0,10}random|random dice", low):
        ch["balanced_dice"] = False
    m = None if "turn_timer_s" in ch else re.search(r"(?:turn\s*(?:time|timer|limit))\D{0,12}(\d{1,3})\s*(s|sec|min)?", low)
    if m:
        v = int(m.group(1))
        ch["turn_timer_s"] = v * 60 if (m.group(2) or "").startswith("min") or v <= 5 else v
    elif re.search(r"no\s*(turn\s*)?(time|timer)\s*limit|unlimited", low):
        ch["turn_timer_s"] = None
    m = re.search(r"players?\D{0,6}([2-6])\b", low) or re.search(r"\(([2-6])\)", low)
    if m:
        ch["players"] = int(m.group(1))

    found.update(ch)
    mode = replace(mode, **ch)
    unconfirmed = [k for k in ("vp_target", "friendly_robber", "balanced_dice", "turn_timer_s",
                               "discard_limit") if k not in ch]
    return mode, {"read": found, "unconfirmed": unconfirmed}


def turn_budget(mode: Mode, default_s: float) -> float:
    """Seconds we allow ourselves per turn: 75% of the table's timer, else the config default."""
    if mode.turn_timer_s:
        return max(10.0, 0.75 * mode.turn_timer_s)
    return default_s
