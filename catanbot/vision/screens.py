"""Classify what the game is asking for, from the OCR'd text of a capture.

Phrases come from calibration captures (data/calib); add new ones as they are seen.
"""
from __future__ import annotations

import re

# (state, regex on lowercased joined text). First match wins, so specific before generic.
PATTERNS = [
    # solo 2026-10-03: the barbarian "Defeat!"/"Victory!" banner is mid-game, not the end screen
    # game 13 18:10: "You've won!" banner (Experience / Ranking, VIEW PROGRESS / CONTINUE) sat 4 min unread
    ("game_result", r"you.?ve (won|lost)|you (won|lost)|^(?=.*experience)(?=.*ranking)(?=.*continue)"),
    ("game_over", r"experience gained|player statistics|game statistics|has won|wins the game|game over"),
    ("discard_progress", r"discard ?progress ?card"),   # 5th progress card (live 2026-10-03)
    ("discard", r"discard|choose \d+ cards? to (give|return)|hand over to the robber|resources to hand over|"
                r"resources? you want to gift|is marrying"),   # live 2026-10-03 game 6: Wedding dialog stalled
    # solo 2026-10-03: "select position for settlement" + the rules panel's "...robber" is not it
    # live 2026-10-09 20:34: OCR "Select position for Robbef®" sat 50 s unread after our discard
    ("place_robber", r"move the robber|select .{0,25}robber|place the robber|sel\w* ?position ?fo\w* ?rob+[a-z]{0,3}\b"),
    ("place_pirate", r"move the pirate|place the pirate"),
    ("steal", r"choose a player to steal|steal from|want to steal|steal resource cards"),
    # stage 2026-10-09 00:40:50: our Commercial Harbor "Choose a player with whom you want to
    # trade." sat unread (the choose_player rule skips "to trade") until a ring fallback hit its X.
    # Master Merchant says 'want to "trade"' (quoted) and stays choose_player.
    ("harbor_player", r"with whom you want to trade"),
    # probe 2026-10-04: Deserter "Choose a player to remove a knight." + a check under each name
    # live 2026-10-09 21:03: the Spy's "Select progress card" showed a Deserter whose text says
    # "Choose an opponent" -- read as choose_player for 80 s and the Spy took nothing
    ("choose_player", r"^(?=.*hide overlay)(?!.*player trade)(?!.*to trade)(?!.*select (a )?progress car).*(choose (a player|an opponent)|which player|select a player)"),
    # Deserter played ON us (text UNCONFIRMED, probe-captured): our own knights are ringed
    ("lose_knight", r"^(?!.*choose (a player|an opponent))(?!.*player trade).*((remove|lose|give up|sacrifice)\w* (one of )?(your |a )?(own )?knights?\b|select (a |your |the )?knight to (remove|lose|give))"),
    # probe 2026-10-04: Master Merchant "Take 2 resources from other player." (their cards on top)
    ("take_cards", r"take \d+ (resources|cards|resource)"),
    # stage 2026-10-08 22:42: a rival took our metropolis -> "Select position to remove City" (9 min unread)
    ("remove_city", r"sel\w* ?position ?to ?remove ?c[il1][tf]y"),
    ("setup_settlement", r"select ?position ?for ?settlement"),
    # game 13 17:45: Engineer "Select position for City wall" (OCR "City waff") was read as setup_city
    ("place_wall", r"sel\w* ?position ?fo\w* ?c[il1][tf]y ?w[a-z]{0,3}"),
    ("setup_city", r"select ?position ?for ?c[il1][tf]y"),   # solo 2026-10-03 OCR: "Cify"
    ("setup_road", r"select ?position ?for ?road"),
    ("place_ship", r"select ?position ?for ?ship"),
    # solo 2026-10-03: Smith/promotion asks "Select position for Strong knight" (read as unknown 18 min)
    ("promote_knight", r"sel\w* ?position ?fo\w* ?(strong|mighty)"),   # probe 2026-10-04 OCR "Selet position focstrong"
    ("place_knight", r"select ?position ?for ?(basic ?)?knight"),   # game 13 16:35 "...for Basic knight" (Deserter)
    ("place_metropolis", r"sel\w* ?position ?fo\w* ?metro"),
    # stage 2026-10-09 game 0959 10:34: Inventor "Select position for Number Token" (read as unknown 9 min)
    ("select_token", r"sel\w* ?position ?fo\w* ?number ?tok"),   # game 13 17:01 OCR "Metropofis", read as unknown
    ("trade_offer_in", r"wants to trade|offer from|offers you"),
    # game 13 16:30: aqueduct "Science bonus -- Choose 1 more resources that you'll gain."
    ("choose_resource", r"choose a resource|select a resource|select resource|resources? that you'?.?ll gain|science bonus"),
    # probe game 1 2026-10-04: "Defender of Catan ... As thanks you receive a progress card." + 3 decks
    # final probe game 2026-10-04 10:36: Spy shows the victim's cards: "Select progress card"
    ("pick_progress", r"select (a )?progress car[a-z]?"),   # live 2026-10-08: OCR "Select progress cara"
    ("choose_deck", r"you receive a progress card|choose a (progress )?(card )?(deck|stack)"),
    ("progress_card", r"progress card|play a card"),
    ("roll", r"roll the dice|tap to roll|click to roll"),
    ("my_turn", r"it'?s your turn|your turn"),
    ("other_turn", r"it'?s (.+?)'?s turn"),
    ("invite", r"invit(e|ation)|wants to play|join (the )?game"),
    # audit 2026-10-03: bare "ready" matched "already" mid game and wiped the game state
    # live 2026-10-08 19:06/19:14: our own trade offer panel says "Waiting for players" -- read as
    # lobby mid-game it wiped the game twice; never lobby while the Player trade panel is up
    ("lobby", r"^(?!.*player trade)(?!.*choose a player to trade).*(start game|\bnot ready\b|(^|\| )ready($| \|)|waiting for players)"),
    ("home", r"leaderboards|almanac|guilds|solo:? ?offline games"),   # 21:15 solo menu read as main
    ("login", r"forgot password|^login\b| login |discord"),   # app restart: it signs in by itself
    ("intro", r"\d+\s*victory points"),
]

# hard rule (config never_matchmaking): never click buttons with these words
FORBIDDEN_BUTTONS = re.compile(r"matchmak|quick ?match|auto ?match|find (a )?game|ranked|online game|"
                               r"quick ?play|league|tournament|public (game|match)|random (match|opponent)", re.I)


# stage 2026-10-09 22:46 (game 221251): our Spy's "Select progress card" showed Marianne's Bishop,
# whose card text says "Move the robber" -- read as place_robber for 10 min (no rings, no way out)
# and the game was left. The Spy dialog's title wins over any card text printed under it.
PICK_PROGRESS_FIRST = True     # config pick_progress_first (runner sets it); False = first pattern wins
_STATES = [s for s, _ in PATTERNS]
_PICK_PROGRESS = dict(PATTERNS)["pick_progress"]
_CARD_TEXT_STATES = set(_STATES[_STATES.index("discard_progress") + 1:_STATES.index("pick_progress")])


# solo 2026-10-09 23:20:13 (data/discard/232013_in.png): the game feed line "Marianne discards 1
# progress card(s)." read as our discard dialog -- 8 of the game's 9 "discards" were this line, the
# bot clicked cards into whatever was open (23:38:03 a trade panel) and pressed its green check.
# Another player's feed line ("<name> discards N ...") is never our prompt.
_FEED_DISCARDS = re.compile(r"\bdiscards\s*\d+\b")


def classify(texts: list[str], me: str = "marlow.systems") -> tuple[str, dict]:
    low = " | ".join(t.lower() for t in texts if not _FEED_DISCARDS.search(t.lower()))
    info: dict = {}
    for state, pat in PATTERNS:
        m = re.search(pat, low)
        if not m:
            continue
        if PICK_PROGRESS_FIRST and state in _CARD_TEXT_STATES and re.search(_PICK_PROGRESS, low):
            return "pick_progress", info
        if state == "other_turn":
            who = m.group(1).strip()
            if who.replace(" ", "") in (me.lower(), "your"):
                return "my_turn", info
            info["player"] = who
        return state, info
    return "unknown", info


# ---- whose turn it is (stage 2026-10-09 00:36:43 / 00:37:20) ----------------------------------
# Off our turn the screen has no turn text at all ("It's X's turn" never shows), so every rival
# turn read as screen=unknown. The game marks the player on turn with a gold down-pointing
# triangle above the portrait: base ~30 px wide, 16 px tall, centred at oval_x+17 (oval = the
# white resource badge banner._find_ovals finds under each avatar). It bobs: its base sits
# anywhere from oval_y-81 to oval_y-74, so the best of those offsets counts.
TURN_TRI_MIN = 0.75         # gold share inside the triangle (real 0.85+)
TURN_TRI_OUT_MAX = 0.25     # gold share in the strip just above its base (empty table/banner art)
_TRI_H, _TRI_HALF = 16, 15


def _gold(hsv):
    return (hsv[..., 0] >= 12) & (hsv[..., 0] <= 35) & (hsv[..., 1] > 90) & (hsv[..., 2] > 170)


def _tri_masks():
    import numpy as np
    xs = np.arange(-22, 23)
    inside = np.zeros((_TRI_H + 5, len(xs)), bool)
    above = np.zeros_like(inside)
    above[0:3] = np.abs(xs) <= _TRI_HALF - 2           # 2-4 rows above the base
    for i in range(_TRI_H + 1):
        half = _TRI_HALF * (_TRI_H - i) / _TRI_H
        inside[i + 4] = np.abs(xs) <= max(half - 2, 0)
    return inside, above


def turn_marker_scores(img) -> list[tuple[float, float]]:
    """(inside, above) gold shares of the best-placed turn triangle over each banner, left to right."""
    import cv2
    from .banner import _find_ovals
    inside, above = _tri_masks()
    out = []
    H, W = img.shape[:2]
    for (ox, oy, _w, _h) in _find_ovals(img):
        best = (0.0, 1.0)
        for dx in range(-2, 3):
            cx = ox + 17 + dx
            for base in range(oy - 81, oy - 73):
                y0 = base - 4
                if y0 < 0 or cx - 22 < 0 or cx + 23 > W or y0 + inside.shape[0] > H:
                    continue
                g = _gold(cv2.cvtColor(img[y0:y0 + inside.shape[0], cx - 22:cx + 23], cv2.COLOR_BGR2HSV))
                a, b = float(g[inside].mean()), float(g[above].mean())
                if a - b > best[0] - best[1]:
                    best = (a, b)
        out.append(best)
    return out


def active_seat(img) -> int | None:
    """Banner index (0 = us, the leftmost) under the gold turn triangle, None if no single one."""
    hits = [i for i, (a, b) in enumerate(turn_marker_scores(img)) if a >= TURN_TRI_MIN and b <= TURN_TRI_OUT_MAX]
    return hits[0] if len(hits) == 1 else None


def dialog_close_x(img, title: dict) -> tuple[int, int] | None:
    """The white X in the top-right corner of a card dialog, found right of its title box
    (Commercial Harbor 2026-10-09: title centre (654, 287), X at (820, 275)). None if no X."""
    import cv2
    import numpy as np
    H, W = img.shape[:2]
    cx, cy = title["x"] + title["w"] / 2, title["y"] + title["h"] / 2
    x0, x1 = int(cx + 60), int(min(W, cx + 320))
    y0, y1 = int(max(0, cy - 40)), int(min(H, cy + 20))
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None
    hsv = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 2] > 215) & (hsv[..., 1] < 50)).astype(np.uint8)
    n, _, st, cen = cv2.connectedComponentsWithStats(m, 8)
    best = None
    for i in range(1, n):
        x, y, w, h, a = st[i]
        # an X: square-ish, 14-30 px, its ink a thin share of its box (a white card would fill it)
        if 14 <= w <= 30 and 14 <= h <= 30 and 0.6 <= w / h <= 1.6 and 0.2 <= a / (w * h) <= 0.7:
            if best is None or cen[i][0] > best[0]:
                best = (float(cen[i][0]), float(cen[i][1]))
    return (int(best[0] + x0), int(best[1] + y0)) if best else None


# live 2026-10-06 02:32 and 2026-10-08 20:09: the OCR reads OK as Cyrillic "ОK", and the
# Commercial Harbor "No player has any commodity cards / OK" popup sat 10+ min unpressed
_LATIN = str.maketrans("АВЕКМНОРСТХаеорсухіІ", "ABEKMHOPCTXaeopcyxiI")


def find_text(ocr: list[dict], pattern: str) -> dict | None:
    """OCR box whose text matches (case-insensitive) -- for clicking buttons by label.
    Cyrillic look-alike letters count as their Latin twins."""
    rx = re.compile(pattern, re.I)
    for r in ocr:
        t = r["text"].translate(_LATIN)
        if rx.search(t) and not FORBIDDEN_BUTTONS.search(t):
            return r
    return None


# ---- build popup options -----------------------------------------------------------------
# Labels under the round option icons when a board spot is tapped (seen live 2026-10-04 on the
# 1920x1080 server: "Build road", "Build ship", "Build city", "Build wall", "Activate knight",
# "Promote knight"; "Build knight" in game 2). Settlement/move-knight labels are not confirmed.
POPUP_LABELS = {"road": "build road", "ship": "build ship", "city": "build city", "wall": "build wall",
                "knight": "build knight", "activate": "activate knight", "promote": "promote knight",
                "settlement": "build settlement", "move": "move knight"}   # probe 2026-10-04: "Move Knigint"
POPUP_MIN_RATIO = 0.72
_NOT_A_LABEL = re.compile(r"needed for|resource:|select position|build\s*cost", re.I)


def _letters(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())


def popup_label_kind(text: str) -> tuple[str | None, float]:
    """Which popup label an OCR box most looks like, and how much (0..1). Live 2026-10-04: the
    small labels over the board OCR'd as "Build foad", "Buildboad", "Promotekmigit"."""
    import difflib
    t = _letters(text)
    if len(text) > 24 or len(t) < 5 or _NOT_A_LABEL.search(text):
        return None, 0.0
    best = max((difflib.SequenceMatcher(None, t, _letters(lab)).ratio(), k) for k, lab in POPUP_LABELS.items())
    return (best[1], best[0]) if best[0] >= POPUP_MIN_RATIO else (None, best[0])


def find_popup_option(ocr: list[dict], kind: str, fallback: str | None = None) -> dict | None:
    """The OCR box of the popup option for `kind`: the closest-looking label wins across all
    kinds (so "Build ship" is never taken as road, "Promote knight" never as knight). Then the
    `fallback` regex, only on short boxes no other kind's label claims."""
    best, best_r = None, 0.0
    for o in ocr:
        k, r = popup_label_kind(o["text"])
        if k == kind and r > best_r:
            best, best_r = o, r
    if best or not fallback:
        return best
    rx = re.compile(fallback, re.I)
    for o in ocr:
        k, _ = popup_label_kind(o["text"])
        if (k in (None, kind) and len(o["text"].strip()) <= 24 and not _NOT_A_LABEL.search(o["text"])
                and rx.search(o["text"]) and not FORBIDDEN_BUTTONS.search(o["text"])):
            return o
    return None


def ocr_near(ocr_fn, img, xy, rx: int = 320, ry: int = 200, scale: int = 3) -> list[dict]:
    """OCR a crop around the tapped spot, enlarged `scale` times, boxes mapped back to `img`.
    Live 2026-10-04: on the 1920x1080 server the frame is shrunk to 1308x790 and the ~11 px popup
    labels over the busy board were unreadable whole-frame; the x3 crop reads them."""
    import cv2
    H, W = img.shape[:2]
    x, y = int(xy[0]), int(xy[1])
    x0, y0, x1, y1 = max(0, x - rx), max(0, y - ry), min(W, x + rx), min(H, y + ry)
    if x1 - x0 < 40 or y1 - y0 < 40:
        return []
    big = cv2.resize(img[y0:y1, x0:x1], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return [dict(o, x=o["x"] / scale + x0, y=o["y"] / scale + y0, w=o["w"] / scale, h=o["h"] / scale)
            for o in ocr_fn(big)]


# ---- build popup option rings (live 2026-10-04, docs/analysis/live_controls_2026-10-04.md) ----
# A tap on a board spot draws round option icons on a row through the spot: one icon sits on
# the spot, two at -64/+61 px, four at -189/-64/+57/+186 (frame px). Each is a 57 px circle with
# a thin ring: green = affordable, red = not. The label under it is only ~11 px tall in the
# shrunk 1920x1080 frame, so the ring + icon decide; the label OCR is a cross-check.
OPTION_RING_R = (25.5, 31.0)       # annulus that holds the ring line (frame px)
OPTION_RING_MIN = 0.36             # ring pixels / annulus pixels (half-size score): real rings 0.40-0.53
OPTION_BAND = (190, 740)           # board rows; the banner portrait ring at y~111 is not an option
ICON_R = 20                        # icon disc radius used for the templates
ICON_MIN, ICON_MARGIN = 0.8, 0.08
_ICONS = None
IMPROVE_KINDS = ("trade", "politics", "science")


def _ring_kernel(f=1):
    import numpy as np
    r0, r1 = OPTION_RING_R[0] / f, OPTION_RING_R[1] / f
    n = int(r1) + 1
    yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
    d = np.hypot(xx, yy)
    k = ((d >= r0) & (d <= r1)).astype(np.float32)
    return k / k.sum()


def option_rings(img, band=OPTION_BAND, near=None) -> list[dict]:
    """Option icons on screen: [{"x", "y", "colour": green|red, "score"}], left to right.
    near=(x, y): only the row through that tapped spot (+-14 px, +-230 px wide)."""
    import cv2
    import numpy as np
    H, W = img.shape[:2]
    y0, y1 = max(0, int(band[0])), min(H, int(band[1]))
    x0, x1 = 0, W
    if near is not None:
        y0, y1 = max(y0, int(near[1]) - 50), min(y1, int(near[1]) + 50)
        x0, x1 = max(0, int(near[0]) - 270), min(W, int(near[0]) + 270)
    if y1 - y0 < 20 or x1 - x0 < 20:
        return []
    hsv = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    lit = (s > 140) & (v > 140)
    f = 2                                    # scored at half size: 4x fewer pixels, 4x smaller kernel
    k = _ring_kernel(f)
    out = []
    for name, m in (("green", lit & (h >= 40) & (h <= 80)), ("red", lit & ((h <= 8) | (h >= 172)))):
        if not m.any():
            continue
        small = cv2.resize(m.astype(np.float32), (m.shape[1] // f, m.shape[0] // f), interpolation=cv2.INTER_AREA)
        sc = cv2.filter2D(small, -1, k, borderType=cv2.BORDER_CONSTANT)
        while True:
            yx = np.unravel_index(int(np.argmax(sc)), sc.shape)
            best = float(sc[yx])
            if best < OPTION_RING_MIN:
                break
            cy, cx = yx
            out.append({"x": int(cx * f + f // 2 + x0), "y": int(cy * f + f // 2 + y0), "colour": name,
                        "score": round(best, 2)})
            sc[max(0, cy - 20):cy + 21, max(0, cx - 20):cx + 21] = 0
    if near is not None:
        out = [o for o in out if abs(o["y"] - near[1]) <= 14 and abs(o["x"] - near[0]) <= 230]
    return sorted(out, key=lambda o: o["x"])


def _icon_templates() -> dict:
    global _ICONS
    if _ICONS is None:
        import numpy as np
        from pathlib import Path
        f = Path(__file__).resolve().parents[2] / "data" / "calib" / "popup_icons.npz"
        _ICONS = {k: v.astype(np.float32) for k, v in np.load(f).items()} if f.exists() else {}
    return _ICONS


def icon_kind(img, x, y) -> tuple[str | None, float]:
    """Which option the icon at ring centre (x, y) is: road, ship, knight, activate, promote, city,
    wall, trade, politics, science (templates in data/calib/popup_icons.npz, cut from the
    2026-10-04 server frames). Grey-level correlation of the icon disc, +-2 px shift."""
    import cv2
    import numpy as np
    tpl = _icon_templates()
    if not tpl:
        return None, 0.0
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    R, sh = ICON_R, 2
    x, y = int(round(x)), int(round(y))
    if y - R - sh < 0 or x - R - sh < 0 or y + R + sh >= g.shape[0] or x + R + sh >= g.shape[1]:
        return None, 0.0
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    m = np.hypot(xx, yy) <= R - 1
    scores = {}
    for kind, t in tpl.items():
        a = t[m] - t[m].mean()
        best = -1.0
        for dy in range(-sh, sh + 1):
            for dx in range(-sh, sh + 1):
                p = g[y + dy - R:y + dy + R + 1, x + dx - R:x + dx + R + 1][m]
                p = p - p.mean()
                best = max(best, float((a * p).sum() / (np.linalg.norm(a) * np.linalg.norm(p) + 1e-6)))
        scores[kind] = best
    fam = lambda k: "improve" if k in IMPROVE_KINDS else k        # noqa: E731
    order = sorted(scores, key=scores.get, reverse=True)
    top = order[0]
    rival = next((k for k in order[1:] if fam(k) != fam(top)), None)
    if scores[top] < ICON_MIN or (rival and scores[top] - scores[rival] < ICON_MARGIN):
        return None, scores[top]
    if fam(top) == "improve":
        # the three improvement icons differ by colour, not shape: trade yellow, politics blue, science green
        hsv = cv2.cvtColor(img[y - R:y + R + 1, x - R:x + R + 1], cv2.COLOR_BGR2HSV)[m]
        hsv = hsv[hsv[:, 1] > 60]
        if len(hsv):
            hue = float(np.median(hsv[:, 0]))
            top = "trade" if hue < 32 else "science" if hue < 85 else "politics"
    return top, scores[top]


def label_under(ocr: list[dict], ring: dict) -> tuple[str | None, float]:
    """The popup label OCR'd under a ring (~45 px below its centre), as popup_label_kind."""
    best = (None, 0.0)
    for o in ocr:
        cx, cy = o["x"] + o["w"] / 2, o["y"] + o["h"] / 2
        if abs(cx - ring["x"]) < 45 and 25 <= cy - ring["y"] <= 65:
            k, r = popup_label_kind(o["text"])
            if k and r > best[1]:
                best = (k, r)
    return best
