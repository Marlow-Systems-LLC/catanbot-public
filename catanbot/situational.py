"""Per-turn situational selector (owner 2026-10-04).

"marlow needs to figure out when one should be used or not, it must do everything it can to stop
the advancing player and/or take the lead." Rules that lose an A/B overall stay in the code; this
module switches them on only in the turns where they help. Two stances:

- BLOCK: a rival is advancing (clearly more win probability or VP than us, or 1-2 VP from
  winning). Every blocking tool aims at that rival (brain.focus): road cuts, knight steals and
  bounces, the robber, barbarian sabotage when it costs that rival a city, no trades with them.
- RACE: we lead or are level. Our own building and VP play, spend the hand, defend. The robber,
  knight steals, bounces and sabotage take the most cards per round from rivals who are NOT
  last-placed (brain.spare); the last-placed player only when nobody else can be hit.
  Steal targets rank: allowed -> expected worth of one stolen card (grain > the commodity our
  next improvement needs > the plan's missing cards > the rest) -> card count -> stronger rival.

A city of ours at risk that costs a lot to lose (owner: "when a city is lost, the developments stop
coming") pushes toward defending in either stance. The friendly robber (never rob a player at
<=4 VP) is applied on top by the existing code paths, never overridden here.

Brain.situational = False (default) leaves play exactly as before.
"""
from __future__ import annotations

from .board import PIPS, TERRAIN_COMMODITY

# ---- thresholds: one table, tune from data ----------------------------------------------------
T = {
    "block_vp_lead": 2,          # BLOCK when the strongest rival shows this many VP more than us
    "block_prob_gap": 0.10,      # ...or this much more win probability
    "near_win_vp": 2,            # ...or is this close to the VP target (always the focus then)
    "phase_mid": 0.40,           # leader VP / target: below = early
    "phase_late": 0.70,          # above = late
    "ship_share": 0.5,           # event die: 3 of 6 faces move the barbarians
    "defend_loss_cost": 1.5,     # our city at risk costing at least this -> defend first
    "defend_turns": 4,           # ...when the attack is at most this many turns away
    "defend_pay_two": 0.25,      # pay 2-for-1 for grain sooner when defending
    "sabotage_min_worth": 1.5,   # stand a knight down when the rival's city loss is worth this
    "advancing_weight": 2.0,     # the focus rival's city loss counts this much more
    # city_loss_cost terms
    "loss_vp": 1.0,              # the VP a pillaged city costs
    "loss_commodity_w": 1.0,     # per commodity a round the city stops producing
    "loss_improve_w": 0.3,       # per open improvement track that commodity feeds (scaled by levels left)
    "loss_last_city": 2.0,       # the last non-metropolis city: improvements/progress stall
    "loss_no_city": 1.0,         # ...and nothing left to buy improvements with at all
    # steal_value: what one random card from a rival is worth to us (owner order)
    "need_grain": 3.0,           # grain first
    "need_improve": 2.5,         # the commodity our next improvement is short of
    "need_plan": 2.0,            # a card the current plan is missing (brain.plan_missing)
    "need_base": 0.5,            # any other card
}

# flags the selector may override for one turn (restored after plan_turn)
FLAG_KEYS = ("road_cuts", "knight_bounce", "barb_sabotage", "displace", "spend_turn",
             "limit_lookahead", "activate_first", "pay_two_danger")
# targeting attributes: kept until the selector runs again (they steer answers on rivals' turns)
TARGET_KEYS = ("focus", "spare", "stance")

COMMODITY_TRACK = {"paper": "science", "cloth": "trade", "coin": "politics"}


def _city_cost(gs, p, v) -> float:
    b = gs.board
    dry = gs.mode.canal and gs.canal_progress < gs.canal_total
    rate, tracks = 0.0, set()
    for h in b.vertex_hexes(v):
        com = TERRAIN_COMMODITY.get(h.terrain)
        if not com or not h.number or (dry and getattr(h, "barren", False)):
            continue
        rate += PIPS[h.number] / 36
        tracks.add(COMMODITY_TRACK[com])
    c = T["loss_vp"] + T["loss_commodity_w"] * rate * len(gs.players)
    c += T["loss_improve_w"] * sum((5 - min(5, p.improvements.get(t, 0))) / 5 for t in tracks)
    return c


def city_loss_cost(gs, player_index: int) -> float:
    """What losing a city to the barbarians costs this player: the VP, the commodities that stop
    (so improvements and progress cards stall), and a large extra for the last non-metropolis
    city (with no city at all, no improvement can be bought). 0 when no city can be pillaged.
    The pillaged city is the cheapest one (the player chooses)."""
    p = gs.players[player_index]
    exposed = [v for v in p.cities if v not in p.metropolises]
    if not exposed:
        return 0.0
    c = min(_city_cost(gs, p, v) for v in exposed)
    if len(exposed) == 1:
        c += T["loss_last_city"]
        if not p.metropolises:
            c += T["loss_no_city"]
    return c


def situation(gs, brain) -> dict:
    me = gs.me()
    players = gs.players
    i_me = players.index(me)
    probs = brain.probs(gs)
    opp = [p for p in players if p is not me]
    target = gs.target
    near = [p for p in opp if target - p.vp_visible <= T["near_win_vp"]]
    if near:
        rival = max(near, key=lambda p: (p.vp_visible, probs.get(p.name, 0)))
    else:
        rival = max(opp, key=lambda p: (probs.get(p.name, 0), p.vp_visible))
    last = min(opp, key=lambda p: (p.vp_visible, probs.get(p.name, 0)))
    frac = max(p.vp_visible for p in players) / max(1, target)
    phase = "early" if frac < T["phase_mid"] else "mid" if frac < T["phase_late"] else "late"
    turns = None
    strength, bottom = {}, set()
    if gs.mode.ck:
        steps = max(0, gs.barbarian_len - gs.barbarian_pos)
        turns = steps / T["ship_share"]
        strength = {p.name: gs.defence_strength(p) for p in players}
        exposed = [p for p in players if any(v not in p.metropolises for v in p.cities)]
        if exposed:
            low = min(strength[p.name] for p in exposed)
            bottom = {p.name for p in exposed if strength[p.name] == low}
    gap = 0
    if gs.mode.ck:
        at_risk = len([v for v in me.cities if v not in me.metropolises])
        gap = max(0, at_risk - strength.get(me.name, 0))
    rivals = [{"index": players.index(p), "name": p.name, "prob": probs.get(p.name, 0.0),
               "vp": p.vp_visible, "at_bottom": p.name in bottom,
               "loss_cost": city_loss_cost(gs, players.index(p)) if p.name in bottom else 0.0}
              for p in opp]
    return {
        "me": i_me, "my_prob": probs.get(me.name, 0.0), "my_vp": me.vp_visible, "target": target,
        "rival": players.index(rival), "rival_prob": probs.get(rival.name, 0.0),
        "rival_vp": rival.vp_visible, "rival_lead": rival.vp_visible - me.vp_visible,
        "rival_to_win": target - rival.vp_visible,
        "last": players.index(last), "phase": phase, "turns_to_attack": turns,
        "my_city_at_risk": gap > 0, "my_loss_cost": city_loss_cost(gs, i_me) if gap > 0 else 0.0,
        "rivals": rivals,
    }


def choose(sit: dict, base_flags: dict) -> dict:
    """Flag overrides for this turn, plus focus/spare/stance."""
    block = (sit["rival_to_win"] <= T["near_win_vp"] or sit["rival_lead"] >= T["block_vp_lead"]
             or sit["rival_prob"] - sit["my_prob"] >= T["block_prob_gap"])
    t = sit["turns_to_attack"]
    defend = (sit["my_city_at_risk"] and sit["my_loss_cost"] >= T["defend_loss_cost"]
              and (t is None or t <= T["defend_turns"]))
    o: dict = {}
    if defend:                    # owner: a lost city stalls our developments -- defend first
        o.update(knight_bounce=False, barb_sabotage=False, activate_first=True,
                 pay_two_danger=min(base_flags.get("pay_two_danger", 0.5) or 1.0, T["defend_pay_two"]))

    def worth(r, advancing):
        return r["at_bottom"] and r["loss_cost"] * (T["advancing_weight"] if advancing else 1.0) \
            >= T["sabotage_min_worth"]

    if block:
        focus = sit["rival"]
        r = next(x for x in sit["rivals"] if x["index"] == focus)
        o.update(stance="block", focus=focus, spare=None, road_cuts=True, displace=True)
        if not defend:
            o.update(knight_bounce=True, barb_sabotage=bool(worth(r, True)))
    else:
        o.update(stance="race", focus=None, spare=sit["last"], displace=False,
                 spend_turn=True, limit_lookahead=True)
        if not defend:
            o.update(knight_bounce=True,       # owner: still rob the most cards per round
                     barb_sabotage=any(worth(x, False) for x in sit["rivals"] if x["index"] != sit["last"]))
    return o


def next_improve_commodity(brain, gs) -> str | None:
    """The commodity our next improvement is short of: the knight ladder's improve step if it
    names one, else the lowest track below 5 (owner ladder: all to 3, then 4, 5). None without
    a city (no improvement can be bought)."""
    from .rules import IMPROVEMENT_TRACKS
    me = gs.me()
    if not gs.mode.ck or not me.cities:
        return None
    plan = brain.knight_plan(gs, me)
    tracks = sorted((t for t in IMPROVEMENT_TRACKS if me.improvements.get(t, 0) < 5),
                    key=lambda t: me.improvements.get(t, 0))
    if plan and plan[0] == "improve" and plan[1]:
        tracks = [plan[1]] + [t for t in tracks if t != plan[1]]
    for t in tracks:
        com = IMPROVEMENT_TRACKS[t]
        if me.hand.get(com, 0) < me.improvements.get(t, 0) + 1:
            return com
    return None


def need_table(brain, gs) -> dict:
    """need(card) for one stolen card, owner order: grain, the next improvement's commodity,
    the plan's missing cards (brain.plan_missing), anything else at a small base value."""
    need = {r: T["need_plan"] for r in brain.plan_missing(gs)}
    com = next_improve_commodity(brain, gs)
    if com:
        need[com] = T["need_improve"]
    need["grain"] = T["need_grain"]
    return need


def steal_value(gs, p, need: dict) -> float:
    """Expected worth to us of one random card from p: sum P(type) * need(type), on the
    brain's hand estimate (gs.est_hand: seen production / known hand)."""
    est = gs.est_hand(p)
    tot = sum(est.values())
    if tot <= 0:
        return 0.0
    return sum(n / tot * need.get(r, T["need_base"]) for r, n in est.items())


def rank_key(brain, gs, need: dict, probs: dict):
    return lambda p: (round(steal_value(gs, p, need), 6), p.total_cards(), p.vp_visible,
                      probs.get(p.name, 0.0))


def prefer(brain, gs, cands: list):
    """The player the targeting stance prefers among cands (PlayerStates already allowed by the
    friendly robber), or None to leave the choice to the tool's own logic.
    1. allowed: BLOCK puts the focus rival first; RACE drops the last-placed rival unless
       nobody else is there. 2. expected value of one steal (grain > next improvement's
    commodity > plan's missing cards > the rest). 3. most cards. 4. the stronger rival."""
    stance = getattr(brain, "stance", None)
    if not getattr(brain, "situational", False) or not stance or not cands:
        return None
    players = gs.players
    if stance == "block":
        f = getattr(brain, "focus", None)
        hit = next((p for p in cands if f is not None and players.index(p) == f), None)
        if hit is not None:
            return hit
        pool = list(cands)              # bounce hops off the focus: the best steal next
    else:
        spare = getattr(brain, "spare", None)
        pool = [p for p in cands if spare is None or players.index(p) != spare] or list(cands)
    return max(pool, key=rank_key(brain, gs, need_table(brain, gs), brain.probs(gs)))


def spy_target(brain, gs, cands: list):
    """Spy (progress card): BLOCK takes from the focus rival if it holds cards; RACE skips the
    last-placed player unless it is the only holder. None = the old choice."""
    stance = getattr(brain, "stance", None)
    if not getattr(brain, "situational", False) or not stance or not cands:
        return None
    players = gs.players
    if stance == "block":
        f = getattr(brain, "focus", None)
        return next((p for p in cands if f is not None and players.index(p) == f), None)
    spare = getattr(brain, "spare", None)
    pool = [p for p in cands if spare is None or players.index(p) != spare]
    return max(pool, key=lambda p: len(p.progress_hand)) if pool else None


def sabotage_ok(brain, gs, low_names: set) -> bool:
    """With the selector on: does a knight stand-down cost the right rival a city?
    BLOCK: only when the focus rival is at the bottom. RACE: only when a rival who is not
    last-placed is. Selector off: True (old rule unchanged)."""
    stance = getattr(brain, "stance", None)
    if not getattr(brain, "situational", False) or not stance:
        return True
    players = gs.players
    names = set(low_names)                  # rivals only (the caller leaves us out)
    if stance == "block":
        f = getattr(brain, "focus", None)
        return f is not None and players[f].name in names
    spare = getattr(brain, "spare", None)
    return any(spare is None or players[spare].name != n for n in names)
