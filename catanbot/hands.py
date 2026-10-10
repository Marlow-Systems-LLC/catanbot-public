"""Per-player hand ledger (owner 2026-10-05): "make bot know what resources exist in everyone's
hand -- they gain, trade and throw away resources, only what is stolen from them is not visible".
Owner 2026-10-06: "make the bot always know what everyone holds at all times, and use that in
its advantage."

Each opponent has an expected count per card kind (resources + commodities) plus an "unknown"
pool: cards we know are there but not what they are. Counts are expectations (floats): a steal
moves probability mass from the victim to the thief instead of inventing certainty. chance()
turns that into P(holds at least n): the known part plus a binomial over the unknown pool.

Events and where they come from (catanbot/runner.py):
  roll       -> on_roll        production off the board (robber, barren canal fields, gold knights,
                               gold fields as unknown picks). Each roll leaves a hint for the
                               city cards the board read can't see (opponent cities look like
                               settlements): a C&K commodity or the second resource.
             -> seven          a 7: every player over its discard limit throws half (spread)
  board read -> observe_pieces new settlements / roads / ships / knights / activations = paid cost
  trades     -> trade          our offer taken, or an incoming offer we accepted (both sides known)
  offers     -> saw_holding    a player's offer shows it holds the cards it gives
  robber     -> expect_take    our steal: spread at once, made exact when our hand shows the card
             -> steal          a steal from us (the thief gets our last hand's shape)
  cards      -> on_my_card     Master Merchant / Wedding / Commercial Harbor we play: spread at
                               once, exact when our hand shows what came in
             -> monopoly       our monopoly: expected take per player (E[min(per, X)]); the next
                               count change says how many each really gave
  banner     -> reconcile_all  the visible card counts. One player up by exactly what the others
                               lost = a steal / taking card between opponents (the spread moves
                               with it); a rise after a roll = the hinted city cards; a drop that
                               matches one hidden build the hand can pay (city, wall, promote,
                               dev, improvement) = that build; the rest -> unknown (up) or spread
                               (down). The total always ends equal to the visible count.
Never visible: what a 1-for-1 trade between two opponents swaps (no count change), the kinds in
a bank trade by others (only the count), what an opponent picks from gold / the aqueduct, what
it discards, and a card stolen from us when the steal text isn't read.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from math import comb

from .board import COMMODITIES, RESOURCES, TERRAIN_COMMODITY, TERRAIN_RES
from .rules import COSTS, discard_limit, production

KINDS = tuple(RESOURCES) + tuple(COMMODITIES)
HOLD_MIN = 0.5          # "plausibly holds": at least half a card expected
SETTLE_S = 3.0          # a banner count read this soon after a ledger event may still be the old one
EPS = 1e-6
KNIGHT_REPEAT_S = 1.0   # a changed knight read must still be there this long after (another frame)
# Great Canal: gold (knights, roll 8) is the only grain until the canal is done, so a gold pick
# leans grain (owner: "gold knight = grain"); the rest is spread over the other resources.
GOLD_PICK_CANAL = {"grain": 0.5, "brick": 0.125, "lumber": 0.125, "wool": 0.125, "ore": 0.125}
EXPECT_S = 15.0         # cards we took must show in our hand this soon after
TRANSFER_MAX = 8        # one player up by what the others lost: at most this many cards moved
MONO_DEDUPE_S = 20.0    # the same monopoly reported twice (play flow + dialog) counts once
# drops the board read can't see: city upgrade (opponent cities look like settlements), wall,
# knight promotion, dev card. C&K improvements (n of one commodity) are added in _drop_costs.
HIDDEN_BUILDS = ("city", "dev", "wall", "promote")
CK_ONLY = ("wall", "promote")
MATCH_MIN, MATCH_MARGIN = 0.5, 1.5   # a drop is a build when the hand can pay it this surely


@dataclass
class PlayerLedger:
    known: dict[str, float] = field(default_factory=lambda: {k: 0.0 for k in KINDS})
    unknown: float = 0.0
    gained: dict[str, float] = field(default_factory=dict)   # everything seen coming in (prior for unknown)
    changed_at: float = 0.0
    seen: bool = False          # any event or visible count so far (else we know nothing about it)
    excl: set = field(default_factory=set)          # kinds the unknown pool can't hold (monopoly emptied them)
    up_hint: dict = field(default_factory=dict)     # what the next unexplained rise is (last roll's city cards)
    up_max: float = 0.0
    down_hint: dict = field(default_factory=dict)   # what the next unexplained drop is (our monopoly)
    down_max: float = 0.0
    mono_kind: str | None = None                    # our monopoly hit this hand; a rise = it had less

    def total(self) -> float:
        return sum(self.known.values()) + self.unknown

    def add(self, cards: dict[str, float]) -> None:
        for k, n in cards.items():
            if n <= 0:
                continue
            self.known[k] = self.known.get(k, 0.0) + n
            self.gained[k] = self.gained.get(k, 0.0) + n
        self.seen = True

    def take(self, cost: dict[str, float]) -> None:
        """Remove these cards: known counts first, the shortfall from unknown; never below zero."""
        for k, n in cost.items():
            have = self.known.get(k, 0.0)
            t = min(have, n)
            self.known[k] = have - t
            self.unknown = max(0.0, self.unknown - (n - t))

    def add_unknown(self, n: float) -> None:
        if n > EPS:
            self.unknown += n
            self.excl.clear()       # new unknown cards may be of any kind again

    def remove_share(self, n: float) -> dict[str, float]:
        """Remove n cards spread over what the hand is made of (known kinds and the unknown pool
        in proportion). Returns {kind: amount} removed, with "?" for the unknown part."""
        tot = self.total()
        if tot <= EPS or n <= 0:
            return {}
        f = min(1.0, n / tot)
        out = {}
        for k, x in self.known.items():
            if x > 0:
                out[k] = x * f
                self.known[k] = max(0.0, x - x * f)
        if self.unknown > 0:
            out["?"] = self.unknown * f
            self.unknown = max(0.0, self.unknown - self.unknown * f)
        return out

    def undo_remove(self, removed: dict[str, float]) -> None:
        for k, x in removed.items():
            if k == "?":
                self.unknown += x
            else:
                self.known[k] = self.known.get(k, 0.0) + x


class HandLedger:
    def __init__(self):
        self.players: dict[str, PlayerLedger] = {}
        self.rolls = 0                        # rolls seen this game (builds count only after the 1st)
        self._pieces: dict[str, dict] = {}    # committed piece snapshot per player
        self._pending: dict[str, tuple] = {}  # knight ((count, active), first seen) waiting to repeat
        self.log: list[str] = []
        self.me: str | None = None            # our name: our hand is read exactly, never ledgered
        self.ck = False                       # Cities & Knights: walls, promotions, improvements exist
        self._my: dict | None = None          # our last hand read (expect_take compares against it)
        self._expect: dict | None = None      # cards we are taking, waiting to show in our hand
        self._mono: tuple = ()                # ((taker, kind), at) of the last monopoly ledgered

    # ---- basics -------------------------------------------------------------------------
    def p(self, name: str) -> PlayerLedger:
        return self.players.setdefault(name, PlayerLedger())

    def has(self, name: str) -> bool:
        return name in self.players

    def knows(self, name: str) -> bool:
        """We have seen something about this hand (an event or a card count)."""
        pl = self.players.get(name)
        return pl is not None and pl.seen

    def count(self, name: str) -> int:
        """Cards in name's hand (equals the visible count after a reconcile)."""
        pl = self.players.get(name)
        return int(round(pl.total())) if pl is not None else 0

    def _note(self, msg: str) -> None:
        self.log.append(msg)
        del self.log[:-200]

    def _touch(self, name: str, now: float | None) -> None:
        pl = self.p(name)
        pl.changed_at = time.time() if now is None else now
        pl.seen = True

    def gain(self, name: str, cards: dict, why: str = "", now: float | None = None) -> None:
        self.p(name).add({k: float(n) for k, n in cards.items()})
        self._touch(name, now)
        if why:
            self._note(f"{name} +{cards} ({why})")

    def pay(self, name: str, cost: dict, why: str = "", now: float | None = None) -> None:
        """name spent these cards: take them from its known counts, the shortfall from unknown
        (it must have held them there); never below zero."""
        self.p(name).take(cost)
        self._touch(name, now)
        if why:
            self._note(f"{name} -{cost} ({why})")

    def needs_recount(self, last_read: float, now: float | None = None) -> bool:
        """An opponent's hand changed after the last banner read and the cards have settled:
        read the counts again so the ledger never drifts from the visible totals."""
        t = time.time() if now is None else now
        return any(pl.changed_at > last_read and t >= pl.changed_at + SETTLE_S
                   for n, pl in self.players.items() if n != self.me)

    # ---- events -------------------------------------------------------------------------
    def on_roll(self, gs, total: int, now: float | None = None) -> dict[str, dict]:
        """Production for every player on this dice total (on a 7: discards). Returns what was
        added."""
        self.rolls += 1
        self.ck = bool(getattr(gs.mode, "ck", False))
        for name, pl in self.players.items():
            pl.up_hint, pl.up_max = {}, 0.0         # hints are about one roll only
            pl.down_hint, pl.down_max, pl.mono_kind = {}, 0.0, None
        if total == 7:
            self.seven(gs, now)
            return {}
        out = gs.roll_yield(total)
        for name, cards in out.items():
            if cards:
                self.gain(name, cards, f"roll {total}", now)
        b = gs.board
        canal_open = gs.mode.canal and gs.canal_progress < gs.canal_total
        for p in gs.players:
            if p.is_me or p.name == self.me:
                continue
            gold, hint = 0, {}
            for v, kind in p.buildings():
                for h in b.vertex_hexes(v):
                    if h.number != total or h.coord == b.robber or h.fog:
                        continue
                    if canal_open and h.barren:
                        continue
                    gold += production(h.terrain, kind, gs.mode).get("gold", 0)
                    res = TERRAIN_RES.get(h.terrain)
                    if kind == "settlement" and res and res != "gold":
                        # the board read can't tell an opponent's city: its 2nd card would be
                        # the commodity (C&K forest/pasture/mountains) or the same resource
                        extra = TERRAIN_COMMODITY.get(h.terrain, res) if gs.mode.ck else res
                        hint[extra] = hint.get(extra, 0) + 1
            pl = self.p(p.name)
            pl.up_hint, pl.up_max = hint, float(sum(hint.values()))
            if gold and not gs.mode.canal:
                # a gold field (Seafarers): cards of the owner's choice, kind unseen
                pl.add_unknown(float(gold))
                self._touch(p.name, now)
                self._note(f"{p.name} +{gold} gold pick")
                out.setdefault(p.name, {})["gold"] = gold
        if gs.mode.canal and total == 8 and gs.canal_progress < gs.canal_total:
            # Great Canal (Almanac 3, sim.produce): a knight on a gold corner collects 1 card of
            # the owner's choice on an 8, at most 1 per player
            for p in gs.players:
                if any(h.terrain == "gold" and h.coord != b.robber for k in p.knights for h in b.vertex_hexes(k)):
                    pl = self.p(p.name)
                    pl.add_unknown(1.0)
                    for k, w in GOLD_PICK_CANAL.items():
                        pl.gained[k] = pl.gained.get(k, 0.0) + w
                    self._touch(p.name, now)
                    self._note(f"{p.name} +1 gold pick")
                    out.setdefault(p.name, {})["gold"] = 1
        return out

    def discard_risk(self, gs) -> dict[str, int]:
        """Cards each opponent must throw on a 7 now (over the discard limit, walls +2 each)."""
        out = {}
        for p in gs.players:
            if p.is_me or p.name == self.me or not self.knows(p.name):
                continue
            n = self.count(p.name)
            if n > discard_limit(gs.mode, p):
                out[p.name] = n // 2
        return out

    def seven(self, gs, now: float | None = None) -> dict[str, int]:
        """A 7: everyone over the limit discards half, kinds unseen (spread). The next count read
        corrects it (an unseen wall raises the limit)."""
        risk = self.discard_risk(gs)
        for name, n in risk.items():
            self.discard(name, n, now)
            self._note(f"{name} discards ~{n} on the 7")
        return risk

    def observe_pieces(self, gs, now: float | None = None) -> list[tuple[str, str]]:
        """Pieces read off the board this frame. Before the first roll they are only a baseline
        (setup pieces are free). After it, a new building / road / ship / knight, or one more
        active knight, means the player paid for it. A changed knight read must still be there
        KNIGHT_REPEAT_S later before it counts (knights are read fresh each frame and can flicker)."""
        self.ck = bool(getattr(gs.mode, "ck", False))
        built = []
        for p in gs.players:
            if p.is_me:
                continue
            snap = {"blds": set(p.settlements) | set(p.cities), "roads": set(p.roads),
                    "ships": set(p.ships), "knights": len(p.knights),
                    "active": sum(1 for k in p.knights.values() if k.active)}
            old = self._pieces.get(p.name)
            if old is None or self.rolls == 0:
                self._pieces[p.name] = dict(snap, kmax=snap["knights"])
                self.p(p.name)
                continue
            for key, kind in (("blds", "settlement"), ("roads", "road"), ("ships", "ship")):
                for _ in snap[key] - old[key]:
                    built.append((p.name, kind))
                old[key] = old[key] | snap[key]
            kn = (snap["knights"], snap["active"])
            t = time.time() if now is None else now
            pend = self._pending.get(p.name)
            if kn == (old["knights"], old["active"]):
                self._pending.pop(p.name, None)
            elif pend is None or pend[0] != kn:
                self._pending[p.name] = (kn, t)     # first sight of this knight read: wait for a repeat
            elif t - pend[1] >= KNIGHT_REPEAT_S:
                self._pending.pop(p.name, None)
                if kn[0] > old["kmax"]:
                    built += [(p.name, "knight")] * (kn[0] - old["kmax"])
                    old["kmax"] = kn[0]
                if kn[1] > old["active"]:
                    built += [(p.name, "activate")] * (kn[1] - old["active"])
                old["knights"], old["active"] = kn
        for name, kind in built:
            self.pay(name, COSTS[kind], f"built {kind}", now)
        return built

    def trade(self, a: str, a_gives: dict, b: str, b_gives: dict, now: float | None = None) -> None:
        """A trade between a and b that we saw happen (either may be us: our side is ignored,
        our hand is read exactly)."""
        for name, out, inc in ((a, a_gives, b_gives), (b, b_gives, a_gives)):
            if name != self.me:
                self.pay(name, out, "trade", now)
                self.gain(name, inc, "trade", now)

    def saw_holding(self, name: str, cards: dict, now: float | None = None) -> None:
        """name offered these cards: it holds at least that many. Raise the known floor, taking
        the difference out of the unknown pool."""
        pl = self.p(name)
        for k, n in cards.items():
            gap = n - pl.known.get(k, 0.0)
            if gap <= EPS:
                continue
            take = min(gap, pl.unknown)
            pl.unknown -= take
            pl.known[k] = pl.known.get(k, 0.0) + gap
            pl.gained[k] = pl.gained.get(k, 0.0) + gap
        self._touch(name, now)
        self._note(f"{name} holds at least {cards} (offer)")

    def steal(self, thief: str | None, victim: str, victim_est: dict | None = None,
              now: float | None = None) -> dict[str, float]:
        """The robber (or a card) takes 1 card from victim. The victim loses one card spread
        over its hand; the thief gains that spread, not a certain card. victim_est: the exact
        hand when the victim is us."""
        if victim_est is not None:
            tot = sum(victim_est.values())
            share = {k: v / tot for k, v in victim_est.items() if v > 0} if tot > 0 else {}
        else:
            pl = self.p(victim)
            removed = pl.remove_share(1.0)
            share = {k: v for k, v in removed.items() if k != "?"}
            if removed.get("?"):
                w = self._weights_for(victim)
                for k, x in w.items():
                    share[k] = share.get(k, 0.0) + removed["?"] * x
            self._touch(victim, now)
        if thief and share and thief != self.me:
            tp = self.p(thief)
            for k, x in share.items():
                tp.known[k] = tp.known.get(k, 0.0) + x
                tp.gained[k] = tp.gained.get(k, 0.0) + x
            self._touch(thief, now)
        self._note(f"{thief or '?'} stole 1 from {victim}: {({k: round(v, 2) for k, v in share.items()})}")
        return share

    def discard(self, name: str, n: int, now: float | None = None) -> None:
        """name threw away n cards of kinds we didn't see."""
        self.p(name).remove_share(n)
        self._touch(name, now)

    # ---- cards we take: spread now, exact once our hand shows them -----------------------
    def expect_take(self, victims: list[str], n_each: int, give: list[str] | None = None,
                    why: str = "take", now: float | None = None) -> None:
        """We take n_each cards from each victim (our robber steal, Master Merchant, Wedding,
        Commercial Harbor). Each loses n_each spread over its hand now; give: the resources we
        hand out in return, one per victim (Commercial Harbor, which goes where unseen). When
        our hand next shows exactly the cards that came in, the victims pay those kinds."""
        victims = [v for v in victims if v and v != self.me]
        if not victims:
            return
        t = time.time() if now is None else now
        removed = {}
        for v in victims:
            pl = self.p(v)
            removed[v] = pl.remove_share(n_each)
            if give:
                pl.add({r: give.count(r) / len(victims) for r in set(give)})
            self._touch(v, t)
        self._expect = {"victims": victims, "n": n_each, "removed": removed, "at": t, "why": why,
                        "base": dict(self._my) if self._my is not None else None}
        self._note(f"we take {n_each} from each of {victims} ({why})")

    def my_hand(self, hand: dict, now: float | None = None) -> dict | None:
        """Our hand as read this frame. Resolves a pending expect_take: returns {kind: n} that
        came in when it matched, else None."""
        t = time.time() if now is None else now
        ex, got = self._expect, None
        if ex is not None:
            base = ex["base"]
            if base is None or t - ex["at"] > EXPECT_S:
                self._expect = None
            else:
                inc = {k: hand.get(k, 0) - base.get(k, 0) for k in hand if hand.get(k, 0) > base.get(k, 0)}
                if inc and sum(inc.values()) == ex["n"] * len(ex["victims"]):
                    got = inc
                    self._resolve(ex, inc, t)
                    self._expect = None
        self._my = {k: v for k, v in hand.items() if v is not None}
        return got

    def _resolve(self, ex: dict, inc: dict, now: float) -> None:
        k = len(ex["victims"])
        for v in ex["victims"]:
            pl = self.p(v)
            pl.undo_remove(ex["removed"].get(v, {}))
            pl.take({r: n / k for r, n in inc.items()})
            for r, n in inc.items():                 # it held them: they count as seen coming in
                pl.gained[r] = pl.gained.get(r, 0.0) + n / k
            self._touch(v, now)
        self._note(f"{ex['why']}: took {inc} from {ex['victims']} (exact)")

    def on_my_card(self, key: str, args: dict, gs, now: float | None = None) -> None:
        """A progress card we just played that takes cards from players."""
        if key == "master_merchant" and args.get("player"):
            self.expect_take([str(args["player"])], 2, why="master merchant", now=now)
        elif key == "wedding" and args.get("from"):
            self.expect_take(list(args["from"]), 2, why="wedding", now=now)
        elif key == "commercial_harbor" and args.get("give"):
            opps = [p.name for p in gs.opponents() if p.total_cards() > 0][:len(args["give"])]
            self.expect_take(opps, 1, give=list(args["give"]), why="commercial harbor", now=now)

    # ---- monopoly -----------------------------------------------------------------------
    def monopoly(self, taker: str, kind: str, per_player: int, names: list[str],
                 now: float | None = None, gs=None) -> float:
        """taker named kind: every other player gives up to per_player of it. Each gives what it
        is expected to (E[min(per, held)], known first, then the unknown pool). Whoever was
        unlikely to hold more has none left; the next count read says how many each really gave
        (down_hint / mono_kind correct it in reconcile)."""
        t = time.time() if now is None else now
        if self._mono and self._mono[0] == (taker, kind) and abs(t - self._mono[1]) < MONO_DEDUPE_S:
            return 0.0                                # the same play reported twice
        self._mono = ((taker, kind), t)
        took = 0.0
        for n in names:
            if n == taker or n == self.me:
                continue
            pl = self.p(n)
            x = self.expected_take(n, kind, per_player, gs)
            more = self.chance(n, kind, per_player + 1, gs)
            pl.take({kind: x})
            if more < HOLD_MIN:
                pl.known[kind] = 0.0
                pl.excl.add(kind)
            pl.down_hint, pl.down_max, pl.mono_kind = {kind: 1.0}, max(0.0, per_player - x), kind
            took += x
            self._touch(n, t)
        if taker != self.me:
            self.gain(taker, {kind: took}, "monopoly", t)
        self._note(f"{taker} monopoly {kind}: took ~{took:.1f}")
        return took

    # ---- the visible count --------------------------------------------------------------
    def reconcile(self, name: str, count: int | None, read_at: float | None = None) -> None:
        """Visible card count. A gap is explained where it can be (city cards after a roll, our
        monopoly, one hidden build the hand can pay), the rest goes to unknown (up) or is spread
        (down). Ends with the total equal to count. Skipped when the count was read before (or
        right after) the last ledger event."""
        if count is None:
            return
        pl = self.p(name)
        if read_at is not None and read_at < pl.changed_at + SETTLE_S:
            return
        pl.seen = True
        gap = count - pl.total()
        keep_excl = False
        if gap > EPS:
            keep_excl = self._explain_rise(name, pl, gap)
        elif gap < -EPS:
            self._explain_drop(name, pl, -gap)
        for k in pl.known:
            pl.known[k] = max(0.0, pl.known[k])
        pl.unknown = max(0.0, pl.unknown)
        gap = count - pl.total()                     # whatever is left: never drift from the count
        if gap > EPS:
            if keep_excl:
                pl.unknown += gap                    # cards we took too many of: not the named kind
            else:
                pl.add_unknown(gap)
        elif gap < -EPS:
            pl.remove_share(-gap)

    def _explain_rise(self, name: str, pl: PlayerLedger, gap: float) -> bool:
        """Explain a count rise; True when the rest is cards we wrongly took off (keep excl)."""
        if pl.mono_kind:
            # it gave fewer than we took off: it had less, so none of that kind is left
            pl.known[pl.mono_kind] = 0.0
            pl.excl.add(pl.mono_kind)
            pl.mono_kind, pl.down_hint, pl.down_max = None, {}, 0.0
            return True                               # the rest is unknown (reconcile tops up)
        x = min(gap, pl.up_max)
        s = sum(pl.up_hint.values())
        if x > EPS and s > 0:
            pl.add({k: x * w / s for k, w in pl.up_hint.items()})
            pl.up_max -= x
            self._note(f"{name} +{x:.0f} city cards {pl.up_hint}")
        return False

    def _explain_drop(self, name: str, pl: PlayerLedger, n: float) -> None:
        if pl.down_max > EPS and pl.down_hint:
            x = min(n, pl.down_max)
            s = sum(pl.down_hint.values())
            pl.take({k: x * w / s for k, w in pl.down_hint.items()})
            pl.down_max -= x
            n -= x
        if n > 1 - EPS and abs(n - round(n)) < 1e-3:
            b = self._match_cost(name, int(round(n)))
            if b is not None:
                pl.take(b[1])
                self._note(f"{name} -{b[1]} (count drop = {b[0]})")
                n = 0.0
        if n > EPS:
            pl.remove_share(n)

    def _drop_costs(self) -> list[tuple[str, dict]]:
        out = [(b, COSTS[b]) for b in HIDDEN_BUILDS if self.ck or b not in CK_ONLY]
        if self.ck:
            out += [(f"{c} improvement {n}", {c: n}) for c in COMMODITIES for n in range(1, 6)]
        return out

    def _match_cost(self, name: str, n: int) -> tuple[str, dict] | None:
        """The one hidden build of n cards this hand can surely pay, else None."""
        cands = sorted(((self.afford_chance(name, c), b, c) for b, c in self._drop_costs()
                        if sum(c.values()) == n), key=lambda x: -x[0])
        if not cands or cands[0][0] < MATCH_MIN:
            return None
        if len(cands) > 1 and cands[0][0] < MATCH_MARGIN * cands[1][0]:
            return None
        return cands[0][1], cands[0][2]

    def reconcile_all(self, counts: dict, read_at: dict | None = None, now: float | None = None) -> None:
        """All visible counts of one banner read. When exactly one player went up and the others
        went down by the same total, the cards moved between them (a steal, Master Merchant,
        Wedding or a monopoly among opponents): the gainer gets the losers' spread. Then each
        hand is reconciled to its count."""
        read_at = read_at or {}
        gaps = {}
        for name, c in counts.items():
            if c is None or name == self.me:
                continue
            pl = self.p(name)
            ra = read_at.get(name)
            if ra is not None and ra < pl.changed_at + SETTLE_S:
                continue
            if not pl.seen:
                gaps[name] = 0.0                     # a first count is a baseline, not a move
                continue
            gaps[name] = c - pl.total()
        ups = [n for n, g in gaps.items() if g > 0.5]
        downs = [n for n, g in gaps.items() if g < -0.5]
        if len(ups) == 1 and downs and abs(sum(gaps[n] for n in ups + downs)) < 0.5 \
                and gaps[ups[0]] <= TRANSFER_MAX and all(abs(gaps[n] - round(gaps[n])) < 1e-3 for n in downs):
            t = max([x for x in read_at.values() if x is not None], default=None) if now is None else now
            for v in downs:
                for _ in range(int(round(-gaps[v]))):
                    self.steal(ups[0], v, now=t)
            self._note(f"count moves: {ups[0]} took {sum(-gaps[n] for n in downs):.0f} from {downs}")
        for name in gaps:
            self.reconcile(name, counts[name], None)

    # ---- reading it back ----------------------------------------------------------------
    def in_game(self) -> set[str]:
        """Kinds anyone has been seen getting this game."""
        return {k for pl in self.players.values() for k, x in pl.gained.items() if x > EPS}

    def _weights_for(self, name: str, gs=None) -> dict[str, float]:
        pl = self.players.get(name)
        excl = pl.excl if pl else set()
        w = {k: x for k, x in (pl.gained if pl else {}).items() if x > EPS and k not in excl}
        if not w and gs is not None:
            p = next((q for q in gs.players if q.name == name), None)
            if p is not None:
                w = {r: x for r, x in gs.income(p).items() if r != "gold" and x > 0 and r not in excl}
        if not w:
            w = {k: 1.0 for k in self.in_game() if k not in excl}
        if not w:
            w = {r: 1.0 for r in RESOURCES if r not in excl} or {r: 1.0 for r in RESOURCES}
        s = sum(w.values())
        return {k: x / s for k, x in w.items()}

    def est(self, name: str, gs=None) -> dict[str, float]:
        """Expected hand: known counts plus the unknown pool split like what the player has been
        seen getting (else its income, else what is in the game)."""
        pl = self.players.get(name)
        if pl is None:
            return {}
        out = {k: x for k, x in pl.known.items() if x > EPS}
        if pl.unknown > EPS:
            for k, x in self._weights_for(name, gs).items():
                out[k] = out.get(k, 0.0) + pl.unknown * x
        return out

    def holds(self, name: str, kind: str, gs=None) -> float:
        return self.est(name, gs).get(kind, 0.0)

    def dist(self, name: str, kind: str, gs=None) -> list[float]:
        """P(name holds exactly i of kind) for i = 0, 1, ...: the whole known cards, a coin flip
        for a known fraction (a steal's spread), a binomial over the unknown pool."""
        pl = self.players.get(name)
        if pl is None:
            return [1.0]
        k = max(0.0, pl.known.get(kind, 0.0))
        base = int(k + 1e-3)
        frac = k - base if k - base > 1e-3 else 0.0
        u = int(round(pl.unknown))
        w = self._weights_for(name, gs).get(kind, 0.0) if u else 0.0
        d = [comb(u, i) * w ** i * (1 - w) ** (u - i) for i in range(u + 1)]
        if frac:
            d = [(d[i] if i < len(d) else 0.0) * (1 - frac) + (d[i - 1] * frac if i else 0.0)
                 for i in range(len(d) + 1)]
        return [0.0] * base + d

    def chance(self, name: str, kind: str, n: int = 1, gs=None) -> float:
        """P(name holds at least n of kind)."""
        if n <= 0:
            return 1.0
        return min(1.0, sum(self.dist(name, kind, gs)[n:]))

    def expected_take(self, name: str, kind: str, per: float, gs=None) -> float:
        """E[min(per, held)]: what a monopoly taking up to per gets from name."""
        return sum(min(per, i) * x for i, x in enumerate(self.dist(name, kind, gs)))

    def afford_chance(self, name: str, cost: dict, extra: dict | None = None, gs=None) -> float:
        """P(name can pay cost), with extra cards added (kinds taken as independent)."""
        out = 1.0
        for k, c in cost.items():
            out *= self.chance(name, k, int(c - (extra or {}).get(k, 0)), gs)
        return out

    def feeds_build(self, name: str, give: dict, builds=("city",), gs=None) -> str | None:
        """Owner 2026-10-06: don't hand a player the last card of a city. The build our cards
        would complete for name (it likely can't pay it now, likely can after), else None."""
        for b in builds:
            before = self.afford_chance(name, COSTS[b], gs=gs)
            after = self.afford_chance(name, COSTS[b], give, gs)
            if after >= HOLD_MIN and after - before >= 0.25:
                return b
        return None

    def best_cards_value(self, name: str, n: int, value, gs=None) -> float:
        """Worth of the n best cards we could pick from name's hand (Master Merchant): expected
        count per kind, best kinds first, under value(kind)."""
        est = sorted(self.est(name, gs).items(), key=lambda kv: -value(kv[0]))
        out, left = 0.0, float(n)
        for k, x in est:
            t = min(left, x)
            out += t * value(k)
            left -= t
            if left <= EPS:
                break
        return out
