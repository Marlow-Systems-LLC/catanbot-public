"""Self-play simulator (base rules + a Cities & Knights core) for testing and pretraining.

Not a full re-implementation of Catan Universe: it covers the economy, building,
robber/blocking, bank and player trades, longest road/largest army and (for C&K)
commodities, improvements, metropolises, knights and barbarians. That's what the
learned weights depend on.
"""
from __future__ import annotations

import datetime
import json
import random
from pathlib import Path

from . import progress_cards as pc
from .board import COMMODITIES, PIPS, RESOURCES, board_for_mode
from .brain import Action, Brain
from .learning import GameLog, Model
from .rules import COSTS, MODES, discard_limit, pay, production
from .state import GameState, PlayerState, award_longest_road, dig_canals, grant_metropolis, new_hand
from .threat import features

EVENT_DIE = ("ship", "ship", "ship", "trade", "politics", "science")


class Sim:
    def __init__(self, mode_key="great_canal", n=4, seed=None, brains=None, log_player=0, record_setup=False):
        self.seed = seed
        self.mode_key = mode_key
        self.record_setup = record_setup        # keep one setup-feature row per seat (setup_rows)
        self.setup_moves: dict[int, list] = {}  # seat -> [(kind, vertex, edge, "road"|"ship")] in placement order
        self.setup_feats: list[dict] = []
        self.rng = random.Random(seed)
        self.mode = MODES[mode_key]
        self.gs = GameState(board_for_mode(mode_key, self.rng.randrange(1 << 30)), self.mode)
        self.gs.canal_total = len(self.gs.board.canal_cells()) or 9
        for i in range(n):
            p = PlayerState(f"P{i}")
            p.hand = new_hand()
            self.gs.players.append(p)
        self.brains = brains or [Brain() for _ in range(n)]
        for b in self.brains:
            b.knight_moves = True               # the sim can carry out knight moves
            b.ship_moves = True                 # ...and ship moves
        self.dev_deck = ["knight"] * 14 + ["vp"] * 5 + ["road"] * 2 + ["plenty"] * 2 + ["mono"] * 2
        self.rng.shuffle(self.dev_deck)
        self.dev_hidden: dict[str, list[str]] = {p.name: [] for p in self.gs.players}
        self.log = GameLog(mode_key, [p.name for p in self.gs.players], f"P{log_player}")
        self.log_player = log_player
        self.decks = {t: [k for k, c in pc.CARDS.items() if c["deck"] == t for _ in range(c["count"])]
                      for t in ("trade", "politics", "science")}
        self.progress_drawn = {}
        self.progress_discarded = 0
        self.displaced = 0                      # knights pushed off by stronger ones
        self.cards_lost = {}                    # name -> cards lost to 7-discards and robbing
        self.attacks = 0                        # barbarian attacks so far
        self.lost: list[tuple[int, str]] = []   # (attack index, player) cities lost
        self.chases = 0                         # knights that chased the robber away
        # adaptation: an opponent took the spot a seat was heading for -> turns to its next settlement
        self.disrupted: dict[int, int] = {}     # seat -> turn its goal spot was taken
        self.recover: list[int] = []            # turns from disruption to the next settlement
        self.disruptions = 0

    # -- perspective: brain i sees itself as "me" (in the sim everyone sees exact hands)
    def as_me(self, i):
        for j, p in enumerate(self.gs.players):
            p.is_me = (i == j)
        return self.gs

    def vp(self, p: PlayerState) -> int:
        v = len(p.settlements) + 2 * len(p.cities) + 2 * len(p.metropolises)
        v += 2 * p.has_longest_road + 2 * p.has_largest_army + p.defender_vp + p.canal_chips + p.special_vp
        v += self.dev_hidden[p.name].count("vp_played")
        v += getattr(p, "merchant_hex", None) is not None     # C&K Merchant: +1 VP while placed
        return v

    def total_vp(self, p):
        return self.vp(p) + self.dev_hidden[p.name].count("vp")

    def refresh(self):
        for p in self.gs.players:
            p.vp_visible = self.vp(p)
            p.hand_count = sum(p.hand.values())
            p.dev_cards = len([c for c in self.dev_hidden[p.name] if c != "vp_played"])

    # -- setup -------------------------------------------------------------------
    def setup(self):
        n = len(self.gs.players)
        order = list(range(n)) + list(range(n - 1, -1, -1))
        if self.mode.canal:
            order += list(range(n - 1, -1, -1))   # Great Canal: 3rd (coastal) settlement + ship, reverse order like the city round (owner)
        last = len(order) - n               # the last round pays the start cards
        for k, i in enumerate(order):
            gs = self.as_me(i)
            p = gs.players[i]
            v = self.brains[i].choose_setup_settlement(gs)
            if n <= k < 2 * n and self.mode.start_with_city:
                p.cities.add(v)
            else:
                p.settlements.add(v)
            e = self.brains[i].choose_setup_road(gs, v)
            # Almanac: a setup piece on the coast gets a ship (choose_setup_road gives a sea edge)
            ship = gs.board.is_sea_edge(e) and (not gs.board.is_land_edge(e)
                                                or (self.mode.canal and gs.board.is_coastal_vertex(v)))
            (p.ships if ship else p.roads).add(e)
            self.setup_moves.setdefault(i, []).append(
                ("city" if v in p.cities else "settlement", v, e, "ship" if ship else "road"))
            if k >= last:
                for h in gs.board.vertex_hexes(v):
                    r = {"hills": "brick", "forest": "lumber", "pasture": "wool",
                         "fields": "grain", "mountains": "ore"}.get(h.terrain)
                    if r:                   # barren fields still pay start grain (owner)
                        p.hand[r] += 1
        self.refresh()
        if self.record_setup:
            self.setup_feats = [setup_features(self.as_me(i), i, self.brains[i], self.setup_moves.get(i, []))
                                for i in range(n)]
            self.as_me(self.log_player)

    # -- turn ------------------------------------------------------------------------
    def produce(self, roll):
        b = self.gs.board
        for i, p in enumerate(self.gs.players):
            gold = got = 0
            for v, kind in p.buildings():
                for h in b.vertex_hexes(v):
                    if h.number == roll and h.coord != b.robber:
                        if self.mode.canal and h.barren and self.gs.canal_progress < self.gs.canal_total:
                            continue
                        for r, x in production(h.terrain, kind, self.mode).items():
                            if r == "gold":
                                gold += x
                                continue
                            p.hand[r] = p.hand.get(r, 0) + x
                            got += x
            if self.mode.canal and roll == 8 and self.gs.canal_progress < self.gs.canal_total:
                # Great Canal (Almanac 3): only knights collect gold (active or not), any 1 card on
                # a roll of 8, at most 1 per player per roll however many gold knights it has; the
                # gold digger stops once the canal is complete
                if any(h.terrain == "gold" and h.coord != b.robber
                       for k in p.knights for h in b.vertex_hexes(k)):
                    gold += 1
            if not gold and not got and self.mode.ck and p.improvements.get("science", 0) >= 3:
                gold = 1                     # aqueduct: a roll that gives nothing pays 1 of choice
            if gold:                         # gold: the player picks the resources
                for r in self.brains[i].choose_resource(self.as_me(i), gold):
                    p.hand[r] = p.hand.get(r, 0) + 1

    def seven(self, i):
        for j, p in enumerate(self.gs.players):
            limit = discard_limit(self.mode, p)
            c = sum(p.hand.values())
            if c > limit:
                gs = self.as_me(j)
                for r, x in self.brains[j].choose_discard(gs, c // 2).items():
                    p.hand[r] -= x
                    self.cards_lost[p.name] = self.cards_lost.get(p.name, 0) + x
                    p.cards_lost += x
        if self.mode.ck and not self.attacks:
            return                          # C&K: the robber stays put until the first attack
        self.move_robber(i)

    def move_robber(self, i, only=None):
        gs = self.as_me(i)
        b = gs.board
        options = [c for c, h in b.hexes.items() if h.is_land and c != b.robber and h.terrain != "desert"
                   and (only is None or c in only)]
        if self.mode.friendly_robber:
            self.refresh()
            options = gs.friendly_robber_hexes(options, gs.players[i])
        hexc, victim = self.brains[i].choose_robber(gs, options)
        gs.board.robber = hexc
        if victim:
            pool = [r for r, x in victim.hand.items() for _ in range(x)]
            if pool:
                r = self.rng.choice(pool)
                victim.hand[r] -= 1
                gs.players[i].hand[r] = gs.players[i].hand.get(r, 0) + 1
                self.cards_lost[victim.name] = self.cards_lost.get(victim.name, 0) + 1
                victim.cards_lost += 1
            self.log.event(f"P{i} robs {victim.name}")

    def barbarians(self):
        gs = self.gs
        gs.barbarian_pos += 1
        if gs.barbarian_pos < gs.barbarian_len:
            return
        gs.barbarian_pos = 0
        cities = sum(len(p.cities) for p in gs.players)
        strength = {p.name: gs.defence_strength(p) for p in gs.players}
        if sum(strength.values()) >= cities:
            best = max(strength.values())
            top = [p for p in gs.players if strength[p.name] == best]
            if len(top) == 1:
                top[0].defender_vp += 1
            else:                           # tie: no VP, each tied player draws a card of choice
                for p in top:
                    j = gs.players.index(p)
                    track = self._pick_deck(p)
                    if track and self._draw(j, track) and j != gs.current:
                        self._progress_limit(p)     # the roller keeps a 5th until its turn ends (play_held)
        else:
            # only players with a city that is not a metropolis can be the weakest
            exposed = [p for p in gs.players if any(v not in p.metropolises for v in p.cities)]
            low = min((strength[p.name] for p in exposed), default=0)
            for p in exposed:
                if strength[p.name] == low:
                    v = next(v for v in p.cities if v not in p.metropolises)
                    p.cities.discard(v)
                    p.walls.discard(v)      # a pillaged city loses its wall
                    p.settlements.add(v)
                    self.lost.append((self.attacks, p.name))
        self.attacks += 1
        gs.barbarians_attacked = True
        for p in gs.players:
            for k in p.knights.values():
                k.active = False

    def execute(self, i, a: Action) -> bool:
        """Apply one plan item; False when it could not be paid (skipped)."""
        p = self.gs.players[i]
        if a.kind == "bank_trade":
            pay(p.hand, a.give)
            for k, v in a.get.items():
                p.hand[k] = p.hand.get(k, 0) + v
            return True
        brain = self.brains[i]
        cost = brain._cost(self.gs, a)
        if any(p.hand.get(k, 0) < v for k, v in cost.items()):
            return False
        chips = p.canal_chips
        if a.kind == "move_knight":
            self._displace(i, a.target[1])
        g2 = brain._apply(self.as_me(i), a)
        me2 = g2.me()
        # copy the structural result back
        for attr in ("hand", "settlements", "cities", "roads", "ships", "knights", "walls",
                     "improvements", "canal_chips", "ships_new", "ship_moved"):
            setattr(p, attr, getattr(me2, attr))
        p.canal_chips = chips
        if a.kind == "improve":             # on the real state: a level-5 steal hits the holder
            grant_metropolis(self.gs, p, a.target)
        if self.mode.canal:
            for c, x, y in dig_canals(self.gs):
                self.log.event(f"canal dug at {c} by {x}+{y}")
        if a.kind == "dev":
            if self.dev_deck:
                self.dev_hidden[p.name].append(self.dev_deck.pop())
        self.update_awards()
        return True

    def _displace(self, i, v):
        """C&K: a knight pushed off v moves (same level and status) to a free corner on its
        owner's roads, the owner's pick; with no such corner it goes back to the supply."""
        gs = self.gs
        for j, o in enumerate(gs.players):
            if j == i or v not in o.knights:
                continue
            k = o.knights.pop(v)
            self.displaced += 1
            spots = {w for e in o.roads | o.ships for w in e if w != v}
            spots = [w for w in sorted(spots) if gs.board.is_land_vertex(w) and not gs.occupied(w)]
            if spots:
                b, g = self.brains[j], self.as_me(j)
                w = max(spots, key=lambda w: b._canal_pair_value(g, Action("knight", w)) if self.mode.canal else 0)
                o.knights[w] = k
            self.as_me(i)
            self.log.event(f"P{i} displaces {o.name}'s knight at {v}")

    def play_dev(self, i):
        p = self.gs.players[i]
        cards = self.dev_hidden[p.name]
        if "knight" in cards and (self.gs.board.robber in self._my_hexes(p) or len(cards) > 0):
            cards.remove("knight")
            p.knights_played += 1
            self.move_robber(i)
            self.update_awards()
            return
        if "plenty" in cards:
            cards.remove("plenty")
            for r in self.brains[i].choose_resource(self.as_me(i), 2):
                p.hand[r] += 1
        elif "mono" in cards:
            cards.remove("mono")
            r = self.brains[i].choose_monopoly(self.as_me(i))
            for q in self.gs.players:
                if q is not p:
                    p.hand[r] += q.hand.get(r, 0)
                    q.hand[r] = 0

    def _my_hexes(self, p):
        return {h.coord for v, _ in p.buildings() for h in self.gs.board.vertex_hexes(v)}

    def update_awards(self):
        gs = self.gs
        award_longest_road(gs)     # owner 2026-10-04: a broken road loses the card (knights break roads)
        if self.mode.has_dev_cards:
            holder = next((p for p in gs.players if p.has_largest_army), None)
            best = max(gs.players, key=lambda p: p.knights_played)
            if best.knights_played >= 3 and (holder is None or best.knights_played > holder.knights_played):
                for p in gs.players:
                    p.has_largest_army = p is best

    def player_trade(self, i):
        gs = self.as_me(i)
        prop = self.brains[i].propose_trade(gs)
        if not prop:
            return
        give, get, partners = prop
        for name in partners:
            j = next(k for k, p in enumerate(gs.players) if p.name == name)
            gj = self.as_me(j)
            # partner gives `get`, receives `give`
            if self.brains[j].evaluate_offer(gj, gs.players[i].name, get, give):
                a, b = gs.players[i], gs.players[j]
                for k, v in give.items():
                    a.hand[k] -= v
                    b.hand[k] = b.hand.get(k, 0) + v
                for k, v in get.items():
                    b.hand[k] -= v
                    a.hand[k] = a.hand.get(k, 0) + v
                self.log.event(f"trade {a.name}->{b.name} {give} for {get}")
                return

    def snapshot(self):
        gs = self.as_me(self.log_player)
        self.refresh()
        brain = self.brains[self.log_player]
        feats = {p.name: features(gs, p, brain.model.params) for p in gs.players}
        self.log.snapshot(gs.turn, feats, brain.probs(gs))

    def turn(self, i):
        gs = self.gs
        gs.current = i
        for k in gs.players[i].knights.values():
            k.acted = False                     # one action per knight per turn
        if getattr(self.brains[i], "situational", False):
            self.refresh()
            self.brains[i].situational_begin(self.as_me(i))   # this turn's stance before the roll
        gs.players[i].ships_new = set()         # ships built last turn may move now
        gs.players[i].ship_moved = False        # one ship move per turn
        if self.mode.has_dev_cards and self.dev_hidden[gs.players[i].name] and self.rng.random() < 0.5:
            self.play_dev(i)
        from . import sim_cards
        d = sim_cards.pre_roll(self, i)         # Alchemist: the holder sets the dice before rolling
        if d:
            white, red = d
        else:
            red, white = self.rng.randint(1, 6), self.rng.randint(1, 6)
        roll = red + white
        if self.mode.ck:
            face = EVENT_DIE[self.rng.randrange(6)]   # owner: event die moves barbarians OR opens a gate
            if face == "ship":
                self.barbarians()
            else:
                self.progress_draw(face, red)
        if roll == 7:
            self.seven(i)
        else:
            gs.record_roll(roll)                # what everyone was seen getting (owner)
            self.produce(roll)
        if self.mode.ck:
            self.play_held(i)
            self.chase_robber(i)
        self.refresh()
        lost = sum(n == gs.players[i].name for _, n in self.lost)
        self.brains[i].update_outlook(self.as_me(i), lost, rank=True)
        self.player_trade(i)
        self.refresh()
        self.brains[i].update_outlook(self.as_me(i), lost)
        # owner 2026-10-06: cards first (play_held above), then the plan from the hand they left;
        # after the builds the held cards are offered again (Medicine/Smith/Road Building may only
        # have a use now) and anything they bring is planned again -- never end holding a usable card
        for _ in range(3):
            plan = self.brains[i].plan_turn(self.as_me(i))
            skipped = []
            for a in plan:
                n0 = len(gs.players[i].settlements) + len(gs.players[i].cities)
                if skipped and not self.brains[i].still_legal(self.as_me(i), [], skipped, a):
                    skipped.append(a)            # Brain.replan_after_skip: needs a skipped item
                    continue
                if not self.execute(i, a):
                    skipped.append(a)
                if len(gs.players[i].settlements) + len(gs.players[i].cities) > n0 and i in self.disrupted:
                    self.recover.append(gs.turn - self.disrupted.pop(i))
                self._check_disrupted(i)
            self.refresh()
            if not (self.mode.ck and plan and self.play_held(i)):
                break
            self.refresh()

    def progress_draw(self, track, red):
        """Gate face: every player whose improvement level on that track reaches the red die
        draws a progress card (level 1 draws on red 1-2, level 5 on 1-6). Cards are held and
        played on the owner's turn (simplified effects, _play_progress). Owner: at most 4 held;
        a 5th drawn on someone else's turn is discarded at once, the roller may keep it until
        its turn ends."""
        gs = self.gs
        for j, p in enumerate(gs.players):
            lvl = p.improvements.get(track, 0)
            if lvl < 1 or red > lvl + 1:
                continue
            if self._draw(j, track) and j != gs.current:
                self._progress_limit(p)

    def _draw(self, j, track) -> bool:
        """Draw one card from the track's deck; True if it went into the hand."""
        p = self.gs.players[j]
        deck = self.decks[track]
        if not deck:
            return False
        card = deck.pop(self.rng.randrange(len(deck)))
        self.progress_drawn[p.name] = self.progress_drawn.get(p.name, 0) + 1
        if card in pc.VP_CARDS:
            p.special_vp += 1                   # shown at once, not held
            return False
        p.progress_hand.append(card)
        return True

    def _pick_deck(self, p):
        """Defender tie: the colour of choice -- the highest track, owner order paper > coin > cloth."""
        order = ("science", "politics", "trade")
        ok = [t for t in order if self.decks[t]]
        return max(ok, key=lambda t: (p.improvements.get(t, 0), -order.index(t)), default=None)

    def _progress_limit(self, p):
        if len(p.progress_hand) <= pc.PROGRESS_HAND_LIMIT:
            return
        was = next((j for j, q in enumerate(self.gs.players) if q.is_me), self.gs.current)
        i = self.gs.players.index(p)
        gs = self.as_me(i)
        dead = getattr(self.brains[i], "wedding_dead_when_leading", False)
        while len(p.progress_hand) > pc.PROGRESS_HAND_LIMIT:
            c = min(p.progress_hand, key=lambda c: pc.keep_value(gs, p, c, dead))
            p.progress_hand.remove(c)
            self.decks[pc.CARDS[c]["deck"]].insert(0, c)   # back under its deck
            self.progress_discarded += 1
        self.as_me(was)                                   # keep the caller's perspective

    def play_held(self, i) -> int:
        """Own turn, after the roll and before the plan: play held cards in the live bot's order
        (pc.MAIN_ORDER: resource cards before Merchant Fleet); warlord waits for the barbarians.
        Returns how many cards were played."""
        gs = self.gs
        p = gs.players[i]
        near = gs.barbarian_pos >= gs.barbarian_len - 2
        rank = {c: n for n, c in enumerate(pc.MAIN_ORDER)}
        played = 0
        for c in sorted(p.progress_hand, key=lambda c: rank.get(c, len(rank))):
            if c not in p.progress_hand:
                continue
            if c == "warlord" and not near and len(p.progress_hand) <= pc.PROGRESS_HAND_LIMIT \
                    and pc.choose_warlord(gs, p) is None:
                continue                         # owner 2026-10-04: idle knights to activate -> play it, no grain
            p.progress_hand.remove(c)
            played += bool(self._play_progress(i, c))
        self._progress_limit(p)
        return played

    def _play_progress(self, j, card):
        gs = self.gs
        p = gs.players[j]
        if card == "warlord":
            for k in p.knights.values():
                k.active = True
            if self.mode.canal:                 # a pair activated by the Warlord digs at once
                for c, x, y in dig_canals(gs):
                    self.log.event(f"canal dug at {c} by {x}+{y} (warlord)")
            return True
        from . import sim_cards                 # real card effects (not playable now -> kept)
        return sim_cards.play(self, j, card)

    def chase_robber(self, i):
        """C&K knight action, own turn after the roll: an ACTIVE knight next to the robber
        deactivates, moves it and steals 1. Old play: one chase, only when the robber blocks one of
        our producing hexes (a gold-corner knight: only off its own gold hex) and the knight isn't
        needed against barbarians within 2 steps.
        Owner 2026-10-04 (brain flags):
        - knight_bounce: chase to steal too, and bounce the robber from knight to knight round the
          table (each chase puts it next to our next active knight, on a rival with cards) while we
          can afford the lost defence: barbarians far, grain/Warlord to reactivate, or no city at risk.
        - barb_sabotage: near an attack, drop a knight when the barbarians still win only against
          a rival (our city stays safe), or when they are held anyway."""
        gs, b = self.gs, self.gs.board
        p = gs.players[i]
        br = self.brains[i]
        if b.robber is None or not self.attacks or not getattr(br, "knight_chase", True):
            return
        bounce = getattr(br, "knight_bounce", False)
        sabotage = getattr(br, "barb_sabotage", False)
        dry = self.mode.canal and gs.canal_progress < gs.canal_total
        mine = {x.coord for v, _ in p.buildings() for x in b.vertex_hexes(v)
                if x.number and not (dry and x.barren)}
        near = gs.barbarian_pos >= gs.barbarian_len - 2
        done = 0
        while True:
            h = b.hexes.get(b.robber)
            if h is None:
                return
            pick = None
            for v, k in p.knights.items():
                if not k.active or k.acted or h not in b.vertex_hexes(v):
                    continue
                gold = self.mode.canal and any(x.terrain == "gold" for x in b.vertex_hexes(v))
                unblock = h.terrain == "gold" if gold else h.coord in mine
                if not unblock and not (bounce and not gold and self._steal_target(i)):
                    continue
                k.active = False                # what if this knight stands down?
                ok = self._can_drop(i, near, sabotage)
                if ok and bounce and (done or not unblock):
                    ok = self._can_bounce(i, near, done)
                k.active = True
                if ok:
                    pick = (v, k)
                    break
            if pick is None:
                return
            v, k = pick
            k.active, k.acted = False, True
            only = self._bounce_hexes(i, v) if bounce else None
            self.move_robber(i, only or None)
            self.chases += 1
            done += 1
            if not bounce:
                return

    def _steal_target(self, i) -> bool:
        """Any rival we may rob (friendly robber) that holds cards."""
        gs = self.gs
        safe = gs.robber_protected() if self.mode.friendly_robber else set()
        return any(q is not gs.players[i] and q.name not in safe and q.total_cards() > 0 for q in gs.players)

    def _can_drop(self, i, near, sabotage) -> bool:
        """With the candidate knight already inactive: is our defence still fine?"""
        gs = self.gs
        p = gs.players[i]
        if not near or Brain.defence_gap(gs, p) <= 0:
            return True
        if not sabotage:
            return False
        cities = sum(len(q.cities) for q in gs.players)
        strength = {q.name: gs.defence_strength(q) for q in gs.players}
        if sum(strength.values()) >= cities:
            return True                         # held anyway: nobody loses a city
        exposed = [q for q in gs.players if any(c not in q.metropolises for c in q.cities)]
        low = min(strength[q.name] for q in exposed)
        if strength[p.name] <= low:
            return False                        # we would lose a city too
        # selector on: only when the right rival loses it (BLOCK: the focus; RACE: not the last-placed)
        from .situational import sabotage_ok
        return sabotage_ok(self.brains[i], gs, {q.name for q in exposed if strength[q.name] == low and q is not p})

    def _can_bounce(self, i, near, done) -> bool:
        """Owner: bounce round the table with grain to reactivate, a Warlord, or no city at risk."""
        p = self.gs.players[i]
        if not near:
            return True
        if not any(c not in p.metropolises for c in p.cities):
            return True
        return p.hand.get("grain", 0) > done or "warlord" in p.progress_hand

    def _bounce_hexes(self, i, chaser):
        """Robber spots next to another of our ready knights that touch a rival with cards."""
        gs, b = self.gs, self.gs.board
        p = gs.players[i]
        safe = gs.robber_protected() if self.mode.friendly_robber else set()
        ready = [v for v, k in p.knights.items() if v != chaser and k.active and not k.acted
                 and not (self.mode.canal and any(x.terrain == "gold" for x in b.vertex_hexes(v)))]
        out = set()
        for v in ready:
            for x in b.vertex_hexes(v):
                if x.coord == b.robber or not x.is_land or x.terrain == "desert":
                    continue
                for u in b.vertices:
                    if x in b.vertex_hexes(u):
                        o = gs.owner_at(u)
                        if o is not None and o is not p and o.name not in safe and o.total_cards() > 0:
                            out.add(x.coord)
        return out

    def _check_disrupted(self, mover):
        """Did the mover just take (or block) another seat's next settlement spot?"""
        for j, b in enumerate(self.brains):
            o = b.outlook
            if j == mover or j in self.disrupted or not o or not o.goals or o.goals[0].kind != "settlement":
                continue
            x = o.goals[0].target
            if x is not None and not self.gs.can_settle(x):
                self.disrupted[j] = self.gs.turn
                self.disruptions += 1
                b.outlook = None                # it notices on its next turn

    def run(self, max_turns=400, learn=False, lr=0.05, share=0.5):
        self.setup()
        n = len(self.gs.players)
        winner = None
        for t in range(max_turns):
            self.gs.turn = t
            i = t % n
            self.turn(i)
            if t % n == 0:
                self.snapshot()
            p = self.gs.players[i]
            if self.total_vp(p) >= self.gs.target:
                winner = p.name
                break
        if self.record_setup:
            self.outcome_winner = winner
        result = None
        if learn and winner:
            result = self.brains[self.log_player].model.learn(self.log, winner, lr=lr, share=share)
        return winner, self.gs.turn, result


    def setup_rows(self, extra: dict | None = None) -> list[dict]:
        """One row per seat: the opening features (setup_features) joined with the outcome.
        Call after run() on a Sim made with record_setup=True."""
        ps = self.gs.players
        n = len(ps)
        vps = [self.total_vp(p) for p in ps]
        winner = getattr(self, "outcome_winner", None)
        rows = []
        for i, p in enumerate(ps):
            better = sum(v > vps[i] for v in vps)
            ties = sum(v == vps[i] for v in vps)
            rank = better + (ties + 1) / 2              # 1 = best, ties share the average rank
            row = {"source": "sim", "mode": self.mode_key, "seed": self.seed, "n_players": n,
                   "seat": i, "player": p.name, **self.setup_feats[i],
                   "outcome": {"final_vp": vps[i], "rank": rank,
                               "rank_norm": round((n - rank) / (n - 1), 4) if n > 1 else 1.0,
                               "won": winner == p.name, "no_winner": winner is None,
                               "turns": self.gs.turn, "chips": p.canal_chips,
                               "lost_city_first_attack": any(a == 0 and nm == p.name for a, nm in self.lost)}}
            if extra:
                row.update(extra)
            rows.append(row)
        return rows


def setup_log_path(day: str | None = None) -> Path:
    day = day or datetime.date.today().strftime("%Y%m%d")
    return Path(__file__).resolve().parent.parent / "data" / "setup_log" / f"sim_{day}.jsonl"


def write_setup_rows(rows: list[dict], path: Path | str | None = None) -> Path:
    path = Path(path) if path else setup_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    return path


def setup_features(gs: GameState, seat: int, brain: Brain, moves: list | None = None) -> dict:
    """Features of one seat's opening, read right after setup (gs seen from that seat: is_me).
    pips_by_res uses the brain's production-weighted pips (canal fields count 0.15); pips_raw is
    plain number pips of every hex under our pieces (a city counted once)."""
    b = gs.board
    me = gs.players[seat]
    blds = me.buildings()
    eff = brain._owned_pips(gs, me)
    raw = sum(PIPS[h.number] for v, _ in blds for h in b.vertex_hexes(v) if h.number)
    harb = [b.harbor_at(v) for v, _ in blds if b.harbor_at(v)]
    moves = moves or []
    last = moves[-1][1] if moves else (blds[-1][0] if blds else None)
    routes = [r for v, _ in blds if (r := brain.gold_route_len(gs, v)) is not None]
    gold_near = 0
    for v, _ in blds:
        ring1 = b.neighbors(v)
        near = ring1 | {w for u in ring1 for w in b.neighbors(u)}
        gold_near = max([gold_near] + [PIPS[h.number] for w in near for h in b.vertex_hexes(w)
                                        if h.terrain == "gold" and h.number])
    dup = 0.0
    vs = [v for v, _ in blds]
    for x in range(len(vs)):
        for y in range(x + 1, len(vs)):
            na = {h.number for h in b.vertex_hexes(vs[x]) if h.number}
            nb = {h.number for h in b.vertex_hexes(vs[y]) if h.number}
            dup += sum(PIPS[n] for n in na & nb)
    start = {r: c for r, c in me.hand.items() if c}
    res_set = {r for r in RESOURCES if eff.get(r, 0) > 0}
    return {
        "pieces": [[k, list(v), [list(x) for x in e], t] for k, v, e, t in moves],
        "weights": {**brain.setup_w, **{k: getattr(brain, k) for k in ("third_harbour", "third_grain", "first_harbour")}},
        "features": {
            "pips_total": round(sum(eff.values()), 3),
            "pips_raw": raw,
            "pips_by_res": {r: round(x, 3) for r, x in sorted(eff.items())},
            "distinct": len(res_set),
            "distinct_commodities": sum(1 for c in COMMODITIES if eff.get(c, 0) > 0),
            "harb_3to1": "3:1" in harb,
            "harb_2to1": sorted(h for h in harb if h != "3:1"),
            "has_canal_hex": any(b.is_canal(h) for v, _ in blds for h in b.vertex_hexes(v)),
            "start_cards": start,
            "start_total": sum(start.values()),
            "start_grain": start.get("grain", 0),
            "grain_hexes_last": brain.fields_hexes(gs, last) if last is not None else 0,
            "gold_route_len": min(routes) if routes else -1,
            "gold_near_pips": gold_near,
            "dup_pips": dup,
            "last_coastal": bool(last is not None and b.is_coastal_vertex(last)),
            "on_gold_corner": any(any(h.terrain == "gold" for h in b.vertex_hexes(v)) for v, _ in blds),
            "seat": seat,
            "n_players": len(gs.players),
        },
    }


def train(games=200, mode="great_canal", n=4, seed=0, lr=0.05, model=None, share=0.5):
    model = model or Model()
    # owner 2026-10-06 "the model must improve with every stage": training never writes a file.
    # learn() saved after every game, so train_loop shards (Model(best.json)) rewrote best.json
    # in parallel even in rejected rounds, and train() with no model rewrote data/model.json.
    # Only a gate (train_loop / live_learn) saves, after a battle.
    path, model.path = model.path, None
    stats = {"games": 0, "no_winner": 0, "turns": 0, "pred": []}
    try:
        for g in range(games):
            sim = Sim(mode, n, seed=seed + g, brains=[Brain(model) for _ in range(n)])
            winner, turns, res = sim.run(learn=True, lr=lr, share=share)
            stats["games"] += 1
            stats["turns"] += turns
            if not winner:
                stats["no_winner"] += 1
            elif res:
                stats["pred"].append(res["winner_predicted_pct"])
    finally:
        model.path = path
    return model, stats
