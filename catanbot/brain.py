"""Decision making. Every choice maximises the bot's win probability under the threat
model and, when it hurts someone, hurts the opponent most likely to win."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

from . import knight_targets as kt
from . import progress_cards as pc
from .board import COMMODITIES, PIPS, RESOURCES, EKey, VKey, ekey, hex_corners
from .learning import Model
from . import moves as mv
from .rules import COSTS, IMPROVEMENT_TRACKS, can_afford, discard_limit, improvement_cost, missing, pay
from .state import GameState, Knight, PlayerState, award_longest_road, dig_canals, grant_metropolis
from .threat import block_target, features, win_probs


@dataclass
class Action:
    kind: str                       # settlement/city/road/ship/dev/knight/activate/promote/wall/improve/canal/bank_trade/end
    target: object = None           # vertex / edge / track
    give: dict = field(default_factory=dict)
    get: dict = field(default_factory=dict)
    note: str = ""


@dataclass
class TurnCtx:
    """What the runner knows about this turn that the board doesn't show."""
    offers_made: set = field(default_factory=set)   # ("want", card) already asked this turn
    refused: dict | None = None                     # our last offer nobody took
    lost_cities: int = 0                            # barbarians took a city; rebuild first
    sheds: int = 0                                  # bank trades made to get under the limit
    late: bool = False                              # turn budget spent: wrap up


# Setup (initial placement) scoring weights. Defaults = the hand-set values the code used before
# 2026-10-03, so with no "setup_weights" in data/config.json nothing changes. scripts/fit_setup.py
# fits them by self-play and only writes the key when a candidate beats this baseline.
SETUP_DEFAULTS: dict[str, float] = {
    # _opening_value
    "gold_value_canal": 2.2,     # value per gold pip in the canal opening (never on a settlement: can_settle)
    "need_none": 2.0,            # x NEED weight: no source at all of a needed resource
    "need_weak": 0.5,            # x NEED weight: only a 2/12 or 3/11 of it
    "distinct": 1.0,             # per distinct base resource
    "no_commodity": 2.0,         # C&K: no commodity hex anywhere
    # _spot_extras
    "harb_3to1_canal": 8.0,      # owner: 3:1 is essential in the canal (4:1 cripples)
    "harb_3to1": 1.5,            # other modes
    "harb_21_base": 1.0,         # 2:1 = base + per_pip x own pips of that resource
    "harb_21_per_pip": 0.4,
    "canal_hex": 3.0,            # corner touches a canal hex (chips + grain unlock)
    "gold_near": 0.5,            # x pips of the best gold corner within 2 steps
    # setup_vertex_score
    "city_extras": 0.3,          # canal city: extras scaled down (harbour goes on the 3rd settlement)
    "city_pips": 0.25,           # canal city: + x own pips
    "dup": 0.15,                 # x pips shared between two openings (same number)
    "three_roll": 6.0,           # x P(first build within 3 rolls)
    "lookahead_coast": 0.3,      # city pick: x best free coastal 3rd spot
    # _start_hand_value
    "start_hand": 1.2,           # overall scale of the start-card value
    "start_grain_backup": 5.0,   # canal start grain when no 3:1 here (or strong city spot)
    "start_grain_with3to1": 1.0, # canal start grain on a 3:1 corner
    "start_grain2": 0.5,         # extra for a second start grain
    # owner-rule terms (0 = off, the old behaviour)
    "third_harbour": 0.0,        # canal 3rd settlement on a 3:1
    "third_grain": 0.0,          # canal 3rd settlement, per fields hex (max 2)
    "first_harbour": 0.0,        # owner 2026-10-03: first settlement on a harbour (owner prior 4.0)
    "grain2": 0.0,               # canal settlement on two fields hexes (owner: "could have chosen 2 grain")
    "gold_route": 0.0,           # canal coastal corner (>= gold_route_min_pips) with a short ship path to a free gold corner
    "deny_3to1": 0.0,            # canal: taking the last free 3:1 corner while an opponent still has to place
    # setup as a set (Brain.setup_set, owner 2026-10-05 game 14; canal only)
    "set_miss": 6.0,             # a core resource (lumber/brick/wool/ore) nobody of ours makes
    "set_weak": 2.0,             # ... made only by a 2/12 or 3/11 (under 3 whole-game pips)
    "set_city_ws": 1.0,          # x city pips on forest + pasture (wood, sheep, paper, cloth)
    "set_city_other": 0.5,       # city pips off forest/pasture count this part (the city is for wood + sheep)
    "set_city_gold": 1.1,        # x pips of the best gold hex within 2 steps / a 3-edge ship path of the city: a knight there
                                 # takes any card on every 8 all game (gold_value_canal 2.2 x 0.5 for the knight)
    "set_canal": 1.5,            # building touches a canal chip hex (knights dig there for VP)
    "set_harb": 8.0,             # last settlement on a 3:1 harbour
    "set_harb_covered": 6.0,     # ... more when the core set is already covered (owner: then just a harbour)
    "set_lookahead_harb": 2.0,
    "set_first_harb": 8.0,       # owner 2026-10-06: first settlement on a 3:1 with good numbers (8/9), x min(1, pips/9)
    # owner 2026-10-09: settle on corners that pay from 3 hexes, then 2, rarely 1 (producing_hexes);
    # one pip ~ one point here, so a 3-hex corner beats a 2-hex one unless that is ~3+ pips better
    "hex_count_3": 3.0,
    "hex_count_2": 0.0,
    "hex_count_1": -4.0,         # a must-have (the 3:1 we lack, 2-grain corner, gold route) counts as 3
    "hex_count_0": -8.0,
    "set_first_harb_min": 6.0,   # ... only from this many whole-game pips on the corner   # 2nd pick: a free 3:1 coastal corner is left for the last settlement
}
# not learned: shape parameters of the terms above
SETUP_SHAPE = {"gold_route_max_edges": 3, "gold_route_min_pips": 5}
# owner prior for terms the code had off (owner-trade-partner-and-city / step5 audit 4D.8)
SETUP_OWNER_PRIOR = {"first_harbour": 4.0}
_SETUP_ATTRS = ("third_harbour", "third_grain", "first_harbour")   # also Brain attributes (selfplay variants)


def load_setup_weights() -> dict[str, float]:
    """data/config.json "setup_weights" over SETUP_DEFAULTS (unknown keys ignored)."""
    w = dict(SETUP_DEFAULTS)
    try:
        from .config import load
        cfg = load().get("setup_weights") or {}
    except Exception:
        cfg = {}
    for k, x in cfg.items():
        if k in w:
            w[k] = float(x)
    return w


def producing_hexes(gs: GameState, v: VKey, kind: str = "settlement") -> int:
    """Owner 2026-10-09: how many hexes at corner v really pay this building -- a number, no
    robber, no fog. Great Canal: gold pays only knights, and the farmland lies face down (pays
    nothing) while the canal is closed."""
    from .rules import production
    b = gs.board
    canal_shut = gs.mode.canal and gs.canal_progress < gs.canal_total
    n = 0
    for h in b.vertex_hexes(v):
        if not h.number or h.coord == b.robber or h.fog:
            continue
        if gs.mode.canal and h.terrain == "gold":
            continue
        if canal_shut and (h.barren or h.terrain == "fields"):
            continue
        if production(h.terrain, kind, gs.mode):
            n += 1
    return n


def hex_count_term(gs: GameState, v: VKey, kind: str, weights: dict, must: bool = False) -> float:
    """setup_w hex_count_<n> for corner v (n capped at 3). A must-have corner (the canal 3:1 we
    lack, the 2-grain corner, the gold route) stands level with a 3-hex corner: the owner's
    harbour and grain/gold rules decide there, and pips still order the must-haves."""
    n = 3 if must else producing_hexes(gs, v, kind)
    return weights.get(f"hex_count_{min(3, n)}", 0.0)


def set_flag(brain, name: str, value) -> None:
    """Review 0.3.1: an A/B or self-play flag must name a real Brain attribute -- a typo would
    otherwise add a new attribute and compare two identical policies."""
    if not hasattr(brain, name):
        raise AttributeError(f"Brain has no flag {name!r}")
    setattr(brain, name, value)


class Brain:
    def __init__(self, model: Model | None = None, setup_weights: dict | None = None):
        """setup_weights: None = read data/config.json "setup_weights"; a dict = SETUP_DEFAULTS
        updated with it (config not read; {} = pure defaults)."""
        self.model = model or Model()
        if setup_weights is None:
            self.setup_w = load_setup_weights()
        else:
            self.setup_w = dict(SETUP_DEFAULTS)
            self.setup_w.update({k: float(x) for k, x in setup_weights.items() if k in SETUP_DEFAULTS})
        self.fleet_resource: str | None = None    # Merchant Fleet: 2:1 on this card for this turn
        self.merchant_res: str | None = None      # our Merchant's hex resource: 2:1 while it stands
        self.keep: set[str] = set()               # cards we're saving (e.g. grain/ore to rebuild a lost city)
        self.outlook = None                       # outlook.Outlook: our next 3 goals + opponents' threats
        self.harbor_3to1 = False                  # owner-confirmed 3:1 harbour (config)
        self.ratio_seen: dict = {}                # live: ratios read off the bank dialog (the truth)
        self.spend_turn = True                    # owner 2026-10-03: bank spare cards toward the target
        self.spend_need_cap = 2                   # ...only when the target is at most this many cards short
        self.note = lambda msg: None              # runner hooks its log in here
        # canal 3rd (coastal) settlement: extra weight for a 3:1 harbour / a start grain
        # (tuned by self-play, scripts/selfplay.py)
        # (values from setup_weights; kept as attributes so selfplay variants can setattr them)
        self.third_harbour = self.setup_w["third_harbour"]
        self.third_grain = self.setup_w["third_grain"]
        # owner 2026-10-03: "pick a harbor with the first settlement, especially if good numbers
        # or good resources, and then focus on getting the rest" (0 = off)
        self.first_harbour = self.setup_w["first_harbour"]
        # defence trades: grain for an idle knight before a new knight; pay 2-for-1 when the
        # barbarians are this close (fraction of the track), 0 = never
        self.activate_first = True
        self.pay_two_danger = 0.5
        # 2026-10-03 self-play: 4 in 10 seats lost a city in the 1st attack with idle knights and no
        # grain; cards went to improvements / 2nd knights instead of a bank set for grain (0 = off)
        self.grain_reserve = 0.0
        # owner 2026-10-05 game 15: the city fell with 0 active knights -- the bot built a 2nd
        # knight and promoted an idle one (wool+ore gone) instead of buying grain to activate
        self.idle_knight_first = True
        # owner 2026-10-08: the city fell after ships and roads ate the cards with no knight up --
        # while a city has no active knight, the knight (and its grain) comes first: hold wool, ore
        # and grain, no roads/ships/settlements; a wall only when the hand nears the 7 limit and
        # the knight can't be made this turn
        self.knight_first = True
        # stage 2026-10-09 game 050606: 0 settlements in 89 turns -- after every attack all knights
        # sit idle, the knight-first hold blocked every settlement and each grain went to an
        # activation. With the ship far (3+ steps from a city in danger) a settlement we can reach
        # takes the grain first; the knight's turn comes when the ship nears.
        self.settle_when_far = True
        # stage 2026-10-09 07:32:59: trade metropolis lost at level 4 after the cloth went to the
        # bank (07:22, 07:31) and a harbour (07:25) -- the commodity of a track whose metropolis we
        # hold at 4 is never given away until level 5
        self.lock_metropolis_commodity = True
        self.robber_steal_first = True          # live 2026-10-09 19:44: robber on an 8 with no victim
        self.discard_keep_improve = True        # live 2026-10-09 19:44: 3 paper + 3 wood (paper 4) thrown away
        # stage 2026-10-09 07:22: Wedding held all game while we led -- leading, it is the first discard
        self.wedding_dead_when_leading = True
        # stage 2026-10-09 game 0834: after a promote was skipped the knight build behind it (no basic
        # piece left without the promotion) was tapped 3 times -- plan items that a skipped item
        # made legal leave the plan (runner._run_builds, sim)
        self.replan_after_skip = True
        # stage 2026-10-09 game 0959: Jean climbed 4 -> 11 while the threat stayed the leader. Each
        # VP a rival gained over the last threat_rise_turns of our turns adds this to its threat
        # score (0 = off; A/B scripts/ab_policy.py threat_rise)
        self.threat_rise_w = 0.08
        self.threat_rise_turns = 6
        # owner 2026-10-03 opening: 1 wool + 1 ore + 3 of a kind banked for grain; bank the set as
        # soon as it's there (wait a round for wool/ore) and keep that grain for the knight
        self.wall_late = False        # owner 2026-10-04: walls before the 1st attack too (was True: bricks to the knight first)
        # owner 2026-10-05 game 14: Great Canal setup chosen as a SET -- whole-game yield per hex,
        # city on the strongest wood+sheep corner, settlements cover what the others lack, the
        # last (coastal) settlement on a 3:1 once covered. False = the old per-spot scorer.
        self.setup_set = True
        self.grain_first = True       # A/B 2026-10-03 (80 games): lost city in 1st attack 29.4% vs 35.0%, wins 40/40
        # owner 2026-10-03: too many cards lost to the robber -> 2 brick for a wall (0 = as before)
        self.wall_boost = 2.0         # A/B 2026-10-03 with wall_late (80 games): wins 41/39, VP 14.20 vs 14.03, walls 1.89 vs 1.62
        # C&K knight moves (moves.knight_moves): the sim turns this on; the live runner has no
        # click flow for a move yet, so live plans never contain one
        self.knight_moves = False
        # Seafarers ship move (moves.ship_moves): the sim turns this on; live it follows the
        # config switch "ship_move" (runner.tap_move_ship, flow UNVERIFIED 2026-10-09)
        self.ship_moves = False
        # owner 2026-10-09 "I offered a lot of resources to marlow, it should have accepted when
        # more than 3:1 and the offering party is not more than 5 points from winning"
        self.generous_accept = True
        # owner 2026-10-09 21:08 "marlow should build settlements now, still not doing that even
        # with 25 resources in hand": the greedy plan let improvements, promotions and their bank
        # trades eat the settlement's wool/grain every step. With a big hand (over the discard
        # limit, or city + road + settlement paid outright) the settlement on the best open spot
        # goes in once the city/knight are done, or a road toward a good spot (_road_target, 7+
        # pips, no rival road) at most settle_big_hand_reach roads away when the hand pays the
        # roads and the settlement. False = the old greedy order.
        self.settle_big_hand = True
        self.settle_big_hand_reach = 2
        # owner 2026-10-09 "even if the canal is done, marlow should continue to build roads and
        # settlements and then cities": with the canal dug its road bonuses stop and a lone road
        # (+0.06) lost to a wall. A road that opens a good spot the same hand settles is planned
        # as one road + settlement move. False = one-step greedy only.
        self.road_settle_lookahead = True
        # owner 2026-10-09 "when deleting a knight of another player, go for the best position if
        # needed or the largest one otherwise": a displacement onto a rival knight whose spot matters
        # (knight_targets.knight_spot_value) is allowed with displace off, and ranks first; else
        # the strongest knight we can push ranks first (False = as before)
        self.knight_target_by_spot = True
        # owner 2026-10-09: "knights can move along the road or ships, also knights can move the
        # robber if active and get deactivated". knight_move_ships: knight move paths run over our
        # ships as well as roads (False = roads only). knight_chase: an active knight on a corner
        # of the robber's hex chases it after the first attack (chase_pick), then goes inactive.
        self.knight_move_ships = True
        self.knight_chase = True
        # owner 2026-10-04 (docs/analysis/sim_three_rounds_2026-10-04.md):
        self.set_for_grain = True     # 1. bank 3:1 into grain; with 2 of a kind, trade for the 3rd
        # solo 2026-10-09 22:58:51: idle knight, no grain, ship at 3/7 -- 1 of 2 wool (3:1 harbour)
        # went to a player for ore toward a 2nd knight; the next roll's wool made 2, not 3, and the
        # only city fell at 23:00:40. True: while an idle knight waits for grain, player offers
        # never give a card of a 3:1 set or pair (grain_set_cards; config grain_set_keep)
        self.keep_grain_set = True
        self.knight_always = 0.01     # 2. always build a knight: grain comes later from 3:1 trades
        self.defender_vp = 0.04       # 3. play for the defender VP (sole top strength), more late in the game
        self.spend_grain = 0.03       # 4. never keep grain after the roll: everyone wants it
        self.gold_route = 0.03        # roads/ships toward a free gold corner (the gold knight = grain)
        # stage 2026-10-09 19:15:31: a road went "up towards grain" to a 2-fields corner whose farmland
        # pays nothing while the canal is shut; the 2-grain must-have is a setup rule, not a road goal
        self.road_skip_farmland = True
        self.limit_lookahead = True   # spend when the cards due before our next turn would pass the limit
        # owner 2026-10-04 (2nd message): a wall (2 brick) on a city cuts 7-losses further; more cities
        # = more wall slots; spending cards off-plan (walls, improvements, progress) saves them from
        # 7s and stealing. 0 = as before.
        self.city_wall_slot = 0.01    # a city that opens a wall slot (walls only stand on cities, max 3)
        self.spend_exposed = 0.004    # per card above 4 after the roll: any build may cost this much
        # owner 2026-10-04 (3rd message): every development track to 3, then 4 and 5. Level 5 locks
        # the metropolis (nobody can take it); at 4 a rival reaching 5 still takes it -- both ways.
        self.improve_ladder = 0.02    # 0 = as before
        # owner 2026-10-09 "for the third metropolis, marlow needs a third city": a city that
        # gives a level-3+ track (whose metropolis we lack) a city to stand on; 0 = off
        self.city_metro_slot = 0.04
        # owner 2026-10-10 "the bot has to keep trying to build cities throughout the game, this is
        # still one of the goals" -- solo 20261009_225719 lost its only city at 23:00:40 and sat at
        # cities=0 for 40 minutes: every grain it banked went to a canal activation ("spend the
        # turn: bank 3 wool -> grain toward the activate") and the ore to promotions.
        # city_always: with a settlement to upgrade and the city cap not hit, the city is one of
        # the outlook goals, its ore/grain are kept from offers, and "spend the turn" banks toward
        # it one card further (config city_always_goal).
        self.city_always = True
        # city_rebuild_w: no city (lost to the barbarians): the rebuild is goal 1, every ore/grain
        # the city needs that a plan spends elsewhere (activation, promotion, a bank trade) costs
        # this much; "spend the turn" banks toward it however far off (0 = off; config city_rebuild_first)
        self.city_rebuild_w = 0.05
        self.lost_cities = 0                   # runner/sim: cities lost and not yet rebuilt (update_outlook)
        # owner 2026-10-03 opening (owner-opening-grain-walls): "holding 3 of a kind and no grain ->
        # bank it for grain now, even without wool/ore yet". Solo 1009 turn 1 (22:58:26): wool 3 +
        # ore 1 at a 3:1, the set was kept as the knight's wool and the knight came first. True =
        # before the 1st attack, with no knight yet, the set is banked even when it is the knight's
        # card (config opening_set_first)
        self.opening_set_first = True
        # owner 2026-10-04 (4th message): longest road = 2 VP; a settlement or knight on a rival's road
        # breaks it; knights are pushed off by stronger knights or progress cards (Intrigue/Deserter)
        # owner 2026-10-05 game 15: a road on the edge a rival needs to reach a free canal corner
        # stops its knight there (Rhea did it to us)
        self.canal_deny = True
        self.share_when_stuck = True   # owner game 15: shared canal chip with the leader when stuck
        self.road_cuts = True         # knight moves/pushes that cut a rival's road are considered
        # owner 2026-10-04 (sim.chase_robber): steal with knights, bounce the robber round the table
        self.knight_bounce = False
        # owner 2026-10-04: stand a knight down near an attack when only a rival loses a city
        self.barb_sabotage = False
        # owner 2026-10-06: "it could have built a road to the right to try to move the knight of the
        # other bot and contend the points on the right side of the map" -- in the Great Canal, a
        # road up to a weaker rival knight on an undug canal corner, then push it off (0 = off)
        self.canal_contest = 0.015
        self.setup_ship_lookahead = True   # live 2026-10-09 20:09: setup ships head for a spot 4 sea steps out
        self.canal_road_front = 0.05    # owner 2026-10-09: roads along the canal toward rivals, not up to grain
        # owner 2026-10-08: "it missed some of the points in the top right corner and it had a ship
        # there where it could add a knight" -- a canal corner with a second corner in reach is worth
        # more, roads/ships toward free canal corners count, and a knight never moves (switches
        # off) when that leaves a city open to the barbarians (False = as before)
        self.canal_hunt = True
        # owner 2026-10-06: cities before settlements -- city first in the trade targets and this
        # much extra on a city build (0 = old order)
        self.city_first = 0.02
        self.displace = False         # knight moves may push a weaker enemy knight; A/B ab_9700: off lost-first 20.0% vs 25.0%, wins 42/38
        # fleet opus 2026-10-04 (grain/knight economy, default off = the play above):
        # grain_gate -- set_for_grain (bank sets / ask the 3rd card / activate first) and the
        # spend_grain bonus on an activation only fire when that activation does something now:
        # a city at risk, the sole-top defender spot, or a canal dig (_activation_useful)
        self.grain_gate = False
        # overnight 2026-10-04: 27% of seats lose a city at the 1st attack and then win 5%. The
        # owner gap rule (1 active knight per city) misses the RELATIVE rule: the barbarians win
        # when all active strength < all cities, and the weakest defender(s) lose a city. Bonus
        # for a knight action that takes us out of the losing spot (scaled by danger; 0 = off)
        self.barb_relative = 0.0
        # knight_guard -- knight_always only when the new knight can be activated (grain in hand
        # beyond the idle knights) or it parks on gold; otherwise this much is taken off (0 = off)
        self.knight_guard = 0.0
        # owner 2026-10-06: "it built two knights in first round when it should have kept the
        # resources to exchange" -- once every city has a knight, another one that cannot be
        # activated now (no spare grain, not on gold) loses this much: keep wool/ore for trades
        self.spare_knight = 0.05
        # grain_engine -- extra value for science up to 3 (aqueduct) and the first gold knight (0 = off)
        self.grain_engine = 0.0
        # owner 2026-10-04: pick the rules per turn (situational.py): BLOCK the advancing rival or
        # RACE when we lead/are level. Off = play exactly as before.
        self.situational = False
        self.focus: int | None = None      # BLOCK: the rival (index in gs.players) every blocking tool aims at
        self.spare: int | None = None      # RACE: the last-placed rival, robbed only when nobody else can be
        self.stance: str | None = None     # "block" / "race" while the selector is on
        self._sit_saved: dict | None = None
        self._sit_key = None
        # a model file can carry its own play (profiles.py): v1 = the old play, v2 = every learning
        for k, v in getattr(self.model, "brain", {}).items():
            if hasattr(self, k):
                setattr(self, k, v)

    # ---- plan 3 moves ahead, re-plan after every move anyone makes (owner 2026-10-02) ----
    def update_outlook(self, gs: GameState, lost_cities: int = 0, rank: bool = False) -> bool:
        """Rebuild the outlook when the board changed, set the keep-cards (and with rank=True
        the biggest threat). Shared by the live runner and the self-play sim.
        Returns True when the outlook was rebuilt."""
        from . import outlook
        me = gs.me()
        sig = outlook.signature(gs)
        rebuilt = False
        if rank or sig != getattr(self, "_outlook_sig", None) or self.outlook is None:
            self.outlook = outlook.build(gs, self.probs(gs))
            self._outlook_sig = sig
            rebuilt = True
        self.lost_cities = int(lost_cities or 0)
        keep = {"grain", "ore"} if lost_cities else set()
        o = self.outlook
        self._pin_city_goal(gs)
        if self.city_rebuild(gs):
            keep |= {"grain", "ore"}
        elif self.city_goal(gs) is not None:
            # owner 2026-10-10: the city stays a goal -- its ore/grain aren't offered (spares are)
            keep |= {k for k, v in COSTS["city"].items() if 0 < me.hand.get(k, 0) <= v}
        if o is not None and o.goals:
            # don't trade away a card our next goal still needs (we hold no spare of it)
            keep |= {k for k, v in o.goals[0].cost.items() if 0 < me.hand.get(k, 0) <= v}
        self.keep = keep
        if rank:
            rise = self._vp_rise(gs) if self.threat_rise_w else None
            ranked = outlook.rank_threats(gs, o, self.probs(gs), rise, self.threat_rise_w)
            if ranked:
                self.threat_name = ranked[0][0]
            return ranked
        return rebuilt

    # ---- owner 2026-10-10: a city is a live goal all game ------------------------------------
    @staticmethod
    def _city_spot(gs: GameState):
        """Our best producing settlement (the next city goes there), or None at the city cap."""
        me = gs.me()
        if not gs.mode.ck or not me.settlements or len(me.cities) >= gs.mode.max_cities:
            return None
        b = gs.board
        return max(sorted(me.settlements), key=lambda v: (b.vertex_pips(v), v))

    def city_goal(self, gs: GameState):
        """city_always: the settlement the next city goes on, else None."""
        return self._city_spot(gs) if self.city_always else None

    def city_rebuild(self, gs: GameState) -> bool:
        """city_rebuild_w: we hold no city (the barbarians took it) or a lost city isn't rebuilt
        yet, and a settlement can take it back."""
        if not self.city_rebuild_w or self._city_spot(gs) is None:
            return False
        return bool(self.lost_cities) or not gs.me().cities

    def _pin_city_goal(self, gs: GameState) -> None:
        """The city is always one of the outlook goals (the first while it is a rebuild). Its
        rounds look long in the Great Canal (no grain hex), so the cheapest-first sort dropped it
        and the log showed only settlement / canal_chip / defender goals after the loss."""
        o = self.outlook
        v = self._city_spot(gs) if (self.city_always or self.city_rebuild(gs)) else None
        if o is None or v is None:
            return
        from .outlook import VPMove, _rounds
        me = gs.me()
        cur = next((g for g in o.goals if g.kind == "city"), None)
        mv_ = cur or VPMove(me.name, "city", 1, _rounds(gs, me, COSTS["city"]), v, dict(COSTS["city"]))
        if self.city_rebuild(gs):
            if o.goals and o.goals[0] is mv_:
                return
            o.goals = [mv_] + [g for g in o.goals if g is not mv_][:2]
            o.keep = {k for k, n in mv_.cost.items() if n > 0}
        elif cur is None:
            o.goals = o.goals[:2] + [mv_]

    def _city_dip(self, gs: GameState, seq: list[Action]) -> int:
        """While the city is a rebuild: ore/grain the city needs that `seq` spends elsewhere (a
        card banked or traded in and then spent counts too). A seq that builds the city costs
        nothing, and neither does the knight a rebuilt city would have no cover without."""
        if not self.city_rebuild(gs):
            return 0
        me = gs.me()
        hand = dict(me.hand)
        avail = dict(hand)
        for a in seq:
            if a.kind == "city":
                return 0
            if a.kind in ("bank_trade", "player_trade"):
                for r, n in (a.give or {}).items():
                    hand[r] = hand.get(r, 0) - n
                for r, n in (a.get or {}).items():
                    hand[r] = hand.get(r, 0) + n
                    avail[r] = avail.get(r, 0) + n
                continue
            if a.kind == "knight" and len(me.knights) <= len(me.cities):
                continue                      # owner: knights still come, so the next attack can't take it
            for r, n in self._cost(gs, a).items():
                hand[r] = hand.get(r, 0) - n
        return sum(max(0, min(avail.get(r, 0), n) - max(0, hand.get(r, 0)))
                   for r, n in COSTS["city"].items())

    def _city_cost(self, gs: GameState, seq: list[Action]) -> float:
        return self.city_rebuild_w * self._city_dip(gs, seq) if self.city_rebuild_w else 0.0

    def _city_save_trade(self, gs: GameState) -> Action | None:
        """Owner 2026-10-10 "keep trying to build cities": one bank trade of spare cards (never
        the city's ore/grain, never grain in the canal) into a card the rebuild city still misses."""
        me = gs.me()
        hand = dict(me.hand)
        need = missing(hand, COSTS["city"])
        if not need:
            return None
        val = self.resource_value(gs, me)
        best = None
        for give in RESOURCES:
            if give in COSTS["city"] or give in need:
                continue
            ratio = self.bank_ratio(gs, give)
            if hand.get(give, 0) < ratio:
                continue
            for res in need:
                gain = val.get(res, 1.0) - ratio * val.get(give, 1.0) * 0.5
                if best is None or gain > best[0]:
                    best = (gain, give, ratio, res)
        if best is None:
            return None
        _, give, ratio, res = best
        return Action("bank_trade", give={give: ratio}, get={res: 1})

    def set_need_cap(self, gs: GameState, kind: str) -> int:
        """Cards a goal may still be short when we ask a player for the 3rd card of a bank set:
        3, or the whole city while it is a rebuild (the set's grain is the way back)."""
        return sum(COSTS["city"].values()) if kind == "city" and self.city_rebuild(gs) else 3

    def _vp_rise(self, gs: GameState) -> dict[str, int]:
        """VP each rival gained over the last threat_rise_turns rankings (one per own turn, live and
        sim). Noisy reads (0959: 11 -> 9 -> 12): now = the lower of the last two readings, then =
        the higher of the two readings that far back; never below 0."""
        from collections import deque
        n = max(1, int(self.threat_rise_turns))
        hist = self.__dict__.setdefault("_vp_hist", {})
        out = {}
        for p in gs.opponents():
            h = hist.get(p.name)
            if h is None or h.maxlen != n + 2:
                h = hist[p.name] = deque(h or (), maxlen=n + 2)
            h.append(int(p.vp_visible))
            if len(h) < 2:
                continue
            seq = list(h)
            now = min(seq[-1], seq[-2])
            i = max(0, len(seq) - 1 - n)
            then = max(seq[i], seq[i - 1]) if i > 0 else seq[i]
            out[p.name] = max(0, now - then)
        return out

    # ---- progress cards (C&K) -------------------------------------------------------
    def progress_play(self, gs: GameState, phase: str, hand_cards: list[str] | None = None):
        """Next progress card to play now as (card_key, args), or None. Call again after each
        card resolves: the hand and board change. Hand defaults to gs.me().progress_hand.
        Bishop is held until the first barbarian attack (gs.barbarians_attacked)."""
        me = gs.me()
        hand = list(me.progress_hand if hand_cards is None else hand_cards)
        gs.plan_missing = self.plan_missing(gs)
        gs.fleet_resource = self.fleet_resource       # monopoly leans to cards we trade 2:1
        self._seen_into(gs)                           # the cards see the bank dialog's ratios too
        if self.merchant_res and getattr(me, "merchant_hex", None) is None:
            me.merchant_hex = next((h.coord for h in gs.board.hexes.values()
                                    if pc.TERRAIN_RES.get(h.terrain) == self.merchant_res), None)
        why: dict = {}
        plays = pc.choose_play(gs, me, hand, phase, why, self.wedding_dead_when_leading)
        if not gs.barbarians_attacked:
            if any(c == "bishop" for c, _ in plays):
                why["bishop"] = "robber locked until the first barbarian attack"
            plays = [(c, a) for c, a in plays if c != "bishop"]
            if phase == "end_turn" and not plays:
                # bishop was the over-the-limit play: discard the least valuable card instead
                keys = [pc.card_key(c) for c in hand]
                left = [c for c in keys if c not in pc.VP_CARDS and c != "alchemist"]
                if len(left) + (1 if "alchemist" in keys else 0) > pc.PROGRESS_HAND_LIMIT and left:
                    plays = [("discard", {"card": min(left, key=lambda c: pc.keep_value(
                        gs, me, c, self.wedding_dead_when_leading))})]
        if why:                                   # owner 2026-10-06: a held card needs a reason
            self.note(f"progress {phase}: holding " + "; ".join(f"{c}: {r}" for c, r in why.items()))
        return plays[0] if plays else None

    def after_spy(self, gs: GameState, card: str, phase: str = "main"):
        """Spy took `card`: it joins our progress hand and the rest of the turn is re-planned, so
        it is played now when it has a use (the Alchemist waits for our next roll). Returns the
        next play, as progress_play."""
        me = gs.me()
        me.progress_hand = list(me.progress_hand or []) + [card]
        me.progress_cards = len(me.progress_hand)
        nxt = self.progress_play(gs, phase)
        self.note(f"spy: took {card}; re-planned -> " + (f"play {nxt[0]}" if nxt else "nothing to play now"))
        return nxt

    def start_fleet(self, resource: str) -> None:
        """Merchant Fleet was played: bank_ratio gives 2:1 on `resource` until end_fleet()."""
        self.fleet_resource = resource

    def end_fleet(self) -> None:
        self.fleet_resource = None

    # ---- helpers --------------------------------------------------------------
    def w(self, gs: GameState) -> dict:
        return self.model.weights_for(gs.mode.key)

    def my_prob(self, gs: GameState) -> float:
        return win_probs(gs, self.w(gs), self.model.params)[gs.me().name]

    # ---- situational selector (situational.py) ---------------------------------------
    def situational_begin(self, gs: GameState) -> dict | None:
        """Selector on: work out this turn's stance and apply its flag overrides (once per turn;
        the sim calls this before the roll so robber/chase use it, plan_turn calls it again)."""
        if not self.situational:
            return None
        key = (gs.turn, gs.me().name)
        if self._sit_key == key and self._sit_saved is not None:
            return None
        from . import situational as st
        self.situational_end()
        base = {k: getattr(self, k) for k in st.FLAG_KEYS}
        o = st.choose(st.situation(gs, self), base)
        self._sit_saved = base
        self._sit_key = key
        for k, v in o.items():
            if k in st.FLAG_KEYS or k in st.TARGET_KEYS:
                setattr(self, k, v)
        self.note(f"situational: {self.stance} focus={self.focus} spare={self.spare}")
        return o

    def situational_end(self) -> None:
        """Restore the flags overridden for this turn. focus/spare/stance stay until the next
        turn's selector run (they steer our answers and robber choices on rivals' turns)."""
        if self._sit_saved is not None:
            for k, v in self._sit_saved.items():
                setattr(self, k, v)
            self._sit_saved = None

    def _focus_player(self, gs: GameState) -> PlayerState | None:
        if not self.situational or self.focus is None or not (0 <= self.focus < len(gs.players)):
            return None
        p = gs.players[self.focus]
        return None if p.is_me else p

    def threat(self, gs: GameState) -> PlayerState | None:
        f = self._focus_player(gs)
        if f is not None:
            return f                          # BLOCK: the advancing rival
        # runner sets this just before each of our turns from the outlook ranking
        name = getattr(self, "threat_name", None)
        p = next((q for q in gs.opponents() if q.name == name), None) if name else None
        if p is not None:
            return p
        return block_target(gs, self.w(gs), self.model.params)

    def probs(self, gs: GameState) -> dict[str, float]:
        return win_probs(gs, self.w(gs), self.model.params)

    def all_features(self, gs: GameState) -> dict[str, dict]:
        return {p.name: features(gs, p, self.model.params) for p in gs.players}

    @staticmethod
    def clone(gs: GameState) -> GameState:
        g = copy.copy(gs)
        g.players = copy.deepcopy(gs.players)
        g.canal_dug = set(gs.canal_dug)
        g.metropolis_owner = dict(gs.metropolis_owner)
        return g

    def resource_value(self, gs: GameState, me: PlayerState) -> dict[str, float]:
        """How much the bot wants one more card of each kind right now (scarcity + need)."""
        board_pips = {r: 0.0 for r in RESOURCES}
        from .board import TERRAIN_RES
        for h in gs.board.hexes.values():
            r = TERRAIN_RES.get(h.terrain)
            if r in board_pips and h.number:
                board_pips[r] += PIPS[h.number]
        total = sum(board_pips.values()) or 1
        inc = gs.income(me)
        val = {}
        for r in RESOURCES:
            scarcity = 1.0 + max(0.0, 0.2 - board_pips[r] / total) * 3
            own = 1.0 / (1.0 + 6 * inc.get(r, 0))
            val[r] = scarcity * (0.6 + own)
        if gs.mode.ck:
            val["ore"] *= 1.25
            val["grain"] *= 1.15
            for c in COMMODITIES:
                val[c] = 1.6
        if gs.mode.canal and gs.canal_progress < gs.canal_total:
            val["grain"] *= 1.5   # grain is locked, so it's rare
        if gs.mode.ck and (self.defence_gap(gs, me) > 0 or len(me.knights) < len(me.cities)):
            val["wool"] *= 1.3    # playbook: a knight is owed -> wool/ore are the cards to ask for
            val["ore"] *= 1.1
        return val

    # ---- setup ------------------------------------------------------------------
    # Missing one of these after setup means trading for it every turn. Lumber+brick
    # build roads/settlements, so without them the bot can't expand.
    NEED = {"lumber": 3.0, "brick": 2.5, "grain": 1.5, "ore": 1.5, "wool": 1.0}
    # C&K commodities (city on forest/pasture/mountains). Science (paper) gives the
    # aqueduct and progress cards that convert to VP; politics (coin) next.
    # Order paper > coin > cloth is the owner's call: blue (politics) cards steal from
    # other players. Online sources rank cloth over coin (docs/strategy_base_sf_ck.md).
    COMMODITY_W = {"paper": 1.6, "coin": 1.3, "cloth": 0.9}

    def _need(self, gs: GameState) -> dict[str, float]:
        need = dict(self.NEED)
        if gs.mode.ck:
            need["ore"] = need["grain"] = 2.0     # knights + cities (strategy doc)
        if gs.mode.seafarers:
            need["lumber"] = 3.4; need["wool"] = 1.6   # ships
        if gs.mode.canal:
            need.pop("grain", None)               # grain is locked early anyway
        return need

    def _setup_kind(self, gs: GameState, me: PlayerState) -> str:
        placed = len(me.settlements) + len(me.cities)
        # C&K: settlement then city; Great Canal adds a third (coastal) settlement
        return "city" if placed == 1 and gs.mode.start_with_city else "settlement"

    def _start_hand_value(self, gs: GameState, me: PlayerState, v: VKey) -> float:
        """Owner 2026-10-02: in the canal game the LAST settlement pays the starting
        resources (the city pays nothing), so two grain hexes under it = two grain to start.
        Grain is the scarce one there, so it counts triple."""
        val = self.resource_value(gs, me)
        s, grain = 0.0, 0
        from .rules import production
        for h in gs.board.vertex_hexes(v):
            if h.terrain in ("sea", "desert", "canal") or not h.number:
                continue
            for r, n in production(h.terrain, "settlement", gs.mode).items():
                if r in RESOURCES and r != "grain":
                    s += n * val.get(r, 1.0)
            if h.terrain == "fields":
                grain += 1          # grain even from barren farmland at the start
        # owner 2026-10-02: start grain is the BACK-UP, not the aim. It matters when this spot
        # gets no 3:1 harbour, or when our settlement is a strong city spot (good numbers on
        # the right hexes) that wants a quick city. A second grain is a last resort.
        if gs.mode.canal and grain:
            harbour = gs.board.harbor_at(v) == "3:1"
            city_spot = any(sum(PIPS[h.number] for h in gs.board.vertex_hexes(w)
                                if h.number and h.terrain in ("mountains", "forest", "pasture", "hills")) >= 9
                            for w in me.settlements)
            w = self.setup_w
            s += (w["start_grain_backup"] if (not harbour or city_spot) else w["start_grain_with3to1"]) \
                + (w["start_grain2"] if grain >= 2 else 0.0)
        else:
            s += 3.0 * grain * val.get("grain", 1.0)
        return self.setup_w["start_hand"] * s

    def _spot_pips(self, gs: GameState, v: VKey, kind: str) -> dict[str, float]:
        from .rules import production
        b = gs.board
        out: dict[str, float] = {}
        for h in b.vertex_hexes(v):
            if not h.number or h.coord == b.robber or h.fog:
                continue
            # owner 2026-10-02: canal farmland is turned face down after setup and pays nothing
            # until the canal is open -- a long time; count it as a small late-game hope only
            w = 0.15 if (gs.mode.canal and (h.barren or h.terrain == "fields")
                         and gs.canal_progress < gs.canal_total) else 1.0
            for r, n in production(h.terrain, kind, gs.mode).items():
                out[r] = out.get(r, 0.0) + w * n * PIPS[h.number]
        return out

    def _opening_value(self, gs: GameState, me: PlayerState, pips: dict[str, float]) -> float:
        val = self.resource_value(gs, me)
        sw = self.setup_w
        s = 0.0
        for r, x in pips.items():
            if r in COMMODITIES:
                s += x * self.COMMODITY_W.get(r, 1.0)
            else:
                s += x * ((sw["gold_value_canal"] if gs.mode.canal else 1.4) if r == "gold" else val.get(r, 1.0))
        wild = pips.get("gold", 0) > 0
        for r, w in self._need(gs).items():
            have = pips.get(r, 0.0)
            if have <= 0 and not wild:
                s -= sw["need_none"] * w    # no source at all
            elif have < 3:
                s -= sw["need_weak"] * w    # only a 2/12 or 3/11
        s += sw["distinct"] * sum(1 for r in RESOURCES if pips.get(r, 0) > 0)
        if gs.mode.ck and not any(pips.get(c, 0) for c in COMMODITIES):
            s -= sw["no_commodity"]         # no city improvements possible
        return s

    def _first_harbour_bonus(self, gs: GameState, v: VKey, pips: dict[str, float]) -> float:
        """Owner rule: the first settlement on a harbour, more so on good numbers; a 2:1 counts
        when this spot itself makes that resource well (4+ pips)."""
        hb = gs.board.harbor_at(v)
        if not self.first_harbour or not hb or hb == "grain" and gs.mode.canal:
            return 0.0
        if hb != "3:1" and pips.get(hb, 0) < 4:
            return 0.0
        return self.first_harbour * (0.5 + 0.5 * min(1.0, sum(pips.values()) / 10))

    def _has_3to1(self, gs: GameState, me: PlayerState) -> bool:
        return any(gs.board.harbor_at(w) == "3:1" for w, _ in me.buildings())

    def _spot_extras(self, gs: GameState, v: VKey, pips: dict[str, float]) -> float:
        b = gs.board
        sw = self.setup_w
        s = 0.0
        hb = b.harbor_at(v)
        if hb == "3:1" and self.first_harbour and self._has_3to1(gs, gs.me()):
            hb = None           # we already trade 3:1: get the rest (grain, numbers) instead
        if hb:
            port = sw["harb_3to1"] if hb == "3:1" else sw["harb_21_base"] + sw["harb_21_per_pip"] * pips.get(hb, 0)
            if gs.mode.canal:
                # owner 2026-10-02: no grain in the canal game -> 4:1 is crippling, a 3:1 is key
                port = sw["harb_3to1_canal"] if hb == "3:1" else (0.0 if hb == "grain" else port)   # owner: rarely playable without 3:1
                s += port
            else:
                s += port * (0.6 if gs.mode.ck else 1.0)   # ports matter less in C&K
        if gs.mode.canal and any(b.is_canal(h) for h in b.vertex_hexes(v)):
            s += sw["canal_hex"]   # canal access = canal-chip VP and the grain unlock
        if gs.mode.canal and not pips.get("gold"):
            # owner 2026-10-02 game 4: the city "could have been lower to give access to the gold
            # mine" -- a gold corner one or two steps away is a later settlement/knight spot (grain)
            ring1 = b.neighbors(v)
            near = ring1 | {w for u in ring1 for w in b.neighbors(u)}
            gold = max((PIPS[h.number] for w in near for h in b.vertex_hexes(w)
                        if h.terrain == "gold" and h.number), default=0)
            s += sw["gold_near"] * gold        # an 8 (5 pips) = +2.5
        return s

    def _city_bonus(self, gs: GameState, v: VKey) -> float:
        """C&K: only forest/pasture/mountains under a city give commodities (the
        development cards / improvements), so the city spot is judged on them."""
        if not gs.mode.ck:
            return 0.0
        from .board import TERRAIN_COMMODITY
        b = gs.board
        kinds = [TERRAIN_COMMODITY[h.terrain] for h in b.vertex_hexes(v)
                 if h.terrain in TERRAIN_COMMODITY and h.number and h.coord != b.robber]
        s = 1.5 * len(kinds) + 1.0 * len(set(kinds))
        if "paper" in kinds:
            s += 2.0
        if not kinds:
            s -= 4.0
        return s

    def city_keep_value(self, gs: GameState, v: VKey) -> float:
        """Owner 2026-10-02: when the barbarians take a city, keep the better one:
        3 different resources, good numbers, commodities (developments)."""
        pips = self._spot_pips(gs, v, "city")
        if gs.mode.canal:
            pips.pop("gold", None)       # Great Canal: only knights collect gold
        kinds = {r for r, x in pips.items() if x > 0}
        return sum(pips.values()) + 3.0 * len(kinds) + self._city_bonus(gs, v)

    def choose_city_to_lose(self, gs: GameState, options: list[VKey] | None = None) -> VKey | None:
        me = gs.me()
        cands = [v for v in (options or me.cities) if v in me.cities] or list(options or me.cities)
        # a metropolis can't be pillaged: never offer it while a plain city is left
        cands = [v for v in cands if v not in me.metropolises] or cands
        return min(cands, key=lambda v: self.city_keep_value(gs, v)) if cands else None

    # ---- knights given or taken by progress cards (owner 2026-10-04, live game 13) ----
    def knight_keep_value(self, gs: GameState, v: VKey) -> float:
        """What our knight at v is worth keeping now: strength, active, a city it keeps safe
        before the attack (and the defender VP), gold it collects, an undug canal tile it still
        digs, a leader's road it cuts. A knight with none of these (stuck, its canal done) is the
        cheapest -- losing it frees a rebuild where it scores more."""
        me = gs.me()
        k = me.knights[v]
        # owner 2026-10-06: a level costs a promotion to rebuild -- the lowest level goes first
        # (it gave up a level 2 with a level 1 on a dug canal beside it); active only breaks ties
        val = 0.04 * k.level + (0.02 if k.active else 0.0)
        g = self.clone(gs)
        g.me().knights.pop(v, None)
        if gs.mode.ck:
            danger = gs.barbarian_pos / max(1, gs.barbarian_len)
            opened = self.defence_gap(g, g.me()) - self.defence_gap(gs, me)
            val += (0.02 + 0.14 * danger) * max(0, opened)     # far off: time to re-activate
            if self.defender_vp:
                val += max(0.0, self._defender_bonus(g, gs))
        val += self._prospector_value(g, g.me(), v)
        if gs.mode.canal and gs.canal_progress < gs.canal_total and any(
                gs.board.is_canal(h) and h.coord not in gs.canal_dug for h in gs.board.vertex_hexes(v)):
            val += 0.05 + (0.02 if k.active else 0.0)
        if self._cut_target(gs, me, v) is not None or self._cuts_focus(gs, v):
            val += 0.02
        return val

    def choose_knight_to_lose(self, gs: GameState, options: list[VKey] | None = None) -> VKey | None:
        """Deserter played on us: give up the knight worth least now (owner 2026-10-04)."""
        me = gs.me()
        cands = [v for v in (options or me.knights) if v in me.knights]
        return min(cands, key=lambda v: (round(self.knight_keep_value(gs, v), 6), me.knights[v].level,
                                         me.knights[v].active, v)) if cands else None

    def choose_free_knight_spot(self, gs: GameState, options: list[VKey]) -> VKey | None:
        """A free knight to place (Deserter, ...): the legal corner a paid knight would pick --
        gold, an open canal tile, a city to defend before the attack, a leader's road to cut."""
        opts = [v for v in options if not gs.occupied(v)] or list(options)
        if not opts:
            return None
        g0 = self.clone(gs)
        for r, n in COSTS.get("knight", {}).items():
            g0.me().hand[r] = g0.me().hand.get(r, 0) + n      # the knight is free: cost is no factor

        def val(v):
            try:
                return self.action_value(g0, Action("knight", v))
            except Exception:                    # noqa: BLE001 -- a spot the model can't score
                return -1.0
        return max(opts, key=lambda v: (val(v), v))

    def warlord_covers(self, gs: GameState) -> bool:
        """Owner 2026-10-04: a held Warlord activates every knight free -- no grain (and no
        3:1 for grain) is spent on activation while one is in hand."""
        me = gs.me()
        return gs.mode.ck and "warlord" in (me.progress_hand or []) \
            and any(not k.active for k in me.knights.values())

    def _dup_penalty(self, gs: GameState, a: VKey, b2: VKey) -> float:
        """Same number on both openings = feast-or-famine income."""
        b = gs.board
        na = [h.number for h in b.vertex_hexes(a) if h.number]
        nb = [h.number for h in b.vertex_hexes(b2) if h.number]
        return self.setup_w["dup"] * sum(PIPS[n] for n in set(na) & set(nb))

    def _owned_pips(self, gs: GameState, me: PlayerState) -> dict[str, float]:
        tot: dict[str, float] = {}
        for v, kind in me.buildings():
            for r, x in self._spot_pips(gs, v, kind).items():
                tot[r] = tot.get(r, 0) + x
        return tot

    # ---- setup as a set (owner 2026-10-05, game 14; Brain.setup_set) ------------------------
    # Owner: "it could have picked 6 sheep 9 wood ... city, and the settlement here best would
    # have been at 5 9 10 where stone and clay were going to give it all it needed, then just a
    # harbor would have been the target for the third settlement". So: every hex counts by what
    # it pays over the whole game (barren canal farmland ~nothing, gold nothing to a building);
    # the city goes on the strongest wood+sheep corner; settlements cover what the others lack;
    # once the core set is covered the last (coastal) settlement is about a 3:1.
    SET_CORE = ("lumber", "brick", "wool", "ore")

    def _set_mode(self, gs: GameState) -> bool:
        return bool(self.setup_set and gs.mode.canal and gs.mode.start_with_city)

    def _life_pips(self, gs: GameState, v: VKey, kind: str) -> dict[str, float]:
        """Whole-game pips at v: _spot_pips (canal farmland 0.15: it pays only once the canal is
        done) without gold -- in the Great Canal only knights collect gold, never a building."""
        p = self._spot_pips(gs, v, kind)
        if gs.mode.canal:
            p.pop("gold", None)
        return p

    def _gold_near_pips(self, gs: GameState, v: VKey) -> int:
        """Pips of the best gold hex with a corner within 2 steps of v, or a short ship path away
        (a knight spot fed from here; owner game 4: the city "could have been lower to give
        access to the gold mine")."""
        b = gs.board
        ring1 = b.neighbors(v)
        near = set(ring1) | {w for u in ring1 for w in b.neighbors(u)} | {v}
        best = max((PIPS[h.number] for w in near for h in b.vertex_hexes(w)
                    if h.terrain == "gold" and h.number), default=0)
        if not best and self.gold_route_len(gs, v) is not None:
            # owner "6/10 with a ship towards the gold mine": a short ship path (<= 3 sea edges) from
            # this coastal corner to a free gold corner brings the knight there
            best = max((PIPS[h.number] for h in b.hexes.values() if h.terrain == "gold" and h.number), default=0)
        return best

    def _set_score(self, gs: GameState, me: PlayerState, extra: list) -> float:
        """Value of our whole opening = the pieces down + extra [(v, kind)]."""
        sw = self.setup_w
        b = gs.board
        spots = list(me.buildings()) + list(extra)
        tot: dict[str, float] = {}
        val = self.resource_value(gs, me)
        worth = lambda r: self.COMMODITY_W.get(r, 1.0) if r in COMMODITIES else val.get(r, 1.0)
        ws = ("lumber", "wool", "paper", "cloth")
        s = 0.0
        for v, kind in spots:
            p = self._life_pips(gs, v, kind)
            for r, x in p.items():
                tot[r] = tot.get(r, 0.0) + x
                # the city is there for wood + sheep: its other hexes count part (set_city_other)
                s += x * worth(r) * (sw["set_city_other"] if kind == "city" and r not in ws else 1.0)
            if kind == "city":
                s += sw["set_city_ws"] * sum(x for r, x in p.items() if r in ws)
                s += sw["set_city_gold"] * self._gold_near_pips(gs, v)
            if any(b.is_canal(h) for h in b.vertex_hexes(v)):
                s += sw["set_canal"]
        for r in self.SET_CORE:
            have = tot.get(r, 0.0)
            if have <= 0:
                s -= sw["set_miss"]
            elif have < 3:
                s -= sw["set_weak"]
        for i in range(len(spots)):
            for j in range(i + 1, len(spots)):
                s -= self._dup_penalty(gs, spots[i][0], spots[j][0])
        return s

    def _free_3to1_left(self, gs: GameState, me: PlayerState, taken: list) -> bool:
        """A legal coastal 3:1 corner survives our pieces at `taken` and the opponents' likely
        picks before our last settlement."""
        b = gs.board
        near = set(taken) | {w for t in taken for w in b.neighbors(t)}
        gone = self.predict_opponent_takes(gs, len(gs.players) - 1, near)
        cut = near | gone | {w for g in gone for w in b.neighbors(g)}
        return any(b.harbor_at(w) == "3:1" and b.is_coastal_vertex(w) and w not in cut
                   for w in self._setup_spots(gs)[0])

    def _set_pair_score(self, gs: GameState, v1: VKey, me: PlayerState) -> float:
        """First settlement: the best city still free after the opponents between our picks
        (hedged with the runner-up when someone picks in between), scored as one set."""
        b = gs.board
        near = set(b.neighbors(v1)) | {v1}
        k = self.opponent_picks_before_next(gs)
        gone = self.predict_opponent_takes(gs, k, near)
        scores = []
        for v2 in gs.legal_settlement_spots(me, setup=True):
            if v2 in near or v2 in gone or any(v2 in b.neighbors(g) for g in gone):
                continue
            sc = self._set_score(gs, me, [(v1, "settlement"), (v2, "city")]) + self._hex_term(gs, v2, "city", me)
            if self._free_3to1_left(gs, me, [v1, v2]):
                sc += self.setup_w["set_lookahead_harb"]
            scores.append(sc)
        scores.sort(reverse=True)
        hedge = 1 if k == 0 else 2
        partner = scores[:hedge] or [self._set_score(gs, me, [(v1, "settlement")])]
        return sum(partner) / len(partner)

    def _set_vertex_score(self, gs: GameState, v: VKey, me: PlayerState) -> float:
        sw = self.setup_w
        b = gs.board
        n = len(me.settlements) + len(me.cities)
        if n == 0:
            s = self._set_pair_score(gs, v, me) + self._hex_term(gs, v, "settlement", me)
            if b.harbor_at(v) == "3:1":
                pips = sum(self._life_pips(gs, v, "settlement").values())
                if pips >= sw.get("set_first_harb_min", 6.0):
                    s += sw.get("set_first_harb", 0.0) * min(1.0, pips / 9)
            return s
        have3 = self._has_3to1(gs, me)          # owner 2026-10-06: harbour first, then the last pick goes on numbers
        kind = self._setup_kind(gs, me)
        s = self._set_score(gs, me, [(v, kind)]) + self._hex_term(gs, v, kind, me)
        if n == 1:
            if not have3 and self._free_3to1_left(gs, me, [w for w, _ in me.buildings()] + [v]):
                s += sw["set_lookahead_harb"]
            return s
        # the last (coastal) settlement: it pays the start cards
        have = self._owned_pips(gs, me)
        covered = all(have.get(r, 0) > 0 for r in self.SET_CORE)
        hb = b.harbor_at(v)
        if hb == "3:1" and have3:
            pass                                # a second 3:1 adds nothing (owner: 5/6 at the bottom instead)
        elif hb == "3:1":
            s += sw["set_harb"] + (sw["set_harb_covered"] if covered else 0.0)
        elif hb and hb != "grain":
            here = self._life_pips(gs, v, "settlement")
            s += sw["harb_21_base"] + sw["harb_21_per_pip"] * (have.get(hb, 0) + here.get(hb, 0))
        s += self._start_hand_value(gs, me, v)
        tot = dict(have)
        for r, x in self._life_pips(gs, v, kind).items():
            tot[r] = tot.get(r, 0) + x
        s += self.three_roll_value(gs, tot, self._start_cards(gs, v))
        return s

    def setup_vertex_score(self, gs: GameState, v: VKey, me: PlayerState) -> float:
        """Value of the whole opening if we place here (existing pieces + this one)."""
        if self._set_mode(gs):
            return self._set_vertex_score(gs, v, me)
        tot = self._owned_pips(gs, me)
        kind = self._setup_kind(gs, me)
        here = self._spot_pips(gs, v, kind)
        for r, x in here.items():
            tot[r] = tot.get(r, 0) + x
        extras = self._spot_extras(gs, v, here)
        if gs.mode.canal and kind == "city":
            # the third (coastal) settlement can take the 3:1 harbor; the city is about production
            extras *= self.setup_w["city_extras"]
            extras += self.setup_w["city_pips"] * sum(here.values())
        s = self._opening_value(gs, me, tot) + extras
        for w, _ in me.buildings():
            s -= self._dup_penalty(gs, w, v)
        n_built = len(me.settlements) + len(me.cities)
        if n_built == 0:
            s += self._first_harbour_bonus(gs, v, here)
        if gs.mode.canal and n_built >= 2:
            s += self._start_hand_value(gs, me, v)
            s += self.third_harbour * (gs.board.harbor_at(v) == "3:1")
            s += self.third_grain * min(2, sum(h.terrain == "fields" for h in gs.board.vertex_hexes(v)))
        s += self._owner_terms(gs, v, kind, me)
        s += self._hex_term(gs, v, kind, me)
        pays = n_built >= 2 if gs.mode.canal else n_built == 1      # the placement that pays start cards
        s += self.three_roll_value(gs, tot, self._start_cards(gs, v) if pays else {})
        if gs.mode.canal and n_built == 1:
            # look one more pick ahead: the coastal 3rd settlement still free after the others pick
            near = set(gs.board.neighbors(v)) | {v}
            gone = self.predict_opponent_takes(gs, len(gs.players) - 1, near)
            coast = [w for w in gs.legal_settlement_spots(me, setup=True)
                     if gs.board.is_coastal_vertex(w) and w not in near and w not in gone
                     and not any(w in gs.board.neighbors(g) for g in gone)]
            if coast:
                s += self.setup_w["lookahead_coast"] * max(self._spot_extras(gs, w, self._spot_pips(gs, w, "settlement"))
                               + self._start_hand_value(gs, me, w) for w in coast)
        return s + (self._city_bonus(gs, v) if kind == "city" else 0.0)

    # ---- owner-rule setup terms (step5 audit 4D; all 0 by default) -------------------------
    @staticmethod
    def fields_hexes(gs: GameState, v: VKey) -> int:
        return sum(1 for h in gs.board.vertex_hexes(v) if h.terrain == "fields" and h.number)

    def gold_route_len(self, gs: GameState, v: VKey, max_edges: int | None = None) -> int | None:
        """Shortest ship path (sea edges, none owned) from coastal corner v to a free corner of a
        gold hex, in edges; None if longer than max_edges or v is not coastal. Owner: "6/10 with a
        ship towards the gold mine" -- the gold corner is a knight spot, never a building."""
        b = gs.board
        max_edges = SETUP_SHAPE["gold_route_max_edges"] if max_edges is None else max_edges
        if not b.is_coastal_vertex(v):
            return None
        gold = self._gold_corners(gs)
        if not gold:
            return None
        if v in gold and not gs.occupied(v):
            return 0
        seen, frontier = {v}, [v]
        for d in range(1, max_edges + 1):
            nxt = []
            for u in frontier:
                for w in b.neighbors(u):
                    if w in seen:
                        continue
                    e = ekey(u, w)
                    if not b.is_sea_edge(e) or gs.edge_owner(e):
                        continue
                    if w in gold and not gs.occupied(w):
                        return d
                    seen.add(w)
                    nxt.append(w)
            frontier = nxt
        return None

    def _gold_corners(self, gs: GameState) -> set:
        b = gs.board
        cache = getattr(b, "_gold_corners", None)
        if cache is None:
            cache = {v for v in b.vertices if any(h.terrain == "gold" for h in b.vertex_hexes(v))}
            b._gold_corners = cache
        return cache

    def last_free_3to1(self, gs: GameState, v: VKey, me: PlayerState) -> bool:
        """v is a 3:1 corner and no other legal 3:1 corner survives taking it, while an opponent
        still has a piece to place (they would take it)."""
        b = gs.board
        if b.harbor_at(v) != "3:1":
            return False
        k = len(me.buildings())
        if not any(len(o.buildings()) <= k for o in gs.players if o is not me):
            return False
        cut = set(b.neighbors(v)) | {v}
        return not any(b.harbor_at(w) == "3:1" and w not in cut
                       for w in gs.legal_settlement_spots(me, setup=True))

    def _hex_must_have(self, gs: GameState, v: VKey, me: PlayerState, grain2: bool = True) -> bool:
        """A 1-hex corner may still be the pick when it is the only way to a must-have: the canal
        3:1 we lack, the 2-grain corner, or the coastal 6/10 with a ship route to gold."""
        if not gs.mode.canal:
            return False
        if gs.board.harbor_at(v) == "3:1" and not self._has_3to1(gs, me):
            return True
        if grain2 and self.fields_hexes(gs, v) >= 2:
            return True
        return gs.board.vertex_pips(v) >= SETUP_SHAPE["gold_route_min_pips"] and self.gold_route_len(gs, v) is not None

    def _hex_term(self, gs: GameState, v: VKey, kind: str, me: PlayerState) -> float:
        """Owner 2026-10-09: settlements on 3 producing hexes, then 2, rarely 1 (hex_count_<n>).
        Settlements only: the canal start city keeps the owner's wood + sheep 6/9 rule (FIXES.md)."""
        if kind != "settlement":
            return 0.0
        return hex_count_term(gs, v, kind, self.setup_w, self._hex_must_have(gs, v, me))

    def _owner_terms(self, gs: GameState, v: VKey, kind: str, me: PlayerState) -> float:
        """grain2 / gold_route / deny_3to1 (canal only; each weight 0 = off)."""
        if not gs.mode.canal or kind != "settlement":
            return 0.0
        sw = self.setup_w
        s = 0.0
        if sw["grain2"] and self.fields_hexes(gs, v) >= 2:
            s += sw["grain2"]
        if sw["gold_route"] and gs.board.vertex_pips(v) >= SETUP_SHAPE["gold_route_min_pips"] \
                and self.gold_route_len(gs, v) is not None:
            s += sw["gold_route"]
        if sw["deny_3to1"] and self.last_free_3to1(gs, v, me):
            s += sw["deny_3to1"]
        return s

    # ---- setup lookahead (owner 2026-10-02: see where everyone sets, recalc every move,
    # estimate 3 dice ahead) ----------------------------------------------------------------
    def _raw_spot(self, gs: GameState, v: VKey) -> float:
        """How an opponent likely ranks a spot: plain pips (+ a 3:1 harbour in the canal)."""
        s = sum(PIPS[h.number] for h in gs.board.vertex_hexes(v) if h.number and h.coord != gs.board.robber)
        if gs.mode.canal and gs.board.harbor_at(v) == "3:1":
            s += 3
        return s

    def opponent_picks_before_next(self, gs: GameState) -> int:
        """Snake order: with N players and s opponent pieces already down before our first pick
        (= our seat), 2*(N-1-s) opponent picks come between our 1st and 2nd placement."""
        n = len(gs.players)
        mine = len(gs.me().buildings())
        theirs = sum(len(p.buildings()) for p in gs.opponents())
        if mine == 0:
            seat = min(theirs, n - 1)
            return 2 * (n - 1 - seat)
        return n - 1                                 # later rounds: about one pick each

    def setup_seat(self, gs: GameState) -> int:
        """1-based seat in the opening order, from how many opponent pieces were down when we
        first placed (1st gets the best spot; last places settlement and city back to back)."""
        if gs.me().buildings():
            return getattr(self, "_seat", 0) or 0
        self._seat = min(sum(len(p.buildings()) for p in gs.opponents()), len(gs.players) - 1) + 1
        return self._seat

    def _setup_spots(self, gs: GameState) -> tuple[list, list]:
        """(legal setup corners, the same sorted by plain pips), cached while no piece or the robber
        moves: one setup pick asked for them ~15k times (half of a simulated game's time)."""
        key = (id(gs.board), gs.board.robber,
               frozenset().union(*(p.settlements | p.cities for p in gs.players)))
        c = getattr(self, "_spots_cache", None)
        if c is None or c[0] != key:
            spots = gs.legal_settlement_spots(gs.me(), setup=True)
            c = self._spots_cache = (key, spots, sorted(spots, key=lambda v: -self._raw_spot(gs, v)))
        return c[1], c[2]

    def predict_opponent_takes(self, gs: GameState, k: int, blocked: set) -> set:
        """The k spots opponents most likely take next (greedy by plain pips, distance rule)."""
        b = gs.board
        taken, cut = set(), set(blocked)
        for v in self._setup_spots(gs)[1]:
            if len(taken) >= k:
                break
            if v in cut:
                continue
            taken.add(v)
            cut |= set(b.neighbors(v)) | {v}
        return taken

    def _start_cards(self, gs: GameState, v: VKey) -> dict:
        from .rules import production
        out: dict[str, int] = {}
        for h in gs.board.vertex_hexes(v):
            if h.number and h.terrain not in ("sea", "desert", "canal"):
                for r, n in production(h.terrain, "settlement", gs.mode).items():
                    if r in RESOURCES:
                        out[r] = out.get(r, 0) + n
        return out

    def three_roll_value(self, gs: GameState, pips: dict[str, float], start: dict | None = None) -> float:
        """Owner: think 3 dice ahead. Chance that the next 3 rolls (plus the start hand) bring
        every card of a first build: road, settlement, or (C&K) a knight. Gold pays any card."""
        start = start or {}
        gold = pips.get("gold", 0) / 36

        def p(r, n):
            q = min(1.0, pips.get(r, 0) / 36 + gold)
            one = 1 - (1 - q) ** 3                   # at least one in 3 rolls
            return one if n == 1 else one * (1 - (1 - q) ** 2) * 0.6   # rough 2nd copy
        goals = [COSTS["road"], COSTS["settlement"]] + ([COSTS["knight"]] if gs.mode.ck else [])
        best = 0.0
        for cost in goals:
            pr = 1.0
            for r, n in missing(dict(start), cost).items():
                pr *= p(r, n)
            best = max(best, pr)
        return self.setup_w["three_roll"] * best

    def setup_pair_score(self, gs: GameState, v1: VKey, me: PlayerState) -> float:
        """First placement: score v1 with the best second spot still free after the opponents
        between our picks take theirs (predicted from where everyone already sits)."""
        p1 = self._spot_pips(gs, v1, "settlement")
        second = "city" if gs.mode.start_with_city else "settlement"
        b = gs.board
        near = set(b.neighbors(v1)) | {v1}
        gone = self.predict_opponent_takes(gs, self.opponent_picks_before_next(gs), near)
        scores = []
        for v2 in gs.legal_settlement_spots(me, setup=True):
            if v2 in near or v2 in gone or any(v2 in b.neighbors(g) for g in gone):
                continue
            p2 = self._spot_pips(gs, v2, second)
            tot = dict(p1)
            for r, x in p2.items():
                tot[r] = tot.get(r, 0) + x
            sc = self._opening_value(gs, me, tot) + self._spot_extras(gs, v2, p2)
            if second == "city":
                sc += self._city_bonus(gs, v2)
            sc -= self._dup_penalty(gs, v1, v2)
            sc += self._hex_term(gs, v2, second, me)
            tot2 = dict(tot)
            start = self._start_cards(gs, v2) if not gs.mode.canal else {}
            sc += self.three_roll_value(gs, tot2, start)
            scores.append(sc)
        scores.sort(reverse=True)
        # last seat places both back to back (owner): nothing can take the partner, plan it exactly
        hedge = 1 if self.opponent_picks_before_next(gs) == 0 else 2
        partner = scores[:hedge] or [0.0]           # best left for us, hedged with the runner-up
        return sum(partner) / len(partner) + self._spot_extras(gs, v1, p1) \
            + self._first_harbour_bonus(gs, v1, p1) + self._owner_terms(gs, v1, "settlement", me) \
            + self._hex_term(gs, v1, "settlement", me)

    def choose_setup_settlement(self, gs: GameState, options: list[VKey] | None = None) -> VKey:
        me = gs.me()
        options = options or gs.legal_settlement_spots(me, setup=True)
        if gs.mode.canal and len(me.settlements) + len(me.cities) >= 2:
            coast = [v for v in options if gs.board.is_coastal_vertex(v)]
            options = coast or options      # canal rule: the third settlement must be on the coast
        if me.settlements or me.cities or self._set_mode(gs):
            # (set mode: the first pick already plans the city with it -- _set_pair_score)
            return max(options, key=lambda v: self.setup_vertex_score(gs, v, me))
        # prefilter by single-spot value, then look ahead to the pair
        top = sorted(options, key=lambda v: self.setup_vertex_score(gs, v, me), reverse=True)[:20]
        return max(top, key=lambda v: self.setup_pair_score(gs, v, me))

    def choose_setup_road(self, gs: GameState, at: VKey, options: list[EKey] | None = None) -> EKey:
        me = gs.me()
        b = gs.board
        # Seafarers: the first piece off a coastal spot may be a ship (owner: "doesn't know how to
        # set a boat"); ships on a sea edge, roads on a land edge
        ok = (lambda e: b.is_land_edge(e) or b.is_sea_edge(e)) if gs.mode.seafarers else b.is_land_edge
        options = options or [e for e in b.vertex_edges(at) if not gs.edge_owner(e) and ok(e)]
        if gs.mode.canal and b.is_coastal_vertex(at):
            # Almanac: every setup piece built on the coast gets a ship (the east lake is not coast)
            options = [e for e in options if b.is_sea_edge(e)] or options

        def reach(e):
            far = e[0] if e[1] == at else e[1]
            best = 0.0
            for n in b.neighbors(far):
                if n != at and gs.can_settle(n):
                    best = max(best, self.setup_vertex_score(gs, n, me))
            return best
        # canal: every setup piece on the coast is a ship (coast edges count as land edges too)
        ships = [e for e in options if b.is_sea_edge(e) and (
            not b.is_land_edge(e) or (gs.mode.canal and b.is_coastal_vertex(at)))]
        if self.setup_ship_lookahead and ships:
            # live 2026-10-09 20:09 ("the second ship is going nowhere"): one corner ahead at sea has
            # nothing to settle, so every ship tied; follow free sea edges up to 4 steps to the best
            # settle spot (or undug canal corner) and take the edge that heads there
            home, todo = {at}, [at]           # our landmass: spots roads reach anyway
            while todo:
                x = todo.pop()
                for f in b.vertex_edges(x):
                    y = f[0] if f[1] == x else f[1]
                    if b.is_land_edge(f) and y not in home:
                        home.add(y)
                        todo.append(y)

            def ship_reach(e):
                far = e[0] if e[1] == at else e[1]
                best, frontier, seen = 0.0, {far}, {at, far}
                for d in range(1, 5):
                    nxt = set()
                    for x in frontier:
                        for f in b.vertex_edges(x):
                            if not b.is_sea_edge(f) or gs.edge_owner(f):
                                continue
                            y = f[0] if f[1] == x else f[1]
                            if y in seen:
                                continue
                            seen.add(y)
                            nxt.add(y)
                            if y not in home and gs.can_settle(y):
                                best = max(best, self.setup_vertex_score(gs, y, me) * 0.85 ** d)
                            if self.canal_road_front and gs.mode.canal and gs.canal_progress < gs.canal_total:
                                best = max(best, 40.0 * self._canal_front(gs, y) * 0.85 ** d)
                    frontier = nxt
                return best
            front = (lambda e: round(self._canal_front(gs, e[0] if e[1] == at else e[1]), 2)) \
                if self.canal_road_front and gs.mode.canal and gs.canal_progress < gs.canal_total else (lambda e: 0)
            return max(options, key=lambda e: (front(e), ship_reach(e) if e in ships else 0.0, reach(e)))
        if self.canal_road_front and gs.mode.canal and gs.canal_progress < gs.canal_total:
            # owner 2026-10-09 (live 20:02): the setup road went up toward grain; it goes along the
            # canal toward the rivals, then toward the best spot
            def front(e):
                far = e[0] if e[1] == at else e[1]
                return round(self._canal_front(gs, far), 2)
            return max(options, key=lambda e: (front(e), reach(e)))
        return max(options, key=reach)

    # ---- main turn --------------------------------------------------------------
    BOARD_BUILDS = ("settlement", "city", "road", "ship", "dev", "improve", "knight", "activate", "promote", "wall")

    def still_legal(self, gs: GameState, before: list, skipped: list, a: Action) -> bool:
        """replan_after_skip: replay the plan items before `a` without the skipped ones and say
        whether `a` is still a legal build there (cards not counted: piece limits, spots, levels)."""
        if not self.replan_after_skip or not skipped or a.kind not in self.BOARD_BUILDS:
            return True
        g = self.clone(gs)
        for b in before:
            if any(b is x for x in skipped) or b.kind not in self.BOARD_BUILDS:
                continue
            try:
                g = self._apply(g, b)
            except Exception:                    # noqa: BLE001 -- a stale item: judge on what we have
                pass
        me = g.me()
        me.hand = {k: 99 for k in list(RESOURCES) + list(COMMODITIES)}
        return any(m.kind == a.kind and m.target == a.target for m in mv.build_moves(g))

    def _apply(self, gs: GameState, a: Action) -> GameState:
        g = self.clone(gs)
        me = g.me()
        if a.kind == "bank_trade":
            pay(me.hand, a.give)
            for k, v in a.get.items():
                me.hand[k] = me.hand.get(k, 0) + v
            return g
        cost = improvement_cost(a.target, me.improvements[a.target]) if a.kind == "improve" else COSTS.get(a.kind, {})
        pay(me.hand, cost)
        if a.kind == "settlement":
            me.settlements.add(a.target)
            me.vp_visible += 1
        elif a.kind == "city":
            me.settlements.discard(a.target)
            me.cities.add(a.target)
            me.vp_visible += 1
        elif a.kind == "road":
            me.roads.add(a.target)
            me.longest_road = g.longest_road_len(me)
        elif a.kind == "ship":
            me.ships.add(a.target)
            me.ships_new.add(a.target)            # can't be moved this turn
            me.longest_road = g.longest_road_len(me)
        elif a.kind == "move_ship":
            src, dst = a.target
            me.ships.discard(src)
            me.ships.add(dst)
            me.ship_moved = True                  # one ship move per turn
            me.longest_road = g.longest_road_len(me)
        elif a.kind == "dev":
            me.dev_cards += 1
        elif a.kind == "knight":
            me.knights[a.target] = Knight(1, False)
        elif a.kind == "activate":
            me.knights[a.target].active = True
            me.knights[a.target].acted = True     # C&K: no knight action the turn it is activated
        elif a.kind == "move_knight":
            src, dst = a.target
            for o in g.players:
                if o is not me:
                    o.knights.pop(dst, None)      # displaced: where it lands is its owner's choice
            k = me.knights.pop(src)
            k.active, k.acted = False, True
            me.knights[dst] = k
        elif a.kind == "chase":
            # C&K: the knight that chased the robber/pirate away is inactive (and has acted)
            k = me.knights[a.target[0]]
            k.active, k.acted = False, True
        elif a.kind == "promote":
            me.knights[a.target].level += 1
        elif a.kind == "wall":
            me.walls.add(a.target)
        elif a.kind == "improve":
            me.improvements[a.target] += 1
            me.vp_visible += grant_metropolis(g, me, a.target)   # per track; level 5 steals
        if a.kind in ("road", "ship", "move_ship", "settlement", "knight", "move_knight"):
            self._award_longest_road(g)          # owner 2026-10-04: knights break roads too
        if g.mode.canal and a.kind in ("knight", "activate", "move_knight"):
            before = me.canal_chips
            dig_canals(g)
            me.vp_visible += me.canal_chips - before
        me.hand_count = sum(me.hand.values())
        return g

    @staticmethod
    def _award_longest_road(g: GameState) -> None:
        """Longest road (2 VP): 5+ and strictly longer than the holder; a settlement on a road
        can break it. Owner 2026-10-02: teach the bot the points from roads."""
        for name, d in award_longest_road(g).items():
            next(p for p in g.players if p.name == name).vp_visible += d

    def candidate_actions(self, gs: GameState) -> list[Action]:
        """Builds worth simulating: the legal ones from moves.build_moves, with knight spots
        pruned to the first plain one plus canal and prospector corners (search cost)."""
        me = gs.me()
        out, plain = [], False
        builds = mv.build_moves(gs)
        # stage 2026-10-09 06:11: the plain knight spot was the first corner in sort order -- (4,2),
        # the corner two roads had just reached; the settlement spot was gone. Corners we could
        # settle go last.
        if gs.mode.ck and any(m.kind == "knight" for m in builds):
            spots = set(gs.legal_settlement_spots(me))
            builds.sort(key=lambda m: m.kind == "knight" and m.target in spots)
        for m in builds:
            if m.kind == "knight":
                special = gs.mode.canal and (
                    any(gs.board.is_canal(h) and h.coord not in gs.canal_dug for h in gs.board.vertex_hexes(m.target))
                    or self._prospector_value(gs, me, m.target) > 0) \
                    or (self.road_cuts and (self._cuts_road(gs, me, m.target) or self._cuts_focus(gs, m.target)))
                if not special and plain:
                    continue
                plain = plain or not special
            out.append(Action(m.kind, m.target))
        if self.warlord_covers(gs):
            out = [a for a in out if a.kind != "activate"]    # the Warlord activates them free
        if self.canal_contest and not self.knight_moves and gs.mode.canal and gs.canal_progress < gs.canal_total:
            # owner: push a weaker rival knight off an undug canal corner, or step onto a free one
            for m in mv.knight_moves(gs, ships=self.knight_move_ships):
                dst = m.target[1]
                if self._undug_canal_corner(gs, dst) and (
                        not gs.occupied(dst) or any(dst in o.knights for o in gs.opponents())):
                    out.append(Action(m.kind, m.target))
        if self.knight_moves and gs.mode.canal:
            # only moves that reach gold or an undug canal corner are worth simulating
            for m in mv.knight_moves(gs, ships=self.knight_move_ships):
                dst = m.target[1]
                cuts = self.road_cuts and (self._cuts_road(gs, me, dst) or self._cuts_focus(gs, dst))
                matters = self.knight_target_by_spot and any(dst in o.knights for o in gs.opponents()) \
                    and kt.spot_matters(gs, me, dst)
                if gs.occupied(dst) and not (self.displace or cuts or matters):
                    continue
                if gs.occupied(dst) or cuts or self._prospector_value(gs, me, dst) > 0 or any(
                        gs.board.is_canal(h) and h.coord not in gs.canal_dug for h in gs.board.vertex_hexes(dst)):
                    out.append(Action(m.kind, m.target))
        if self.ship_moves and gs.mode.seafarers:
            out += [Action(m.kind, m.target) for m in mv.ship_moves(gs)]
        return out

    @staticmethod
    def _touches_canal(gs: GameState, me: PlayerState) -> bool:
        pts = {v for v, _ in me.buildings()} | {v for e in me.roads for v in e} | set(me.knights)
        return any(gs.board.is_canal(h) for v in pts for h in gs.board.vertex_hexes(v))

    def _needs_grain(self, gs: GameState) -> bool:
        """Great Canal, canal open, a city at risk: a knight still has to be activated."""
        if not (gs.mode.ck and gs.mode.canal and gs.canal_progress < gs.canal_total):
            return False
        if self.warlord_covers(gs):
            return False                         # a held Warlord activates them free
        return self.defence_gap(gs, gs.me()) > 0

    def _activation_useful(self, gs: GameState) -> bool:
        """fleet opus: an idle knight's activation changes something now -- a city at risk, the
        sole-top defender spot (one activation lifts us above every rival), or a canal dig."""
        if not gs.mode.ck:
            return True
        me = gs.me()
        idle = [(v, k) for v, k in me.knights.items() if not k.active]
        if not idle:
            return False
        if self.defence_gap(gs, me) > 0:
            return True
        b = gs.board
        mine = gs.defence_strength(me)
        best = max((gs.defence_strength(o) for o in gs.opponents()), default=0)
        gain = max((k.level for v, k in idle if not (gs.mode.canal and (
            b.is_small_island_vertex(v) or self._on_gold(gs, v)))), default=0)
        if mine <= best < mine + gain:
            return True
        if gs.mode.canal and gs.canal_progress < gs.canal_total:
            for v, _ in idle:
                if self._apply(gs, Action("activate", v)).me().canal_chips > me.canal_chips:
                    return True
        return False

    def _grain_engine_value(self, gs: GameState, a: Action) -> float:
        """fleet opus: the only steady grain in the canal is a gold knight (any card on an 8) and
        the aqueduct (science 3). Science 1-3 earns grain_engine (double for the level that gives the
        aqueduct); the first knight on a gold corner earns 2x grain_engine."""
        me = gs.me()
        if a.kind == "improve" and a.target == "science":
            lvl = me.improvements["science"] + 1
            return self.grain_engine * (2.0 if lvl == 3 else 1.0) if lvl <= 3 else 0.0
        if a.kind == "knight" and self._on_gold(gs, a.target) \
                and not any(self._on_gold(gs, v) for v in me.knights):
            return 2.0 * self.grain_engine
        return 0.0

    def _knight_activatable(self, gs: GameState, v: VKey) -> bool:
        """fleet opus (C5): a new knight is worth its nudge only when it can be activated -- grain in
        hand beyond the knights already idle -- or it parks on gold (it collects without activation).
        Outside the canal (or once it is complete) grain is no longer scarce."""
        if not (gs.mode.canal and gs.canal_progress < gs.canal_total):
            return True
        me = gs.me()
        if self._on_gold(gs, v):
            return True
        idle = sum(not k.active for k in me.knights.values())
        return me.hand.get("grain", 0) > idle

    def _improve_keep(self, gs: GameState) -> dict[str, int]:
        """Commodities that pay for an improvement now (or one card short), kept out of bank trades while the
        barbarians are more than 2 steps away (owner: there was a round to spare)."""
        me = gs.me()
        if not (gs.mode.ck and me.cities and gs.barbarian_len - gs.barbarian_pos > 2):
            return {}
        keep = {}
        for track, com in IMPROVEMENT_TRACKS.items():
            lvl = me.improvements.get(track, 0)
            if lvl < 5 and me.hand.get(com, 0) >= lvl:      # affordable, or one card short
                keep[com] = min(me.hand.get(com, 0), lvl + 1)
        return keep

    def _grain_first_trade(self, gs: GameState) -> Action | None:
        """Owner 2026-10-03: holding 3 of a kind (at a 3:1) and no grain -> bank it for grain now,
        keeping 1 wool + 1 ore for the knight if it isn't built yet."""
        me = gs.me()
        idle = any(not k.active for k in me.knights.values())
        canal = gs.mode.ck and gs.mode.canal
        if self.warlord_covers(gs):
            return None                          # owner 2026-10-04: play the Warlord, no 3:1 for grain
        if not self.grain_first or me.hand.get("grain", 0) or not (
                self._needs_grain(gs) or (self.set_for_grain and canal and idle
                                          and (not self.grain_gate or self._activation_useful(gs)))):
            return None
        keep = {} if any(not k.active for k in me.knights.values()) else {"wool": 1, "ore": 1}
        if canal and not me.cities:
            keep["ore"] = max(keep.get("ore", 0), COSTS["city"].get("ore", 0))   # the ore rebuilds the city
        # owner 2026-10-06: sold 3 paper for grain with the ship at 3/7 instead of taking
        # science 3 -- with a round to spare, an improvement we can pay now keeps its commodities
        for com, n in self._improve_keep(gs).items():
            keep[com] = max(keep.get(com, 0), n)
        val = self.resource_value(gs, me)
        locked = self.reserved_commodities(gs)
        opts = []
        for r in list(RESOURCES) + list(COMMODITIES):
            ratio = self.bank_ratio(gs, r)
            if r != "grain" and r not in locked and me.hand.get(r, 0) - keep.get(r, 0) >= ratio:
                opts.append((ratio * val.get(r, 1.0), r, ratio))
        if not opts and self.opening_set_first and canal and not me.knights and not gs.barbarians_attacked:
            # owner 2026-10-03 "bank it for grain now, even without wool/ore yet; wait another
            # round for them": the knight's wool/ore may be part of the set (city ore still kept)
            for r in ("wool", "ore"):
                ratio = self.bank_ratio(gs, r)
                left = COSTS["city"].get(r, 0) if r == "ore" and not me.cities else 0
                if r not in locked and me.hand.get(r, 0) - left >= ratio:
                    opts.append((ratio * val.get(r, 1.0), r, ratio))
        if not opts:
            return None
        _, r, ratio = min(opts)
        return Action("bank_trade", give={r: ratio}, get={"grain": 1})

    def _grain_reserve_cost(self, gs: GameState, a: Action) -> float:
        """An idle knight, no grain, a city at risk (Great Canal: grain only comes from the bank,
        gold or the aqueduct): don't spend the cards of our nearest bank set for grain."""
        me = gs.me()
        if not (gs.mode.ck and gs.mode.canal and gs.canal_progress < gs.canal_total):
            return 0.0
        if me.hand.get("grain", 0) or self.defence_gap(gs, me) <= 0 \
                or not any(not k.active for k in me.knights.values()):
            return 0.0
        if self.warlord_covers(gs):
            return 0.0                           # a held Warlord activates them free
        sets = [(me.hand.get(r, 0) / self.bank_ratio(gs, r), r) for r in list(RESOURCES) + list(COMMODITIES)
                if r != "grain" and me.hand.get(r, 0)]
        if not sets:
            return 0.0
        frac, r = max(sets)
        if not self._cost(gs, a).get(r):
            return 0.0
        danger = gs.barbarian_pos / max(1, gs.barbarian_len)
        return (0.04 + 0.08 * danger) * min(1.0, frac)

    def _cost(self, gs: GameState, a: Action) -> dict:
        if a.kind == "improve":
            return improvement_cost(a.target, gs.me().improvements[a.target])
        return COSTS.get(a.kind, {})

    @staticmethod
    def barb_loss_risk(gs: GameState) -> bool:
        """Would we lose a city if the barbarians attacked now? (C&K: they win when the total
        active strength is below the number of cities; every weakest player with a city that can
        be pillaged loses one.)"""
        if not gs.mode.ck:
            return False
        pill = {p.name: [v for v in p.cities if v not in p.metropolises] for p in gs.players}
        me = gs.me()
        if not pill.get(me.name):
            return False
        # owner 2026-10-05: enough knights now is not safe -- a rival can move (switch off) a
        # knight right before the attack and still stay above us, so count the worst case
        mine = gs.defence_strength(me)
        worst = {p.name: (mine if p is me else min(gs.defence_strength(p), mine + 1) if pill.get(p.name) else 0)
                 for p in gs.players}
        if sum(worst.values()) >= sum(len(p.cities) for p in gs.players):
            return False
        weakest = min(worst[p.name] for p in gs.players if pill.get(p.name))
        return mine <= weakest

    def grain_set_cards(self, gs: GameState) -> set:
        """Cards a player offer must not give away: Great Canal (canal still open), no grain, an
        inactive knight that would cover a city at risk -> every non-grain resource whose bank
        set is complete or one card short (2 of a kind at a 3:1). The next roll or one asked
        card turns that pair into the grain the knight needs. Empty when keep_grain_set is off."""
        if not self.keep_grain_set or not (gs.mode.ck and gs.mode.canal) \
                or gs.canal_progress >= gs.canal_total:
            return set()
        me = gs.me()
        if me.hand.get("grain", 0) or not any(not k.active for k in me.knights.values()):
            return set()
        if self.defence_gap(gs, me) <= 0 or self.warlord_covers(gs):
            return set()
        out = set()
        for r in RESOURCES:
            if r == "grain":
                continue
            ratio = self.bank_ratio(gs, r)
            if ratio <= 3 and me.hand.get(r, 0) >= max(1, ratio - 1):
                out.add(r)
        return out

    @staticmethod
    def defence_gap(gs: GameState, me: PlayerState) -> int:
        """C&K owner rule: one ACTIVE knight per city, all game long. Cities that are
        metropolises can't be pillaged. >0 means a city is at risk."""
        if not gs.mode.ck:
            return 0
        at_risk = len([v for v in me.cities if v not in me.metropolises])
        return max(0, at_risk - gs.defence_strength(me))

    def wall_value(self, gs: GameState, me) -> float:
        """Owner 2026-10-02: the bot kept losing half its hand to 7s. Each city wall (and the
        Engineer card, a free wall) lifts the hand limit by 2, so it is worth roughly the cards
        it saves: a 7 every ~6 rolls, and a hand that sits above the limit loses half of it."""
        limit = discard_limit(gs.mode, me)
        hand = sum(me.hand.values())
        per_round = gs.income_total(me) * max(2, len(gs.players))     # cards between our turns
        pressure = max(0.0, min(2.0, (hand + per_round - limit + 2) / 3))
        boost = self.wall_boost if (gs.barbarians_attacked or not self.wall_late) else 0.0
        if boost:
            # owner 2026-10-03: losing cards to the robber and the 7s is the signal for a wall
            pressure = max(pressure, min(2.0, me.cards_lost / 6))
        return 0.006 + 0.012 * pressure * (1 + boost)

    def action_value(self, gs: GameState, a: Action) -> float:
        g = self._apply(gs, a)
        v = self.my_prob(g)
        if a.kind == "dev":
            v += 0.012                       # expected knight/VP/progress value
        v += self._outlook_bonus(gs, g, a)
        if gs.mode.ck and a.kind in ("knight", "activate", "promote"):
            danger = gs.barbarian_pos / max(1, gs.barbarian_len)
            idle_promote = (a.kind == "promote" and self.idle_knight_first and self.defence_gap(gs, gs.me()) > 0
                            and not gs.me().knights.get(a.target, Knight(1, True)).active)
            if not idle_promote:                 # an idle knight defends nothing, however strong
                v += 0.015 * danger * (1 + len(gs.me().cities))
            # saving cities beats everything else: count a build of an (inactive)
            # knight as half-closing the gap, since activation must follow
            gap0, gap1 = self.defence_gap(gs, gs.me()), self.defence_gap(g, g.me())
            if gap0 > 0:
                idle = sum(not k.active for k in gs.me().knights.values())
                half = 0.5 if a.kind == "knight" and not ((self.grain_reserve or self.idle_knight_first)
                                                         and idle >= gap0) else 0.0
                closed = gap0 - gap1 + half
                v += (0.08 + 0.08 * danger) * closed
        if self.barb_relative and gs.mode.ck and a.kind in ("knight", "activate", "promote") \
                and self.barb_loss_risk(gs) and not self.barb_loss_risk(g):
            danger = gs.barbarian_pos / max(1, gs.barbarian_len)
            v += self.barb_relative * (0.5 + danger)
        if self.grain_first and a.kind not in ("activate", "bank_trade") and self._needs_grain(gs) \
                and gs.me().hand.get("grain", 0) and not g.me().hand.get("grain", 0):
            danger = gs.barbarian_pos / max(1, gs.barbarian_len)
            v -= 0.1 + 0.08 * danger            # that grain is for the knight
        if self.grain_reserve and a.kind not in ("activate", "bank_trade"):
            v -= self.grain_reserve * self._grain_reserve_cost(gs, a)
        if gs.mode.ck and a.kind == "knight" and self.knight_always:
            v += self.knight_always
        if gs.mode.ck and a.kind == "knight" and self.knight_guard and not self._knight_activatable(gs, a.target):
            v -= self.knight_always + self.knight_guard
        if gs.mode.ck and a.kind == "knight" and self.spare_knight \
                and len(gs.me().knights) >= max(1, len(gs.me().cities)) \
                and not self._knight_activatable(gs, a.target):
            v -= self.knight_always + self.spare_knight
        if gs.mode.canal and a.kind in ("activate", "knight") and not gs.me().cities \
                and g.me().canal_chips <= gs.me().canal_chips and self._defender_bonus(gs, g) <= 0 \
                and not (a.kind == "knight" and self._on_gold(gs, a.target)):
            v -= 0.05                      # no city to defend: grain and sheep go to the city
        if gs.mode.ck and self.defender_vp and a.kind in ("activate", "promote", "move_knight"):
            v += self._defender_bonus(gs, g)
        if self.spend_grain and gs.mode.canal and a.kind != "bank_trade" \
                and gs.me().hand.get("grain", 0) > g.me().hand.get("grain", 0) \
                and not (self.grain_gate and a.kind == "activate" and not self._activation_useful(gs)):
            v += self.spend_grain
        if self.road_cuts and gs.mode.ck and a.kind in ("road", "ship"):
            v += self._cut_reach_value(gs, g)
        if self.road_cuts and gs.mode.ck and a.kind == "knight" and self._cut_target(gs, gs.me(), a.target):
            v += 0.01                    # owner: the knight that breaks a rival's road
        if self.road_cuts and gs.mode.ck and a.kind == "knight" and self._cuts_focus(gs, a.target):
            v += 0.01                    # selector BLOCK: ...most of all the advancing rival's
        if self.canal_contest and gs.mode.canal and a.kind in ("road", "ship") and gs.canal_progress < gs.canal_total:
            v += self._contest_reach_value(gs, g)
        if self.canal_hunt and gs.mode.canal and a.kind in ("road", "ship") and gs.canal_progress < gs.canal_total:
            v += self._canal_reach_value(gs, g)
        if self.canal_deny and gs.mode.canal and a.kind == "road" and gs.canal_progress < gs.canal_total:
            v += self._canal_deny_value(gs, g)
        if self.canal_road_front and gs.mode.canal and a.kind in ("road", "ship") and gs.canal_progress < gs.canal_total:
            new = {x for e in g.me().roads | g.me().ships for x in e} - {x for e in gs.me().roads | gs.me().ships for x in e}
            v += self.canal_road_front * max((self._canal_front(gs, x) for x in new), default=0.0)
        if self.gold_route and gs.mode.canal and a.kind in ("road", "ship", "move_ship"):
            v += self._gold_route_value(gs, g)
        if a.kind == "settlement" and len(gs.me().settlements) + len(gs.me().cities) >= 2:
            # owner 2026-10-09: settle where 3 hexes pay, then 2, rarely 1 (same scale as city_keep_value)
            v += 0.004 * self._hex_term(gs, a.target, "settlement", gs.me())
        if a.kind == "city" and self.city_first:
            v += self.city_first
        if a.kind == "city":
            # owner 2026-10-03: the city went on a gold/sea corner with no resources or
            # developments -- upgrade the settlement that produces most (3 kinds, commodities)
            v += 0.004 * self.city_keep_value(gs, a.target)
        if gs.mode.ck and a.kind == "city" and self.defence_gap(g, g.me()) > 0:
            danger = gs.barbarian_pos / max(1, gs.barbarian_len)
            v -= 0.02 + 0.04 * danger   # a new city without a knight is a pillage target
        if gs.mode.canal and a.kind == "road" and gs.canal_progress < gs.canal_total:
            if any(gs.board.is_canal(h) for x in a.target for h in gs.board.vertex_hexes(x)) \
                    and not self._touches_canal(gs, gs.me()):
                v += 0.03
        if gs.mode.canal and a.kind in ("knight", "activate") and gs.canal_progress < gs.canal_total:
            v += self._canal_pair_value(gs, a)
        if gs.mode.canal and a.kind == "knight":
            v += self._prospector_value(gs, gs.me(), a.target)
        if self.grain_engine and gs.mode.canal and gs.canal_progress < gs.canal_total:
            v += self._grain_engine_value(gs, a)
        if a.kind == "move_knight":
            src, dst = a.target
            # gold gained at the new corner, minus the gold given up at the old one
            v += self._prospector_value(gs, gs.me(), dst) - self._prospector_value(g, g.me(), src)
            if gs.canal_progress < gs.canal_total:
                if self.canal_hunt:
                    # value the corner without the mover still at src (same-hex moves aren't pairs),
                    # minus a pair it leaves behind at src
                    v += self._canal_pair_value(g, Action("knight", dst))
                    v -= 0.03 * any(gs.board.is_canal(h) and h.coord not in gs.canal_dug
                                    and any(c in gs.me().knights and c != src for c in hex_corners(*h.coord))
                                    for h in gs.board.vertex_hexes(src) if dst not in hex_corners(*h.coord))
                else:
                    v += self._canal_pair_value(gs, Action("knight", dst))
            danger = gs.barbarian_pos / max(1, gs.barbarian_len)
            opened = self.defence_gap(g, g.me()) - self.defence_gap(gs, gs.me())
            v -= (0.08 + 0.08 * danger) * max(0, opened)   # the move deactivates a defender
            if self.canal_hunt and gs.mode.ck and (opened > 0 or self.barb_loss_risk(g)) \
                    and not gs.me().hand.get("grain", 0):
                v -= 1.0    # owner 2026-10-08: never switch a defender off when no grain can switch it back on
            victim = next((o for o in gs.players if dst in o.knights and o is not gs.me()), None)
            if victim is not None:
                # pushing a knight off an undug canal corner breaks that player's dig there
                v += 0.01 + 0.03 * any(gs.board.is_canal(h) and h.coord not in gs.canal_dug
                                       for h in gs.board.vertex_hexes(dst))
                if self.knight_target_by_spot:
                    # owner 2026-10-09: the best spot when one matters, else the largest knight
                    s = kt.knight_spot_value(gs, gs.me(), dst)
                    v += 0.004 * min(s, 6.0) if s >= kt.SPOT_MATTERS else 0.001 * victim.knights[dst].level
        if gs.mode.ck and self.knight_plan(gs, gs.me()) == (a.kind, a.target if a.kind == "improve" else None):
            v += 0.03
        if gs.mode.ck and a.kind == "promote" and gs.me().improvements["politics"] >= 3:
            v += 0.01
        # (Great Canal: gold corners can't be built on, only knights collect gold -- see
        # _prospector_value; owner 2026-10-03)
        if a.kind in ("road", "ship", "move_ship"):
            v += self._road_value(gs, g)
        if a.kind == "move_ship":
            v += self._ship_move_value(gs, g)
        if gs.mode.ck and a.kind == "wall":
            v += self.wall_value(gs, gs.me())
        if gs.mode.ck and a.kind == "city" and self.improve_ladder:
            v += self._new_commodity_value(gs, a.target)
        if gs.mode.ck and a.kind == "city" and self.city_wall_slot:
            me0 = gs.me()
            if len(me0.walls) < 3 and len(me0.walls) >= len(me0.cities):
                v += self.city_wall_slot          # every city already walled: this one opens a slot
        if gs.mode.ck and a.kind == "city" and self.city_metro_slot:
            me0 = gs.me()
            held = {t for t, h in (getattr(gs, "metropolis_owner", {}) or {}).items() if h and h[0] == me0.name}
            if not [c for c in me0.cities if c not in me0.metropolises] and \
                    any(lv >= 3 and t not in held for t, lv in me0.improvements.items()):
                v += self.city_metro_slot        # every city holds a metropolis: this one hosts the next
        if gs.mode.ck and a.kind == "improve":
            lvl = gs.me().improvements[a.target] + 1
            # owner order paper (science) > coin (politics) > cloth (trade); a commodity
            # sitting in hand is robber food, so an improvement must beat the spend penalty
            order = {"science": 1.2, "politics": 1.0, "trade": 0.85}.get(a.target, 1.0)
            v += order * (0.03 + 0.008 * lvl) + (0.02 if lvl == 3 else 0)   # level-3 abilities
            if self.improve_ladder:
                v += self._ladder_value(gs, a.target, lvl)
            if gs.mode.canal and a.target == "science" and lvl <= 3 and gs.canal_progress < gs.canal_total:
                # owner 2026-10-02: with no grain fields, the aqueduct (science 3: a free card of
                # our choice on any roll that gives us nothing) and gold are the only grain
                v += 0.02 + (0.04 if lvl == 3 else 0.0)
        return v

    def _canal_pair_value(self, gs: GameState, a: Action) -> float:
        """Playbook: a dig pays 2 chips (2 VP) if both knights are ours, 1 each if shared.
        Prefer setting up our own pair; never hand the leader a chip while we hold a city."""
        me = gs.me()
        threat = self.threat(gs)
        best, extra = 0.0, 0.0
        # owner 1006: no 2nd knight build while one is idle -- the hunt bonus waits for that too
        hunt = self.canal_hunt and not (a.kind == "knight" and any(not k.active for k in me.knights.values()))
        reach = self._canal_corner_reach(gs, me) if hunt else set()
        for h in gs.board.vertex_hexes(a.target):
            if not gs.board.is_canal(h) or h.coord in gs.canal_dug:
                continue
            corners = set(hex_corners(*h.coord)) - {a.target}
            mine = [v for v in corners if v in me.knights]
            others = [p for p in gs.opponents() for v, k in p.knights.items() if v in corners and k.active]
            if others and not mine:
                # activating next to an opponent's active knight digs a shared hex at once
                partner = others[0]
                if self._city_out_of_reach(gs):
                    # owner game 15: city out of reach -> share the dig with anyone, else others take it
                    val = 0.03 if a.kind == "activate" else 0.02
                elif threat is not None and partner is threat:
                    val = -0.03 if a.kind == "activate" else -0.01
                else:
                    val = 0.01
            elif mine:
                # second corner of our own pair: the knight now + activation later = 2 VP
                val = 0.05 if a.kind == "knight" else 0.02
            else:
                val = 0.02          # first corner: half of a dig, the canal also unlocks grain
                if corners & reach:
                    val += 0.015    # canal_hunt: a second corner of this hex is in reach -> our own pair
            if best != 0.0 and hunt and min(best, val) > 0:
                extra = 0.01        # canal_hunt: a corner of two undug hexes can dig both
            best = val if best == 0.0 else max(best, val)
        return best + extra if best > 0 else best

    def _city_out_of_reach(self, gs: GameState) -> bool:
        """Owner 2026-10-05 game 15: "if the game advances too fast, bot gets stuck trying to build
        the city" + "sharing points with anyone when the city is out of reach, otherwise the others
        will take them". No city on the board (lost or never rebuilt) = take shared canal chips."""
        return self.share_when_stuck and not gs.me().cities

    def _on_gold(self, gs: GameState, v: VKey) -> bool:
        return gs.mode.canal and any(h.terrain == "gold" for h in gs.board.vertex_hexes(v))

    def _defender_bonus(self, gs: GameState, g: GameState) -> float:
        """Owner 2026-10-04: always look for the defender VP -- the attack is held and our strength
        is the only top one. Ties give nobody the point (sim: 6 of 8 held attacks). Worth more as
        the barbarians close in and late in the game."""
        def top(x: GameState) -> int:
            me = x.me()
            mine = x.defence_strength(me)
            best = max((x.defence_strength(o) for o in x.opponents()), default=0)
            held = sum(x.defence_strength(p) for p in x.players) >= sum(len(p.cities) for p in x.players)
            return 2 if mine > best and held else (1 if mine > best else 0)
        gain = top(g) - top(gs)
        if not gain:
            return 0.0
        danger = gs.barbarian_pos / max(1, gs.barbarian_len)
        late = gs.me().vp_visible / max(1, gs.target)
        return 0.5 * self.defender_vp * gain * (0.5 + danger) * (1 + late)

    @staticmethod
    def _cuts_road(gs: GameState, me: PlayerState, v: VKey) -> bool:
        """v sits inside a rival's road of 4+ (a corner with 2 of their road edges)."""
        for p in gs.opponents():
            if p.longest_road >= 4 and sum(1 for e in p.roads | p.ships if v in e) >= 2:
                return True
        return False

    @staticmethod
    def _cut_target(gs: GameState, me: PlayerState, v: VKey) -> PlayerState | None:
        """A rival whose road v sits inside, when cutting it reduces their advantage: they hold
        the longest road, or their road is 4+ and they are level with us or ahead."""
        for p in gs.opponents():
            if sum(1 for e in p.roads | p.ships if v in e) < 2:
                continue
            if p.has_longest_road or (p.longest_road >= 4 and p.vp_visible >= me.vp_visible):
                return p
        return None

    def _cut_reach_value(self, gs: GameState, g: GameState) -> float:
        """Owner 2026-10-04: build a road to reach a rival's road with a knight when that cuts
        their advantage. Rewards a road/ship that opens a free corner where a knight would cut."""
        me0, me1 = gs.me(), g.me()
        reach0 = {v for e in me0.roads | me0.ships for v in e} | {v for v, _ in me0.buildings()}
        reach1 = {v for e in me1.roads | me1.ships for v in e}
        best = 0.0
        for v in reach1 - reach0:
            if gs.occupied(v) or not gs.board.is_land_vertex(v) or not mv.home_island(gs, v):
                continue
            p = self._cut_target(gs, me0, v)
            if p is not None:
                best = max(best, 0.02 if p.has_longest_road else 0.01)
            if self._cuts_focus(gs, v):
                best = max(best, 0.01) + 0.01   # selector BLOCK: the advancing rival's road first
        return best

    def _undug_canal_corner(self, gs: GameState, v: VKey) -> bool:
        return any(gs.board.is_canal(h) and h.coord not in gs.canal_dug for h in gs.board.vertex_hexes(v))

    def _contest_targets(self, gs: GameState) -> set:
        """Rival knights on undug canal corners that one of our knights could push off: the corner
        ends one of our roads and a stronger knight of ours is joined to it by our roads."""
        me = gs.me()
        adj: dict = {}
        for e in me.roads | me.ships:
            adj.setdefault(e[0], []).append(e[1])
            adj.setdefault(e[1], []).append(e[0])
        others = {v for p in gs.opponents() for v in list(p.knights) + [x for x, _ in p.buildings()]}
        out = set()
        for src, k in me.knights.items():
            seen, stack = {src}, [src]
            while stack:
                v = stack.pop()
                for w in adj.get(v, []):
                    if w not in seen and w not in others:
                        seen.add(w)
                        stack.append(w)
            ends = {w for v in seen for w in adj.get(v, [])}
            # a promotion can still make it stronger (mighty needs politics 3)
            could = k.level + 1 if k.level < 2 or (k.level < 3 and me.improvements.get("politics", 0) >= 3) else k.level
            for o in gs.opponents():
                for v, ok in o.knights.items():
                    if v in ends and ok.level < could and self._undug_canal_corner(gs, v):
                        out.add((v, o.name))
        return out

    def _contest_reach_value(self, gs: GameState, g: GameState) -> float:
        """A road that brings our knight's network up to a rival knight on an undug canal corner."""
        new = self._contest_targets(g) - self._contest_targets(gs)
        if not new:
            return 0.0
        threat = self.threat(gs)
        lead = any(threat is not None and name == threat.name for _, name in new)
        return self.canal_contest * (1.5 if lead else 1.0)

    def _canal_corner_reach(self, gs: GameState, p: PlayerState) -> set:
        """Free corners of undug canal tiles that p reaches with one more road (a knight spot)."""
        b = gs.board
        reach = {v for e in p.roads | p.ships for v in e} | {v for v, _ in p.buildings()}
        out, verts = set(), set(b.vertices)
        for h in b.hexes.values():
            if not b.is_canal(h) or h.coord in gs.canal_dug:
                continue
            for c in hex_corners(*h.coord):
                if c not in verts or c in out or gs.occupied(c) or not b.is_land_vertex(c):
                    continue
                if c in reach or any(not gs.edge_owner(e) and b.is_land_edge(e)
                                     and (e[0] if e[1] == c else e[1]) in reach for e in b.vertex_edges(c)):
                    out.add(c)
        return out

    def _canal_reach_value(self, gs: GameState, g: GameState) -> float:
        """Owner 2026-10-08: build roads/ships to where the canal points are. A new road/ship end on
        a free corner of an undug canal hex is a knight spot (knights are built at road and ship
        ends); most of all the second corner of a hex where our knight already waits."""
        me0, me1 = gs.me(), g.me()
        if len(me0.knights) >= 6:
            return 0.0                                   # every knight piece is on the board
        reach0 = {v for e in me0.roads | me0.ships for v in e}
        best = 0.0
        for v in {v for e in me1.roads | me1.ships for v in e} - reach0:
            if gs.occupied(v) or not gs.board.is_land_vertex(v) or not mv.home_island(gs, v):
                continue
            for h in gs.board.vertex_hexes(v):
                if gs.board.is_canal(h) and h.coord not in gs.canal_dug:
                    ours = any(c in me0.knights for c in hex_corners(*h.coord))
                    best = max(best, 0.03 if ours else 0.015)
        return best

    def _canal_front(self, gs: GameState, v: VKey) -> float:
        """Owner 2026-10-09 "it should go on the canal towards the other players to restrict how much
        of the canal they can get": 0 unless v is a free corner of an undug canal hex; then 0.4..1,
        higher the closer v is to a rival's piece (a corner we take there is one they lose)."""
        b = gs.board
        if gs.occupied(v) or not b.is_land_vertex(v):
            return 0.0
        if not any(b.is_canal(h) and h.coord not in gs.canal_dug for h in b.vertex_hexes(v)):
            return 0.0
        me = gs.me()
        theirs = {x for p in gs.players if p is not me
                  for x in [y for e in p.roads | p.ships for y in e] + [w for w, _ in p.buildings()] + list(p.knights)}
        frontier, seen = {v}, {v}
        for d in range(5):
            if frontier & theirs:
                return 1.0 - 0.15 * d
            nxt = set()
            for x in frontier:
                nxt |= b.neighbors(x) - seen
            seen |= nxt
            frontier = nxt
        return 0.4

    def _canal_deny_value(self, gs: GameState, g: GameState) -> float:
        """Owner 2026-10-05 game 15: Rhea added a road just to stop our knight reaching the
        canal. Do the same: reward a road that takes away a rival's last way to a free canal
        corner, most of all the advancing rival's."""
        f = self._focus_player(gs)
        v = 0.0
        for i, o0 in enumerate(gs.players):
            if o0 is gs.me():
                continue
            lost = len(self._canal_corner_reach(gs, o0) - self._canal_corner_reach(g, g.players[i]))
            v += 0.01 * lost * (2 if f is not None and f.name == o0.name else 1)
        return min(v, 0.03)

    def _cuts_focus(self, gs: GameState, v: VKey) -> bool:
        """Selector BLOCK: v sits inside the focus rival's road (2 of its edges, road 3+)."""
        f = self._focus_player(gs)
        return f is not None and f.longest_road >= 3 and sum(1 for e in f.roads | f.ships if v in e) >= 2

    def _ladder_value(self, gs: GameState, track: str, lvl: int) -> float:
        """Owner 2026-10-04: all tracks to 3 first (the lowest track first), then 4 and 5.
        Level 4 takes a free metropolis but a rival who reaches 5 first takes it away; level 5
        locks it. Going to 5 also takes a rival's metropolis that is still at 4."""
        me = gs.me()
        w = self.improve_ladder
        mine = me.improvements
        rivals = [o.improvements.get(track, 0) for o in gs.opponents()]
        top_rival = max(rivals, default=0)
        holder = gs.metropolis_owner.get(track)
        v = 0.0
        if lvl <= 3:
            if mine.get(track, 0) <= min(mine.get(t, 0) for t in IMPROVEMENT_TRACKS):
                v += 0.5 * w                                  # lift the lowest track first
        elif lvl == 4:
            if not holder or holder[0] == me.name:
                v += 0.5 * w * (1 + (top_rival >= 3))         # take it before a rival does
            else:
                v += 0.5 * w                                  # a step toward taking theirs at 5
        elif lvl == 5:
            if holder and holder[0] == me.name and top_rival >= 3:
                v += w * (1 + (top_rival >= 4))               # lock ours: a rival is closing in
            elif holder and holder[0] != me.name:
                owner = next((o for o in gs.opponents() if o.name == holder[0]), None)
                if owner is not None and owner.improvements.get(track, 0) < 5:
                    v += w                                    # take theirs: a 4-point swing
        return v

    def _new_commodity_value(self, gs: GameState, v: VKey) -> float:
        """A city that gives a commodity we don't produce yet opens another development track."""
        from .board import TERRAIN_COMMODITY
        b = gs.board
        have = {TERRAIN_COMMODITY[h.terrain] for c in gs.me().cities for h in b.vertex_hexes(c)
                if h.terrain in TERRAIN_COMMODITY and h.number}
        new = {TERRAIN_COMMODITY[h.terrain] for h in b.vertex_hexes(v)
               if h.terrain in TERRAIN_COMMODITY and h.number} - have
        return 0.5 * self.improve_ladder * len(new)

    def _gold_distance(self, gs: GameState, me: PlayerState, cap: int = 6) -> int:
        """Edges (land or sea, not owned by others) from our network to a free gold corner."""
        b = gs.board
        gold = {v for v in self._gold_corners(gs) if not gs.occupied(v)}
        if not gold:
            return cap
        frontier = {v for e in list(me.roads) + list(me.ships) for v in e} | {v for v, _ in me.buildings()}
        seen = set(frontier)
        for d in range(cap):
            if frontier & gold:
                return d
            nxt = set()
            for u in frontier:
                o = gs.owner_at(u)
                if o is not None and o is not me:
                    continue
                for w in b.neighbors(u):
                    e = ekey(u, w)
                    if w in seen or gs.edge_owner(e):
                        continue
                    nxt.add(w)
            seen |= nxt
            frontier = nxt
        return cap

    def _gold_route_value(self, gs: GameState, g: GameState) -> float:
        """No gold knight yet, canal open: a road/ship that brings a free gold corner closer."""
        me0 = gs.me()
        if gs.canal_progress >= gs.canal_total or any(self._on_gold(gs, v) for v in me0.knights):
            return 0.0
        d0, d1 = self._gold_distance(gs, me0), self._gold_distance(g, g.me())
        return self.gold_route * (d0 - d1) if d1 < d0 else 0.0

    def knight_plan(self, gs: GameState, me: PlayerState) -> tuple[str, str | None] | None:
        """Owner 2026-10-02 knight ladder: one knight, a back-up (opponents hold Deserter and
        Intrigue), promote one to strong, Politics to 3 (needed for mighty), then mighty and
        more knights. A knight parked on a gold corner pays grain but is not a defender, and it
        still uses one of the 2 basic-knight pieces."""
        if not gs.mode.ck:
            return None
        field = {v: k for v, k in me.knights.items() if not self._on_gold(gs, v)}
        if len(field) < 2:
            return ("knight", None)
        if all(k.level < 2 for k in field.values()):
            return ("promote", None)
        if me.improvements["politics"] < 3:
            return ("improve", "politics") if me.cities else ("city", None)
        if all(k.level < 3 for k in field.values()):
            return ("promote", None)
        return ("knight", None)

    def _prospector_value(self, gs: GameState, me: PlayerState, v: VKey) -> float:
        """Great Canal (Almanac 3), a knight on a small-island corner, active or not:
        gold digger -- any 1 card on every roll of 8 while the canal is open, at most 1 card per
        player per roll however many gold knights it has; merchant -- a harbour corner of a
        small island gives that harbour's rate (kept after the canal is complete). A knight on a
        home-island harbour corner gives nothing."""
        if not gs.mode.canal:
            return 0.0
        b = gs.board
        s = 0.0
        held = set(me.knights)
        gold_now = any(h.terrain == "gold" and h.coord != b.robber
                       for k in held for h in b.vertex_hexes(k))
        if gs.canal_progress < gs.canal_total and not gold_now and any(
                h.terrain == "gold" and h.coord != b.robber for h in b.vertex_hexes(v)):
            # owner: the gold knight is grain when nobody else has any
            s += 0.010 * PIPS[8]
        hb = b.harbor_at(v) if b.is_small_island_vertex(v) else None
        if hb:
            ports = {b.harbor_at(x) for x, _ in me.buildings()} | \
                    {b.harbor_at(x) for x in held if b.is_small_island_vertex(x)}
            if hb == "3:1" and "3:1" not in ports:
                s += 0.04           # owner rule: 4:1 cripples the canal game
            elif hb not in ports and hb != "3:1" and hb != "grain":
                inc = gs.income(me).get(hb, 0)
                s += 0.01 + 0.05 * inc
        return s

    def _ship_move_value(self, gs: GameState, g: GameState) -> float:
        """Owner 2026-10-09 "build into the game the way to move ships": move the open-end ship of a
        line that leads nowhere to an end that reaches something -- a small-island gold or harbour
        corner a knight can sail to, a free corner of an undug canal hex, a settle spot (the road
        value adds that). What the old end reached is subtracted, so a move never gives up a
        corner, a spot or the gold route; with nothing gained it is not made. (moves.ship_moves
        already only offers the last ship of an open line: no connection is broken.)"""
        me0, me1 = gs.me(), g.me()
        b = gs.board

        def reach(p, gg):
            ends = {v for e in p.ships for v in e}
            return max((self._prospector_value(gg, p, v) for v in ends
                        if b.is_small_island_vertex(v) and not gg.occupied(v)), default=0.0)
        v = 0.5 * (reach(me1, g) - reach(me0, gs))
        if gs.mode.canal and self.canal_hunt and gs.canal_progress < gs.canal_total:
            # canal corners the new end reaches, minus the ones only the old end reached
            v += self._canal_reach_value(gs, g) - self._canal_reach_value(g, gs)
        d0, d1 = self._spot_distance(gs, me0), self._spot_distance(g, me1)
        if d1 > d0 and d0 < 5:
            v -= 0.06 * (d1 - d0)                 # the old end was nearer a settle spot
        if self.gold_route and gs.mode.canal:
            g0, g1 = self._gold_distance(gs, me0), self._gold_distance(g, me1)
            if g1 > g0:
                v -= self.gold_route * (g1 - g0)  # the old end was nearer free gold
        if me1.longest_road < me0.longest_road and me0.longest_road >= 5:
            v -= 0.02                             # the move shortens a road worth points
        return v if v > 0 else v - 0.001

    # owner 2026-10-08 "the roads on the right side, going down, were a waste": a road only
    # counts as progress toward a spot worth a settlement -- 7+ pips (two real numbers, or one
    # number plus the canal 3:1 harbour), not a lone coast corner
    ROAD_SPOT_MIN = 7

    def _road_target(self, gs: GameState, me, v: VKey) -> bool:
        """An open, legal spot worth a road: enough pips, and no rival road already touching it
        (they settle it before our road gets there)."""
        if not (gs.can_settle(v) and not gs.occupied(v)) or self._raw_spot(gs, v) < self.ROAD_SPOT_MIN:
            return False
        shut = gs.mode.canal and gs.canal_progress < gs.canal_total
        if producing_hexes(gs, v) <= 1 and not self._hex_must_have(
                gs, v, me, grain2=not (self.road_skip_farmland and shut)):
            return False          # owner 2026-10-09: a one-hex corner is no settlement goal
        return not any(v in e for p in gs.players if p is not me for e in list(p.roads) + list(p.ships))

    def _spot_distance(self, gs: GameState, me) -> int:
        """Road steps from my network to the nearest open spot worth settling (_road_target)."""
        b = gs.board
        frontier = {v for e in list(me.roads) + list(me.ships) for v in e} | {v for v, _ in me.buildings()}
        seen = set(frontier)
        for d in range(5):
            for v in frontier:
                if self._road_target(gs, me, v):
                    return d
            nxt = set()
            for v in frontier:
                o = gs.owner_at(v)
                if o is not None and o is not me:
                    continue          # can't build through an opponent's building
                nxt |= b.neighbors(v) - seen
            seen |= nxt
            frontier = nxt
        return 5

    def _road_value(self, gs: GameState, g: GameState) -> float:
        """Roads earn nothing in my_prob, so greedy play never built them and games
        stalled once spots ran out. Value progress toward the next settlement spot."""
        me0, me1 = gs.me(), g.me()
        m = gs.mode
        if len(me0.settlements) + len(me0.cities) >= m.max_settlements + m.max_cities:
            return 0.0
        if len(me0.settlements) >= m.max_settlements:
            return 0.0
        d0, d1 = self._spot_distance(gs, me0), self._spot_distance(g, me1)
        if d0 == 0 or d0 >= 5:
            return 0.0                # a spot ready: save for it; none worth it in reach: no road
        # must beat the ~0.03-0.04 my_prob drop from spending the cards (seen in greater_catan)
        return 0.06 * (d0 - d1) if d1 < d0 else 0.0

    def _seen_into(self, gs: GameState) -> None:
        """The bank dialog's ratios live on the state, so moves/progress_cards see them too."""
        if self.ratio_seen and getattr(gs, "ratio_seen", None) is not self.ratio_seen:
            gs.ratio_seen = self.ratio_seen   # the same dict: later reads and pops show at once

    def ratio_source(self, gs: GameState, res: str) -> tuple[int, str, bool]:
        """(ratio, where it comes from, certain) -- moves.ratio_source, the one source of truth."""
        self._seen_into(gs)
        if self.merchant_res and res == self.merchant_res and self.fleet_resource != res \
                and res not in (getattr(gs, "ratio_seen", None) or {}):
            return 2, "merchant", True        # owner 2026-10-06: Merchant placed -> 2:1 on its resource
        return mv.ratio_source(gs, res, self.harbor_3to1, self.fleet_resource)

    def bank_ratio(self, gs: GameState, res: str) -> int:
        # live 2026-10-04 / 2026-10-08 21:54: the dialog said 4:1 where the board read guessed 3:1
        return self.ratio_source(gs, res)[0]

    def trade_routes(self, gs: GameState, res: str, hand: dict | None = None) -> list[dict]:
        """Every way to get one `res` now or soon (moves.trade_routes), best first."""
        self._seen_into(gs)
        routes = mv.trade_routes(gs, res, hand, self.harbor_3to1, self.fleet_resource)
        if self.merchant_res and self.merchant_res != res:
            h = dict(gs.me().hand if hand is None else hand)
            if h.get(self.merchant_res, 0) >= 2 and not any(r["give"] == {self.merchant_res: 2} for r in routes):
                routes.insert(0, {"how": "merchant", "give": {self.merchant_res: 2}, "ratio": 2,
                                  "now": True, "certain": True})
        return routes

    def reserved_commodities(self, gs: GameState) -> set:
        """Stage 2026-10-09 07:32:59: commodities of the tracks whose metropolis we hold at level 4.
        A rival at 5 takes it; ours at 5 locks it. Never given to the bank, a harbour or a player."""
        if not self.lock_metropolis_commodity or not gs.mode.ck:
            return set()
        me = gs.me()
        if not me.cities:
            return set()
        return {com for track, com in IMPROVEMENT_TRACKS.items()
                if me.improvements.get(track, 0) == 4
                and (gs.metropolis_owner.get(track) or (None,))[0] == me.name}

    def bank_trades_for(self, gs: GameState, cost: dict) -> list[Action] | None:
        """Bank/harbor trades that make `cost` affordable, or None if impossible."""
        me = gs.me()
        hand = dict(me.hand)
        need = missing(hand, cost)
        if not need:
            return []
        trades = []
        val = self.resource_value(gs, me)
        locked = self.reserved_commodities(gs)
        for res, n in need.items():
            for _ in range(n):
                options = []
                for give in list(RESOURCES) + list(COMMODITIES):
                    if give in locked:
                        continue
                    ratio = self.bank_ratio(gs, give)
                    spare = hand.get(give, 0) - cost.get(give, 0)
                    if give != res and spare >= ratio:
                        options.append((ratio * val.get(give, 1.0), give, ratio))
                if not options:
                    return None
                _, give, ratio = min(options)
                hand[give] -= ratio
                hand[res] = hand.get(res, 0) + 1
                trades.append(Action("bank_trade", give={give: ratio}, get={res: 1}))
        return trades

    def _outlook_bonus(self, gs: GameState, g: GameState, a: Action) -> float:
        """Owner 2026-10-02: plan 3 moves ahead, stop the others scoring, score ourselves.
        +goal: the action is (or builds toward) our next planned scoring move.
        +block: it takes the spot / road / track an opponent is about to score with."""
        o = self.outlook
        if o is None:
            return 0.0
        bonus = 0.0
        if o.goals:
            g0 = o.goals[0]
            if a.kind == g0.kind and (g0.target is None or a.target == g0.target):
                bonus += 0.02
            elif a.kind == "road" and g0.kind == "settlement" and a.target in getattr(g0, "path", []):
                bonus += 0.012
            elif a.kind == "road" and g0.kind == "longest_road":
                bonus += 0.008
            elif a.kind in ("knight", "activate") and g0.kind in ("canal_chip", "defender"):
                bonus += 0.01
            elif a.kind == "improve" and g0.kind == "metropolis" and a.target == g0.target:
                bonus += 0.015
        b = o.blocks
        bonus += 0.02 * min(1.0, b.get((a.kind, a.target), 0.0))
        if a.kind == "settlement":
            # a settlement on a rival's road path cuts it too
            bonus += 0.01 * min(1.0, sum(v for (k, t), v in b.items() if k == "road" and t and a.target in t))
        if a.kind in ("road", "ship") and ("longest_road", None) in b:
            me0, me1 = gs.me(), g.me()
            if me1.longest_road > me0.longest_road:
                bonus += 0.01 * min(1.0, b[("longest_road", None)])
        if a.kind in ("knight", "activate", "promote") and ("knight", None) in b:
            bonus += 0.006 * min(1.0, b[("knight", None)])
        return bonus

    def plan_turn(self, gs: GameState, max_steps: int = 12) -> list[Action]:
        """Greedy plan: keep doing the best-value action (with bank trades if needed)."""
        if self.situational:
            self.situational_begin(gs)
            try:
                return self._plan_turn(gs, max_steps)
            finally:
                self.situational_end()
        return self._plan_turn(gs, max_steps)

    def _plan_turn(self, gs: GameState, max_steps: int = 12) -> list[Action]:
        plan: list[Action] = []
        g = gs
        pre = self._grain_first_trade(g)
        if pre:
            self.note(f"opening: bank {pre.give} -> 1 grain for the knight")
            plan.append(pre)
            g = self._apply(g, pre)
        big_started = False
        for _ in range(max_steps):
            forced = self._knight_first_step(g) if self.knight_first else None
            if forced:
                for a in forced:
                    g = self._apply(g, a)
                    plan.append(a)
                continue
            hold = self.knight_first and self._knight_short(g)
            base = self.my_prob(g)
            best, best_v, best_seq = None, base + 1e-4, None
            # owner lesson 2026-10-02: no trades -> use the cards before the robber takes them.
            # Near the discard limit, any build beats holding (a 7 costs half the hand).
            me_ = g.me()
            limit = discard_limit(g.mode, me_)
            n_hand = sum(me_.hand.values())
            slack = 0.0
            if n_hand >= limit - 1:
                slack = 0.06
            elif self.limit_lookahead and g.mode.ck and \
                    n_hand + g.income_total(me_) * (len(g.players) - 1) > limit:
                slack = 0.03                  # the cards due before our next turn pass the limit
            near_limit = slack > 0             # a 7 would cost cards: the only case for a wall before the knight
            if self.spend_exposed and g.mode.ck and n_hand > 4:
                # owner: held cards are robber, spy and monopoly food -- a build off the plan
                # that saves them can be worth a little probability now
                slack = max(slack, self.spend_exposed * (n_hand - 4))
            if slack:
                best_v = base - slack
            if hold and near_limit:
                # live 2026-10-08 21:55: near the limit with the knight held, a 2nd knight
                # (cities > knights) comes first, then a wall, then the best other build below
                cands = self.candidate_actions(g)
                pick = None
                for pk in ("knight", "wall"):
                    opts = [a for a in cands if a.kind == pk and self._held_ok(g, [a], True)]
                    if opts:
                        pick = max(opts, key=lambda x: self.action_value(g, x))
                        break
                if pick is not None:
                    self.note(f"hand {n_hand} near limit {limit}: {pick.kind} while the knight waits")
                    g = self._apply(g, pick)
                    plan.append(pick)
                    continue
            # direct actions
            cands = self.candidate_actions(g)
            settle_now = self._settle_window(g) and any(
                a.kind == "settlement" and self.action_value(g, a) > best_v for a in cands)
            for a in cands:
                if hold and not self._held_ok(g, [a], near_limit):
                    continue
                if settle_now and a.kind in ("activate", "knight", "promote"):
                    continue                  # stage 2026-10-09: ship far, the settlement before the knight
                v = self.action_value(g, a) - self._city_cost(g, [a])
                if v > best_v:
                    best, best_v, best_seq = a, v, [a]
            # trade-enabled builds
            for kind in self._trade_targets(g):
                if hold and kind not in ("wall",) and not (kind == "settlement" and self._settle_window(g)):
                    continue                  # nothing but the knight is worth banking for
                g2_hand = dict(g.me().hand)
                trades = self.bank_trades_for(g, COSTS[kind])
                if not trades:
                    continue
                if hold and kind == "settlement" and not self._settle_trades_ok(trades):
                    continue                  # review: never the knight's cards or commodities
                ikeep = self._improve_keep(g)
                if ikeep and any(g.me().hand.get(c, 0) - sum(t.give.get(c, 0) for t in trades) < n
                                 for c, n in ikeep.items()):
                    continue                  # the improvement comes first; trade next round
                gt = g
                for t in trades:
                    gt = self._apply(gt, t)
                for a in self.candidate_actions(gt):
                    if a.kind != kind:
                        continue
                    v = self.action_value(gt, a) - 0.004 * len(trades) - self._city_cost(g, trades + [a])
                    if v > best_v:
                        best, best_v, best_seq = a, v, trades + [a]
                _ = g2_hand
            # owner 03:23: 3 paper + a 3:1 set of wood = the 4th paper and science 4 -- bank toward
            # an improvement level too, not only toward the board builds
            if g.mode.ck and g.me().cities:
                for track in IMPROVEMENT_TRACKS:
                    lvl = g.me().improvements.get(track, 0)
                    if lvl >= 5:
                        continue
                    trades = self.bank_trades_for(g, improvement_cost(track, lvl))
                    if not trades:
                        continue
                    gt = g
                    for t in trades:
                        gt = self._apply(gt, t)
                    a = Action("improve", track)
                    if a not in self.candidate_actions(gt):
                        continue
                    if hold and not self._held_ok(g, trades + [a], near_limit):
                        continue
                    v = self.action_value(gt, a) - 0.004 * len(trades)
                    if v > best_v:
                        best, best_v, best_seq = a, v, trades + [a]
            if self.road_settle_lookahead:
                rs = self._road_settle_seq(g)
                if rs and self.city_rebuild_w:
                    rs = (rs[0] - self._city_cost(g, rs[1]), rs[1])
                if rs and rs[0] > best_v and (not hold or self._held_ok(g, rs[1], near_limit)):
                    best, best_v, best_seq = rs[1][-1], rs[0], rs[1]
            if self.city_rebuild(g):
                # owner 2026-10-10: after a lost city the rebuild beats ships, roads and digs --
                # bank spare cards toward it before they go on a ~0 build
                t = self._city_save_trade(g)
                tv = base + 0.5 * self.city_rebuild_w - 0.004
                if t is not None and tv > best_v and (not hold or self._held_ok(g, [t], near_limit)):
                    self.note(f"save toward the city: bank {t.give} -> {t.get}")
                    best, best_v, best_seq = t, tv, [t]
            if self.settle_big_hand and (best is None or best.kind not in self.BIG_HAND_FIRST):
                seq = self._big_hand_settle(g, hold, near_limit, base - max(slack, 0.06), big_started)
                if seq:
                    big_started = seq[-1].kind == "road"
                    self.note(f"big hand ({n_hand} cards): {' + '.join(a.kind for a in seq)} "
                              f"toward {seq[-1].target} before {best.kind if best else 'nothing'}")
                    best_seq = seq
            if not best_seq:
                break
            for a in best_seq:
                g = self._apply(g, a)
                plan.append(a)
        if self.spend_turn:
            plan += self._spend_toward_target(g)
        return plan

    KNIGHT_CARDS = ("wool", "ore", "grain")

    def _knight_gap(self, gs: GameState) -> bool:
        """A city stands with no active knight for it, and a knight can still help."""
        if not gs.mode.ck or self.defence_gap(gs, gs.me()) <= 0:
            return False
        me = gs.me()
        if any(not k.active for k in me.knights.values()):
            return True                       # an idle knight waits for its grain
        return any(a.kind == "knight" for a in self._knight_spots(gs))

    def _knight_spots(self, gs: GameState) -> list[Action]:
        g = gs
        if missing(dict(gs.me().hand), COSTS["knight"]):
            g = copy.deepcopy(gs)              # spot check only: pretend the cards are there
            for r, n in COSTS["knight"].items():
                g.me().hand[r] = max(g.me().hand.get(r, 0), n)
        return [a for a in self.candidate_actions(g) if a.kind == "knight"]

    def _knight_need(self, gs: GameState) -> tuple[str, dict]:
        idle = any(not k.active for k in gs.me().knights.values())
        return ("activate", dict(COSTS["activate"])) if idle else ("knight", dict(COSTS["knight"]))

    def _settle_window(self, gs: GameState) -> bool:
        """Barbarians far and no city in danger, with a settlement spot we can build on."""
        if not self.settle_when_far or not gs.mode.ck:
            return False
        from .progress_cards import barbarians_near
        if barbarians_near(gs) or self.city_in_danger(gs):
            return False
        g = gs
        if missing(dict(gs.me().hand), COSTS["settlement"]):
            g = copy.deepcopy(gs)              # spot check only: pretend the cards are there
            for r, n in COSTS["settlement"].items():
                g.me().hand[r] = max(g.me().hand.get(r, 0), n)
        return any(a.kind == "settlement" for a in self.candidate_actions(g))

    def _settle_trades_ok(self, trades) -> bool:
        """Toward a settlement while the knight waits: no knight cards, no commodities banked."""
        return all(not t.give.get(c) for t in trades for c in self.KNIGHT_CARDS + tuple(COMMODITIES))

    # builds the big-hand settlement waits for: the owner's city first, the knight's needs
    BIG_HAND_FIRST = ("city", "settlement", "knight", "activate")

    def _big_hand(self, gs: GameState) -> bool:
        """Over the discard limit, or city + road + settlement paid from the hand outright."""
        me = gs.me()
        if sum(me.hand.values()) > discard_limit(gs.mode, me):
            return True
        both: dict = {}
        for k in ("city", "road", "settlement"):
            for r, n in COSTS[k].items():
                both[r] = both.get(r, 0) + n
        return bool(me.settlements) and can_afford(me.hand, both)

    def _big_hand_trades(self, gs: GameState, cost: dict) -> list[Action] | None:
        trades = self.bank_trades_for(gs, cost)
        if trades is None or (gs.mode.canal and any(t.give.get("grain") for t in trades)):
            return None                       # grain is never banked away in the Great Canal
        return trades

    def _big_hand_settle(self, gs: GameState, hold: bool, near_limit: bool,
                         floor: float, started: bool = False) -> list[Action] | None:
        """Owner 2026-10-09 21:08 (settle_big_hand): a big hand builds the settlement on the best
        open spot (bank trades included), or the road toward a good spot (_road_target) at most
        settle_big_hand_reach roads away when the hand also pays those roads and the settlement."""
        me = gs.me()
        m = gs.mode
        if len(me.settlements) >= m.max_settlements or not (started or self._big_hand(gs)):
            return None                       # started: a road this turn already heads for the spot

        def ok(seq):
            return not hold or self._held_ok(gs, seq, near_limit)

        if gs.legal_settlement_spots(me):
            trades = self._big_hand_trades(gs, COSTS["settlement"])
            if trades is None:
                return None
            gt = gs
            for t in trades:
                gt = self._apply(gt, t)
            opts = [(self.action_value(gt, a), a) for a in self.candidate_actions(gt) if a.kind == "settlement"]
            opts = [(v, a) for v, a in opts if v > floor and ok(trades + [a])]
            if not opts:
                return None
            return trades + [max(opts, key=lambda x: x[0])[1]]
        d0 = self._spot_distance(gs, me)
        if not 1 <= d0 <= self.settle_big_hand_reach:
            return None                       # owner: no roads toward weak or far spots
        total = dict(COSTS["settlement"])
        for r, n in COSTS["road"].items():
            total[r] = total.get(r, 0) + n * d0
        if self._big_hand_trades(gs, total) is None:
            return None                       # the roads must leave the settlement paid
        trades = self._big_hand_trades(gs, COSTS["road"])
        if trades is None:
            return None
        gt = gs
        for t in trades:
            gt = self._apply(gt, t)
        opts = []
        for a in self.candidate_actions(gt):
            if a.kind != "road" or not ok(trades + [a]):
                continue
            g2 = self._apply(gt, a)
            if self._spot_distance(g2, g2.me()) < d0:
                v = self.action_value(gt, a)
                if v > floor:
                    opts.append((v, a))
        if not opts:
            return None
        return trades + [max(opts, key=lambda x: x[0])[1]]

    def _road_settle_seq(self, gs: GameState) -> tuple[float, list[Action]] | None:
        """Owner 2026-10-09 (road_settle_lookahead): an open good spot (_road_target) one road
        away and the hand (bank trades included, never grain in the canal) pays road + settlement:
        the pair is one move, valued as the settlement. Greedy one-step play valued the lone road
        at +0.06 and a wall beat it once the canal was dug and its road bonuses stopped."""
        me = gs.me()
        if len(me.settlements) >= gs.mode.max_settlements or gs.legal_settlement_spots(me) \
                or self._spot_distance(gs, me) != 1:
            return None
        total = dict(COSTS["settlement"])
        for r, n in COSTS["road"].items():
            total[r] = total.get(r, 0) + n
        trades = self._big_hand_trades(gs, total)
        if trades is None:
            return None
        gt = gs
        for t in trades:
            gt = self._apply(gt, t)
        best = None
        for a in self.candidate_actions(gt):
            if a.kind != "road":
                continue
            g2 = self._apply(gt, a)
            for s in self.candidate_actions(g2):
                if s.kind != "settlement" or a.target and s.target not in a.target:
                    continue
                v = self.action_value(g2, s) - 0.004 * len(trades)
                if best is None or v > best[0]:
                    best = (v, trades + [a, s])
        return best

    def _knight_first_step(self, gs: GameState) -> list[Action] | None:
        """Owner 2026-10-08: knight first. With a city uncovered, bank toward the knight (or its
        activation) and build it before anything else, if the cards can be found this turn."""
        if not self._knight_gap(gs):
            return None
        from .progress_cards import barbarians_near
        me = gs.me()
        if not barbarians_near(gs) and me.settlements and can_afford(me.hand, COSTS["city"]):
            return None                       # barbarians far: a city we can pay for outright goes first
        if self._settle_window(gs):
            st = self.bank_trades_for(gs, COSTS["settlement"])
            if st is not None and self._settle_trades_ok(st):
                return None                   # ship far: the settlement takes the grain first
        kind, cost = self._knight_need(gs)
        trades = self.bank_trades_for(gs, cost)
        if trades is None:
            if barbarians_near(gs) or self.city_in_danger(gs):
                # owner 2026-10-08 "know all trade routes it can do at a specific time"
                for r in missing(dict(me.hand), cost):
                    routes = self.trade_routes(gs, r)
                    self.note(f"save the city: no bank route to {r} for the {kind}; routes: "
                              + ("; ".join(f"{x['how']} {x['give']}" + ("" if x["now"] else " (later)")
                                           for x in routes) or "none"))
            return None
        if gs.mode.canal and any(t.give.get("grain") for t in trades):
            return None                       # grain is never banked away in the Great Canal
        if not barbarians_near(gs) and any(t.give.get(c) for t in trades for c in COMMODITIES):
            return None                       # owner 03:18: with time left, commodities go to improvements
        gt = gs
        for t in trades:
            gt = self._apply(gt, t)
        opts = [a for a in self.candidate_actions(gt) if a.kind == kind]
        if not opts:
            return None
        a = max(opts, key=lambda x: self.action_value(gt, x))
        self.note(f"knight first: {' + '.join(f'bank {t.give} -> {t.get}' for t in trades) + ' + ' if trades else ''}{kind}")
        return trades + [a]

    @classmethod
    def city_in_danger(cls, gs: GameState) -> bool:
        """Stage 2026-10-09: the only city fell at 7/7 with one idle knight. A city is in danger
        when the defence is short (or would lose a tie) and the ship is at least 3 steps out."""
        if not gs.mode.ck:
            return False
        me = gs.me()
        if not [v for v in me.cities if v not in me.metropolises]:
            return False
        if gs.barbarian_pos < max(1, gs.barbarian_len - 4):
            return False
        return cls.defence_gap(gs, me) > 0 or cls.barb_loss_risk(gs)

    def save_city_offer(self, gs: GameState, made: set | frozenset = frozenset(),
                        order: list | tuple | None = None) -> Action | None:
        """Stage 2026-10-09: "routes: player trade" was only written to the log. With the city in
        danger and no bank route, ask the table for the missing knight card: 2 spare cards for 1,
        never grain, no commodities, never a card the knight itself needs."""
        if not self.city_in_danger(gs):
            return None
        me = gs.me()
        kind, cost = self._knight_need(gs)
        if kind == "knight" and len(me.knights) > len(me.cities):
            return None
        need = missing(dict(me.hand), cost)
        if not need:
            return None
        bt = self.bank_trades_for(gs, cost)
        if bt is not None and not (gs.mode.canal and any(t.give.get("grain") for t in bt)):
            return None                       # the bank does it this turn (knight first step)
        want = next((r for r in sorted(need, key=lambda r: r != "grain") if ("want", r) not in made), None)
        if want is None:
            return None
        keep = dict(COSTS["knight"])
        for r, n in COSTS["activate"].items():
            keep[r] = keep.get(r, 0) + n
        val = self.resource_value(gs, me)
        names = [r for r in (order or list(RESOURCES)) if r in RESOURCES]
        sets = self.grain_set_cards(gs) if want != "grain" else set()
        spare = sorted((r for r in names if r != "grain" and r != want and r not in keep
                        and r not in sets and me.hand.get(r, 0) > 0),
                       key=lambda r: (val.get(r, 1.0), -me.hand.get(r, 0)))
        give: dict = {}
        for r in spare:
            while sum(give.values()) < 2 and me.hand.get(r, 0) > give.get(r, 0):
                give[r] = give.get(r, 0) + 1
        if not give:
            # only knight cards to spare: an ore beyond the knight's own still buys the grain
            for r in ("ore", "wool"):
                extra = me.hand.get(r, 0) - keep.get(r, 0)
                while extra > 0 and sum(give.values()) < 2 and r != want:
                    give[r] = give.get(r, 0) + 1
                    extra -= 1
        if not give:
            return None
        self.note(f"save the city: ship at {gs.barbarian_pos}/{gs.barbarian_len}, {kind} short of {want}: "
                  f"player offer {give} -> {{'{want}': 1}}")
        return Action("player_trade", give=give, get={want: 1})

    def _knight_short(self, gs: GameState) -> bool:
        """The knight is needed but can't be made this turn: hold its cards."""
        return self._knight_gap(gs)

    def _held_ok(self, gs: GameState, seq: list[Action], near_limit: bool) -> bool:
        """While the knight waits: no road/ship/settlement/dev, nothing that spends wool,
        ore or grain; a city only while the barbarians are far; a wall only when the hand nears
        the limit; improvements on commodities are fine."""
        need, need_cost = self._knight_need(gs)
        danger = self.city_in_danger(gs)
        if near_limit:
            # live 2026-10-08 21:55: 12 cards over an 11 limit, knight idle with no grain to be
            # had -> every build was held and the turn ended with plan=[]. Near the limit a 7
            # costs half the hand: spend, keeping only the cards the held knight needs. A 2nd
            # knight (cities > knights) is fine even while the 1st waits for its grain.
            me = gs.me()
            want = dict(need_cost)
            if danger:
                # stage 2026-10-09 00:37: near the limit with the ship coming, the wool went into
                # a ship -- with the city in danger the whole knight set (wool, ore, grain) stays
                for r in self.KNIGHT_CARDS:
                    want[r] = max(want.get(r, 0), 1)
            keep = {r: min(me.hand.get(r, 0), n) for r, n in want.items()}
            hand = dict(me.hand)
            for a in seq:
                if a.kind == "bank_trade":
                    for r, n in a.give.items():
                        hand[r] = hand.get(r, 0) - n
                    for r, n in a.get.items():
                        hand[r] = hand.get(r, 0) + n
                    continue
                if a.kind == "knight" and need == "activate" and len(me.knights) >= len(me.cities):
                    # owner 03:13: no 3rd knight while one is idle -- but with the city in danger a
                    # 2nd knight (one more than the cities) beats spending its cards elsewhere
                    if not (danger and len(me.knights) <= len(me.cities)):
                        return False
                if a.kind == "knight" and danger:
                    for r, n in COSTS["knight"].items():
                        keep[r] = max(0, keep.get(r, 0) - n)     # the cards were kept for it
                if a.kind == "improve":
                    continue                  # commodities only; checked by candidate_actions
                for r, n in COSTS.get(a.kind, {}).items():
                    hand[r] = hand.get(r, 0) - n
            return all(hand.get(r, 0) >= n for r, n in keep.items())
        for a in seq:
            if a.kind == "activate" or (a.kind == "knight" and need == "knight"):
                continue                      # owner 03:13: no 2nd knight while the 1st is idle
            if a.kind == "bank_trade":
                if any(a.give.get(r) for r in self.KNIGHT_CARDS):
                    return False
                if any(x.kind == "settlement" for x in seq) and not self._settle_trades_ok([a]):
                    return False              # commodities stay for the improvements
                continue
            if a.kind == "wall":
                if not near_limit:
                    return False
                continue
            if a.kind == "improve":
                continue
            if a.kind == "settlement" and self._settle_window(gs):
                continue                      # stage 2026-10-09: ship far, a settlement may go first
            if a.kind == "city":
                # a city still beats waiting while the barbarians are far (the owner's worry
                # was ships and roads); with them close, the knight's ore comes first
                from .progress_cards import barbarians_near
                if barbarians_near(gs):
                    return False
                continue
            return False
        return True

    def _spend_toward_target(self, gs: GameState, max_trades: int = 3) -> list[Action]:
        """Owner 2026-10-03: end the turn with the cards used. Nothing more is buildable, so
        bank spare cards (ones the target doesn't use) into the target's missing cards until
        no bank trade is left. Grain is never banked away in the Great Canal."""
        out: list[Action] = []
        g = gs
        for _ in range(max_trades):
            me = g.me()
            hand = dict(me.hand)
            kind = None
            targets = self._trade_targets(g)
            if self.knight_first and self._knight_gap(g):
                targets = [self._knight_need(g)[0]]      # owner 2026-10-08: knight first
                if self._settle_window(g):
                    targets = ["settlement"] + targets   # stage 2026-10-09: ship far, settle first
            for k in targets:
                need = missing(hand, COSTS.get(k, {}))
                if not need:
                    continue
                # owner 2026-10-06 "put the merchant on wood (5 of them), then kept all the wood": a
                # 2:1 card (Merchant / Fleet / 2:1 harbour) may buy toward a target up to 1 card
                # short, and every target is tried, not only the first
                cheap = sum((hand.get(r, 0) - COSTS[k].get(r, 0)) // 2 for r in RESOURCES
                            if r not in need and not (g.mode.canal and r == "grain")
                            and self.bank_ratio(g, r) == 2 and hand.get(r, 0) - COSTS[k].get(r, 0) >= 2)
                cap = self.spend_need_cap
                if k == "city" and self.city_rebuild(g):
                    cap = sum(COSTS["city"].values())   # owner 2026-10-10: save toward the rebuild
                elif k == "city" and self.city_goal(g) is not None:
                    cap += 1                            # owner 2026-10-10: the city stays a goal
                if sum(need.values()) <= cap or (cheap and sum(need.values()) - cheap <= 1):
                    kind = k
                    break
            if kind is None:
                break
            cost = COSTS[kind]
            need = missing(hand, cost)
            val = self.resource_value(g, me)
            best = None
            for give in RESOURCES:
                if give in need or (g.mode.canal and give == "grain"):
                    continue
                ratio = self.bank_ratio(g, give)
                if hand.get(give, 0) - cost.get(give, 0) < ratio:
                    continue
                for res in need:
                    gain = val.get(res, 1.0) - ratio * val.get(give, 1.0) * 0.5
                    if best is None or gain > best[0]:
                        best = (gain, give, ratio, res)
            if best is None:
                break
            _, give, ratio, res = best
            a = Action("bank_trade", give={give: ratio}, get={res: 1})
            self.note(f"spend the turn: bank {ratio} {give} -> {res} toward the {kind}")
            out.append(a)
            g = self._apply(g, a)
        return out

    def plan_missing(self, gs: GameState) -> dict[str, int]:
        """Cards still missing for the nearest build we want (first target wins ties)."""
        me = gs.me()
        best = None
        for i, kind in enumerate(self._trade_targets(gs)):
            cost = COSTS.get(kind, {})
            if not cost:
                continue
            miss = {r: n - me.hand.get(r, 0) for r, n in cost.items() if n > me.hand.get(r, 0)}
            key = (sum(miss.values()), i)
            if best is None or key < best[0]:
                best = (key, miss)
        if self.city_rebuild(gs):
            # owner 2026-10-10: the rebuild's cards first (gold knight pick, Merchant, Monopoly)
            miss = missing(dict(me.hand), COSTS["city"])
            if miss:
                best = ((0, -1), miss)
        if gs.mode.ck:
            # owner 2026-10-06: Merchant Fleet in hand -> trade for the 2nd card of a pair first,
            # then play it (cards are tried again after every trade and before End turn)
            r = pc.fleet_pair_target(gs, me)
            if r is not None:
                return {r: 1}
        return best[1] if best else {}

    def _trade_targets(self, gs: GameState) -> list[str]:
        # owner 2026-10-06: "bot should strive to build more cities if possible and then settlements"
        kinds = ["city", "settlement"] if self.city_first else ["settlement", "city"]
        if not gs.legal_settlement_spots(gs.me()):
            kinds.append("road")
            if gs.mode.seafarers and gs.legal_road_edges(gs.me(), ship=True):
                kinds.append("ship")          # owner 2026-10-02: it never set a boat
        if gs.mode.has_dev_cards:
            kinds.append("dev")
        if gs.mode.ck:
            me = gs.me()
            # owner rule: one active knight per city, all game -> knight is the first target
            if len(me.knights) < len(me.cities):
                kinds.insert(0, "knight")
            else:
                kinds += ["knight"]
            idle = any(not k.active for k in me.knights.values()) and not self.warlord_covers(gs)
            if idle:
                # owner 2026-10-06: "it already has a knight and that knight is not active, it is
                # trying to build more knights but that's not priority" -- activate first, a new
                # knight only after the others
                kinds.insert(0, "activate")
                if "knight" in kinds:
                    kinds.remove("knight")
                    kinds.append("knight")
            if not me.cities and gs.mode.canal:
                # owner 2026-10-06: after losing the city it banked ore for grain, activated a knight
                # it no longer needed and hunted sheep for more knights -- the city comes first;
                # knights only when an activation digs the canal (or wins the defender point)
                kinds = ["city"] + [k for k in kinds if k not in ("city", "knight")]
                if "activate" in kinds and not self._activation_useful(gs):
                    kinds.remove("activate")
            # owner 2026-10-06 (cards report): Medicine in hand makes the city 2 ore + 1 grain --
            # trade toward that price, ahead of the full city
            if "medicine" in (me.progress_hand or []) and me.settlements \
                    and len(me.cities) < gs.mode.max_cities and missing(dict(me.hand), COSTS["city"]):
                kinds.insert(kinds.index("city") if "city" in kinds else 0, "medicine")
        return kinds

    # ---- picking from legal moves (owner 2026-10-02: code owns legality, the model picks) --
    def pick(self, gs: GameState | None, point: str, legal: list, **ctx):
        """One Move from `legal` for this decision point. Never invents a move."""
        if len(legal) == 1:
            return legal[0]
        if point == "turn":
            return self.choose_turn(gs, legal, ctx.get("turn") or TurnCtx())
        if point == "answer":
            return self.choose_answer(gs, legal, ctx.get("partner", ""))
        if point == "discard":
            hand = {m.target: gs.me().hand.get(m.target, 0) for m in legal}
            g = self.clone(gs)
            g.me().hand.clear(); g.me().hand.update(hand)
            r = next(iter(self.choose_discard(g, 1)))
            return next(m for m in legal if m.target == r)
        if point == "steal":
            name = self.choose_victim(gs, [m.target for m in legal]) if gs is not None else legal[0].target
            return next(m for m in legal if m.target == name)
        if point == "robber":
            hx, _ = self.choose_robber(gs, [m.target for m in legal])
            return next(m for m in legal if m.target == hx)
        if point == "resource":
            r = self.choose_resource(gs, 1)[0] if gs is not None else "grain"
            return next((m for m in legal if m.target == r), legal[0])
        if point == "take":
            return next((m for m in legal if m.kind == "take"), legal[-1])
        raise ValueError(point)

    def choose_turn(self, gs: GameState, legal: list, ctx: TurnCtx):
        L = set(legal)
        me = gs.me()
        hand = dict(me.hand)
        end = mv.Move("end")
        if ctx.late:
            return self._shed_move(gs, L, ctx) or end
        builds = [m for m in L if m.kind in mv.BUILD_KINDS]
        if ctx.lost_cities and me.settlements and ctx.sheds < 4 and not builds \
                and sum(missing(hand, COSTS["city"]).values()) <= 2:
            # owner 2026-10-02: rebuild the city -- 3 spare wool at the harbor = 1 grain; not while
            # something is buildable, and only 1-2 cards short (game 1 banked a ship's cards away)
            for res in sorted(missing(hand, COSTS["city"]), key=lambda r: r != "grain"):
                bt = self._cheapest_bank(gs, L, res, exclude=set(COSTS["city"]))
                if bt:
                    self.note(f"city lost: bank {bt.give_d} -> {bt.get_d} toward rebuilding it")
                    return bt
        builds = [m for m in L if m.kind in mv.BUILD_KINDS]
        if ctx.refused and not builds:
            # owner 2026-10-02: players refused -> use the bank/harbor (3:1) for what we need, only
            # while a goal still lacks it and never paying with that goal's cards (game 1: banked
            # 3 ore for a 3rd grain while holding the whole city)
            want = next(iter(ctx.refused))
            goals = [k for k in self._offer_kinds(gs, ctx) if want in missing(hand, COSTS.get(k, {}))]
            if goals:
                bt = self._cheapest_bank(gs, L, want, exclude=set(COSTS[goals[0]]))
                if bt:
                    return bt
        o = self._offer_move(gs, L, ctx)
        if o:
            return o
        for a in self.plan_turn(gs)[:1]:
            m = mv.Move.of(a.kind, give=a.give, get=a.get) if a.kind == "bank_trade" else mv.Move(a.kind, a.target)
            if m in L:
                return m
            self.note(f"plan step {m} not legal now: skipped")
        return self._shed_move(gs, L, ctx) or end

    def _protected(self, ctx: TurnCtx, gs: GameState | None = None) -> set:
        # owner 2026-10-02: "giving out hey for cley when it needs to build a knight" -- in the
        # Great Canal grain only comes from gold/aqueduct, so it is never offered to players
        # owner 2026-10-08: grain is never offered to a player in any mode
        return set(self.keep) | {"grain"} | ({"grain", "ore"} if ctx.lost_cities else set()) \
            | (self.reserved_commodities(gs) if gs is not None else set())

    def _cheapest_bank(self, gs: GameState, L: set, res: str, exclude: set = frozenset()):
        val = self.resource_value(gs, gs.me())
        exclude = set(exclude) | self.reserved_commodities(gs)
        opts = [m for m in L if m.kind == "bank_trade" and m.get_d == {res: 1}
                and not set(m.give_d) & exclude]
        return min(opts, key=lambda m: sum(n * val.get(k, 1.0) for k, n in m.give), default=None)

    def _offer_kinds(self, gs: GameState, ctx: TurnCtx) -> list[str]:
        kinds = self._trade_targets(gs)
        if ctx.lost_cities and gs.me().settlements:
            # owner 2026-10-02: after the barbarians take our city, getting it back comes first
            kinds = ["city"] + [k for k in kinds if k != "city"]
        return kinds

    def _offer_move(self, gs: GameState, L: set, ctx: TurnCtx):
        """A 1:1 offer toward the top trade target (picked from the legal offers)."""
        if not any(m.kind == "offer" for m in L):
            return None
        me = gs.me()
        made = ctx.offers_made
        val = self.resource_value(gs, me)
        protect = self._protected(ctx, gs)
        ratio = {r: self.bank_ratio(gs, r) for r in RESOURCES}
        kinds = self._offer_kinds(gs, ctx)
        if self.grain_rush(gs) and ("want", "grain") not in made:
            for give in sorted((r for r in RESOURCES if me.hand.get(r, 0) and r != "grain"), key=lambda r: val.get(r, 1.0)):
                m = mv.Move.of("offer", give={give: 1}, get={"grain": 1})
                if m in L:
                    return m
        for i, kind in enumerate(kinds):
            need = missing(dict(me.hand), COSTS.get(kind, {}))
            # the top target may be 2 short (knight = wool + ore); side targets only 1
            if not need or sum(need.values()) > (3 if kind == "city" else 2 if i == 0 else 1):
                continue
            if any(("want", r) in made for r in need) and kind != "city" and i > 0:
                continue                              # refused once this turn: don't re-ask
            # owner 2026-10-05: never ask the table for a card nobody plausibly holds
            want = next((r for r in need if ("want", r) not in made and self.table_has(gs, r)), None)
            if want is None:
                continue
            # owner 2026-10-02: never give away the card that keeps a 3:1 trade alive, nor a
            # card our goal or a lost city needs ("gave the hay away for a sheep")
            spare = [r for r in RESOURCES if me.hand.get(r, 0) - COSTS[kind].get(r, 0) > 0 and r != want
                     and me.hand.get(r, 0) != ratio[r] and r not in protect]
            for give in sorted(spare, key=lambda r: val.get(r, 1.0)):
                m = mv.Move.of("offer", give={give: 1}, get={want: 1})
                if m in L:
                    return m
        return self._bank_set_offer(gs, L, ctx, kinds)

    def _bank_set_offer(self, gs: GameState, L: set, ctx: TurnCtx, kinds: list[str]):
        """Owner 2026-10-02: "bot should see it has 2 clay and could ask for a third to be able
        to get the resource it needs". One short of a bank set of a card the goal doesn't use."""
        me = gs.me()
        val = self.resource_value(gs, me)
        protect = self._protected(ctx, gs)
        for kind in kinds:
            cost = COSTS.get(kind, {})
            need = missing(dict(me.hand), cost)
            if not need or sum(need.values()) > self.set_need_cap(gs, kind):
                continue
            for r in sorted(RESOURCES, key=lambda r: val.get(r, 1.0)):
                ratio, have = self.bank_ratio(gs, r), me.hand.get(r, 0)
                if r in need or ("want", r) in ctx.offers_made or have < 1 or have + 1 - cost.get(r, 0) != ratio:
                    continue
                if not self.table_has(gs, r):
                    continue
                spare = [x for x in RESOURCES if me.hand.get(x, 0) - cost.get(x, 0) > 0 and x != r
                         and x not in need and x not in protect and me.hand.get(x, 0) != self.bank_ratio(gs, x)]
                for give in sorted(spare, key=lambda x: val.get(x, 1.0)):
                    m = mv.Move.of("offer", give={give: 1}, get={r: 1})
                    if m in L:
                        self.note(f"bank set: {have} {r}, asking for 1 more to bank {ratio} {r} -> "
                                  f"{next(iter(need))} for the {kind}")
                        return m
        return None

    def _shed_move(self, gs: GameState, L: set, ctx: TurnCtx):
        """Owner 2026-10-02: never end a turn over the hand limit -- bank the surplus into
        what we need until we're at or under the limit."""
        me = gs.me()
        limit = discard_limit(gs.mode, me)
        total = sum(me.hand.values())
        if total <= limit or ctx.sheds >= 4:
            return None
        want = None
        kinds = (["city"] if me.settlements and ctx.lost_cities else []) + self._trade_targets(gs)
        for kind in kinds:
            need = missing(dict(me.hand), COSTS.get(kind, {}))
            if need:
                want = next(iter(need))
                break
        want = want or min(("grain", "ore", "wool", "brick", "lumber"), key=lambda r: me.hand.get(r, 0))
        bt = self._cheapest_bank(gs, L, want)
        self.note(f"hand {total} > limit {limit}: " + (f"bank {bt.give_d} -> {bt.get_d}" if bt else "no bank trade possible"))
        return bt

    def choose_answer(self, gs: GameState, legal: list, partner: str):
        """Accept / decline / counter an incoming offer (only moves in `legal`)."""
        decline = mv.Move("decline")
        acc = next((m for m in legal if m.kind == "accept"), None)
        if acc is not None:
            return acc if self.evaluate_offer(gs, partner, acc.give_d, acc.get_d) else decline
        counters = [m for m in legal if m.kind == "counter"]
        if counters:
            give = counters[0].give_d
            if give.get("grain"):
                return decline      # owner 2026-10-08: never give grain, not even for a named card
            want = self.wanted_resource(gs, give)
            n = 2 if any(k in COMMODITIES for k in give) else 1     # owner: 2 back for a commodity
            if want and not self.commodity_ok(gs, partner, give, {want: n}):
                return decline      # owner 2026-10-08: no commodity to a leader / a level-4 reach
            if want and self.trade_locked(gs, partner, give, {want: n}):
                return decline      # owner 2026-10-08: nobody at 14+ VP unless it wins us the game
            p = next((q for q in gs.opponents() if q.name == partner), None)
            if p is not None and p.vp_visible >= gs.target - 2 and not (want and self.sure_win_after(gs, give, {want: n})):
                return decline
            if want and self.past_half(gs, partner) and not self.evaluate_offer(gs, partner, give, {want: n}):
                self.note(f"wildcard from {partner}: past halfway, {give}->{want} not worth it")
                return decline
            return next((m for m in counters if m.get_d == {want: n}), decline)
        return decline

    def wanted_resource(self, gs: GameState, giving: dict) -> str | None:
        """The one card that best completes a build after giving `giving` away."""
        g = self.clone(gs)
        pay(g.me().hand, giving)
        L = set(mv.offer_moves(g))
        o = self._offer_move(g, L, TurnCtx()) if L else None
        if o is not None:
            return next(iter(o.get_d))
        best, bv = None, -1.0
        for r in RESOURCES:
            g2 = self.clone(g)
            g2.me().hand[r] = g2.me().hand.get(r, 0) + 1
            v = self._best_followup(g2)
            if v > bv:
                best, bv = r, v
        return best

    # ---- robber / blocking ---------------------------------------------------------
    def chase_pick(self, gs: GameState):
        """Owner 2026-10-09: an ACTIVE knight on a corner of the robber's hex may chase it (after
        the first barbarian attack); it moves the robber as after a 7, steals, and goes inactive.
        Chase when the robber blocks one of our producing hexes (a gold-corner knight: only off
        its own gold hex -- it never defends), or, with knight_bounce, to steal from a rival with
        cards. Never when standing the knight down leaves a city open with the barbarians 2 steps
        away or closer (the sim's chase_robber rule). The most blocked production wins.
        Returns (Action("chase", (knight, hex, piece)), (new_hex, victim)) or None; new_hex is
        choose_robber's pick (None for the pirate: the live prompt picks the sea hex)."""
        if not self.knight_chase:
            return None
        opts = mv.chase_moves(gs)
        if not opts:
            return None
        b, me = gs.board, gs.me()
        canal = gs.mode.canal
        dry = canal and gs.canal_progress < gs.canal_total
        near = gs.mode.ck and gs.barbarian_pos >= gs.barbarian_len - 2
        safe = gs.robber_protected() if gs.mode.friendly_robber else set()
        robbable = any(q is not me and q.name not in safe and q.total_cards() > 0 for q in gs.players)
        best, best_score = None, 0.0
        for m in opts:
            v, at, piece = m.target
            h = b.hexes[at]
            gold = canal and any(x.terrain == "gold" for x in b.vertex_hexes(v))
            blocked = 0.0
            for u, kind in me.buildings():
                if h in b.vertex_hexes(u) and h.number and not (dry and h.barren):
                    blocked += (2 if kind == "city" else 1) * PIPS.get(h.number, 0)
            if piece == "pirate":
                blocked += sum(1 for e in me.ships if h in b.edge_hexes(e))
            if gold:
                blocked = 5.0 if h.terrain == "gold" else 0.0    # the gold knight's own grain
            steal = 0.5 if (getattr(self, "knight_bounce", False) and not gold and robbable) else 0.0
            score = blocked + steal
            if score <= 0:
                continue
            k = me.knights[v]
            k.active = False                      # what if this knight stands down?
            gap = self.defence_gap(gs, me)
            k.active = True
            if near and gap > 0:
                continue
            if score > best_score:
                best, best_score = m, score
        if best is None:
            return None
        v, at, piece = best.target
        if piece == "pirate":
            return Action("chase", best.target), (None, None)
        options = [c for c, x in b.hexes.items() if x.is_land and c != at and x.terrain != "desert"]
        return Action("chase", best.target), self.choose_robber(gs, options)

    def choose_robber(self, gs: GameState, options: list[tuple[int, int]] | None = None):
        """Return (hex, victim). Hurt the most likely winner, never ourselves."""
        me = gs.me()
        probs = self.probs(gs)
        b = gs.board
        options = options or [c for c, h in b.hexes.items()
                              if h.is_land and c != b.robber and h.terrain != "desert"]

        def damage(c):
            h = b.hexes[c]
            if not h.number:
                return -1.0
            if gs.mode.canal and h.barren and gs.canal_progress < gs.canal_total:
                return -0.5         # barren farmland produces nothing: blocking it wastes the 7
            d = 0.0
            for v in b.vertices:
                if c not in [x.coord for x in b.vertex_hexes(v)]:
                    continue
                owner = gs.owner_at(v)
                if owner is None:
                    continue
                mult = 2 if v in owner.cities else 1
                w = -3.0 if owner is me else probs[owner.name] * 4
                d += w * mult * PIPS[h.number]
            if d > 0 and gs.mode.ck and chasers(c):
                # owner 2026-10-06: a rival's active knight on the hex chases the robber off next
                # turn (and steals from us) -- the block lasts no time, pick a hex nobody guards
                d *= 0.3
            return d

        def chasers(c):
            h = b.hexes[c]
            return [p for p in gs.players if p is not me
                    and any(k.active and h in b.vertex_hexes(v) for v, k in p.knights.items())]
        if gs.mode.friendly_robber:
            # owner 2026-10-03: no hex touching a player with 4 VP or fewer (state.FRIENDLY_ROBBER_VP)
            protected = gs.robber_protected()
            options = gs.friendly_robber_hexes(options, me)
        def on_hex(c):
            vs = {gs.owner_at(v) for v in b.vertices
                  if c in [x.coord for x in b.vertex_hexes(v)]} - {None, me}
            if gs.mode.friendly_robber:
                vs = {p for p in vs if p.name not in protected}   # never stolen from
            return vs
        if self.robber_steal_first:
            # live 2026-10-09 19:44: the robber went on an 8 with nobody on it. Owner "steal then
            # spend": a hex with a rival holding cards beats a hex with nobody to rob.
            robbable = [c for c in options if any(p.total_cards() > 0 for p in on_hex(c))]
            options = robbable or options
        pick = None
        if self.situational and self.stance:
            # selector: aim at the preferred rival (focus / best steal), its hexes only
            from . import situational as st
            reach = {p for c in options for p in on_hex(c) if p.total_cards() > 0}
            pick = st.prefer(self, gs, sorted(reach, key=lambda p: gs.players.index(p)))
            if pick is not None:
                options = [c for c in options if pick in on_hex(c)] or options
        best = max(options, key=damage)
        victims = on_hex(best)
        if pick is not None and pick in victims:
            return best, pick
        victims = [p for p in victims if p.total_cards() > 0] or list(victims)
        victim = max(victims, key=lambda p: (probs[p.name], p.total_cards()), default=None)
        if len(victims) > 1 and getattr(gs, "hands", None) is not None:
            # owner 2026-10-06: the hand ledger knows who holds what we need -- steal there
            v = self.choose_victim(gs, [p.name for p in victims])
            victim = next((p for p in victims if p.name == v), victim)
        return best, victim

    def choose_victim(self, gs: GameState, names: list[str]) -> str:
        probs = self.probs(gs)
        if gs.mode.friendly_robber:          # owner: never steal from a player with 4 VP or fewer
            names = [n for n in names if n not in gs.robber_protected()] or names
        if self.situational and self.stance:
            from . import situational as st
            pick = st.prefer(self, gs, [p for p in gs.players if p.name in names and not p.is_me])
            if pick is not None:
                return pick.name
        # owner 2026-10-03: steal where the likely hand has what we need (seen production
        # + card count), still leaning on the leader; nobody steals from an empty hand
        missing = self.plan_missing(gs)

        def worth(n):
            p = next((q for q in gs.players if q.name == n), None)
            est = gs.est_hand(p) if p is not None else {}
            tot = sum(est.values())
            if tot <= 0:
                return probs.get(n, 0) - 5.0
            ev = sum(c * (2.0 if r in missing else pc.card_value(gs, r)) for r, c in est.items()) / tot
            return ev + 1.5 * probs.get(n, 0)
        return max(names, key=worth)

    def choose_discard(self, gs: GameState, n: int) -> dict[str, int]:
        me = gs.me()
        hand = dict(me.hand)
        out: dict[str, int] = {}
        # keep what our best plan needs; drop the least valuable surplus
        targets = ["city", "settlement"] + (["knight"] if gs.mode.ck else ["dev"])
        keep_need = {}
        for t in targets:
            for k, v in COSTS[t].items():
                keep_need[k] = max(keep_need.get(k, 0), v)
        if gs.mode.ck and self.defence_gap(gs, me) > 0:
            keep_need["grain"] = max(keep_need.get("grain", 0), 1 + sum(1 for k in me.knights.values() if not k.active))
            keep_need["ore"] = max(keep_need.get("ore", 0), 1)
            keep_need["wool"] = max(keep_need.get("wool", 0), 1)
        for k in self.keep:                     # outlook goal / lost city (runner sets keep)
            keep_need[k] = max(keep_need.get(k, 0), 2 if k in ("grain", "ore") else 1)
        firm = self.improve_keep(gs, hand, n) if self.discard_keep_improve and gs.mode.ck else {}
        val = self.resource_value(gs, me)
        for _ in range(n):
            cands = [k for k, v in hand.items() if v > 0]
            # owner 2026-10-08: never throw grain away -- it goes only when nothing else is left
            k = min(cands, key=lambda r: (r == "grain", hand[r] <= firm.get(r, 0), hand[r] <= keep_need.get(r, 0),
                                          val.get(r, 1) / max(hand[r], 1)))
            hand[k] -= 1
            out[k] = out.get(k, 0) + 1
        return out

    def improve_keep(self, gs: GameState, hand: dict, n: int) -> dict[str, int]:
        """Cards to keep through a discard of `n`: the commodity of the best city improvement the
        hand can still pay after the discard, plus the bank-trade cards for what it is short (live
        2026-10-09 19:44: paper 3 -> 4 was 3 paper + 3 wood at 3:1; both went and the metropolis
        with them). A level that wins a metropolis comes first. {} = nothing reachable."""
        me = gs.me()
        room = sum(hand.values()) - n - hand.get("grain", 0)     # grain always stays
        held = getattr(gs, "metropolis_owner", {}) or {}
        best, best_key = {}, None
        for track, com in IMPROVEMENT_TRACKS.items():
            if not mv.improve_allowed(gs, me, track) or not hand.get(com):
                continue
            lv = me.improvements.get(track, 0)
            cost = improvement_cost(track, lv)[com]
            keep = {com: min(hand[com], cost)}
            for _ in range(cost - keep[com]):
                opts = [(self.bank_ratio(gs, r), r) for r, v in hand.items()
                        if r not in ("grain", com) and v - keep.get(r, 0) >= self.bank_ratio(gs, r)]
                if not opts:
                    keep = None
                    break
                ratio, r = min(opts)
                keep[r] = keep.get(r, 0) + ratio
            if keep is None or sum(keep.values()) > room:
                continue
            owner = held.get(track)
            rival = next((q for q in gs.players if owner and q.name == owner[0]), None)
            metro = lv + 1 >= 4 and (owner is None or owner[0] == me.name
                                     or (lv + 1 == 5 and rival is not None and rival.improvements.get(track, 0) < 5))
            key = (metro, lv + 1, -sum(keep.values()))
            if best_key is None or key > best_key:
                best, best_key = keep, key
        return best

    # ---- trading with players ------------------------------------------------------
    TRADE_LOCK_VP = 14

    def sure_win_after(self, gs: GameState, give: dict, get: dict, on_turn: bool = False, depth: int = 4) -> bool:
        """After trading `give` for `get`, does our hand alone build our way to gs.target (no
        roll, no trade needed)? A hand over the discard limit off our turn is not sure: a 7 cuts it."""
        g = self.clone(gs)
        me = g.me()
        if not can_afford(me.hand, give):
            return False
        pay(me.hand, give)
        for r, n in get.items():
            me.hand[r] = me.hand.get(r, 0) + n
        if not on_turn and sum(me.hand.values()) > discard_limit(g.mode, me):
            return False

        def reach(g, d):
            if g.me().vp_visible >= g.target:
                return True
            if d == 0:
                return False
            for m in mv.build_moves(g):
                if m.kind in ("road", "ship", "dev"):
                    continue            # no sure point in a road or a random card
                try:
                    g2 = self._apply(g, Action(m.kind, m.target))
                except Exception:       # noqa: BLE001
                    continue
                if g2.me().vp_visible > g.me().vp_visible and reach(g2, d - 1):
                    return True
            return False
        return reach(g, depth)

    def trade_locked(self, gs: GameState, partner: str, give: dict, get: dict, on_turn: bool = False) -> bool:
        """Owner 2026-10-08 "when players have 14 points or more, trade should be rejected unless
        it gives you a 100% win next round": no trade at all with a partner at 14+ VP, unless
        the trade's cards let us build to the target on our own."""
        p = next((q for q in gs.opponents() if q.name == partner), None)
        if p is None or p.vp_visible < self.TRADE_LOCK_VP:
            return False
        return not self.sure_win_after(gs, give, get, on_turn)

    def accepter_refusal(self, gs: GameState, partner: str, give: dict, get: dict) -> str | None:
        """Our own offer went to the whole table and `partner` said yes: the partner rules still
        hold (review 0.3.1 -- the accept used to check only the 1.5x rule). None = take it,
        else the reason to pass on this accepter."""
        if give.get("grain"):
            return "grain never goes to a player"
        if self.trade_locked(gs, partner, give, get, on_turn=True):
            return "14+ VP and no sure win"
        if not self.commodity_ok(gs, partner, give, get):
            return "a commodity they must not get"
        p = next((q for q in gs.opponents() if q.name == partner), None)
        if p is not None and p.vp_visible >= gs.target - 2 and not self.sure_win_after(gs, give, get, True):
            return "2 VP from winning"
        if self.past_half(gs, partner) and not self.evaluate_offer(gs, partner, give, get):
            return "past halfway and not great for us"
        return None

    def evaluate_offer(self, gs: GameState, partner: str, give: dict, get: dict) -> bool:
        """Accept an offer where we give `give` and receive `get` from `partner`?"""
        me = gs.me()
        if not can_afford(me.hand, give):
            return False
        if give.get("grain"):
            # owner 2026-10-08 "gave away grain for a trade when asked by a player with more
            # points": grain never goes to another player, in any mode, to anyone
            return False
        if set(give) & self.reserved_commodities(gs):
            return False            # stage 2026-10-09 07:32:59: the metropolis at 4 keeps its commodity
        if self.trade_locked(gs, partner, give, get):
            return False
        p = next((q for q in gs.opponents() if q.name == partner), None)
        if p is not None and p.vp_visible >= self.TRADE_LOCK_VP:
            return True             # a sure win: take it, whoever offers
        if not self.commodity_ok(gs, partner, give, get):
            return False
        p = next((q for q in gs.opponents() if q.name == partner), None)
        if self.generous_accept and sum(get.values()) > 3 * max(1, sum(give.values())) \
                and (p is None or p.vp_visible < gs.target - 5):
            # owner 2026-10-09 (Rhea, 10 of 19, offered brick + 2 wool + ore for one lumber and
            # was declined): better than 3:1 (2:1 when it helps us and they are not in front) from a player more than 5 VP from winning is taken
            # over the leader / halfway / focus rules -- grain, the 14+ lock and commodities still hold
            return True
        if self.generous_accept and sum(get.values()) >= 2 * max(1, sum(give.values())) \
                and p is not None and p.vp_visible < gs.target - 5 and not self._is_winning(gs, p) \
                and not (any(k in self.keep for k in give) and not any(k in self.keep for k in get)):
            # owner 2026-10-09 "even 2:1 if it makes sense and it does not help the winning player":
            # 2:1 or better from anyone but the leader is taken when the cards raise our chances
            g = self.clone(gs)
            m2 = g.me()
            pay(m2.hand, give)
            for k, v in get.items():
                m2.hand[k] = m2.hand.get(k, 0) + v
            m2.hand_count = sum(m2.hand.values())
            if self.rare_gain(gs, give, get) or self._best_followup(g) - self.my_prob(gs) > 0:
                return True
        if gs.mode.canal and get == {"grain": 1} and sum(give.values()) == 1 and "grain" not in give \
                and (not me.hand.get("grain") or any(not k.active for k in me.knights.values())):
            # owner 2026-10-06 "still rejected grain for sheep that it had": no grain fields in the
            # canal; a 1-for-1 into grain we need beats the leader rules -- only a player one
            # point from winning is still refused
            return p is None or p.vp_visible < gs.target - 1
        if p is not None and p.vp_visible >= gs.target - 2:
            return False            # playbook: never feed a player 2 VP from winning
        if p is not None and p.vp_visible > 1.5 * max(1, me.vp_visible):
            return False            # owner 2026-10-03: never trade with a player >1.5x our points
        led = getattr(gs, "hands", None)
        if p is not None and led is not None and led.knows(partner) and p.vp_visible >= me.vp_visible \
                and led.feeds_build(partner, give, gs=gs):
            return False            # owner 2026-10-06: never hand a rival the last card of a city
        f = self._focus_player(gs)
        if f is not None and f.name == partner:
            return False            # selector BLOCK: no cards to the advancing rival
        if self.rare_gain(gs, give, get):
            return True             # owner 2026-10-06: a card we almost never roll is always worth it
        before_me = self.my_prob(gs)
        g = self.clone(gs)
        m2 = g.me()
        pay(m2.hand, give)
        for k, v in get.items():
            m2.hand[k] = m2.hand.get(k, 0) + v
        m2.hand_count = sum(m2.hand.values())
        after_me = self._best_followup(g)
        probs = self.probs(gs)
        p_partner = probs.get(partner, 0)
        leader = max(probs, key=probs.get)
        # the partner gains roughly what we give them
        partner_gain = 0.02 * sum(give.values()) - 0.01 * sum(get.values())
        penalty = partner_gain * (3.0 if partner == leader else 1.0) * (1 + 2 * p_partner)
        gk, rk = next(iter(give)), next(iter(get))
        if any(k in self.keep for k in give) and not any(k in self.keep for k in get):
            # owner 2026-10-02: never hand away the grain we just banked to rebuild our city
            return False
        if (sum(give.values()) == 1 and sum(get.values()) == 1 and not self.past_half(gs, partner)
                and me.hand.get(gk, 0) >= 2 and me.hand.get(gk, 0) > me.hand.get(rk, 0) + 1
                and after_me - before_me > -0.002):
            # owner: accept fair trades -- a spare card (we hold 2+ more) for a scarcer one
            return True
        if self.past_half(gs, partner):
            # owner 2026-10-02: past halfway, only trade if it's extremely good for us,
            # never just to give a front-runner more steam
            return (after_me - before_me) - 3 * penalty > 0.03
        return (after_me - before_me) - penalty > 0.004

    @staticmethod
    def _is_winning(gs: GameState, p) -> bool:
        """The player in front: most visible VP at the table (ties count), us included."""
        top = max([q.vp_visible for q in gs.opponents()] + [gs.me().vp_visible])
        return p.vp_visible >= top

    def commodity_ok(self, gs: GameState, partner: str, give: dict, get: dict) -> bool:
        """Owner 2026-10-08 "should not give out development cards for just one resource, ask at
        least for 2 ... avoid giving them to players that have more points or can get the 4th
        level": a commodity we give needs 2 cards back (a commodity back counts as 2), and never
        goes to a player ahead of us or one whose track for it is at level 3+."""
        given = {k: n for k, n in give.items() if k in COMMODITIES and n}
        if not given:
            return True
        w = lambda d: sum(n * (2 if k in COMMODITIES else 1) for k, n in d.items())
        if w(get) < w(give):
            return False
        p = next((q for q in gs.opponents() if q.name == partner), None)
        if p is None:
            return True
        if p.vp_visible > gs.me().vp_visible:
            return False
        track = {c: t for t, c in IMPROVEMENT_TRACKS.items()}
        return all(p.improvements.get(track[k], 0) < 3 for k in given)

    RARE_INCOME = 3 / 36    # at most one 2/12 number (or nothing at all: Great Canal grain)

    def rare_gain(self, gs: GameState, give: dict, get: dict) -> bool:
        """Owner 2026-10-06 (solo: refused wool->grain from Lin and Marianne three times): always
        take a trade that brings a resource we rarely or never produce (bad numbers, no building on
        it), paid with cards we produce more of. At most 2 cards for 1; never pay with kept cards."""
        if not get or not give or sum(give.values()) > 2 * sum(get.values()):
            return False
        inc = gs.income(gs.me())
        if any(inc.get(r, 0.0) > self.RARE_INCOME for r in get):
            return False
        worst_get = max(inc.get(r, 0.0) for r in get)
        hand = gs.me().hand
        for g in give:
            if g in get or g in self.keep:
                return False
            if inc.get(g, 0.0) <= worst_get and hand.get(g, 0) < 2:
                return False    # just as rare for us, and not a spare
        return True

    @staticmethod
    def past_half(gs: GameState, partner: str) -> bool:
        """Partner has passed half the VP target (and isn't behind us)."""
        p = next((q for q in gs.opponents() if q.name == partner), None)
        return p is not None and p.vp_visible > gs.target / 2 and p.vp_visible >= gs.me().vp_visible

    def _best_followup(self, gs: GameState) -> float:
        """Best value after up to two of our own actions (owner 2026-10-02: a 2-for-1 is fair
        if it lets us deploy a knight AND activate it this turn)."""
        best = self.my_prob(gs)
        for a in self.candidate_actions(gs):
            best = max(best, self.action_value(gs, a))
            if a.kind in ("bank_trade", "player_trade"):
                continue
            try:
                g2 = self._apply(gs, a)
            except Exception:           # noqa: BLE001
                continue
            for b in self.candidate_actions(g2):
                if b.kind not in ("bank_trade", "player_trade"):
                    best = max(best, self.action_value(g2, b))
        return best

    def plausible_partners(self, gs: GameState, want: str, names: list[str] | None = None) -> list[str]:
        """Owner 2026-10-05: ask for a card only from players who plausibly hold it (hand ledger:
        at least HOLD_MIN expected). Without a ledger every candidate stays (old behaviour)."""
        names = [p.name for p in gs.opponents()] if names is None else list(names)
        if getattr(gs, "hands", None) is None:
            return names
        from .hands import HOLD_MIN
        by = {p.name: p for p in gs.players}
        # a player the ledger knows nothing about yet (no event, no card count) stays a partner
        return [n for n in names if n in by and (not gs.hands.knows(n)
                                                  or gs.est_hand(by[n]).get(want, 0.0) >= HOLD_MIN)]

    DECKS = ("trade", "politics", "science")

    def choose_progress_deck(self, gs: GameState, counts: dict | None = None) -> str:
        """owner 2026-10-06 (defender tie, any deck): pick by the game's stage, not just our best
        track. Building up -> science (crane, medicine, smith, engineer, road building); short of
        grain or one card from a big build -> trade (merchant, fleet, harbor, monopoly); a rival
        close to winning, or our own end race -> politics (saboteur, deserter, intrigue,
        constitution). A track near its metropolis pulls toward its deck; empty decks never."""
        me = gs.me()
        lv = me.improvements or {}
        score = {d: 0.0 for d in self.DECKS}
        others = [p.vp_visible for p in gs.players if p.name != me.name]
        close = gs.target - 4
        if (others and max(others) >= close) or me.vp_visible >= close:
            score["politics"] += 3
        hand = dict(me.hand)
        one_short = any(sum(missing(hand, COSTS[k]).values()) == 1 for k in ("city", "knight") if k in COSTS)
        if (gs.mode.canal and not hand.get("grain")) or one_short:
            score["trade"] += 2
        if any(lv.get(t, 0) < 3 for t in self.DECKS):   # the ladder: every track to 3 first
            score["science"] += 1
        for d in self.DECKS:
            if 3 <= lv.get(d, 0) <= 4:
                # metropolis race (owner: 5 locks it, at 4 it can still be taken): that track's cards
                score[d] += 0.3 + 0.1 * lv[d]
            if counts and counts.get(d) is not None:
                if counts[d] <= 0:
                    score[d] = -99.0
                elif counts[d] == 1:
                    score[d] -= 1
        pick = max(self.DECKS, key=lambda d: (score[d], d == "science", d == "politics"))
        self.note(f"progress deck: {pick} (scores {score}, counts {counts})")
        return pick

    def grain_rush(self, gs: GameState) -> bool:
        """Solo 2026-10-06: the knight stood inactive (no grain) for 4 rolls while the barbarians
        sailed 2 -> 7 and the only city was lost; the offers asked for lumber and ore because the
        hand ledger said nobody held grain (canal: start grain, gold and aqueduct still give it).
        True: a city is exposed, an inactive knight would cover it, the ship is past halfway."""
        if not gs.mode.ck:
            return False
        me = gs.me()
        if me.hand.get("grain", 0) or not any(not k.active for k in me.knights.values()):
            return False
        return self.defence_gap(gs, me) > 0 and gs.barbarian_pos / max(1, gs.barbarian_len) >= 0.5

    def table_has(self, gs: GameState, want: str) -> bool:
        """Someone we would trade with (owner: never a player over 1.5x our points) plausibly
        holds want. Always True without a hand ledger."""
        if getattr(gs, "hands", None) is None:
            return True
        my_vp = max(1, gs.me().vp_visible)
        names = [p.name for p in gs.opponents() if p.vp_visible <= 1.5 * my_vp]
        return bool(self.plausible_partners(gs, want, names))

    def propose_trade(self, gs: GameState) -> tuple[dict, dict, list[str]] | None:
        """1-for-1 (or 2-for-1) offer that unlocks our best build. Returns (give, get, partners)."""
        me = gs.me()
        val = self.resource_value(gs, me)
        threat = self.threat(gs)
        kinds = ["city", "settlement"] + (["knight"] if gs.mode.ck else ["dev"])
        if gs.mode.ck and (self.defence_gap(gs, me) > 0 or len(me.knights) < len(me.cities)):
            # owner rule: ask the table for wool (knight) / grain (activation) first
            kinds = ["knight", "activate"] + [k for k in kinds if k != "knight"]
            if self.activate_first and any(not k.active for k in me.knights.values()):
                kinds = ["activate", "knight"] + kinds[2:]     # an idle knight only needs grain
        danger = gs.barbarian_pos / max(1, gs.barbarian_len)
        for kind in kinds:
            need = missing(me.hand, COSTS[kind])
            if sum(need.values()) != 1:
                continue
            want = next(iter(need))
            # review 0.3.1: grain is never offered to a player, not even a spare one
            spare = [r for r in RESOURCES if me.hand.get(r, 0) - COSTS[kind].get(r, 0) > 0
                     and r not in (want, "grain")]
            if not spare:
                continue
            give_r = min(spare, key=lambda r: val[r])
            give = {give_r: 1}
            two_at = min(self.pay_two_danger, 0.3) if self.idle_knight_first else self.pay_two_danger
            if kind == "activate" and self.pay_two_danger and danger >= two_at \
                    and self.defence_gap(gs, me) > 0:
                # saving the city is worth a second card (the cheapest spare we have)
                more = sorted((r for r in RESOURCES for _ in range(me.hand.get(r, 0) - COSTS[kind].get(r, 0) - give.get(r, 0))
                               if r not in (want, "grain")), key=lambda r: val[r])
                if more:
                    give[more[0]] = give.get(more[0], 0) + 1
            partners = [p.name for p in gs.opponents() if (not threat or p.name != threat.name)
                        and not self.trade_locked(gs, p.name, give, {want: 1}, on_turn=True)]
            if gs.hands is not None:
                partners = self.plausible_partners(gs, want, partners)
            if not partners:
                continue            # nobody holds it (e.g. no grain in the game): next plan
            partners.sort(key=lambda n: -self.model.trade_accept_rate(n))
            return give, {want: 1}, partners
        return self._third_of_set(gs, val, threat)

    def _third_of_set(self, gs: GameState, val: dict, threat) -> tuple[dict, dict, list[str]] | None:
        """Owner 2026-10-04: no grain, 2 of a kind at a 3:1 -> ask the table for the 3rd card
        (1 spare card for it), then bank the set into grain."""
        me = gs.me()
        if not (self.set_for_grain and gs.mode.canal) or me.hand.get("grain", 0):
            return None
        if self.grain_gate and not self._activation_useful(gs):
            return None
        pairs = [r for r in list(RESOURCES) + list(COMMODITIES)
                 if r != "grain" and self.bank_ratio(gs, r) == 3 and me.hand.get(r, 0) == 2]
        pairs = [r for r in pairs if self.table_has(gs, r)]     # owner 2026-10-05: only what someone holds
        if not pairs:
            return None
        want = max(pairs, key=lambda r: -val.get(r, 1.0))
        spare = [r for r in RESOURCES if r not in ("grain", want) and me.hand.get(r, 0) > 0
                 and not (r in pairs)]
        if not spare:
            return None
        give_r = min(spare, key=lambda r: val.get(r, 1.0))
        partners = [p.name for p in gs.opponents() if (not threat or p.name != threat.name)
                    and not self.trade_locked(gs, p.name, {give_r: 1}, {want: 1}, on_turn=True)]
        if not partners:
            return None
        if gs.hands is not None:
            partners = self.plausible_partners(gs, want, partners)
            if not partners:
                return None
        partners.sort(key=lambda n: -self.model.trade_accept_rate(n))
        return {give_r: 1}, {want: 1}, partners

    def choose_resource(self, gs: GameState, n: int = 1) -> list[str]:
        """Gold field / Year of Plenty / free pick: what completes our best build."""
        me = gs.me()
        picks = []
        hand = dict(me.hand)
        for _ in range(n):
            best = None
            if gs.mode.canal and gs.canal_progress < gs.canal_total and (
                    "grain" in self.keep or any(missing(hand, COSTS[k]).get("grain") for k in ("city", "settlement", "activate"))):
                best = "grain"                  # owner: gold/aqueduct are the only grain
            for kind in ([] if best else ["city", "settlement", "knight", "dev"]):
                need = missing(hand, COSTS[kind])
                if need:
                    best = max(need, key=lambda r: self.resource_value(gs, me).get(r, 1))
                    break
            best = best or max(RESOURCES, key=lambda r: self.resource_value(gs, me)[r])
            picks.append(best)
            hand[best] = hand.get(best, 0) + 1
        return picks

    def choose_monopoly(self, gs: GameState) -> str:
        """Resource the opponents (weighted toward the leader) hold most of: estimate from income."""
        probs = self.probs(gs)
        score = {r: 0.0 for r in RESOURCES}
        for p in gs.opponents():
            for r, x in gs.income(p).items():
                if r in score:
                    score[r] += x * (1 + probs[p.name])
        return max(score, key=score.get)
