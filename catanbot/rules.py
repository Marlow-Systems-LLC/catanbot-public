"""Mode descriptors, costs and production rules.

Catan Universe tells us legal spots by highlighting them, so these rules are the bot's
*model* of the game (used for planning and simulation), not the referee. VP targets
are read from the game screen when possible; values here are fallbacks.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .board import RESOURCES, TERRAIN_COMMODITY, TERRAIN_RES

Cost = dict[str, int]

COSTS: dict[str, Cost] = {
    "road": {"brick": 1, "lumber": 1},
    "ship": {"lumber": 1, "wool": 1},
    "settlement": {"brick": 1, "lumber": 1, "wool": 1, "grain": 1},
    "city": {"grain": 2, "ore": 3},
    "dev": {"wool": 1, "grain": 1, "ore": 1},
    "wall": {"brick": 2},                                     # C&K city wall
    "knight": {"wool": 1, "ore": 1},                          # C&K basic knight
    "promote": {"wool": 1, "ore": 1},                         # C&K knight upgrade
    "activate": {"grain": 1},                                 # C&K activate knight
    "medicine": {"ore": 2, "grain": 1},                       # C&K Medicine card: a cheap city
}

# C&K city improvement tracks: commodity per track; level n costs n commodities
IMPROVEMENT_TRACKS = {"trade": "cloth", "politics": "coin", "science": "paper"}


@dataclass
class Mode:
    key: str
    name: str
    vp_target: int = 10
    ck: bool = False             # Cities & Knights
    seafarers: bool = False      # ships, gold, islands (The Great Canal uses ships + gold islands, but it is
                                 # its own C&K scenario, not Seafarers: no pirate, no island bonus)
    canal: bool = False          # The Great Canal scenario
    discard_limit: int = 7
    max_settlements: int = 5
    max_cities: int = 4
    max_roads: int = 15
    max_ships: int = 15
    start_with_city: bool = False    # C&K: second setup piece is a city
    # table options read from the lobby / intro / info screens each game (see rules_reader)
    friendly_robber: bool = False    # robber may not touch the lowest-VP players (state.robber_protected)
    balanced_dice: bool = False      # Catan Universe "Balanced" dice: fewer streaks
    turn_timer_s: int | None = None  # None = no turn limit
    players: int | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def has_dev_cards(self) -> bool:
        return not self.ck


# Owner 2026-10-04: marlow plays ONLY The Great Canal. Other scenarios exist on the platform
# (names below, so a lobby/solo screen of another game is recognised and refused), but their
# rules, boards and training are deliberately not in the bot.
MODES: dict[str, Mode] = {
    # Catan Universe Almanac "Great Canal" (confirmed): 21 VP (3p) / 18 VP (4p) on your own turn;
    # starts with 2 settlements + 1 city. Host may change the VP in the scenario table, so
    # GameState.target (screen value, else 21/18 by player count) is what code should use.
    "great_canal": Mode("great_canal", "The Great Canal", 18, ck=True, seafarers=True,
                        canal=True, start_with_city=True, friendly_robber=True,   # owner's usual table
                        notes=["win: 18 VP with 4 players, 21 with 3 (Almanac); host may change it",
                               "grain locked until canal finished",
                               "dig = 2 activated knights on a canal hex's corners, no cost",
                               "canal dig: 2 VP one owner, 1+1 two owners",
                               "9 canal sections; the 9th is built automatically at the end of the turn "
                               "of the 8th dig and gives no VP",
                               "gold: only on an 8; any 1 resource; max 1 per player; knights only; "
                               "ends when the canal is complete",
                               "small-island harbour: a knight there gives the harbour rate, no activation needed",
                               "no settlements on the small islands",
                               "no pirate",
                               "robber only from the first barbarian attack"]),
}
GREAT_CANAL = MODES["great_canal"]

# Other games on Catan Universe (recognition only; marlow never plays them).
OTHER_SCENARIOS: dict[str, list[str]] = {
    "Greater Catan": ["greater catan"], "Enchanted Land": ["enchanted"], "Rise of the Inka": ["inka"],
    "Heading for New Shores": ["new shores"], "Oceania": ["oceania"], "Treasure Island": ["treasure island"],
    "The Oases": ["oases"], "The First Island": ["first island"], "Drought": ["drought"],
    "Rivals for Catan": ["rivals", "first catanians", "duel of the princes", "theme game"],
    "Big Game": ["big game"],
}


def other_scenario(text: str) -> str | None:
    """Name of a non-Great-Canal scenario named in `text`, else None."""
    t = re.sub(r"great canal", " ", text.lower())
    for name, words in OTHER_SCENARIOS.items():
        if any(w in t for w in words):
            return name
    return None


def mode_from_text(text: str) -> Mode | None:
    """The Great Canal when named, None otherwise (another scenario or no name on screen)."""
    return GREAT_CANAL if "great canal" in text.lower() else None


def can_afford(hand: dict[str, int], cost: Cost) -> bool:
    return all(hand.get(k, 0) >= v for k, v in cost.items())


WALL_LIMIT_BONUS = 2      # C&K: each city wall raises the 7-discard limit by two cards


def discard_limit(mode, p) -> int:
    """The one source for the 7-discard limit: the mode's limit + 2 per city wall (review 0.3.1:
    it was written out in 9 places)."""
    return mode.discard_limit + WALL_LIMIT_BONUS * len(p.walls)


def pay(hand: dict[str, int], cost: Cost) -> None:
    for k, v in cost.items():
        hand[k] = hand.get(k, 0) - v


def missing(hand: dict[str, int], cost: Cost) -> dict[str, int]:
    return {k: v - hand.get(k, 0) for k, v in cost.items() if hand.get(k, 0) < v}


def production(terrain: str, building: str, mode: Mode) -> dict[str, int]:
    """What one building on one producing hex yields."""
    res = TERRAIN_RES.get(terrain)
    if res is None:
        return {}
    if building == "settlement":
        return {res: 1}
    # city
    if mode.ck and terrain in TERRAIN_COMMODITY:
        return {res: 1, TERRAIN_COMMODITY[terrain]: 1}
    return {res: 2}


def improvement_cost(track: str, level: int) -> Cost:
    """Cost to raise a C&K improvement track from `level` to `level+1` (levels 0..5)."""
    return {IMPROVEMENT_TRACKS[track]: level + 1}


def all_resources() -> tuple[str, ...]:
    return RESOURCES
