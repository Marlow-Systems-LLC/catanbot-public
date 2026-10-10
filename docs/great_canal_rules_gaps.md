# Great Canal: rules gaps (audit 2026-10-03)

Inputs: docs/great_canal_rules_research.md (printed rules: Treasures, Dragons & Explorers scenario 5 + C&K 2020 rulebook; no Catan Universe rule page exists online) and docs/great_canal_rules_implemented.md (code map with file:line).

Legend: DONE = fixed 2026-10-03 (tests/test_rules_audit.py, tests/test_canal_chips.py); FIX = was being fixed (all FIX rows are now DONE in the sim) (source-backed); ASK = needs the owner or a screen check; LATER = known, not started.

## Implemented wrongly
| # | Rule | Ours | Status |
|---|---|---|---|
| W1 | Canal is 9 tiles, finished when the 8th is placed | Fixed 2026-10-03: board.canal_hexes = the 9 fixed chip hexes (they keep producing; channel crosses their edges), canal_total 9, done at the 8th dig; generator and live read use data/boards/canal_template.json | DONE |
| W2 | Before the first barbarian attack nobody moves the robber or steals on a 7 (discards still happen) | Fixed (owner confirmed Universe skips the robber move) | DONE |
| W3 | Defender tie: no VP, each tied player draws a progress card of choice | Nothing | FIX |
| W4 | A pillaged city loses its wall | Wall stays | FIX |
| W5 | One metropolis per track (a player can hold several); level 5 first takes it from a lower holder | Max one per player, no per-track owner, no steal | FIX |
| W6 | Knights are built on the home island only (reach small islands by moving along ships) | Fixed (owner confirmed): built on main-island corners, knight-move action reaches islands (sim/brain; live clicks not calibrated) | DONE (sim) |
| W7 | No knight on a small island defends | Only gold-corner knights excluded | FIX |
| W8 | Gold prospector stops when the canal is finished | Keeps paying | FIX |
| W9 | Discarded progress cards go under the deck | Removed from the game | FIX |
| W10 | Trader knight (harbour use without a building) is a small-island rule | Any knight on any harbour corner | LATER |

## Not implemented
| # | Rule | Status |
|---|---|---|
| N1 | Aqueduct (science 3): a roll that gives you nothing gives 1 resource of choice (a grain source) | FIX |
| N2 | Knight chases the robber (gold knights only on their gold hex, owner) | FIX (sim) |
| N3 | Knight move along own roads/ships | LATER |
| N4 | Knight displaces a weaker knight | DONE (sim: moves.knight_moves, Sim._displace; live has no knight-move clicks) |
| N5 | Merchant piece: 2:1 on its hex + 1 VP | LATER |
| N6 | Real progress-card effects in the sim (now generic: warlord / steal / +2 cards) | LATER |
| N7 | Ship moving; one unbranched ship line per player | LATER |
| N8 | Piece limits 15 roads / 15 ships | LATER |
| N9 | Leftover canal chips leave the game when it finishes (no chip pool) | LATER (minor) |
| N10 | Live: off-turn 5th progress card discard screen; progress-card clicks are off (progress_live=false) | LATER (needs a game) |

## Confirmed right
Setup structure (settlement, city, coastal settlement + ship); start cards only from the last settlement, barren fields pay start grain; nothing built on gold corners, 1 gold per hex for knights only; robber only, no pirate; event die 3 ships + 3 gates, draw on red <= level+1; progress hand 4, VP cards not held; barbarians at 7, cities vs active knights, lowest exposed lose a city, metropolises safe, all knights deactivate after an attack; defender +1 VP; costs; commodities; walls +2 hand limit; trading house 2:1; canal dig by 2 active knights at no cost, knights stay active, 2 chips to one owner or 1 each; 18 VP (4 players) / 21 (3).

## Unconfirmed in Catan Universe (no source)
3rd setup round order (owner says reverse; cardboard text is unclear), VP target (screen value wins; one live table showed 19), friendly robber setting, progress deck counts, Irrigation while barren, whether Merchant counts toward the VP target.
