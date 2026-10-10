# The Great Canal - rules research

## Name (owner check 2026-10-03)
- Real name: **The Great Canal**. In Catan Universe: Solo > Special Scenarios > The Great Canal;
  the start panel says "Victory Points: 18". Every live/solo game log reads `rules: great_canal target=18`.
- Origin: scenario 5 of the Treasures, Dragons & Adventurers pack (German "Schatze, Drachen & Entdecker";
  older English "Treasures, Dragons & Explorers"), https://www.catan.com/game/treasures-dragons-adventurers.
  The printed pack needs the base game + Seafarers + Cities & Knights, but this is its own scenario,
  NOT plain Seafarers: Cities & Knights rules (knights, barbarians, commodities, progress cards, city
  improvements) plus ships only for the coastal settlement and the ship line to the gold islands.
  No pirate, no island bonus, no settlements on the small islands, grain fields barren until the canal is done.
- NOT "Greater Catan": that is a different scenario (pack scenario 4, 20 VP, impoverishment).
  Code: rules.MODES["great_canal"]; "greater_catan" exists only for completeness and is never played.

Date: 2026-10-03. Method: WebSearch/WebFetch; official PDFs downloaded and text-extracted.
Key sources (abbrev):
- TDA = Treasures, Dragons & Explorers rulebook (English translation of the German rulebook), scenario 5, https://cdn.1j1ju.com/medias/23/0f/1b-catan-treasures-dragons-explorers-rulebook.pdf
- CK = official Cities & Knights 2020 rulebook, https://www.catan.com/sites/default/files/2021-06/catan_c_k_2020_rule_book_200708.pdf
- STEAM = Catan Universe Steam thread, https://steamcommunity.com/app/544730/discussions/0/1639787494950905885
- CATAN = https://www.catan.com/treasures-dragons-adventurers
Confidence: C = confirmed in a source, I = inferred, U = unconfirmed, NS = no source found.

NOT FOUND: any Catan Universe help/wiki/patch-note page, BGG thread or Reddit post stating Universe's Great Canal rules. (Fandom wiki returned HTTP 402; Seafarers appendix PDF has no Great Canal content; BGG/Reddit not reachable via search.) So all "Universe differs from cardboard" items are NS unless noted. The cardboard rules below are the reference; the Universe build may differ.

## Owner's known facts: confirm / correct
- 3 setup rounds (fwd settlement, rev city, then a 3rd settlement+ship): PARTLY CONFIRMED. Cardboard: round 1 all place a settlement; round 2 a city (in place of the 2nd settlement) "for the order of placement the base game rules apply" (i.e. reverse); then the LAST player to place a city immediately places their 2nd settlement with road/ship, then the others place their 2nd settlement clockwise. Net effect: 3 placements per player; the 3rd round order is not stated as "reverse" in text (starts with the last city placer = reverse-order first, then clockwise). Owner's "reverse" for round 3 is I/unverified. TDA p.13. C (for structure), U (exact order of round 3).
- Start cards only from last round: CONFIRMED. "Each player receives the raw materials of the landscape hexes surrounding their last established settlement (including any grain for a barren farmland)." TDA. C. Note: grain from barren fields IS given at setup per the text (a barren field still pays grain in the start hand).
- Gold "8" hexes at the bottom, only knights collect, nothing built on gold corners: CONFIRMED. Gold prospector knight on an intersection of a gold-river hex on a small island, reached by a line of ships, gets any 1 resource on a roll of 8, active not needed; 2 knights on the same gold hex still give only 1 resource. "No settlement may be built on the small islands." TDA. C.
- No grain fields produce (barren): CONFIRMED until the canal is finished; farmland number tokens are turned face down, then flipped when the canal completes. TDA. C.
- Progress hand limit 4: CONFIRMED (C&K base rule; VP cards do not count). CK. C.
- Event die has 3 ship faces: CONFIRMED (3 ship, 3 city gates: 1 each blue/green/yellow). CK p.5 and catan.com C&K FAQ https://www.catan.com/cities-knights. C.

## Setup / board
- Players: 3 or 4 only. TDA. C.
- Each player starts with 2 settlements and 1 city (city replaces 2nd settlement slot), each with one adjacent road or ship, on the home island. TDA. C.
- Restriction: each player must build at least one settlement/city on the coast (the east lake does not count as coast); a coastal settlement/city requires placing a ship (not a road). TDA. C. (Owner note "3rd coastal settlement + ship" is consistent.)
- Land layout: farmland and mountain hexes fixed; desert "A" (basin, west) is the double-sided hex; mountain lake "B" east. Light-brown area gets 4 brick, 5 wool, 4 wood shuffled. Two Gold River hexes and small islands. Number tokens of gold river and farmland placed as printed; farmland tokens face down; remaining tokens random with no adjacent red numbers. Harbours and Catan VP chips placed per the map. TDA. C. Exact positions are only in the map image; NS in text.
- Robber only; no pirate ("played only with the robber"). The robber starts on any desert. TDA. C.
- Knights may only be built on the home island. TDA. C.
- No ship branching: "shipping lines should not be branched" (one open ship line only). TDA. C.
- Ships on lakes: STEAM says rules do not forbid ships on lakes but in practice not allowed. Source: poster, partial. U.

## Canal and VP
- 9 canal tiles, each canal hex marked with Catan (VP) chips. TDA. C.
- Building a canal: when 2 ACTIVATED knights stand on intersections of a hex with Catan chips, the canal is dug immediately on activation of the second knight. The 2 chips are removed, a canal tile (trench side) is placed linking an adjacent canal or chip hex. Strength does not matter. Knights stay active (not deactivated). Both knights may belong to one player or two different players. TDA. C.
- Rewards: if both knights belong to one player, that player gets both removed chips (2 VP); if different players, each gets one chip (1 VP each). Each chip = 1 VP. TDA. C. (Universe displays these as "Canal Points": STEAM, C.) STEAM also notes Master Merchant ignores Canal Points in the VP comparison (bug report). U.
- Completion: as soon as the penultimate (8th) tile is placed, the canal is finished; the last (9th) tile is placed too if not simultaneously built (a knight activation on a vertex between two chip hexes can build two). The 2 remaining chips leave the game. All tiles flip to water side, the desert and farmland tokens flip; grain is produced again. TDA. C.
- Strategy (not rules): roads between two canal hexes plus knights planted there are a potent VP source; 3rd settlement on grain hex helps first active knight and Irrigation. STEAM. C (opinion).

## Knight roles in this scenario
- Canal builder (above). Gold prospector: see above; loses function when the desert is flooded. Trader: a knight at a harbour intersection on a small island uses that harbour, activation not required. TDA. C.
- Knights on the small islands do not count toward barbarian defence. TDA. C.
- Irrigation progress card works even while farmland is barren (gives 2 grain per adjacent farmland hex, cities do not double). TDA, CK. C.

## Victory
- 21 VP (3 players) / 18 VP (4 players), reached on your own turn. Cardboard TDA. C. Universe: CATAN-adjacent search summary and STEAM-linked listing repeated 21/18 but Universe-specific page not read. U for Universe.
- C&K base game default is 13 VP; this scenario overrides. CK. C.

## Seafarers rules (as combined with C&K, CK p.13 "Seafarers variant")
- Rules about roads also apply to ships. C.
- Barbarians attack all islands; all cities and knights on all islands are counted (except this scenario: small-island knights excluded). C.
- Knights may move across sea if roads/ships connect start and end; a knight may be moved to a sea intersection (not placed new) if a ship line reaches it. A knight must always be connected to a settlement/city of its colour; ships may not be moved if that breaks the link. C.
- Knights on sea intersections can chase the pirate; not relevant here (no pirate). C.
- Ship cost/move rules (1 lumber + 1 wool; move one ship per turn; no moving a ship built that turn; open end only) are base Seafarers rules, not fetched this run. NS.
- Gold: in this scenario gold is collected only by knights (above), not by settlements. C.

## Cities & Knights (CK, official)
- Dice: yellow + red + event die each turn. Event die: 3 ship faces (barbarians advance 1), 3 city gate faces (blue/green/yellow). Gate: players whose matching improvement shows the red die number draw a progress card of that colour. CK p.5. C.
- Barbarian track: 7 spaces; attack when ship reaches the last space. The Barbarian ship returns to the start and all knights flip inactive after each attack. Robber/pirate does not move until the first barbarian attack (the 7: discard check still applies, no robber move, no steal). CK p.4/11. 7 spaces from catan.com C&K FAQ. C.
- Attack: barbarian strength = number of cities (metropolis counts) on the whole board. Defence = sum of ACTIVE knights (basic 1, strong 2, mighty 3). Barbarians win if stronger: the player(s) with the lowest active knight total lose one city (to settlement; wall destroyed); players with only metropolises or no cities are skipped. Ties/defence win (knights >= barbarians): highest total = Defender of Catan, +1 VP card (6 in deck); tie for highest = no VP, each tied player draws a progress card of choice. CK p.11. C.
- Improvements: 3 tracks (trade yellow/cloth, politics blue/coin, science green/paper), 5 levels; cost of level n = n commodities. Level 3 abilities: Trading House (2:1 commodities), Fortress (strong -> mighty), Aqueduct (if no resource/commodity from a non-7 roll, take any 1 resource). Level 4 = metropolis (city becomes 4 VP total, immune to pillage; 1 metropolis per track; stolen by someone reaching level 5 first). No improvements with no cities on board; no improvements beyond level 3 without a city to hold a metropolis. CK p.7-8. C.
- Progress cards: hand limit 4 (5th card must be played or one discarded; VP cards - Constitution/Printer - played at once, not counted). Cards may be played the turn drawn, never traded or stolen by robber; played after rolling, except Alchemist before. Three decks (blue/green/yellow); exact deck contents (54 cards, counts per card) are in the Almanac, not extracted. Deck contents: U/NS. CK p.9. C for rules.
- Knights: hire 1 wool + 1 ore (basic, placed inactive) on an unoccupied intersection connected to your roads; activate 1 grain (can be same turn, but no action that turn); promote 1 wool + 1 ore (strong -> mighty only with Fortress), once per turn. Actions (active knight, then deactivates): move along own roads/ships (may pass through own pieces); displace a weaker enemy knight (the displaced owner relocates it to a connected empty intersection, else it is removed; same-strength cannot displace; basic cannot displace); chase the robber (any strength, only if the robber is on one of the 3 adjacent hexes; move robber to any numbered hex, steal 1 card). Max 2 knights per strength per player (6 total). CK p.6/10. C.
- City walls: 2 brick, only under a city, one per city, max 3 per player (rule text "3 on board"), each adds +2 to the safe hand limit on a 7; destroyed if the city is pillaged. CK p.2/6. C.
- Commodities: cloth, coin, paper; produced only by cities: pasture 1 wool + 1 cloth, mountains 1 ore + 1 coin, forest 1 lumber + 1 paper (hills 2 brick, fields 2 grain). Settlements give resources only. Commodities count toward the 7 limit, are tradable and stealable; 4:1 for anything, 3:1 at a 3:1 harbour, 2:1 only with the Trading House. CK p.7. C.
- Merchant: placed by Merchant/Master Merchant progress card on a land hex adjacent to your city/settlement; 2:1 on that hex's resource; 1 VP while you control it; next merchant card transfers it. CK p.12. C.
- Hand limit on a 7: base game is more than 7 cards -> discard half (round down); each wall +2. CK states "discard half" and the example 2 cities with walls = 11 cards safe. Base 7 safe from the example (7 + 2x2). C (derived from example).
- Spend-before-robber etc. are owner strategy, not rules.

## Friendly robber / discard / Universe differences
- Friendly Robber: optional Catan variant (no robbing players with <=2 VP or similar); Universe implementation and whether it is on in Great Canal: NS.
- Universe differences from cardboard (VP target, setup order, canal point UI, no pirate, discard limit): NS beyond the STEAM remarks on Canal Points and ships on lakes.
- Hand limit on a 7 in Universe: assumed 7 (+2 per wall). I.

## Gaps to check in-game
Setup order of round 3, exact VP target in Universe, friendly-robber setting, whether Irrigation works while barren, positions of harbours/gold hexes, progress deck counts, whether the Merchant is included in the 21/18 VP.

## In-game text (Catan Universe localization, read 2026-10-03)

Source: English strings in `CatanUniverse.app/Contents/Resources/Data/resources.assets` on the server
(read with `strings`, not by opening menus, so game 8 was not disturbed).

- Menu name: "(C&K+Sf) The Great Canal"; rules keyword: "Special rules".
- Description: played with the Cities & Knights and Seafarers rules, **without the pirate**. Win with
  **21 VP (3 players) or 18 VP (4 players), on your own turn**. Reformed robber: the robber may be
  moved **only upon or after the first barbarian attack**.
- Board: the canal tiles are "desert basin" hexes; at the start "every value chip on a desert is turned
  upside down" (grain fields dry); when the canal is fully dug "now grain can be grown".
- "Canal Points" tooltip: victory points earned by building canal parts.
- Gold hexes are "gold rivers" ("receives X from a gold river").
- Game log actions: build ship, **move ship**, build canal piece, Inventor swaps two number chips,
  Deserter places a knight, bank trade.
- Friendly robber tooltip (generic, all scenarios): the robber may not go next to a player who has
  **only 2 VP**. Our code uses 4 VP (owner's table setting, memory owner-canal-rules-confirmed): OPEN.

Confirmed vs the code: 18/21 VP (state.py:83), win only on own turn (sim.py:494), no pirate, robber
lock until the first attack, canal chips = VP (sim.py:71), grain barren until the canal is done.
The "19 VP" table seen earlier is wrong. Not in the text: setup order, deck counts, Irrigation while
barren, Merchant VP, how many active knights dig one piece.

## In-game Almanac: "Great Canal" (authoritative for Catan Universe, read 2026-10-04)

Source: `StreamingAssets/Almanac` in the Catan Universe install (base64; decoded copy at
data/ref/almanac_en.txt, entry almanac_115). Paraphrased:

1. C&K + Seafarers rules, **no pirate**. Win: 21 VP (3p) / 18 VP (4p), **on your own turn**.
   Robber can be moved only at/after the first barbarian attack.
2. Setup: each player starts with **2 settlements + 1 city**, all on the home island, placed in the
   order **settlement, then city, then 2nd settlement**. At least one must be on the coast; every piece
   built on the coast also gets a **ship** placed there. The eastern **lake is not coast**. Start
   resources come from the **last settlement placed** (infertile grain fields pay grain too).
3. Knights: built **only on the home island**. Three jobs:
   - **Canal builders**: when 2 **activated** knights stand on corners of a marked canal hex, that canal
     section is dug **immediately**. One owner: **2 VP**; two owners: **1 VP each**. Knights stay
     active. Strength does not matter. A third player cannot join once the 2nd knight activates.
   - **Gold diggers**: with a ship route to a gold-field corner on a small island, a player may **move a
     knight** there. While it stands there the player gets **any 1 resource on every roll of 8**
     (knight need not be active; 2 knights on gold still give only 1).
   - **Merchants**: a knight moved onto a **harbour corner of a small island** gives its owner that
     harbour's trade rate (need not be active).
   - Knights on the small islands **do not defend** against the barbarians.
4. Canal completion: **9 sections**. If only one is missing at the **end of a turn** it is built
   automatically, **no VP** for it. Then fields produce grain again; small-island knights stop being
   gold diggers but **keep the merchant (harbour) job**.
5. Other: **no settlements on the small islands**; ship routes **may not branch**; **Irrigation** pays
   even while fields are infertile.
- Owner decision 2026-10-04: friendly robber threshold is 4 VP (players with 4 VP or fewer cannot be robbed and their hexes are off limits). This overrides the generic tooltip's 2 VP. Code: state.FRIENDLY_ROBBER_VP=4.
