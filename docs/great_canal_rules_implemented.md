# Rules implemented for mode "great_canal" (C&K + Seafarers + canal)

Paths relative to . The bot is not a referee: rules.py:1-6 says these rules are the bot's
*model* for planning and simulation; Catan Universe highlights the legal spots on screen. Two consumers:
sim.py (self-play) and brain.py (planning, via Brain._apply). Several rules exist in only one of them (flagged below).

## 0. Mode flags
- rules.py:66-69  MODES["great_canal"] = Mode("great_canal","The Great Canal", vp_target=18, ck=True, seafarers=True, canal=True, start_with_city=True); notes: "grain locked until canal finished", "canal chips = 1 VP each", "dig = 2 activated knights on a canal hex's corners, no cost".
- rules.py:31-50  Mode defaults: discard_limit=7, max_settlements=5, max_cities=4, max_roads=15, max_ships=15, friendly_robber=False, balanced_dice=False.
- rules.py:75-82  mode_from_text: "great canal" in screen text -> great_canal.
- rules.py:52-54  has_dev_cards = not ck -> no dev cards, no largest army in this mode.
- board.py:315-322 board_for_mode("great_canal") -> real_canal_board.

## 1. Setup rounds and start cards
- sim.py:72-94 setup(): order = 0..n-1 then n-1..0 (snake), plus for canal a THIRD round n-1..0 (sim.py:75-76, "3rd (coastal) settlement + ship, reverse order like the city round (owner)").
- Round 1 = settlement; round 2 (indices n..2n-1) = CITY because start_with_city (sim.py:82-83; brain.py:185 `_setup_kind` returns "city" when placed==1; brain.py:441). So start = 2 settlements + 1 city (rules.py:65 comment, printed rules).
- Start cards: only the last placement (k >= len(order)-n) pays one card per adjacent producing hex, for hills/forest/pasture/fields/mountains (sim.py:77,88-93). Barren fields still pay start grain (sim.py:92). Gold/canal pay nothing at start. The city pays nothing (brain.py:188-190 docstring).
- Third settlement must be coastal: brain.py:471-473 (choose_setup_settlement filters board.is_coastal_vertex, board.py:136-138 = touches sea or <3 hexes) when player already has >=2 buildings.
- Setup road/ship: brain.py:477-495 choose_setup_road; with seafarers a piece off a sea edge becomes a ship (sim.py:87: sea edge and not land edge -> ship). Nothing enforces that the coastal settlement MUST get a ship (docs/strategy_lessons.md "Third settlement starts with a ship" is runner behaviour only).
- Setup legality: state.py:139-146 setup=True skips road-connection check; state.py:123-128 can_settle = land vertex + distance rule + (canal) no gold hex corner.
- Seat-order logic (which of 4 seats), harbour-first heuristics: brain.py:42-52,300-460 (not rules, strategy).
- Start-spot scoring for the canal (not rules): +3 canal-hex access (brain.py:277-278), 3:1 harbour weight 8.0 (brain.py:271-272), barren fields weight 0.15 (brain.py:222-225).

## 2. VP target
- rules.py:66 fallback 18; state.py:69-75 GameState.target: screen-read `vp_target` always wins (state.py:71-72); else great_canal with exactly 3 players -> 21 (state.py:73-74); else mode value 18.
- runner reads it from the lobby/info text: rules_reader.py:33-41 (accepts 5..25). docs/canal_playbook.md notes the live table showed 19 (app value authoritative).
- Win check in sim: sim.py:381 total_vp(p) >= gs.target after own turn (sim.py:62-63 includes hidden dev "vp", none in C&K).
- VP counted (sim.py:56-60): settlement 1, city 2, metropolis +2 (on top of the city = 4 for that city), longest road 2, largest army 2 (never held in C&K), defender_vp, canal_chips (1 each), special_vp (progress VP cards).
- brain.py:_apply tracks vp_visible: settlement +1, city +1, metropolis +2 (brain.py:524-535), longest road +/-2 (brain.py:561-565), canal chips via dig_canals (brain.py:538-541).

## 3. Building costs (rules.py:15-25 COSTS)
| piece | cost | file:line |
|---|---|---|
| road | 1 brick + 1 lumber | rules.py:16 |
| ship | 1 lumber + 1 wool | rules.py:17 |
| settlement | brick, lumber, wool, grain (1 each) | rules.py:18 |
| city | 2 grain + 3 ore | rules.py:19 |
| dev card | wool+grain+ore (unused in C&K: moves.py:97 only if has_dev_cards) | rules.py:20 |
| city wall | 2 brick | rules.py:21 |
| basic knight | 1 wool + 1 ore | rules.py:22 |
| promote knight (any level) | 1 wool + 1 ore | rules.py:23 |
| activate knight | 1 grain | rules.py:24 |
- City improvements: rules.py:27-28,111-113: tracks trade=cloth, politics=coin, science=paper; going from level L to L+1 costs L+1 of that commodity (levels 0..5: 1,2,3,4,5). improvement_cost(track, level) rules.py:111. Max level 5: moves.py:101. Needs at least one city: moves.py:100. (brain.py also uses `IMPROVEMENT_TRACKS` mapping, rules.py:28; progress_cards.py:92 has the same map with different key names.)
- City upgrade: Move("city", v) for every settlement (moves.py:93). Piece limits: settlements+cities < 5+4 (moves.py:91, brain.py:755-757, progress_cards.py:268 medicine). max_roads/max_ships are DEFINED (rules.py:42-43) but never checked anywhere (grep: no use).
- Settlement legality: state.py:139-146; roads (land edges) state.py:148-164; ships on sea edges only (state.py:155-156) but a road/ship may extend from any own building or own network end owned by nobody else (state.py:160-161).
- Walls: moves.py:115-116 max 3 walls per player, one per city, cost 2 brick. Each wall +2 discard limit (sim.py:123, brain.py:606, 836, 1031, progress_cards.py:127-128, runner.py:1428).
- Knight piece limits: moves.py:105-114: max 2 knights of each level (basic/strong/mighty).
- Everything affordable is generated by moves.build_moves (moves.py:86-117) filtered by can_afford (rules.py:85).

## 4. Production, commodities, gold
- rules.py:98-108 production(terrain, building, mode): settlement = 1 resource. City: in C&K, forest/pasture/mountains give 1 resource + 1 commodity (paper/cloth/coin; board.py:25 TERRAIN_COMMODITY); hills and fields cities give 2 of the resource; gold -> {"gold": 1 or 2}.
- Dice: sim.py:271 red,white = two d6; roll = sum. produce(roll) sim.py:97-119: every building on a hex with that number and not under the robber produces (sim.py:103).
- Barren farmland: board.py:61, 274 (canal_board marks fields with q<=0), real_canal_board keeps the template's barren flag (board.py:296). Produces nothing while canal_progress < canal_total (sim.py:104-105, state.py:107-108, brain.py:222-225, 1096). After the canal is done they produce normally (the "flip back" is just the condition failing).
- Gold: board.py:22 "gold" terrain; real_canal_board keeps gold hexes fixed (board.py:295). Settlements/cities cannot touch a gold hex (state.py:123-128 can_settle). Only KNIGHTS collect gold: sim.py:111-116: for each player, each gold hex whose number == roll and not robbed and with any of the player's knights on a corner pays exactly 1 card, regardless of active state and of how many knights (2 knights on one hex = 1). The player picks the card (brain.choose_resource, brain.py:1249-1267: grain first while the canal is open and grain is needed, else the card that completes the best build).
- Expected income (planning): state.py:99-113 (robber-aware, barren-aware; gold counted as wildcard weight 1.0; note it counts gold only via buildings, so it ignores the knight-gold income; brain._prospector_value brain.py:708-730 values knight-gold separately: 0.010*pips while canal open, 0.006 after).
- Grain lock: no special global lock beyond barren fields; grain also comes from start cards, gold knights, aqueduct (see 7), Irrigation (progress), bank trades.
- Commodities: stored in hand (state.py:199-200 new_hand has paper/cloth/coin). Discard counts them (hand sum, sim.py:124).

## 5. The 7: discard, robber, steal
- Roll 7: sim.py:279-280 -> seven(i). No production on 7.
- Discard: sim.py:121-129 for every player with more than limit = 7 + 2*walls cards (resources + commodities): discards floor(count/2) cards (c // 2), chosen by brain.choose_discard (brain.py:1130-1152; keeps cards for city/settlement/knight and defence).
- Robber move: sim.py:131-141 choose_robber (brain.py:1084-1127): any land hex except desert and current robber hex (brain.py:1090-1091), hurting the most likely winner, never itself, a barren-farmland hex is worth -0.5 (brain.py:1096-1097).
- Steal: 1 random card from a chosen victim with a building on the hex (sim.py:134-140); victim = highest win prob among those with cards (brain.py:1123-1126). Knights do not count as "players on the hex" for stealing.
- Robber start: board.robber = None for canal boards (board.py:280, 311; printed rule says "any desert"; boards have no desert in the code).
- Robber blocks gold-knight pay (sim.py:114) and production (sim.py:103).
- PIRATE: not implemented. board.py:83 declares `pirate` field, never used (grep: no other hit). Printed Great Canal rules have no pirate (docs/rules_research.md:"Only the robber is used, with no pirate"), so this is consistent.
- Friendly robber: Mode.friendly_robber (rules.py:46), read from lobby text (rules_reader.py:47-49). brain.choose_robber only: players with vp_visible <= 2 are protected (brain.py:1108-1121). The simulator ignores it (sim.move_robber never checks). Not part of MODES["great_canal"] defaults (False).
- Knight chase of the robber: NOT implemented (only a comment in docs/canal_playbook.md and state.py:131-133 docstring; "Knight robber-chase ... aren't verified" docs/canal_playbook.md last line; "playbook only (runner flow)" in the table).
- Robber lock until first barbarian attack: GameState.barbarians_attacked (state.py:63) only gates the Bishop progress card (brain.py:83-98); choose_robber does NOT lock (docs/progress_wiring_RESULT.md "Robber lock: not applied").

## 6. Event die and barbarians
- Event die: sim.py:20 EVENT_DIE = ship x3, trade, politics, science; rolled every turn when mode.ck (sim.py:273-278), BEFORE the production roll/7 (sim.py:273-282).
- Ship face: sim.py:143-169 barbarians(): barbarian_pos += 1; track length gs.barbarian_len = 7 (state.py:62); attack when pos >= 7, then pos resets to 0 (sim.py:146-148). (state.py:61 comment "0..7, attack at 7".) Starting position 0.
- Attack strength comparison: sim.py:149-151: cities = total cities of ALL players (metropolises are cities, they are counted); strength[p] = gs.defence_strength(p) = sum of levels of ACTIVE knights (state.py:46-48, 130-137). In great_canal, knights standing on a corner of a gold hex are excluded (state.py:134-137). Defenders win if sum(strength) >= cities (sim.py:151).
- Defender reward: sim.py:152-155: if exactly one player has the maximum strength, defender_vp += 1 (1 VP). Ties: nothing (the printed rule's progress-card for tied defenders is NOT implemented). 0 total strength with 0 cities also "wins" with strength tie.
- Loss: sim.py:156-165: `exposed` = players with at least one city that is not a metropolis; those with the minimum strength among exposed each lose ONE city (first non-metropolis one in set order) -> back to a settlement (-1 VP). Metropolises are immune (sim.py:157-158, brain.py:595-600 defence_gap docstring).
- Knights deactivate: sim.py:167-169 ALL knights of ALL players set inactive after every attack (win or lose), and attack counter increments (sim.py:166).
- Nothing about the "rebuilt city loses its wall" etc. Walls are not pillaged (walls don't matter for barbarians here).
- Sim pre-condition: barbarians only run in ck modes. First-attack flag: runner.py:516-525,767-771 sets barbarians_attacked from screen text ("loses a city|barbarians attack|defender of catan").
- Defence planning: brain.py:593-600 defence_gap = non-metropolis cities - defence_strength (the "one active knight per city" owner rule); brain.py:616-629 danger terms; progress_cards.py:145-153 barbarians_near / defence_gap.
- Gate faces (trade/politics/science): see progress cards (section 9).

## 7. Improvements (city improvement levels, abilities, metropolises)
- Levels 0..5, cost L+1 commodities per level (rules.py:111-113). State: PlayerState.improvements dict {"trade","politics","science"} (state.py:38).
- Draw-gate effect (the main rule use of the level): sim.py:300-321 and section 9.
- Level-3 abilities:
  - Trading house / 2:1 commodity trade: moves.py:72-73 bank_ratio: commodities trade 2:1 when improvements["trade"] >= 3 (flat 2:1 for all commodities; the printed rule is: trade level 3 = 2:1 commodity trade). Applies in planning (brain.bank_ratio brain.py:765-766) only; sim's bank_trade is applied as given.
  - Aqueduct (science 3): NOT implemented as a game mechanic. It is only mentioned as a value bonus in brain.action_value (brain.py:654-657: +0.02, +0.04 more at level 3 when canal open) and docs ("a free card of our choice on any roll that gives us nothing"). Sim does not give the card; GameState does not either. Level-3 science and politics also add a flat +0.02 (brain.py:652).
  - Fortress / knight upgrade (politics 3): mighty promotion requires politics >= 3 (moves.py:113; brain.py:640-641, 702-705). Nothing like a "fortress" (printed C&K politics-3 = mighty knights) exists beyond that. No "metropolis immunity" from fortress.
  - Level 5 effects / Level-4 effects: no extra ability. No "free road/ship" ability etc.
- Metropolis: brain.py:531-535 `_apply`: when a track reaches level >= 4 and the player owns no metropolis yet and has a city -> the FIRST city (set iteration order) becomes a metropolis, +2 vp_visible. Sim uses this via execute copying `metropolises` (sim.py:186-187). Implications (simplifications): only one metropolis per player, no per-track bookkeeping, no check that another player already holds a metropolis on that track, no level-5 steal, no choice of city. outlook.py:157-171 plans metropolis as 4 (or 5 if someone has it) with 2 VP. VP table outlook.py:20-33.
- Cranes/Engineer/Smith etc. are progress cards (section 9).

## 8. Walls
- Cost 2 brick (rules.py:21), max 3 (moves.py:115), on a city without a wall (moves.py:116); +2 hand limit each (sim.py:123); brain.wall_value brain.py:602-611 values by discard pressure. Engineer card builds one free (progress_cards.py:40-41, choose_engineer 258-264). Walls have no role against the barbarians (correct per rules).

## 9. Progress cards
- Deck: sim.py:39-40 self.decks per colour built from progress_cards.CARDS (54 cards: 18 science, 18 trade, 18 politics; counts at progress_cards.py:35-87; all listed with effect/timing).
- Draw rule: sim.py:300-321 progress_draw(track, red): on a trade/politics/science face, every player whose improvement level in that track L >= 1 and red <= L+1 draws one card (level 1 draws on red 1-2, level 5 on 1-6; sim.py:309). Red die used is the production red die (sim.py:271,278). Also stated in docs/progress_cards.md:"General rules" and docs/strategy_lessons.md:"The event die".
- VP cards (Printer, Constitution; progress_cards.py:90): special_vp += 1 immediately, not held (sim.py:316-318).
- Hand limit: PROGRESS_HAND_LIMIT = 4 (progress_cards.py:91). Rule (sim.py:300-304,320-321,323-327,329-339): a card drawn on someone else's turn that brings the hand over 4 forces an immediate discard of the lowest KEEP_VALUE card (progress_cards.py:482); on the roller's own turn it may keep a 5th until play_held ends the turn, where `_progress_limit` trims to 4. Discarded cards are counted (progress_discarded) but not returned to the deck (docs say to the bottom).
- Playing in the sim: play_held(i) sim.py:329-339, own turn after roll, plays EVERYTHING except Warlord (held until barbarians within 2 steps: pos >= len-2). _play_progress sim.py:341-356 is heavily simplified: warlord = activate all knights free (sim.py:344-346); any politics-deck card = steal one random card from the VP leader with cards (sim.py:347-353); every other (science/trade) card = +2 resources of choice (sim.py:354-356). No specific effects (no Alchemist, Crane, Smith, Irrigation, Merchant VP, etc.). The sim ignores 'printed in hand' timing (pre-roll).
- Live play: progress_cards.choose_play (progress_cards.py:487-542) with per-card choosers (188-460): alchemist pre-roll only (docs line 3-14 phases), others after roll, end_turn discards to 4 (progress_cards.py:497-498,531-541). Effects documented in CARDS (progress_cards.py:33-88): e.g. Crane -1 commodity, Engineer free wall, Irrigation 2 grain/fields hex next to your building (works on barren fields, progress_cards.py:44-45), Mining 2 ore/mountains hex, Medicine city for 2 ore + 1 grain, Smith 2 free promotions, Road Building 2 free roads/ships, Merchant 2:1 + 1 VP, Merchant Fleet 2:1 for one resource this turn (brain.start_fleet brain.py:100-101; moves.bank_ratio moves.py:60-61), Resource Monopoly 2 each, Trade Monopoly 1 each, Bishop move robber (gated by barbarians_attacked, brain.py:83-98), Deserter, Intrigue (displace knight), Diplomat, Saboteur, Spy, Warlord, Wedding.
- progress_live gate: live clicking is OFF (docs/progress_wiring_RESULT.md "progress_live is false").
- NOT in the sim: Merchant 1 VP, Merchant 2:1 hex, Merchant Fleet; sim.vp does not count a merchant (sim.py:56-60).
- Unconfirmed-edition caveats: progress_cards.py:23 ("5th-edition names; Catan Universe uses the same set: unconfirmed"), docs/progress_cards.md last note on Saboteur.
- Progress cards do not count towards the resource discard limit: sim.py:124 counts p.hand only (docs/progress_cards.md says so).

## 10. Knights
- Representation: state.py:11-14 Knight(level 1..3, active bool); PlayerState.knights: dict[vertex, Knight] (state.py:30). A knight occupies a vertex: GameState.occupied(v) (state.py:89-90) counts knights as occupying it for settlements / other knights; but distance_rule_ok (state.py:118-121) uses only buildings.
- Build (basic knight): moves.py:102-109: at a land vertex reachable via own roads (and, in canal mode, own ships: moves.py:103-104 `reach |= ships' vertices`), not occupied, only while the player has < 2 level-1 knights (moves.py:107). Cost wool+ore. Created inactive (brain.py:523-524). NOT restricted to the home island by rule (printed rule "knights may only be built on the home island" is not enforced; ships make small-island corners reachable on purpose as prospector spots, brain.py:708-730).
- Activate: 1 grain (rules.py:24); only inactive knights (moves.py:111-112). Active knights count for defence and canal digging.
- Promote: wool+ore (rules.py:23); level < 2, or level 2 only with politics >= 3, and fewer than 2 knights of the target level (moves.py:113-114); level +1 (brain.py:527-528). The promoted knight keeps its active flag.
- Move knight: NOT implemented (no move/relocate action; BUILD_KINDS moves.py:17-18 has none).
- Displace (attack another knight): NOT implemented as an action; only the Intrigue progress card chooser (progress_cards.py:425-440) and Deserter (393-398) exist as knight-affecting effects (planned in choose_play; live clicks off).
- Chase the robber: NOT implemented (see section 5).
- Deactivate after barbarian attack: sim.py:167-169 (all knights, every attack).
- Gold-corner knights: state.py:134-137 do not defend; canal knights: dig only if on canal corners.
- Knight ladder (strategy): brain.py:690-706 (one knight, a back-up, promote to strong, politics 3, mighty). One active knight per city: brain.py:593-600.
- Warlord card: activate all knights free (sim.py:344-346; progress_cards.py:84-85).
- VP from knights: none directly (outlook.py:28-31).

## 11. Ships
- Build: cost lumber+wool (rules.py:17); moves.py:95-96 Move("ship", e) for e in legal_road_edges(me, ship=True) when mode.seafarers; sea edges only (state.py:155-156 board.is_sea_edge = at least one adjacent sea hex), must connect to an own building or own ship end that no opponent building blocks (state.py:159-161). Ships are placed as PlayerState.ships (state.py:29); count towards edge_owner (state.py:92-96) and longest road.
- Ships open vertices for knights (moves.py:103-104) and for settlement connection (state.py:144).
- NOT implemented: moving ships (open-ended ship relocation), pirate, "no branching of ship lines in this scenario" (docs/rules_research.md says it exists), no ships on lakes, max 15 ships not checked, ship-to-road transitions require a settlement (not checked: roads and ships form one network in longest_road_len).
- The brain nudges for ship building: brain.py:870-873 kinds.append("ship") ("owner 2026-10-02: it never set a boat"), brain.py:644-645 _road_value.
- Initial ship: state.py/brain.py:477-495 choose_setup_road uses ships on sea edge.

## 12. The canal (digging, chips, VP, rewards)
- Board: canal hexes are terrain "canal" (board.py:58, 270 canal_board approximation: row of 7 + (-1,1),(1,-1) = 9; real_canal_board keeps canal hexes from data/boards/gc_*.json, board.py:285-312). A canal hex is LAND (board.py:26 LAND includes "canal") but produces nothing (no resource in TERRAIN_RES).
- Progress counters: GameState.canal_progress (tiles dug), canal_total default 8 (state.py:64-65 "9 tiles, but the canal completes at the penultimate one (TDE rulebook)"), canal_dug set of hex coords (state.py:66). Sim overrides canal_total = number of canal hexes on the board (or 9 if none) at sim.py:28 (inconsistent with the state.py default of 8, see below).
- dig_canals(gs) state.py:206-229: whenever called (after every knight/activate in Brain._apply brain.py:538-541, in Sim.execute sim.py:190-192), for each undug canal hex find all active knights standing on its 6 corners (state.py:216-217). If >= 2: take the first two (`knights[0]`, `knights[1]`); each of those two owners gets canal_chips += 1 (so the same owner gets +2, two owners +1 each; state.py:220-222); hex marked dug; canal_progress += 1 (state.py:223-224). No cost; knight level irrelevant; knights stay active (printed rule).
  - Completion: when canal_progress >= canal_total - 1 the counter jumps to canal_total (state.py:226-228 "last tile goes in with the 8th"), the loop breaks, and dig_canals returns immediately for future calls (state.py:210). Effect: barren farmland produces (state.py:107, sim.py:104). Remaining chips are lost implicitly (no chip pool is modelled).
  - Note: with the default canal_total=8 the canal completes after 7 digs; in the Sim canal_total = N canal hexes, so it completes after N-1 digs (e.g. N=9 -> after 8 digs, matching the printed rule). The state.py default of 8 would complete one tile too early on a 9-tile board.
  - Chips: PlayerState.canal_chips (state.py:40), each = 1 VP (sim.py:58, brain.py:538-541). There is no chip stock: chips are created per dig, and the "last two chips removed" rule is not modelled.
  - A hex can be dug by any two active knights of any players (even three players?): only the first two active knights found win; a third party cannot intervene (digging is immediate on second activation, matching printed rules). `dig_canals` loops over `gs.players` in seat order, so when two owners are mixed the order of the first two might not match "the two that arrived", but chips are symmetric.
  - Building the knight on a canal corner: a knight on a canal corner that is active digs when a partner is active on another corner. Inactive knights do not count (state.py:217). Knights built on corners also dig immediately if they were already... no, new knights start inactive (brain.py:524).
- Wiring: brain.py:569-581 candidate_actions keeps knight spots at canal or prospector corners; brain.py:584-586 _touches_canal; brain.py:630-633 first road touching canal bonus; brain.py:634-637 _canal_pair_value (660-685) values pairs (+0.05 second own knight, +0.02 activation, -0.03 activating next to the threat's knight, +0.01 shared); outlook.py:182-186 canal_chip goal.
- Unlock of grain: only the barren flag (section 4).
- Gold prospector, Trader (harbour knights): see sections 4 and 13. The prospector stops when the canal floods in the printed rules; the code does not stop it (no canal_progress check in sim.py:111-116).

## 13. Harbours and trade ratios
- Harbours: board.py:73-76 Harbor(edge, kind "3:1" or a resource). Real canal board takes fixed harbour edges/kinds from data/boards/gc_*.json (board.py:310); canal_board uses _harbors(count=9) random (board.py:207-216). harbor_at(v) returns the kind if v is an endpoint of a harbour edge (board.py:147-151).
- Ratios: moves.py:56-76 bank_ratio(gs, res, harbor_3to1, fleet): default 4; 3:1 harbour at any own building -> 3; the resource's own harbour -> 2 (returns immediately); commodities 2:1 with trade level >= 3 (moves.py:72-73); Merchant Fleet resource -> 2 (moves.py:60-61); an owner-confirmed 3:1 flag overrides (moves.py:74-75).
- Trader knight: moves.py:64-65 in canal mode `spots += list(me.knights)` -- a knight (active or not) on a harbour corner gives that harbour (printed rule).
- Bank trades: moves.bank_moves moves.py:121-129 (give `ratio` of one card, get 1 of any other RESOURCE, commodities cannot be bought); sim bank trades use brain._apply / sim.execute bank_trade (sim.py:173-177).
- Player-to-player trade: sim.py:238-256 propose_trade / evaluate_offer; brain.py:1154-... evaluate_offer; moves.offer_moves 1-for-1.
- Merchant: 2:1 on its resource is only in the progress card descriptions, not in bank_ratio (merchant hex is not tracked in state) -> NOT implemented as a ratio.
- Harbour picks at setup: brain.py:253-262 first_harbour, 262-275 canal weights.

## 14. Longest road
- state.py:166-190 longest_road_len: DFS over the union of roads AND ships (edges set), longest trail; vertices occupied by an opponent's settlement/city BLOCK passing through (state.py:174, 180-181), knights do NOT block, ships and roads are not distinguished.
- Award: 2 VP at >= 5 and strictly longer than the holder; ties keep the holder (sim.py:222-230, brain.py:555-575 _award_longest_road).
- Recomputed after each road/ship/settlement (brain.py:536-537).

## 15. Friendly robber
- See section 5. Flag only (rules.py:46), read from the lobby; used in brain.choose_robber (brain.py:1108-1121) to avoid targeting <= 2 VP players; ignored in sim.

## 16. Explicitly MISSING / simplified (with code admissions)
- sim.py:1-7 docstring: "Not a full re-implementation of Catan Universe: covers the economy, building, robber/blocking, bank and player trades, longest road/largest army and (for C&K) commodities, improvements, metropolises, knights and barbarians."
- rules.py:1-6: rules are the bot's model, not the referee; VP target fallback values.
- rules.py:61 "In-game value is read from the screen (unconfirmed)"; rules.py:65 printed 21/18.
- Not implemented at all: pirate (board.py:83 field only); knight movement, displacement and robber chasing; ship movement; ship-line no-branch rule; home-island-only knight rule; gold-knight stops after canal flood; merchant piece (2:1 and 1 VP); aqueduct effect; metropolis steal / per-track holders / city choice; defender tie rule; discard rule counting of 'cards per player' details; trader-knight for non-harbour; city-wall pillage rules; balanced dice (flag only, rules.py:47); turn timer (flag only); two-owner "dig" ordering; chip pool; hex-by-hex canal tile orientation; "knights don't count for defence on islands" is implemented only for gold-corner knights (state.py:134-137), not for harbour-corner knights on small islands.
- Simplified: progress-card effects in sim (sim.py:341-356); improvements beyond cost/draw gate; hidden VP; start order.
- Comments admitting limits: state.py:65 (canal_total 8), board.py:257-259 canal_board "approximation", board.py:166 etc.; brain.py:41-49 knobs tuned by self-play; docs/progress_cards.md "UNCONFIRMED"; docs/rules_research.md "[UNCONFIRMED]" for Catan Universe details; docs/progress_wiring_RESULT.md "barbarians_attacked is never set from the screen" (since superseded by runner.py:523-525 regex) and "Robber lock: not applied".
- Rule differences between sim and brain: sim is the only place with barbarian resolution, gold-knight production, discard on 7, event die. brain._apply is the only place with metropolis granting and longest-road VP bookkeeping (sim uses update_awards plus the copied attributes).
