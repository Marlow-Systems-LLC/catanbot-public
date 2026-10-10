# Removed from docs/strategy_scenarios.md: not The Great Canal (kept for reference, NOT ingested by the RAG)

<!-- moved from docs/strategy_scenarios.md lines 106-233 (original numbering) -->
## 2. Enchanted Land (3-4 players, Seafarers + C&K)

### Rules [SRC rulebook]
- VP target: win by defeating your 6th dragon OR reaching 18 VP (4p) / 21 VP (3p) on
  your turn. catan collector page words the 3p rule as "6 treasures AND 21 VP" which
  contradicts the rulebook; the rulebook wording ("or") is the one used here.
- Setup like Great Canal: settlement, then city, then the last player's extra
  settlement. At least one piece on the coast; a coastal piece may take a ship OR road.
- Robber only, starts on the home-island desert. No pirate.
- Enchanted island: 5 deserts + 2 gold river hexes + 9 random land hexes, 19 dragon
  tokens (strength 1 flag x6, 2 flags x7, 3 flags x6) placed on 19 intersections.
- Building on the Enchanted island: settlements only on the coast, never upgraded to
  cities; no roads on it at all.
- Knight travel: you may move ONE activated knight (only one per player, ever) from
  the home island to the island if a ship line reaches an intersection there and the
  knight is connected by roads/ships to your own coastal settlement on that island.
  It deactivates after moving and can never return. If your own settlement is on the
  target intersection, it lands on a free adjacent coast intersection. Enemy settlement
  or unmovable enemy knight blocks the target.
- On the island a knight moves up to 3 intersections, freely (no roads), but may not
  end on a coast intersection. Stronger knight can expel a weaker one; the expeller
  places the loser anywhere free on the island. Moving deactivates.
- Fighting: knight on a dragon intersection; activate it, fight next turn. Flag 1: any
  knight. Flag 2: strong or mighty (level 2+). Flag 3: mighty only (level 3). Win =
  take the token, 1 VP. Lose = knight deactivated, token reflipped and stays. Can
  reactivate and retry, or move to another dragon.
- Barbarians: island knights do not defend the home island and are never deactivated.
  Inventor cannot touch island number tokens. Deserter and Intrigue cannot hit island
  knights.
- Steam notes: Saboteur/Wedding ignore dragon points when counting; dragons seen as the
  surest path (ships + politics level 3 then kill 6 dragons); body-blocking the island.

### Heuristics [INF unless marked]
Opening
- It is a 6-dragon race for the one knight you may send, so prefer ~6 successful
  fights over 18 VP. The path needs: (a) one ship line to the island coast quickly,
  (b) one knight that can reach level 2-3, i.e. lots of ore+wool and the knight
  upgrade (needs Politics track level 3 for mighty knights in C&K: level 3 knights need
  Politics 3, SRC C&K rules).
- Opening placement: ore+wool+grain emphasis (knights), plus a coast spot facing the
  island. Weights: NEED lumber 2.0 (ships only), brick 1.5, ore 2.5, wool 2.0, grain 2.0.
  Coastal spot with a short ship path to the enchanted island: +2.0.
- Coin commodity (mountains->coin? in C&K mountains give coin, pasture cloth, forest
  paper) matters: Politics 3 is needed for level 3 (mighty). Value coin x1.6 (up from 1.2).
Build order
1. Build knight(s), 1-2 ships toward the island (ship line must reach an island
   intersection that is free and not enemy-owned).
2. Level the lead knight to strong (level 2) before sending: flag-1 and flag-2 dragons
   are 13 of 19 tokens. Send it as soon as it is level 2 and a ship line is connected
   (earlier sends dominate the "first on the island" blocking position).
3. Each dragon kill takes: move (1 turn), activate (1 grain), fight next turn. So
   ~3 turns per dragon if the next dragon is within 3 intersections. Choose the path
   through clusters of flag-1/2 tokens; avoid walking into 3-flag dragons until mighty.
4. Parallel track: still play for normal VP (cities, metropolis, Longest road does not
   exist for island) so a dragon stall does not lose the game. Dragon kills ~0.8 VP
   each only if you can reach 6; otherwise treat each as 1 VP.
Special-piece valuation
- Dragon token: 1 VP, plus progress toward the 6-dragon instant win. Value
  1.0 + 0.4*(kills_so_far/6)^2 [INF].
- Island knight position: value of the dragon within 3 intersections of the weakest
  flag you can beat; a knight without a beatable dragon in range is worth ~0.
- Blocking: placing your single island knight (and expelling) on the crossing leading to
  the opponent's next dragon. Only worth it against a leader with >= 3 dragons.
Robber
- Standard (leader's best hex). Island hexes can be robbed too, but only if you have
  buildings there; the home island is the main target.
Trading
- Ore and wool are rare because everyone wants knights: raise val["ore"] and
  val["wool"] multipliers to 1.4/1.2 in this mode [INF]. Grain 1.15 already in code.
Common mistakes
- Sending a level-1 knight and being stuck behind a flag-2 dragon.
- Building roads on the island (illegal) or trying to upgrade island settlements.
- Counting island knights for barbarian defence (they do not defend).
- Forgetting only ONE knight per player may go to the island and never return.
- Assuming the 6-dragon win needs 6 distinct trips: kills are one token each.

## 3. Rise of the Inkas ("Inka")

### Rules [SRC publisher pages + opinionatedgamers review, physical game]
Caveat: I could not retrieve the actual Catan Universe rule text; everything below is
the physical standalone "Catan Histories: Rise of the Inkas" that Universe adapts. It is
NOT a Seafarers scenario.
- 3 tribes ("eras") per player in sequence. VP target 11 total: first tribe ends at 4
  VP, second at 4 more (8 total), the third finishes at 3 more (11). Players START with
  2 VP (two settlements).
- VP come only from settlements (1) and cities (2).
- When you reach the era target ("tribe at its apex") your tribe declines: all your
  roads are removed, thicket markers cover your settlements/cities, and you place a
  NEW roadless settlement anywhere on the board (the next tribe). Declined pieces keep
  producing for you but cannot be upgraded or start roads; opponents may build a
  settlement over a thicket piece, removing it.
- Longest Trade Road and Mightiest Combat Arts give gameplay advantages, not VP:
  Longest trade route = one 1:1 trade per turn; Mightiest combat arts = may remove the
  robber each turn if it sits on a hex touching your buildings.
- Resources: feathers, coca, fish plus sea/jungle goods. No port needed to trade, three
  permanent bank rates: 3:1 identical goods -> any, 2:1 matching goods -> any, 3:2
  different goods for two. Exact resource list, build costs, and robber/dev-card
  specifics: NOT confirmed. Mark UNKNOWN.

### Heuristics [INF, low confidence]
Opening
- Placement of the first tribe is the foundation of the whole game: the declined pieces
  keep producing. So settle on 6/8 hexes early and aim for fast first-era completion
  only when your second tribe's start location is prepared.
- Dale Yu tip [SRC]: build roads next to opponents' settlements so you can claim their
  sites when they decline. Weight: road toward an opponent settlement that is at
  3+ VP of their era target: +0.04 (win-prob units used in brain.py).
Build order
- Era 1: 2 settlements -> expand to 4 VP via 1 city + 1 settlement (cheapest: 2 more
  settlements = 4 VP; cities cost 2 VP for 5 cards). Prefer settlements to cities when
  the era target is close (VP/card).
- Do not rush an era end if an opponent can take your resource spots first: simulate
  "which hexes would I lose" before crossing the threshold.
Special pieces
- Thicket (declined) settlement: still produces, so keep valuing its pips at 100% for
  income but 0% for expansion; value of destroying an opponent's thicket by building
  over it = their pips x 1.0 + your new site pips.
- Longest-trade-road card: value ~ 1 card per turn = 0.8 of a 3:1 harbor [INF].
- Combat-arts card: value ~ 0.5x a robber-block avoidance for your best hex.
Robber
- Standard: block the highest-pips hex of the opponent closest to finishing their
  current era; the bot's own hexes touching a thicket/declined pieces are still
  vulnerable, so remove the robber (if you hold the card) every turn.
Common mistakes
- Ending an era while the next tribe has no good open spot.
- Leaving declined settlements next to strong opponent roads (they overbuild).
- Planning on ports: no ports needed; the 3:1/2:1/3:2 bank rates are permanent.


<!-- moved from docs/strategy_scenarios.md lines 271-280 (original numbering) -->
7. rules.py:70 `enchanted_land` is C&K only, vp 13, no seafarers flag. Real: Seafarers
   + C&K, 18 (4p) / 21 (3p) VP, or 6 dragons. Dragons, the island, the one-knight rule,
   no-roads-on-island, no-city-on-island, flag strengths: nothing exists in the code
   (grep "dragon|treasure" has zero hits). Also `start_with_city` = True is right.
8. rules.py:71 `inka` is modelled as Seafarers with 12 VP and base building rules.
   Real (physical): 11 VP across three eras (4/4/3), starts with 2 VP, VP from
   settlements/cities only, tribe decline with thickets, roads removed, new roadless
   settlement placement, new commodities, permanent 3:1/2:1/3:2 bank rates, and
   longest-road/army replaced by effects. None modelled. board.py:281 maps `inka` to the
   seafarers board.

<!-- moved from docs/strategy_scenarios.md lines 302-302 (original numbering) -->
- Enchanted Land summary: https://catancollector.com/enchanted-land

<!-- moved from docs/strategy_scenarios.md lines 304-308 (original numbering) -->
- Catan Universe Steam discussion (Great Canal/Enchanted Land strategy): https://steamcommunity.com/app/544730/discussions/0/1639787494950905885
- Rise of the Inkas overview: https://www.catan.com/game/catan-rise-inkas
- Rise of the Inkas (Catan Universe page): https://catanuniverse.com/en/rise-of-the-inkas/
- Dale Yu first impressions: https://opinionatedgamers.com/2019/01/15/dale-yu-first-impressions-of-catan-histories-rise-of-the-inkas/
- Catan Studio announcement: https://www.catanstudio.com/2018/02/08/catan-rise-of-the-inkas-announced/

EDITED, original: Rules for Great Canal and Enchanted Land come from

EDITED, original: Rise of the Inkas rules are only partly confirmed (see section 3).


EDITED, original: ## 4. Rules mismatches

EDITED, original: ## 5. Sources

EDITED, original: unbranched ship lines, Enchanted Land and Inka rules)

EDITED, original: miss that. Also "enchanted" ok, "inka" would not match "Inkas"? It does ("inka" in
   "inkas"), fine.

EDITED, original: # Catan Universe scenarios: rules and bot heuristics

EDITED, original: ## 1. The Great Canal
