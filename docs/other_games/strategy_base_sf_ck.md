# Removed from docs/strategy_base_sf_ck.md: not The Great Canal (kept for reference, NOT ingested by the RAG)

<!-- moved from docs/strategy_base_sf_ck.md lines 12-54 (original numbering) -->
## 1. Base Catan

### Opening placement
- Rule of thumb: total pips of first settlement >= 11 is good, 9-10 is minimum. [S mykindofmeeple] A 6-8-9 / 5-8-9 spot is
  13 pips, the practical maximum. [S]
- Prefer many different numbers across the two settlements (4-5 distinct), because duplicates give boom/bust and
  correlated droughts. [S ultraboardgames CK guide; mykindofmeeple] Code: penalty -0.6 per number of the second spot that
  already appears in the owned set (using the pip value of that number as scale: -0.15*pips_of_dup_number). [I]
- Cover >= 4 of 5 resources across both spots; 5/5 if a port covers the gap. [S in spirit; matches existing +1.0 per type]
- Scarcity: brick and ore are each one hex short (3 vs 4), so they are scarcer. Ranking quoted: brick > ore ~ wheat ~ wood > sheep. [S mykindofmeeple]
  Codeable: value multipliers brick 1.25, ore 1.2, grain 1.1, lumber 1.1, wool 0.85. [I, scaled from the ranking]
  Note that brain.py's NEED has lumber above brick; sources say brick is rarer. Keep both high; lumber matters
  mainly in road-heavy openings. [I]
- Ore+wheat(+sheep) spot is the city/dev-card engine; strong when brick/wood are reachable by a port or the second settlement. [I, common community wisdom]
- Ports: only worth it when you produce the matching resource (2:1) or hold a surplus-prone resource. Value of 2:1 port
  = 0.4*pips_of_that_resource (existing formula is fine); 3:1 = flat 1.0-1.5, lower than the existing if pips are poor. [I]
  Ore/wheat 2:1 port plus a 6/8 on that resource is the best port case. [I]
- Never pick 2/3/11/12 on a needed resource over 6/8 on an unneeded one. [S ultraboardgames "better to secure 6 or 8 on unwanted resources"]
- Second pick (snake order): complete the missing resources first, then pips. Weight "fills missing resource" x1.5 vs pip sum. [I]
- Expansion room: reject spots where both adjacent edges are blocked within 2 roads; subtract 1.5 if the best next
  settlement spot within 2 edges is already denied. [S "don't accept total encirclement"; number I]

### Build priorities
- Early: 2 roads + settlement toward the best 3rd spot before cities, unless ore/wheat is high. Don't build a long
  road before you can afford the settlement it leads to, since it telegraphs the target. [S ultraboardgames]
- Mid: cities on your best 6/8 wheat/ore hex; dev cards when you have ore+wheat+sheep surplus.
- Late: count VP paths: 10 = 2 settlements + cities + LR/LA/dev VPs. Hide how close you are; jump 2 VP at once (e.g.
  settlement + city) so opponents can't block. [S ultraboardgames]
- Longest road ~ 5 roads: pursue only if it also builds toward a settlement spot. [I]

### Robber, 7s, trading
- 7 comes once every 6 rolls on average (6/36). [S] Keep hand <= 7 when probability of 7 is imminent: discard risk
  rule: if hand >= 8, spend before ending turn (build, trade to bank) even at 4:1. [I]
- Robber target: the leader's highest-pip hex that is not also yours; prefer hexes serving 2+ opponents. [S mykindofmeeple]
  Don't steal from players with >= 8 cards when you'd rather they hold a big hand (steal helps them avoid discard). [S]
  Target score: value = opp_pips_on_hex * (1 + 0.5*leader_bonus) - 0.7*own_pips_on_hex. [I]
- Trade as close to your own turn as possible so resources are used immediately. [S] Avoid needing 3+ resources from trades. [S]
- Don't trade with the leader (>= 8 VP) unless you gain a build this turn; refuse trades that give them a build. [I]

### Common mistakes
- Chasing 2/12 hexes; stacking one number; no brick or no wood access; building roads before knowing target; building
  longest road late with no settlement behind it; holding > 7 cards into a 7.


<!-- moved from docs/strategy_base_sf_ck.md lines 68-69 (original numbering) -->
- Pirate: blocks ship building and moving; often better than robber early when rivals are racing to an island. Don't
  choose a worse pirate position just to steal a card. [S catanrulesguide, Unboxed blog]

<!-- moved from docs/strategy_base_sf_ck.md lines 124-139 (original numbering) -->
## 4. Greater Catan (Seafarers + C&K, 20 VP in Catan Universe)

All [I]; no source found for this mode (searched; catan.com and Wikipedia only describe the combination in general).
- Longer game: 20 VP is much longer than 13, so economy weight rises. Increase pip weight x1.1 and diversity bonus x1.2;
  raise 'city on commodity hex' bonus (+1.0) since improvements need many commodities over a long game.
- Both dials of VP: there are metropolis (2 each, 3 tracks), Seafarers island bonuses (2 per first island), longest
  road/route (2), defender of Catan. Expected path: 3 cities (3) + 2 metropolises (4) + island VP (2-4) + LR/route (2) + defender (1-3) + settlements.
- Opening: C&K rules dominate (city + settlement start, wheat + ore, commodity hex). Add Seafarers coastal access
  bonus +1.0 for a spot with sea edges toward an island, since ships need lumber + wool (also paper: forest yields both).
  Forest city = lumber + paper, the best fit for ships plus the green track. [I; consistent with section 2/3 sources]
- Barbarians and pirates both run; the pirate also may be blocked, so keep the knight count >= barbarian strength while
  ships expand. Knights are cheap; never leave a city undefended while you expand. [I]
- Island expansion is the main differentiator versus plain C&K: keep >= 1 ship route open for each turn until the first island is claimed. [I]
- Time horizon: build cities first (they are safe from barbarians only with walls... actually only metropolises are safe), delay roads.
  Cities lost to barbarians are far more costly at 20 VP. [S rules re: metropolis safe; I for emphasis]


EDITED, original: # Catan strategy as bot numbers: base, Seafarers, C&K, Greater Catan

EDITED, original: strategy articles for these modes. "Greater Catan" has no sourced strategy at all, so section 4 is entirely [I].

EDITED, original: ## 2. Seafarers

EDITED, original: ## 5. Proposed
EDITED, original: (search summary; pirate use)

