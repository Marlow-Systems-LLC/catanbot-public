# Catan strategy as bot numbers: Cities & Knights + ships/gold, for The Great Canal

Tags: [S] = stated by a source (URL list at end), [I] = my inference or a number to tune, [?] = not confirmed.
Source coverage caveat: BGG threads returned 403 to the fetcher, and I found no colonist.io or Catan Universe
strategy articles for these modes.
Weights are written against brain.py `NEED`, `COMMODITY_W`, `_opening_value`, `_spot_extras` (lines ~74-125).

Current brain.py baseline: NEED lumber 3.0, brick 2.5, grain 1.5, ore 1.5, wool 1.0; COMMODITY_W paper 1.6, coin 1.2,
cloth 0.8; -2*w for a missing resource, -0.5*w if below 3 pips; +1.0 per distinct resource; -2.0 if C&K with no
commodity terrain; ports 1.5 (3:1) or 1.0+0.4*pips (2:1); canal +3.

## 2. Seafarers (ships and gold)

- Opening: wool + lumber matter more because ships cost 1 lumber + 1 wool and you need many. [S catanrulesguide]
  Seafarers deltas: lumber 3.0 -> 3.4, wool 1.0 -> 1.6 in NEED when the map has islands to reach; keep brick. [I numbers]
- Gold: low numbers usually (2,3,11,12), so a gold hex is worth less than a plain 6/8; take it only with a decent number
  or to fill a gap. [S catanrulesguide, BGG-summary result] Codeable: gold weight 1.4 (existing) but multiplied by 0.8 if the gold hex pip <= 2; and
  treat gold as a wild covering ONE missing resource, not all. [I]
- New islands: first settlement on a foreign island gives special VP (scenario dependent, usually 2). Get there early;
  it also takes exclusive resources and denies rivals. [S wikipedia, catanrulesguide]
  Rule: ship-route extension scores +2.0 per ship toward an unclaimed island with first-landing bonus; +0.5 otherwise. [I]
- Ships: one open-ended ship can be moved free per turn; keep a route open until you commit. [S] Don't spend 3+ ships
  when a land road reaches an equal spot. [I]
- Longest trade route: roads and ships join through settlements, so it is easier than base longest road. [S] Value same as LR (+2 VP), but contested more. [I]
- Opening spots on the coast are better than inland because they have ship access: +1.0 for a spot with >= 2 sea edges
  that leads toward unclaimed land. [I]
- Mistakes: overbuilding ships (each ship is a card pair, no production); settling the home island out and never reaching an island; chasing gold on 2/12.

## 3. Cities & Knights (13 VP in base C&K)

### Setup
- Each player starts with one settlement and one city (brain's `start_with_city`). The city picks the commodity hex(es). Production: city on forest = 1 wood + 1 paper, pasture = 1 sheep + 1 cloth,
  mountain = 2 coin... actually 1 ore + 1 coin; so a city on ore keeps ore scarcer. [S ultraboardgames: ore is scarcer because cities give 1 ore not 2]
- Priorities in order: (1) ore on a strong number, (2) wheat reliably (to activate knights before the first barbarians), (3) 4-5
  distinct numbers, (4) >= 4 resources, (5) spread, (6) expansion. [S ultraboardgames CK strategy guide]
- Ports matter less than in base: commodity trading and trading house substitute. [S ultraboardgames] Weight ports x0.6 in C&K. [I number]
- City placement: put the city where it touches 2 commodity-producing hexes if possible; 1 is enough only when
  the hex is 6/8. Numbers: city spot pips on commodity terrain >= 5 (pips of forest/pasture/mountain under the city). [I]
- No commodity under city at all: existing -2.0 is fine; raise to -3.0 if the settlement also has none. [I]

### Commodity ranking (paper vs coin vs cloth)
- Sources agree: green track (science/paper) is strongest: aqueduct, and the alchemist and inventor are the two best progress cards. [S ultraboardgames CK guide] The BGG-search result echoes: aqueduct is "widely touted as the most powerful" ability, trading house sometimes ranked higher. [S search-result summary of BGG/ultraboardgames considerations; I did not read the thread]
- Yellow/trade (cloth): most valuable early when resources are scarce; trading house is 2:1 commodities (cloth/anything) and the merchant cards. [S]
- Blue/politics (coin): weakest, since fortress adds little; but you need some coin for metropolis races and for a defence-heavy plan. [S]
- Ranking: paper > cloth > coin for general purpose. This changes brain.py's current order (paper 1.6, coin 1.2, cloth 0.8), which ranks coin above cloth. Suggested: paper 1.6, cloth 1.2, coin 0.9. [S for order, I for numbers]
  Exception: bot that plans a defender/knight build (mighty knights need sheep) values wool higher, not coin. Coin placement of a city on a mountain is mostly an ore decision.
- Race logic: metropolis = +2 VP once you reach level 4 first on a track, steal-able only by reaching level 5 first (never wasted improvements). [S rules] So concentrate on ONE track; a split 2/2/2 yields nothing. [I from rules]
- Flip the first page (level 1) of all three tracks early: it gives 2/36 per track = progress-card chances, cheap (1 commodity), ~15% total per roll. Subsequent levels add ~2.75% each. [S ultraboardgames CK guide]
  Rule: buy level 1 of each track by turn ~8; then commit to paper to level 3 (aqueduct), keep going to 4 if you hold the lead. [I timing]

### Builds
1. Early: one knight before the first barbarian attack. Losing the first city is "almost impossible to win from". [S]
   Knight activation costs 1 wheat (grain); one active basic knight is the minimum. Barbarians attack when ships hit a count of 7 steps; the weakest contributor loses a city. [S rules]
   Rule: ensure active strength >= barbarian strength (= number of cities+metropolises on the board) before the track ends;
   if short by 1-2 and you are the weakest contributor, buy a knight (and pay to activate). If you are not weakest, do not over-invest. [I]
2. Mid: cities, then level 1-3 on one track; 1-2 city walls only if rolls of 7 hurt (each wall +2 hand size, max 3 walls; = 11 cards with three). [S] Source ultraboardgames says walls are minimal value because brick is abundant; take them when you hold > 7 cards at a 7 is likely. [S/I]
3. Late: metropolis (+2 VP) and longest road: "You will likely need either the longest road or a metropolis" to reach 13. [S] Defender-of-Catan VP (+1 per defense win to the best contributor). [S]
- Barbarian defender VP: knights on the board with >= 2 more strength than the next player makes defender plan viable. [S]
- Deactivating knights late: if you lead the barbarian count by >= 2, deactivate to avoid awarding the VP to a weaker opponent. [S]
- Progress cards: not optional; flip early. [S]

### Robber, 7s
- The robber does not move until barbarians first reach Catan; a 7 only triggers the discard check. [S errata]
  Code: before the first barbarian attack, a 7 only costs hand-limit; so hold more commodities early. [S]
- Hand limit is 7 (+2 per city wall). Commodities count. [S] Spend or trade down at hand >= 8 without walls.
- Aqueduct: if you produce nothing (including robber-blocked), you draw 1 card of your choice; not on a 7. [S]
- Robber placement later: block opp's commodity hex that leads their track; steals lose cards that matter. [I]

### Trading
- Scarcity: ore (high need, low supply) > wheat > wood > sheep > brick. [S ultraboardgames]
  Numbers for C&K resource_value: ore 1.5, grain 1.3, lumber 1.1, wool 1.0, brick 0.8. [I]
- Use 2:1 ports for brick/wood/sheep; keep ore for your own cities/knights; trade wheat when abundant. [S]
- Merchant/Trading house: 2:1 commodity conversions; value of trading house level 3 in the bot: +2.5. [I]

### Common mistakes
- No knight before barbarians; overvaluing ports; spreading improvements across all three tracks; ignoring progress cards;
  announcing longest road too early (invites an embargo); letting wheat run out (blocks knights and cities); stacking low numbers. [S]

## 4. Proposed concrete changes for brain.py (all [I] unless tagged)

| Item | Current | Proposed | Basis |
|---|---|---|---|
| COMMODITY_W | paper 1.6, coin 1.2, cloth 0.8 | paper 1.6, cloth 1.2, coin 0.9 | S: green best, yellow strong early, blue weakest |
| NEED (C&K) | as base | ore 2.0, grain 2.0 (knights need wheat; ore scarce) | S ultraboardgames |
| NEED (Seafarers) | as base | lumber 3.4, wool 1.6 | S catanrulesguide |
| Duplicate-number penalty | none | -0.15 * pips of the number if already owned | S diversity advice |
| Port value in C&K | 1.5 / 1.0+0.4p | multiply by 0.6 | S ports less important in C&K |
| Gold on 2/3/11/12 | 1.4*pips | 1.4*pips*0.8 and wild covers only one missing resource | S gold is low-numbered |
| Min pips first spot | none | prefer >= 11, hard-floor penalty if < 9: -1.5 | S 11 good, 9-10 minimum |
| Second-spot gap fill | +1.0 per type | +1.5 per previously missing type | I |
| Knight before barbarians | n/a | force a knight build at barbarian step >= 4 if strength < attack | S |
| Hand size | n/a | spend when hand >= 8 (no wall); >= 10 / 12 with walls | S rules |
| Robber score | n/a | opp_pips * (1+0.5*leader) - 0.7*own_pips; no steal from hand >= 8 | S+I |

## Sources
- https://ultraboardgames.com/catan/cities-and-knights-strategy-guide.php (read; main C&K source)
- https://www.ultraboardgames.com/catan/cities-and-knights-strategic-considerations.php (read; strategies and wall notes)
- https://mykindofmeeple.com/catan-strategy-tips-how-to-win-beginner-advanced/ (read; base pips, ranking, robber)
- https://catanrulesguide.com/seafarers (read; Seafarers tips)
- https://en.wikipedia.org/wiki/Catan:_Seafarers and https://en.wikipedia.org/wiki/Catan:_Cities_%26_Knights (search summaries)
- https://en.doc.boardgamearena.com/Gamehelpcatanck (C&K rules; search summary only)
- https://www.unboxedtheboardgameblog.com/post/2017/06/08/catan-seafarers (search summary)
- https://colonist.io/catan-rules/seafarers (rules, search listing only)
- https://boardgamegeek.com/thread/2166619/catan-cities-and-knight-advanced-tactics (403, not read; listed only)
- https://boardgamegeek.com/thread/515290/6p-catan-wseafarers-and-c-and-k-play-by-play-with (403, not read; listed only)
- https://catancollector.com/images/rules-archive/6e-2025/05-CN3087-CATAN-CitiesKnights-Rulebook.pdf (rulebook, not read)
