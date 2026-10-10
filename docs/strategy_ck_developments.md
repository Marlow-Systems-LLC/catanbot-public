# Cities & Knights: commodities, city improvements and progress cards

## Commodities
- A city next to a forest makes paper, next to a pasture makes cloth, and next to mountains makes coin. On those hexes it takes 1 resource + 1 commodity instead of 2 resources. Hills and fields cities still give 2 resources.
- Paper builds the science (green) track, coin builds politics (blue), and cloth builds trade (yellow).
- Spend commodities fast. They are robber and 7 food like any card. They are only worth a lot inside an improvement.
- Other uses: trade them to the bank (4:1, 3:1 at a harbour, 2:1 once trade is level 3), or offer them to players who need a track.
- Order of value: paper > coin > cloth.

## City improvements
Three tracks: trade (yellow, paid in cloth), politics (blue, paid in coin), science (green, paid in paper). Rules: C&K p.7-8, checked in great_canal_rules_research.md.

### What each level gives
| Level | Cost to reach it | Progress cards (event die shows the track's gate) | Extra |
|---|---|---|---|
| 0 | - | never | - |
| 1 | 1 commodity | red die 1-2 | - |
| 2 | 2 commodities | red die 1-3 | - |
| 3 | 3 commodities | red die 1-4 | the track's ability (below) |
| 4 | 4 commodities | red die 1-5 | first player at 4 takes the track's metropolis: +2 VP, that city can't be lost to the barbarians |
| 5 | 5 commodities | red die 1-6 (every gate roll) | first player at 5 takes the metropolis from a holder at 4; at 5 nobody can take it |

All five levels of one track cost 1+2+3+4+5 = 15 commodities. Reaching level 3 costs 6, and level 4 costs 10.

### Level-3 abilities (they stay for the rest of the game)
- **Trade 3, the trading house:** commodities trade 2:1 with the bank.
- **Politics 3, the fortress:** strong knights can be promoted to mighty (strength 3).
- **Science 3, the aqueduct:** when a roll other than a 7 gives you nothing, take 1 resource of your choice. In the Great Canal it is a grain source.

### Limits
- You need at least one city to build any improvement. A city lost to the barbarians keeps your levels, but with no city left you can't add more.
- The maximum level is 5.
- Levels 4 and 5 need a city that can hold the metropolis, meaning one that is not already a metropolis. The exception is raising a track whose metropolis you already hold. With one city that already holds a metropolis, other tracks stop at 3.
- There is one metropolis per track, three in total. One player can hold several, each on a different city.
- Improvements are bought on your own turn with commodities only. Resources never pay for them. The Crane progress card makes one improvement cost 1 commodity less.
- You can hold at most 4 progress cards. The roller may keep a 5th only until the end of their turn. A higher level gives more cards, but none past the limit.
- Commodities count toward the 7-card limit and can be stolen. Spend them on improvements before a 7 hits.
- Planner and live play: the bot must know its own levels (game 17 priced every level at 1 commodity and failed 114 times). The live level is counted from landed improvements and saved in data/game_flags.json.

### Plan
- Science to 3 early (the aqueduct gives grain).
- Politics to 3 before building mighty knights.
- Trade when cloth piles up.
- Race the metropolis in a track where nobody else is close.
- Lock it at 5 when a rival is at 4.

## Progress cards: how to use each
Hand limit 4 (VP cards don't count). Play on your own turn after the roll, except Alchemist (before the roll).

### Science (green)
- **Alchemist:** set both dice before rolling. Pick a number that pays you and not the leader; never 7.
- **Crane:** the next improvement costs 1 commodity less.
- **Engineer:** a free city wall (+2 hand limit).
- **Inventor:** swap two number tokens (not 2, 12, 6 or 8). Better numbers on your hexes, worse on the leader's.
- **Irrigation:** 2 grain per fields hex next to you. It works on canal farmland, so it's a big grain source.
- **Mining:** 2 ore per mountains hex next to you.
- **Medicine:** a city for 2 ore + 1 grain.
- **Road Building:** 2 free roads or ships.
- **Smith:** promote 2 knights free.
- **Printer:** 1 VP.

### Trade (yellow)
- **Commercial Harbor:** each opponent takes 1 of your resources and gives you 1 commodity.
- **Master Merchant:** take 2 cards from a player with more VP.
- **Merchant:** 2:1 on that hex's resource and 1 VP while you hold it.
- **Merchant Fleet:** 2:1 on one card type for the turn.
- **Resource Monopoly:** 2 of a resource from each player.
- **Trade Monopoly:** 1 of a commodity from each player.

### Politics (blue)
- **Bishop:** move the robber and take 1 card from each player there.
- **Deserter:** an opponent removes a knight and you place one of the same strength free.
- **Diplomat:** remove an open road.
- **Intrigue:** push away an opponent knight touching your road.
- **Saboteur:** players with at least your VP discard half.
- **Spy:** steal a progress card.
- **Warlord:** activate all your knights free.
- **Wedding:** each player ahead gives you 2 cards.
- **Constitution:** 1 VP.

### Defence
Keep a Deserter or Warlord when the barbarians are close. Expect opponents to play Deserter and Intrigue on a lone knight, so keep a back-up knight.
