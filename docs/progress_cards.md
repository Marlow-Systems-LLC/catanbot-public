# Cities & Knights progress cards

54 progress cards: 18 Science (green, paper track), 18 Trade (yellow, cloth track), 18 Politics (blue, coin track).
Code: `catanbot/progress_cards.py` (`CARDS`, `choose_play`). Tests: `tests/test_progress_cards.py`.

## General rules

- Draw: when the event die shows a city gate, each player whose level in that colour is high enough for the red die draws one card of that colour. Level 1 draws on red 1-2, level 2 on 1-3, up to level 5 on 1-6. So a low red die helps every player with improvements (this is why Alchemist picks a low red die).
- Play: on your own turn only. Alchemist is the only card played before the roll; all others after the roll. Any number per turn. [colonist.io, BGG cheat sheet]
- Hand limit: 4 progress cards. You may hold a 5th during your own turn, but must play or discard down to 4 when the turn ends. A 5th drawn on someone else's turn is discarded at once (to the bottom of its deck).
- VP cards (Printer, Constitution) are revealed when drawn. They don't count toward the limit and can't be stolen by Spy.
- Progress cards don't count toward the resource discard limit on a 7.
- Seafarers combination (no pirate in The Great Canal): Road Building may build ships. **[UNCONFIRMED]** whether Diplomat and Intrigue can also target ships.
- Great Canal: Irrigation still pays 2 grain per adjacent fields hex while the farmland is barren (docs/rules_research.md). That makes it one of the best cards in this mode, because grain is otherwise locked.

## Science (green), 18

| Card | # | Effect | Timing | Best use (bot heuristic) | Bot must choose |
|---|---|---|---|---|---|
| Alchemist | 2 | Choose both production dice; event die still rolled and resolved first | Own turn, **before** rolling | Sum that pays us most minus half the leader's gain; never 7; red die as low as possible. Hold it unless the best roll is worth 2+ cards, beats 1.5x our expected roll, or our hand is over the limit (avoids a 7). | dice (white, red) |
| Crane | 2 | One city improvement this turn costs 1 commodity less (no stacking) | After roll | When it makes the next improvement affordable now; science > politics > trade | track |
| Engineer | 1 | Free city wall (max 3; +2 discard limit each) | After roll | Always, on the best-pip unwalled city | city vertex |
| Inventor | 2 | Swap two number tokens (not 2, 12, 6, 8) | After roll | Put a 5/9 on our hex and a weak number on the leader's. Needs at least 1.5 pips of net gain. | two hexes |
| Irrigation | 2 | 2 grain per fields hex next to any of your buildings (not doubled for cities) | After roll | Play early in the main phase so the grain gets spent; works on barren canal fields | — |
| Medicine | 2 | Settlement → city for 2 ore + 1 grain | After roll | When we hold 2 ore + 1 grain; upgrade the best-pip settlement | settlement vertex |
| Mining | 2 | 2 ore per mountains hex next to your buildings | After roll | Same as Irrigation | — |
| Printer | 1 | 1 VP | Immediately on draw | Automatic | — |
| Road Building | 2 | 2 free roads (or ships) | After roll | Toward settlement spots, undug canal corners, longest road | 2 edges |
| Smith | 2 | Promote up to 2 knights one level free (level 3 needs politics 3) | After roll | Always; active knights first, lowest level first | up to 2 knight vertices |

## Trade (yellow), 18

| Card | # | Effect | Timing | Best use | Bot must choose |
|---|---|---|---|---|---|
| Commercial Harbor | 2 | Each opponent takes 1 resource from you and gives 1 commodity if they have one | After roll | Turns spare resources into commodities (paper > coin > cloth) | which resource to give each opponent (worst first) |
| Master Merchant | 2 | See the hand of a player with **more VP** than you and take 2 cards | After roll | Hit the leader | player |
| Merchant | 6 | Merchant on a land hex next to your building: 2:1 trades in that resource, +1 VP while you hold it | After roll | Always (1 VP, takes it from whoever holds it); pick the hex of the resource we hold most of | hex |
| Merchant Fleet | 2 | This turn, 2:1 bank trades in one resource or commodity of your choice | After roll | 4+ of one type, or hand over the discard limit (spend before the robber) | resource |
| Resource Monopoly | 4 | Name a resource; each player gives you 2 of it if they have them | After roll | Expected take of 3+ cards, weighted by need | resource |
| Trade Monopoly | 2 | Name a commodity; each player gives you 1 | After roll | Expected take of 1.5+ cards; paper > coin > cloth | commodity |

## Politics (blue), 18

| Card | # | Effect | Timing | Best use | Bot must choose |
|---|---|---|---|---|---|
| Bishop | 2 | Move the robber; take 1 random card from each player with a building on the new hex | After roll | Leader's best hex, never our own | hex |
| Constitution | 1 | 1 VP | Immediately on draw | Automatic | — |
| Deserter | 2 | Opponent removes one knight of their choice; you place one of the same strength free | After roll | Player with the most knight strength (leader on ties). Jumps the queue when barbarians are near and we are short of knights | player |
| Diplomat | 2 | Remove an open road (an end with no piece); your own road may be moved | After roll | Break the leader's longest road or a road touching ours | edge |
| Intrigue | 2 | Displace an opponent's knight on a junction your road touches | After roll | Leader's knight, a higher level, an undug canal corner | knight vertex |
| Saboteur | 2 | Every player with VP ≥ yours discards half their hand (5th ed.) | After roll | When we trail and targets lose 3+ cards together | — |
| Spy | 3 | Look at a player's progress cards, take one (not VP) | After roll | Most progress cards, leader on ties | player (then card in UI) |
| Warlord | 2 | Activate all your knights free | After roll | 2+ idle knights, or 1 when barbarians are near, we are short on defence, or canal digging needs active knights | — |
| Wedding | 2 | Each player with more VP gives you 2 cards of their choice | After roll | When 1+ players are ahead | — |

Edition note: the German original and older printings use other names: Printing Press, Forge, Trade Fleet, Trade Master, Trade Port, Trader, Defectors, Plot, Field Marshall. Its "Saboteur" puts a city on its side instead of forcing discards. `card_key()` maps these names to the keys above. The bot assumes the 5th-edition English effects. **[UNCONFIRMED which edition Catan Universe follows for Saboteur.]**

## Owner rules applied in `choose_play`

- **Spend before the robber.** In the main phase, cards that bring in cards (Wedding, Master Merchant, monopolies, Commercial Harbor, Irrigation, Mining) come first, so the brain can spend what they bring that turn. At end of turn they are skipped if the hand would end over the discard limit. Merchant Fleet is forced when the hand is over the limit.
- **Robber and blocks target the leader.** Bishop, Master Merchant, Spy, Diplomat and Intrigue all aim at the leader, chosen by visible VP with income as the tie-break.
- **One active knight per city; losing a city is near fatal.** With barbarians 2 steps or fewer away and a defence gap, Warlord, Smith and Deserter are played first.
- **Paper > coin > cloth.** Crane track order, commodity values and Trade Monopoly choice.
- **No grain in the Great Canal.** Grain is valued higher while the canal is open, which makes Irrigation and Resource Monopoly on grain worth more.

## Sources

- ultraboardgames.com Science/Trade/Politics card pages (cities-and-knights-*-cards.php)
- colonist.io/catan-rules/cities-and-knights (hand limit, effects)
- jonskeet.uk/siedler/cheat2.html (German-edition names/effects)
- BGG rules cheat sheet thread 295621 (play timing, 5th-card discard)
