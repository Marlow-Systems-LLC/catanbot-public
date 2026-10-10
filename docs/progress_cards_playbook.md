# Progress cards playbook (Cities & Knights)

Owner request, 2026-10-06: "research every development card and how it can be used for itself
and against others and teach the bot how to use every card it has. should never hold cards it
can use."

This covers all 54 progress cards: 18 Science (green), 18 Trade (yellow) and 18 Politics (blue).
For each card it gives the effect, how it helps us, how to use it against others, the best timing,
and the only reasons to hold it. Code: `catanbot/progress_cards.py` (one `choose_<card>` each,
`choose_play` for the order, `NO_USE` for the logged hold reasons, `spy_pick` for the Spy).
Card data and sources: `docs/progress_cards.md`.

Marks: **[UNCONFIRMED]** means a rule we could not check in the rulebook text or in Catan Universe.
**[owner]** means an owner rule that overrides the plain rules.

## Rules that apply to every card

- You play progress cards only on your own turn, and any number of them per turn.
- The Alchemist is played before the dice. Every other card is played after the roll, at any
  point in the trade and build phase, so it can come before or between builds.
- Hand limit: 4 progress cards, not counting VP cards. You may hold a 5th during your own turn
  but must play or discard down to 4 when the turn ends. A 5th card drawn on another player's
  turn is discarded at once. **[owner]** (owner-event-die)
- Printer and Constitution are VP cards. They are revealed when drawn and can't be stolen.
- Progress cards don't count toward the 7-roll discard limit, but the cards they bring in do.
- A card received this turn (drawn, or taken with the Spy) may be played this same turn. The
  C&K rules don't bar it the way base-game development cards are barred.
  **[UNCONFIRMED in Catan Universe.]** The Spy flow below depends on it. If the game refuses the
  play, the runner's refused-card check marks the card, and the next turn plays it.
- Before the first barbarian attack the robber doesn't move. **[owner]** The Bishop is held
  until the first attack.
- Great Canal: there are no grain fields while the canal is open, so grain comes only from
  trades, gold and the aqueduct. Any card that brings in grain is worth more. **[owner]**

**When holding is right (owner 2026-10-06).** Hold a card only for a stated reason:
1. A better timing this turn or next. The Alchemist goes before the roll. Gain cards wait at end
   of turn when the cards they bring would sit through a 7. A Warlord with one idle knight and
   the barbarians far away is saved for the turn before an attack.
2. The Bishop waits for the first barbarian attack.
3. There is no legal or useful target.

Every hold is logged as `progress <phase>: holding <card>: <reason>` (brain.progress_play).

**Can't be played (owner 2026-10-06).** Each card below has a "Can't be played when" line: the
game's precondition, not strategy. `pc.can_play(gs, me, card, phase)` checks it before every
choice (`choose_play`) and again before the runner drags a card (`runner.play_progress`), so a
card that can't be played is never tried. The hold is logged as `holding <card>: can't play —
<reason>`. VP cards (Printer, Constitution) are never played; they count when drawn.

---

## Science (green): 18 cards

### Alchemist (2)
- **Effect:** Before rolling, choose the result of both production dice. The event die is still
  rolled and resolved first.
- **Can't be played when:** After the dice are rolled (before the roll only). (owner-confirmed or card text; `pc.can_play`)
- **For us:** Pick the sum that pays us most. Pick a low red die, because a low red die makes
  every city gate we've built pay a progress card. Never pick 7.
- **Against others:** Pick the sum that pays the leader least. The bot scores our gain minus
  half the leader's gain. A non-7 roll also keeps the robber away from our hand.
- **Timing:** Own turn, before the roll only.
- **Hold when:** No sum pays us more than the leader. Otherwise play it: a chosen roll always
  beats a random one. After the roll it waits for the next turn.
- **Bot change (2026-10-06):** It used to be held unless the roll was worth 2+ cards. It now plays
  on any positive net gain.
- Owner-confirmed 2026-10-06: a 7 may be chosen; the bot picks it only to clear the robber off its own hex, never over the hand limit.

### Crane (2)
- **Effect:** One city improvement this turn costs 1 commodity less. Two Cranes can't be stacked.
- **Can't be played when:** We have no city, or no track below level 5 whose next level we can pay with 1 commodity less (level 1 is free). (owner-confirmed or card text; `pc.can_play`)
- **For us:** Push the development tracks in the order paper > coin > cloth. A level-1
  improvement costs 1 commodity, so with the Crane it's free (owner-confirmed 2026-10-06: a Crane
  on a level-0 track is a free level).
- **Against others:** None directly. Reaching level 4 first takes the metropolis race.
- **Timing:** After the roll. The runner taps the improvement right after playing the card.
- **Hold when:** No improvement is affordable this turn even at 1 commodity less, or we have
  no city.

### Engineer (1)
- **Effect:** Build one city wall free. Max 3 walls, and each wall adds 2 to the discard limit.
- **Can't be played when:** We have no city without a wall, or 3 walls already. (owner-confirmed or card text; `pc.can_play`)
- **For us:** We can hold 2 more cards safely on a 7 (owner rule: walls on cities to cut 7
  losses). A wall does not save a city from the barbarians, and a pillaged city loses its wall
  (owner-confirmed 2026-10-06; the bot and sim drop the wall, so the discard limit falls by 2).
- **Against others:** None.
- **Timing:** After the roll, as soon as we have an unwalled city.
- **Hold when:** No unwalled city, or 3 walls already.

### Inventor (2)
- **Effect:** Swap two number tokens. The 2, 12, 6 and 8 can't be moved.
- **Can't be played when:** Fewer than two number chips that may move (3/4/5/9/10/11, not on gold). (owner-confirmed or card text; `pc.can_play`)
- **For us:** Move a 5 or 9 (or a 4 or 10) onto our hex.
- **Against others:** Move a 3 or 11 onto the leader's best hex. The bot
  scores our income gain minus half the leader's gain.
- **Timing:** After the roll. The new numbers pay from the next roll.
- **Hold when:** No legal swap gains us at least 1 pip net. The threshold was 1.5 pips and is now
  1 (2026-10-06).
- **Great Canal (owner-confirmed 2026-10-06):** The face-down chips of barren farmland MAY be
  swapped while the canal is dug; gold hexes may not. A barren chip pays 0 while barren, so the
  swap helps only by moving a good number away from a rival or onto our producing hex. The live
  click flow is still **[UNCONFIRMED]** (PLAY_HELD).

### Irrigation (2)
- **Effect:** Take 2 grain for each fields hex next to at least one of your buildings. A hex counts
  once, and cities don't double it.
- **Can't be played when:** No fields hex next to our settlements or cities (barren canal fields count). (owner-confirmed or card text; `pc.can_play`)
- **For us:** In the Great Canal this is one of the best cards: grain is locked, and barren
  fields still count (owner-confirmed 2026-10-06).
- **Against others:** None. Grain we hold is grain they can't trade from us.
- **Timing:** Early in the main phase, so the grain gets spent this turn (spend before the
  robber).
- **Hold when:** No fields hex next to our buildings. At end of turn it also waits if the hand
  would end over the discard limit.

### Medicine (2)
- **Effect:** Upgrade a settlement to a city for 2 ore and 1 grain instead of 3 ore and 2 grain.
- **Can't be played when:** No settlement, the city cap is reached, or we can't pay 2 ore + 1 grain. (owner-confirmed or card text; `pc.can_play`)
- **For us:** A cheap city on the best-pip settlement. Owner rule: put cities on the commodity
  hexes.
- **Against others:** None.
- **Timing:** After the roll, whenever we hold 2 ore and 1 grain.
- **Hold when:** We can't pay 2 ore and 1 grain, have no settlement, or are at the city cap.
  Follow-up (not built): the planner could trade toward the Medicine price.

### Mining (2)
- **Effect:** Take 2 ore for each mountains hex next to at least one of your buildings.
- **Can't be played when:** No mountains hex next to our settlements or cities. (**[UNCONFIRMED]** in Catan Universe; permissive check; `pc.can_play`)
- **For us:** Ore for cities and knights.
- **Against others:** None.
- **Timing:** Same as Irrigation.
- **Hold when:** No mountains next to our buildings, or at end of turn the hand would end over
  the limit.

### Printer (1)
- **Effect:** 1 VP, revealed at once when drawn, can't be stolen.
- **Can't be played when:** Never played: 1 VP, revealed when drawn. (owner-confirmed or card text; `pc.can_play`)
- **Timing:** Immediately. Never held.

### Road Building (2)
- **Effect:** Build 2 roads free. With Seafarers rules, ships are allowed (The Great Canal).
- **Can't be played when:** No legal road or ship edge, or all road and ship pieces are out. (owner-confirmed or card text; `pc.can_play`)
- **For us:** Reach a settlement spot, an undug canal corner, or Longest Road.
- **Against others:** Cut a rival's road line, or take a spot first.
- **Timing:** After the roll, before other builds, so the new roads can carry a settlement.
- **Hold when:** No legal road or ship edge, or the piece cap is reached.

### Smith (2)
- **Effect:** Promote up to 2 knights one level, free. A mighty knight (level 3) needs politics 3
  (the fortress).
- **Can't be played when:** No knight can go up: level cap (strong without politics 3) or both pieces of the next level are on the board. (owner-confirmed or card text; `pc.can_play`)
- **For us:** Barbarian defence (one active knight per city, **[owner]**), and stronger knights
  for chasing the robber and displacing rivals.
- **Against others:** A stronger knight can displace a rival's knight.
- **Timing:** After the roll. With the barbarians 2 steps away or less and a defence gap, it jumps
  the queue.
- **Hold when:** No knight can go up: the level cap applies (2 without politics 3), and there are
  only 2 pieces per level.

---

## Trade (yellow): 18 cards

### Commercial Harbor (2)
- **Effect:** Each opponent must take 1 resource from you and give you 1 commodity of their
  choice, if they have one.
- **Can't be played when:** We hold no resource to give, or no opponent holds cards. (owner-confirmed or card text; `pc.can_play`)
- **For us:** Turns spare resources into commodities, and paper, coin and cloth drive the
  development tracks.
- **Against others:** Takes commodities from rivals racing for a metropolis. Give each rival the
  card we value least.
- **Timing:** Main phase, before building.
- **Hold when:** We have no spare resource, or no opponent holds cards. At end of turn it waits
  over the limit.

### Master Merchant (2)
- **Effect:** Choose a player with more VP than you, look at their hand, and take 2 cards.
- **Can't be played when:** No opponent has more VP than us and holds cards. (owner-confirmed or card text; `pc.can_play`)
- **For us:** Take the 2 cards our plan is missing (`on_take_cards` picks by `resource_value`).
- **Against others:** Always aim it at the leader, ideally the one with the biggest hand.
- **Timing:** Main phase, before building.
- **Hold when:** Nobody has more VP than us, or the players ahead hold no cards.

### Merchant (6)
- **Effect:** Place the merchant on a land hex next to one of your buildings. You trade that
  hex's resource 2:1, and you get 1 VP while the merchant stays yours. It moves to whoever plays
  the next Merchant.
- **Can't be played when:** No land hex (not gold) next to our settlements or cities. (owner-confirmed or card text; `pc.can_play`)
- **For us:** +1 VP, and 2:1 on the card we get most of (`merchant_hex_value`, owner-approved
  2026-10-06).
- **Against others:** Playing it takes the VP and the 2:1 from the current holder.
- **Timing:** Main phase, first after VP cards, so the 2:1 is used this turn.
- **Hold when:** No land hex next to our buildings. Otherwise always play it.

### Merchant Fleet (2)
- **Effect:** This turn, trade one resource or commodity of your choice 2:1 with the bank, as
  often as you like.
- **Can't be played when:** Never refused: it applies for the turn, so we can trade into a pair after. (owner-confirmed or card text; `pc.can_play`)
- **For us:** Play it when 2:1 on one card finishes a build we can't make now (`fleet_build`,
  owner-approved), when we hold 4 or more of one card, or when the hand is over the discard
  limit.
- **Against others:** None.
- **Timing:** Main phase, last, after the other cards have changed the hand.
- **Hold when:** No 2:1 trade finishes a build and there's no stack of 4. The live UI is
  unconfirmed (PLAY_HELD).

### Resource Monopoly (4)
- **Effect:** Name a resource. Each other player gives you 2 of it, or what they have if fewer.
- **Can't be played when:** No opponent holds cards (the game allows it, but it gains nothing: hold it). (owner-confirmed or card text; `pc.can_play`)
- **For us:** First the haul that covers what the plan is missing. Then a resource every rival
  is known to hold 2 of. Then the most cards, leaning to grain (canal) and to cards we trade 2:1
  (owner-approved 2026-10-06).
- **Against others:** It drains the card a rival needs for their next build. The hand ledger
  estimates each hand.
- **Timing:** Main phase, before building. At end of turn it waits over the limit.
- **Hold when:** The expected haul is under 2 cards and doesn't finish the plan. The live UI is
  unconfirmed (PLAY_HELD).

### Trade Monopoly (2)
- **Effect:** Name a commodity. Each other player gives you 1 of it.
- **Can't be played when:** No opponent holds cards (the game allows it, but it gains nothing: hold it). (owner-confirmed or card text; `pc.can_play`)
- **For us:** The commodity our next improvement needs.
- **Against others:** Slows a rival's metropolis race.
- **Timing:** Main phase, before building.
- **Hold when:** The expected haul is under 1.5 commodities and no rival is known to hold one.

---

## Politics (blue): 18 cards

### Bishop (2)
- **Effect:** Move the robber, then take 1 random card from each player with a settlement or city
  next to its new hex.
- **Can't be played when:** Before the first barbarian attack (robber locked), or no hex to move it to. (owner-confirmed or card text; `pc.can_play`)
- **For us:** Free the robber off our hex, and steal from several players at once.
- **Against others:** Put the robber on the leader's best hex. Friendly robber: no hex touching
  a player with 4 VP or fewer. **[owner]**
- **Timing:** Main phase, before building, so the stolen cards get spent.
- **Hold when:** Before the first barbarian attack **[owner]** (the robber is still inactive), or
  when no hex hurts the rivals more than it hurts us.

### Constitution (1)
- **Effect:** 1 VP, revealed at once when drawn, can't be stolen. Never held.
- **Can't be played when:** Never played: 1 VP, revealed when drawn. (owner-confirmed or card text; `pc.can_play`)

### Deserter (2)
- **Effect:** Choose an opponent. They remove one of their knights (their choice), and you may
  place a knight of the same strength free.
- **Can't be played when:** No opponent has a knight on the board, or we have no free knight piece to set (a taken knight drops a level when we have both of its level out). (owner-confirmed or card text; `pc.can_play`)
- **For us:** A free knight for barbarian defence or a canal corner. Owner rule: place it where
  it scores.
- **Against others:** It cuts the strongest knight player's defence, which can cost them a city
  when the barbarians land. Owner rule: aim at the runaway leader first.
- **Timing:** Main phase. With the barbarians near and a defence gap, it jumps the queue.
- **Hold when:** No opponent has a knight.
- **Owner-confirmed 2026-10-06:** The knight placed keeps the level and the active/inactive
  state it had. With no piece of that strength left it drops to the next free level (a strong
  knight becomes basic when both our strong knights are out); with no free piece at all the card
  can't be played (owner-confirmed 2026-10-06).

### Diplomat (2)
- **Effect:** Remove an open road or ship, meaning one with an end touching none of its owner's
  pieces. If it's your own, you may place it somewhere else (a ship on a ship edge).
- **Can't be played when:** No open road or ship on the board (one end touching none of its owner's pieces). (owner-confirmed or card text; `pc.can_play`)
- **For us:** Move our own dead-end road or ship toward a better spot (`_diplomat_own_move`).
  The live UI for the move is **[UNCONFIRMED]** (probe-captured).
- **Against others:** Break the Longest Road holder's chain (2 VP), shorten the leader's road,
  or clear a road that blocks our way.
- **Timing:** Main phase, before roads, so our road can take the freed edge.
- **Hold when:** No opponent has an open road.
- **Bot change (2026-10-06):** It used to play only on the leader's road when the leader held
  Longest Road or the road touched ours, and was held otherwise. It now takes any rival's open
  road, ranked Longest Road holder +3, leader +2, touching our network +1.5.
- **Owner-confirmed 2026-10-06:** Diplomat can target ships, rivals' and our own.

### Intrigue (2)
- **Effect:** Displace an opponent's knight on a junction that one of your roads or ships touches
  (ships owner-confirmed 2026-10-06). Its owner must move it, or it's removed. You don't need a
  knight of your own.
- **Can't be played when:** No opponent knight on a junction one of our roads or ships touches. (owner-confirmed or card text; `pc.can_play`)
- **For us:** Reopen our road (Longest Road) or a canal corner.
- **Against others:** Take the leader's active, high-level knight off the board ahead of an attack.
- **Timing:** Main phase.
- **Hold when:** No opponent knight stands on a junction our roads touch.

### Saboteur (2)
- **Effect:** Each player with as many VP as you or more discards half their hand of resources
  and commodities, rounded down (owner-confirmed 2026-10-06; ties are hit).
- **Can't be played when:** No opponent with as many VP as us or more holds 2+ cards. (owner-confirmed or card text; `pc.can_play`)
- **For us:** Only when we trail or tie. It never hits us.
- **Against others:** Hits the leader's saved-up hand before a big build.
- **Timing:** Main phase. Best when the targets hold big hands, such as after their production.
- **Hold when:** The targets would lose fewer than 2 cards in total. The threshold was 3 and is
  now 2 (2026-10-06).
- **Edition note [UNCONFIRMED]:** The German original's "Sabotage" turns a city on its side
  instead. The bot assumes the English 5th-edition effect.

### Spy (3)
- **Effect:** Look at another player's progress cards and take one. VP cards can't be taken.
- **Can't be played when:** No opponent holds progress cards. (owner-confirmed or card text; `pc.can_play`)
- **Whom:** The player with the most progress cards, the leader on ties. With the situational
  selector on, BLOCK targets the focus rival and RACE skips the last-placed player
  (`situational.spy_target`, sim only).
- **Which card (2026-10-06, `spy_pick`):**
  1. A card we can play now for a gain scores keep value + 8, so it beats any card for later.
  2. Otherwise the card's keep value. A card with no use now counts half.
  3. Taken from the leader, half the card's keep value is added, because the leader loses it.
  4. The Alchemist (pre-roll) and the Bishop before the first attack are never "playable now".
     They're taken only for later.
- **After the steal:** The card joins our progress hand, and `brain.after_spy` re-runs
  `progress_play`, so the card is played this turn when it has a use. The Alchemist waits for
  the next pre-roll.
- **Timing:** Main phase, early (steal-and-gain cards first), so the stolen card can still be
  played.
- **Hold when:** No opponent holds progress cards.

### Warlord (2)
- **Effect:** Activate all your knights free.
- **Can't be played when:** We have no inactive knight (the game allows it, but it gains nothing: hold it). (owner-confirmed or card text; `pc.can_play`)
- **For us:** Saves 1 grain per knight (grain is gold in the canal), adds barbarian defence, and
  gives active knights for robber chases and canal digging.
- **Against others:** None directly. Active knights can chase the robber and displace rivals.
- **Timing:** Main phase. It jumps the queue when the barbarians are near and we're short of
  defence.
- **Hold when:** We have only one idle knight, the barbarians are far, we have no defence gap and
  the canal is closed. Owner wording: "a Warlord saved for the turn before an attack". While held,
  the planner never buys grain to activate knights (`warlord_covers`).

### Wedding (2)
- **Effect:** Each player with more VP than you gives you 2 cards of their choice.
- **Can't be played when:** No opponent has more VP than us and holds cards. (owner-confirmed or card text; `pc.can_play`)
- **For us:** Up to 6 free cards with 3 players ahead.
- **Against others:** Only the players ahead of us pay.
- **Timing:** Main phase, before building.
- **Hold when:** Nobody has more VP than us, or the players ahead hold no cards.

---

## Play order the bot uses (`MAIN_ORDER`)

1. VP cards.
2. Merchant.
3. The steal and gain cards: Wedding, Master Merchant, Spy, Saboteur, Bishop, the monopolies,
   Commercial Harbor, Irrigation, Mining.
4. Knights and blocks: Warlord, Deserter, Smith, Intrigue, Diplomat, Inventor.
5. Build helpers: Medicine, Crane, Engineer, Road Building.
6. Merchant Fleet, last, on the final hand.

If the barbarians are 2 steps away or less and we have a defence gap, Warlord, Smith and Deserter
go first. At end of turn, if we hold more than 4 progress cards, the bot plays a card that has a
use. If none has, it discards the card least worth keeping.
