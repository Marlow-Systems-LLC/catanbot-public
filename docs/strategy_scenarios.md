# The Great Canal scenario: rules and bot heuristics

Researched 2026-10-02. Rules for Great Canal come from the printed
"Treasures, Dragons & Explorers" rulebook (Catan Universe implements it; in-game values
may differ slightly, so the bot should still read the VP target from the screen).
Tags: [SRC] = from a source, [INF] = my own inference/guess, to be tuned in sim.

## 1. The Great Canal (3-4 players, Seafarers + C&K)

### Rules [SRC rulebook]
- VP target: 18 (4 players), 21 (3 players), checked on your own turn.
  Catan Universe may show a different number; trust the screen.
- Setup: each player gets 2 settlements + 1 city. Place settlement first, then a CITY
  (second round), each with one road or ship, on the home island. The last player to
  place the city places their second settlement immediately. Clockwise the others place
  their second settlement. Starting resources come from the LAST piece placed (and that
  includes grain from barren farmland hexes).
- Must have at least one settlement or city on the coast (the eastern lake does NOT
  count as coast). A coastal piece must have a SHIP (not a road) placed with it.
- Robber only, no pirate. Robber starts on a desert.
- Farmland (fields) hexes are barren: their number tokens start face down and they
  produce NO grain until the canal is finished.
- Canal: 9 canal tiles. Catan chips (1 VP each) sit on the hexes to be dug. When 2
  ACTIVATED knights stand on intersections of a chip-marked hex, the canal is dug at
  once (on activation of the second knight): both chips are removed and a canal tile
  placed. If both knights are the same player's, that player takes both chips (+2 VP).
  If different players, each gets one chip (+1 VP each). Knight strength is irrelevant.
  Knights stay activated afterwards. A third player cannot slip in before it is dug.
- Canal finishes when the penultimate (8th) tile is built; the last tile is then also
  placed (could be 2 tiles at once if a knight sits on a crossing of two chip hexes).
  Remaining chips leave the game, tiles flip to water, desert and farmland tokens flip,
  and grain is produced again from then on.
- Knights may only be BUILT on the home island. Three jobs:
  1. Canal builder (above).
  2. Gold prospector: with a ship line to an intersection of a small-island gold hex, a
     knight placed there gives any 1 resource whenever 8 is rolled. Needs no activation.
     Two knights on one gold hex still give only 1 resource. Lost once the canal is done.
  3. Trader: a knight on a small-island harbor intersection gives that harbor's special
     port. No activation needed.
  Knights on small islands do NOT count for barbarian defence and are not deactivated.
- No settlements on the small islands. Ship lines must not branch in this scenario.
- Irrigation progress card works even on barren farmland (grain card gain).
- Community note (Steam thread): ships cannot be built on lakes. Gold hexes and
  Irrigation are called strong; third settlement next to a grain hex is the usual plan
  to field the first active knight (grain activates knights).

### Heuristics [INF unless marked]
Opening
- Weight grain access LOW on pips for production (locked), but keep at least one
  adjacent fields hex for later and for the starting hand: the final placement pays
  grain from barren fields (SRC) so a last-placed spot touching 1-2 fields = free
  starting grain. Value: +1.0 per adjacent fields hex on the last placement only.
- Grain is needed to ACTIVATE knights (1 grain) and every city/dev card. Before the
  canal is done the sources are: starting hand, bank/port trade, gold knights (any
  resource on 8), Irrigation, commodity trades. So value ports and gold prospecting
  higher than normal: 2:1 grain port is worthless pre-canal only if you cannot reach
  grain, so prefer 3:1 or a brick/ore port. Weight 3:1 = 1.5, specific port = 1.0 +
  0.4*pips (existing code), gold access +3.0 (existing canal_adj bonus is similar).
- Canal-adjacent intersections: a settlement or city touching a chip hex lets roads/
  knights reach it. Bonus +2.0 for a chip-hex corner, +3.0 if the corner is also on a
  hex that borders two chip hexes (a crossing: one activation can yield 2 tiles).
- Need lumber+brick for roads/ships to reach the canal and coast; ore/wool for knights
  (knight 1 wool + 1 ore). Ore+wool pips are worth ~1.2x vs base. Keep NEED weights:
  lumber 3, brick 2.5, ore 2, wool 1.5, grain 1.0 until canal done.
- Place the second-round city on the best ore/wool/brick spot (city doubles it) and
  keep the settlement for the coast/ship requirement. The coast piece should have the
  ship direction toward a gold hex or special harbor.
Build order
1. Round 1-3: build 1-2 knights (home island), activate with grain from the starting
   hand. Aim for 2 activated knights adjacent to the nearest chip hex.
2. Dig priority: any hex where YOU can supply both knights (+2 VP, strongly best).
   Second best: a hex where one opponent supplies one (+1 VP each, symmetric, so only
   do it when you are ahead or both of you are behind the leader).
3. Never complete the canal if you are not the VP leader or if the leader owns more
   grain-producing buildings (finishing hands grain to whoever is on fields). Stall:
   keep the last two hexes undug unless you own >= 2 fields corners.
4. After the canal: city on a fields hex becomes the best city target (grain +
   commodity-free: fields have no commodity in C&K, so a city gives 2 grain).
Special-piece valuation
- Canal chip (1 VP) = a settlement's VP but costs only knight time: value 0.9 VP each,
  but +0.3 for the denial effect on the opponents.
- Knight on a chip-hex corner: 0.5 VP expected if the second knight can arrive within
  2 turns, else 0.15.
- Gold prospector knight: expected 1 resource per 8-roll = 5/36 of turns ~ 0.14 card
  per turn; worth ~ one 5-pip hex corner, but only until canal completion. Rate
  0.6 of a settlement's pip value if canal is >5 turns away, 0.2 if <3 turns.
- Harbor knight: special port for free: value = existing harbor bonus x 0.9.
Robber
- The robber starts on the desert; put it on the leader's best NON-fields hex (fields
  produce nothing before canal, so robbing fields is wasted until the canal completes
  and a perfect pre-emptive spot after).
- After canal completion the robber on the opponent's best fields hex is a premium
  target (everyone is starved of grain at that moment).
Trading
- Players short on grain pay a premium: sell grain at 2:1 vs normal 1:1 if you are the
  only grain holder (weight val["grain"] *= 1.5 exists; keep it, raise to 1.8 early).
- Do not give grain to the opponent who has knights adjacent to a chip hex.
Common mistakes
- Spending the starting grain on a settlement rather than a knight activation.
- Digging a canal for the opponent's benefit (symmetric +1 VP) when they lead.
- Settling next to gold islands (illegal: no settlements on small islands) or relying
  on a branching ship line (illegal).
- Forgetting that gold prospecting ends when the canal is finished.

## 2. Rules mismatches (our model vs the real rules)

File:line references are in catanbot.

Status 2026-10-02: #1 fixed (GameState.target, 21 VP with 3 players); #2, #3 and #5 fixed by
state.dig_canals (two active knights, no cost, done at the 8th tile); #4 fixed (Hex.barren,
only the western fields in canal_board). Still open: #6 to #8 (gold/harbor knights,
unbranched ship lines). Live board: a barren field
reads as no token, so it produces nothing until the tokens flip (UNCONFIRMED on screen).

1. rules.py:67-69 `great_canal` VP fixed at 18. Real: 18 for 4 players, 21 for 3. The
   comment on line 66 mentions this but the fallback ignores player count. Fix: set
   vp_target from player count (Mode.players exists, rules.py:50).
2. rules.py:69 note "canal chips = 1 VP each" is right, but the code in brain.py:245-248
   gives +1 VP and +1 canal_progress for every `canal` action. Real: a dig needs TWO
   activated knights on a chip hex and yields 2 chips total (+2 VP to the diggers
   together, split if different owners), and builds exactly one tile (canal_progress +1
   per dig). Model counts each action as a lone 1-VP chip. sim.py:159-160 likewise.
3. rules.py:25 `"canal": {"brick": 1, "lumber": 1}` is invented. Real cost: none.
   Digging costs knight activation (1 grain each) and positions, not brick+lumber.
   brain.py:287-288 offers canal when "touches" (road/building/knight on a corner,
   `_touches_canal` brain.py:292-294) with no knight requirement; real requirement is
   2 activated knights on the same chip hex.
4. state.py:100-101, sim.py:88: grain locked on ALL `fields` hexes. Correct only if every
   fields hex is a barren farmland; real board has only the western farmland hexes
   barren (SRC says "crop fields ... as shown"); other fields hexes presumably produce.
   Board approximation board.py:253-266 treats `canal` as a hex type producing nothing,
   but in the real scenario the canal is dug ON land hexes (a tile is laid over a
   landscape with chips), and the desert A and farmland are flooded. Model differs.
5. state.py:63 `canal_total = 9` and completion at 9. Real: finished when the 8th
   (penultimate) tile is placed, then the 9th is placed too. Completion tests at
   brain.py:287, 398, 309 use `< canal_total`; sim.py:25 derives it from the count of
   canal hexes, which will not be 9 on the approximate board.
6. rules.py:61 and 67 `ck=True` + `seafarers=True` on great_canal is right (S+C&K), but
   knights built only on the home island, gold-prospector (any resource on 8) and
   harbor-knight rules (SRC) are absent from brain.py/state.py/sim.py (grep finds no
   gold-knight or harbor-knight handling). Ship lines must not branch: not modelled.
9. rules.py:77-79 `mode_from_text`: "Great Canal" in Universe may be spelled "Grand
   Canal" (Steam thread titles it "The Grand Canal"); the keyword "great canal" would
   miss that.
10. brain.py:70-71 `val["grain"] *= 1.5` for locked grain. OK in spirit. brain.py:74-125
    setup: `_setup_kind` (82-84) correctly gives a city second; BUT real setup order is
    settlement, city, then the LAST player places an extra second settlement right after
    their city and gets starting resources from the last piece. The code ignores starting
    resources from the last piece and the "extra settlement for last player" sequence.
    `_opening_value` (97-115) also does not treat the coast/ship requirement
    (at least one coastal piece, with ship) at all, and `_spot_extras` (117-125)
    gives +3.0 for any canal-hex corner but not for the gold-island/harbor-knight route.
11. brain.py:107-110 NEED penalties assume grain pips are needed; in the pre-canal
    phase fields pips are 0 production, so `have <= 0` for grain applies -2.0*1.5 to
    every opening in great_canal, which wrongly punishes all spots. `_spot_pips`
    (brain.py:86-95) does not filter locked grain for canal mode (state.py:100 does).
    Make `_spot_pips` skip fields when mode.canal and canal unfinished, and skip the
    grain NEED penalty.

## 3. Sources
- Printed rulebook (Treasures, Dragons & Explorers, translated): https://cdn.1j1ju.com/medias/23/0f/1b-catan-treasures-dragons-explorers-rulebook.pdf
- Scenario pack page: https://www.catan.com/treasures-dragons-adventurers
