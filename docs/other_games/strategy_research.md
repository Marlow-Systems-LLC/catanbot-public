# Removed from docs/strategy_research.md: not The Great Canal (kept for reference, NOT ingested by the RAG)

<!-- moved from docs/strategy_research.md lines 14-45 (original numbering) -->
## 1. Base Catan

Opening placement
- Aim for 11+ pips on the first settlement; 9-10 acceptable [S: MKM]. Pips: 6/8=5, 5/9=4, 4/10=3, 3/11=2, 2/12=1 [S: https://mykindofmeeple.com/?p=1018253].
- Prefer ore + wheat access; "no strategies can do without ore and wheat", while strategies needing little brick/wood exist [S: https://mykindofmeeple.com/catan-strategy-tips-how-to-win-beginner-advanced/ search summary, MKM].
- Balance pips of wood vs brick and ore vs wheat (a shared number gives a free road/city step) [S: same].
- Diversify numbers across the two settlements so more rolls pay you; avoid both on the same number [S: MKM, HEX].
- Remember brick and ore have one fewer hex than the other resources (3 vs 4) [S: MKM] - scarce resources are worth more in trades.
- Place near a port matching your abundant resource; a 2:1 port next to a high-output hex of that resource is very strong [S: MKM, HEX].
- Leave room to expand; point roads toward open territory; roads toward the board edge are less risky than through the middle [S: HEX, MKM]. Last seat gets back-to-back placements [S: HEX].

Build order
- Early: brick/wood for roads and settlements; late: ore/wheat for cities and dev cards [S: HEX].
- Upgrade early settlements to cities (double yield) [S: MKM].
- Ore/wheat/sheep (cities + dev cards) is a standard fast path; wool-port strategy only works if you own the wool port and need just 1-2 other resources [S: MKM].
- Late game pick one path: longest road (5+), largest army (3+ knights), dev-card VPs [S: BGA102].

Trading
- Trade before you build (players refuse once they see what you built); trade as close to your own turn as possible [S: HEX, MKM].
- Prefer trading with players lagging / furthest from their turn; avoid trades that help the leader; avoid 4:1 bank unless needed [S: MKM, HEX].
- Needing two different resources by trade is manageable; three is "close to impossible" [S: MKM]. Ask for more than one card for the one you want [S: BGA102].

Robber
- Block the highest-pip hex shared by 2+ opponents; target leader or resource competitor [S: MKM, HEX].
- Do not steal from a player holding 8+ cards (let them discard instead) [S: MKM].
- Keep own hand under the discard limit (7 before a 7 is rolled) [S: HEX]. Matches owner rule "never sit near the discard limit".

End game / hidden VP
- Do not reveal VP; make the winning jump in one turn (e.g., city + VP card, or road/army swing) to avoid being blocked [S: MKM].
- Track opponents' longest road/army and block with settlements that break roads [S: BGA102].
- [G] Count hidden dev-card VPs: opponent with many unplayed dev cards at 8-9 VP is likely at 10. Bot should assume +1 VP per 3 unplayed dev cards held late.


<!-- moved from docs/strategy_research.md lines 52-52 (original numbering) -->
- Pirate: moved on a 7 instead of the robber; steals from adjacent ships; blocks building/moving ships on its hex; does not block harbours [S: Wikipedia, same].

<!-- moved from docs/strategy_research.md lines 92-98 (original numbering) -->
Greater Catan (official rulebook, TDE)
- Seafarers scenario; with C&K the game ends at 20 VP (18 without) [S: TDE, Greater Catan section].
- 8 cities per player (4 extra) [S: TDE].
- Impoverishment rule: placing a road/ship next to a hex with no number token forces you to place a token from the reserve, or take one from your own home-island hex (not red next to red, only from hex adjacent to your own settlement, never leave a settlement with zero production) [S: TDE]. Bot implication: extending toward new islands can cost you home-island pips; expand early before the reserve runs out [S: TDE: "if one develops too late ... no opportunity to build"].
- Placement on home island only; ships allowed from coastal settlements; both robber and pirate in play [S: TDE].
- Whether Catan Universe's "Greater Catan" exactly matches this text [?].


<!-- moved from docs/strategy_research.md lines 115-136 (original numbering) -->
## 5. Other Catan Universe scenarios

Enchanted Land (C&K + Seafarers; official TDE)
- Target: 6 dragons defeated or 21 VP (3p) / 18 VP (4p) [S: TDE].
- Start: 2 settlements + 1 city on home island, at least one on the coast; ship or road choice [S: TDE]. Robber only [S: TDE].
- Enchanted island: settlements on coast only, no city upgrades there, no roads on the island [S: TDE].
- Move one activated knight (once per player, permanent) to the island via a connected ship line; thereafter it moves up to 3 intersections, never ending on the coast; expelled knights are placed by the expeller on any free intersection [S: TDE].
- Dragons: knight moves to a dragon intersection, re-activates, fights the next turn. Flag 1 = any knight, 2 = strong/mighty, 3 = mighty; win = token worth 1 VP; loss = knight deactivated, dragon stays [S: TDE]. Island knights do not defend against barbarians; Inventor cannot touch island tokens; Defector/Intrigue cannot target island knights [S: TDE].
- Tips: ships plus politics level 3 (mighty knights), chew through 6 dragons for an easy win [S: STEAM]. Island knights have limited reach (3 intersections) and can body-block [S: STEAM]. Hence bot: invest in Fortress and one knight shuttled early; the 6-dragon route competes with 18/21 VP.

Rise of the Inkas
- Tribes decline twice: you remove your roads and cover settlements with thickets, but those settlements/cities keep producing and can be replaced by opponents; timing the advance is the key decision; rushing exposes your resources [S: https://catanuniverse.com/en/rise-of-the-inkas/ and product pages e.g. https://catanshop.com/inkas search summary].
- [?] No concrete expert strategy found; exact point values and era triggers not confirmed.

Explorers & Pirates
- Three islands, only one known at start; ships and settlers to explore; gold is a second currency (given when you collect nothing on a roll or reveal hexes; cannot be stolen; 2 gold for a resource twice per turn); VP paths: Pirate Lairs, fish, native diplomacy [S: https://officialgamerules.org/game-rules/catan-exploreres-and-pirates/ search summary].
- [?] No expert strategy sources found. Whether the app ships the five scenarios unconfirmed [?].

Traders & Barbarians
- Event cards replace dice in some variants; scenarios: Fishermen, Rivers (gold, bridges; settle next to rivers early), Caravans, Barbarian Attack (captured barbarian = half VP), Traders & Barbarians (deliveries = VP + gold) [S: https://officialgamerules.org/game-rules/catan-traders-and-barbarians/ search summary].
- [?] No expert strategy found; presence of these scenarios in Catan Universe is not confirmed (the Steam page only lists Seafarers, Cities & Knights, Rise of the Inkas, Enchanted Land and Great Canal pack) [S: STEAM / https://store.steampowered.com/app/544730].


EDITED, original: ## 2. Seafarers

EDITED, original: ## 3. Cities & Knights (including Greater Catan)

EDITED, original: ## 4. Seafarers scenario: The Great Canal

EDITED, original: - No expert/tournament write-ups found for Seafarers, Inkas, E&P, T&B; entries are [G] or [?].
EDITED, original:  Keep the pirate on a leader's ship-line chokepoint.

EDITED, original: the official rulebook for the Great Canal / Enchanted Land / Greater Catan is the strongest source here.
