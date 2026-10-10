# Great Canal playbook (C&K + Seafarers, 4 players)

Table for the live game: 4 players, **19 VP (the app's value is authoritative; the printed rulebook says 18)**, 130 s turns, friendly robber (players with 2 VP or less can't be hit).

Sources: docs/strategy_research.md section 4 (TDE rulebook and Steam notes), docs/strategy_base_sf_ck.md section 3 (C&K), the BGA C&K help page and C&K rulebook summaries found by web search (2026-10-02), and the owner's rules. Web search turned up almost no Great Canal strategy beyond the Steam thread already cited, so most canal-specific lines below are owner rules or inference. Tags: **[O]** owner rule, **[S]** sourced, **[I]** inference, **[code]** encoded in `catanbot/brain.py`.

## Rules that shape everything
- **No grain until the canal is finished [S].** Barren farmland on the west gives nothing, and grain only flows again once the penultimate canal tile is dug. Every activation (1 grain) and every city (2 grain) is scarce.
- **Canal chips [S].** Two *activated* knights on corners of the same canal hex dig it at once, at no cost. Both knights yours = 2 chips (2 VP); knights of two owners = 1 chip each. Knight strength doesn't matter, and nobody can cut in, because the dig happens on the second activation.
- **Island knights [S].** A knight (active or not) on a corner of a small-island **gold** hex gives any 1 card on its number (8). A knight on a small-island **harbour** corner gives that harbour. Island knights do **not** defend against barbarians. They stop working once the canal floods.
- Robber only, no pirate. Knights may only be built on the home island, plus island corners reached by your ship line [S].

## Opening
1. **Settle a 3:1 harbour [O][code].** With no grain, a 4:1 bank cripples you. A 3:1 harbour scores +5 at setup, a grain 2:1 scores 0 (`_spot_extras`).
2. Put the city on commodity terrain: paper (forest) first, then coin (mountains), then cloth (pasture) [O][code `COMMODITY_W`, `_city_bonus`].
3. Get ore and wool on good numbers, because knights cost wool + ore [S]. A third settlement bordering a *producing* grain hex (an eastern field) is the best source for the first activation, and Irrigation works on it [S Steam].
4. Touch at least one canal hex. The canal access bonus is +3 at setup [code].
5. Lumber and brick for ships and roads toward canal hexes and the island gold or harbour [S].

## Mid-game
- **One ACTIVE knight per city, all game [O][code `defence_gap`].** Losing a city to the barbarians is close to fatal [S]. A knight is the first trade target while knights < cities, and activation is first while the gap is open (`_trade_targets`).
- **Dig canals in pairs [I][code `_canal_pair_value`].** Put your second knight on another corner of a hex where you already stand (+0.05), so its later activation gives you both chips. Don't activate next to the leader's waiting active knight, which would hand them 1 VP (-0.03). Sharing with a weaker player is fine (+0.01).
- **Island knights [S][code `_prospector_value`].** Reach the gold "8" or a 3:1 harbour by ship and park a knight there: no activation needed, worth about one extra card every 7 rolls. One knight per gold hex is enough, because a second gives nothing more.
- **Improvements: paper > coin > cloth [O][code].** Spend paper and coin on improvements before a 7 can take them. Flip level 1 cheaply and aim for science 3 (aqueduct) [S].
- **Trading [O][code].** Ask the table for **wool** when a knight needs it (`propose_trade` asks for knight or activation cards first, and wool's value rises by 30%). A 2-for-1 is fair if it lets you deploy **and** activate a knight in the same turn (`_best_followup` looks two actions ahead). If players refuse, use the 3:1 harbour (`bank_trades_for`).

## End-game
- Canal chips run out when the canal completes (about 18 chips over 9 tiles). Before the last tiles, prefer finishing your own pairs over starting new hexes [I].
- After completion, grain flows: convert straight to cities and activations [I].
- Rush a metropolis (+2 VP, and it can't be pillaged) in paper if you are at science 3+ [S].
- Watch everyone's VP against the app target. At target − 2, stop trading with that player [code].

## If X happens, do Y

| Event | Reaction | Code |
|---|---|---|
| Robber on my hex | An active knight next to the robber can chase it away (C&K): use it if the hex is your best producer, then re-activate later. Otherwise build out of it. | playbook only (runner flow) |
| Robber is mine to move | Block the leader (highest win probability) on their best hex, steal from them. Never block myself. Barren farmland is worthless to block before the canal is done. Friendly robber: never target ≤2 VP. | `choose_robber` (barren check new) |
| 7 rolled, many cards | Never sit near the discard limit [O]. With hand ≥ limit−1 the planner accepts any build over holding. On a discard, keep knight and activation cards if a city is undefended. | `plan_turn`, `choose_discard` |
| Lost a city to barbarians | Rebuild that city's knight first, then re-upgrade the city. Keep 1 active knight per remaining city. | `defence_gap`, `_trade_targets` |
| Barbarians close (track ≥ 5/7) | Activate and build knights before anything else. A new city without a knight is penalised more as the ship nears. Don't over-invest if you aren't the weakest defender [S]. | `action_value` danger terms |
| Opponent near win (target − 2) | Refuse all trades with them. Robber, Deserter, Intrigue and Diplomat on them. Don't activate next to their canal knight. | `evaluate_offer`, `choose_robber`, `_canal_pair_value` |
| Opponent leads canal digging | Take free corners of hexes where they have one knight before their second arrives, and pair your own instead of sharing. If they are the leader, don't complete a shared dig with them. | `_canal_pair_value` |
| Wildcard offer received | Counter with the card the bot needs most (wool for an owed knight, then the missing card of the top build). Accept only if it helps the bot more than the partner. | runner counter + `evaluate_offer` / `resource_value` |
| Trade refused | Use the 3:1 harbour, then 4:1. Don't keep re-asking within one 130 s turn. | `bank_trades_for` |
| Grain needed (activation or city) | Ask players, then use the 3:1 harbour, the gold prospector, Irrigation, or the Merchant. Never rely on a 4:1. | `resource_value` grain ×1.5 |
| Progress cards | Play on your turn before building. Alchemist when a set roll gives your city a commodity; Irrigation (works on barren fields); Inventor on an 8 next to your city; Deserter and Intrigue on the leader; Merchant on your best 2:1 [S doc §3]. | separate agent (`progress_cards.py`) |
| Commodities in hand, no build | Spend paper, then coin, on improvements before the next 7. | `action_value` improve order |
| Hand at limit with walls | Each wall raises the limit by 2. Walls are cheap brick insurance for a big commodity hand. | `plan_turn` limit uses walls |

## Unconfirmed
- VP target: the app says 19, the printed rulebook says 18 for 4 players. The bot uses the app value.
- Whether Catan Universe lets an island knight count as a harbour for the bank-trade UI (rules say yes) [?].
- Knight robber-chase and the exact counter-offer UI flow aren't verified in this app.
