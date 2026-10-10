"""Bot configuration (data/config.json overrides these defaults)."""
from __future__ import annotations

import json
from pathlib import Path

PATH = Path(__file__).resolve().parent.parent / "data" / "config.json"

DEFAULTS = {
    "bot_name": "MARLOW.SYSTEMS",
    # Only invites from these Catan Universe names are accepted (case-insensitive).
    # Empty list = accept nobody (the bot just waits).
    "friends": [],
    "turn_budget_s": 45,        # finish our own turn within this many seconds
    "trade_wait_s": 12,         # how long to wait for answers to our trade offer
    "max_trade_offers": 3,      # per turn
    "response_budget_s": 6,     # answering prompts on other players' turns
    "click_delay_s": 0.35,      # pause after each click so animations settle
    "poll_s": 0.8,              # screen polling interval while waiting
    "dry_run": False,           # True = read and decide but never click
    "never_matchmaking": True,  # hard rule: never press public matchmaking buttons
    "pause_corner": True,       # move the mouse to the top-left corner to pause the bot
    # stage 2026-10-09 game 0738 (round 1); each switch turns its change off alone
    "textless_ring_wait_s": 5,      # ring fallback this soon after our own card play (45 s otherwise)
    "offer_repeat_after_turns": 6,  # a (give, get) offer declined by all waits this many own turns; 0 = off
    "progress_skip_confirmed": True,  # a card flow that played once this game: no preview/probe captures
    "discard_reread": True,         # an all-unreadable hand is read again (and then the tracked hand) before a discard
    # stage 2026-10-09 game 0834 (round 2)
    "confirm_by_colour": True,      # confirm/cancel buttons found by colour in the button bar (False = fixed point + old latch)
    "improve_by_order": True,       # city popup with repeated improve icons: label wall/trade/politics/science by position
    "knight_acted_memory": True,    # a knight activated or moved this turn can't act again this turn
    "offer_max_declines": 2,        # a (give, get) offer declined by all this often stays blocked for the game; 0 = off
    "hand_settle_s": 0.3,           # an unreadable hand slot just after a screen change: wait this long, read once more; 0 = off
    # stage 2026-10-09 game 0959 (round 3)
    "prompt_cap_s": 90,             # a prompt we can't clear gets this long, then the exit ladder (X, OK, End turn); 0 = off
    "fallback_blind_confirm": False,  # True = the old ring fallback tapped the bar's check spot with no check showing
    "inventor_ringed_only": True,   # Inventor swaps only number chips the game rings
    "inventor_barren": True,        # owner-confirmed 2026-10-06: barren chips may move; ringed-only covers 0959
    "grain_ask_cap": True,          # grain asks declined offer_max_declines times: no more grain asks this game
    "grain_set_keep": True,         # idle knight, no grain (canal): player offers never give a card of a 3:1 set or pair
    # owner 2026-10-10 "the bot has to keep trying to build cities throughout the game, this is still one of the goals":
    "city_always_goal": True,       # a city on our best settlement stays in the outlook goals + keep, before and after the canal
    "city_rebuild_first": True,     # after a lost city (or 0 cities): rebuild beats digs/ships/promotes, bank spare cards toward it
    "opening_set_first": True,      # before the 1st attack, no knight yet: bank a wool/ore 3:1 set into grain (turn 1 of 1009 g3)
    "knight_gone_memory": True,     # a knight spot whose popup offers only builds is forgotten
    "ring_fallback_our_turn": True,  # textless ring fallback only on our turn (not under a rival's turn marker)
    # stage 2026-10-09 games 1548/1632 (round 4)
    "rescue_in_game_exits": True,   # stall rescue tries road rings, the red X and End turn before the table click
    "leave_hosted_party_s": 30,     # front page as host of a custom party (inviter left): LEAVE after this; 0 = off
    "stall_lead_extra_s": 300,      # stalled 10 min while we lead (or trail by <= 2 VP): one more window; 0 = off
    "prompt_cap_unknown": True,     # a textless screen that looks like our prompt falls under prompt_cap_s
    "edge_rings_in_fallback": True,  # the Diplomat's road rings count for the textless ring fallback
    "diplomat_reread": True,        # no road ring seen: re-read the board, then a wider search, before giving up
    "token_ring_snap": 0.35,        # Inventor: a ring maps to a chip within this many hex sizes (0.6 = old)
    # live 2026-10-09 game 1906
    "chat_on_turn": True,           # read and answer chat on our own turn too (before planning), not only after End turn
    "robber_ring_reread": True,     # few robber rings read (right after a discard): read the screen again
    # owner 2026-10-09: "knights can move along the road or ships, also knights can move the robber
    # if active and get deactivated"
    "knight_move_ships": True,      # knight move paths run over our ships as well as our roads (False = roads only)
    "knight_chase_robber": True,    # an active knight next to the robber chases it (UNVERIFIED click flow); False = never
    # owner 2026-10-09 "build into the game the way to move ships" (flow UNVERIFIED live)
    "ship_move": True,              # plans may move the open-end ship of a line (once a turn); False = never
    "ship_move_max_misses": 2,      # failed ship moves in a game before they leave the plan for the game; 0 = never
    # owner 2026-10-09 "it should have accepted when more than 3:1 and the offering party is not
    # more than 5 points from winning"
    "generous_offer_accept": True,  # get > 3x give from a player more than 5 VP from winning: accept
    # owner 2026-10-09 "largest road gives 2 points, a knight can interrupt the road, a larger knight
    # can move a smaller knight, when deleting a knight of another player, go for the best position
    # if needed or the largest one otherwise"
    "knight_target_by_spot": True,  # Deserter/displacement go for a rival knight on a spot that matters, else the strongest (False = old picks)
    "offer_columns": True,          # read every card column of an incoming offer (False = rightmost only)
    # owner 2026-10-09 21:09 "took too much time to trade, if it can do multiple trades at once,
    # it should try that, build it in"
    "bank_batch": True,             # the plan's bank trades the hand pays for together: back to back, no re-plan between
    "bank_combined": True,          # ... first as one bank dialog, every want + give, one press (UNVERIFIED live; a miss = off for the game)
    "bank_combined_people_only": True,  # owner 2026-10-09 23:20: no combined bank trade in solo games, only with friends
    "spend_before_slow_cards": True,  # over the discard limit: Spy/Intrigue/Diplomat/... wait until after the builds
    # owner 2026-10-09 21:08 "marlow should build settlements now, still not doing that even with
    # 25 resources in hand"
    # owner 2026-10-09 "even if the canal is done, marlow should continue to build roads and
    # settlements and then cities"
    "road_settle_lookahead": True,  # a road that opens a good spot the hand also settles is planned as road + settlement; False = one-step greedy
    "settle_big_hand": True,        # over the discard limit (or city+road+settlement paid): settlement / road to a good spot (<=2 roads) before improvements; False = old greedy order
    # stage 2026-10-09 22:00 (game 212844): Jean's Deserter on us rang our knights, no text; read
    # as Jean's turn for 15 min and the game was left -- owner: find the cause of each stall, fix it
    "deserter_rings_by_corner": True,   # rival's turn + our knights ringed (read per knight corner) = our Deserter prompt; False = blob rings only
    # stage 2026-10-09 22:46 (game 221251): the Spy's card list showed a Bishop ("Move the robber")
    # and read as place_robber for 10 min
    "pick_progress_first": True,    # "Select progress card" wins over the card texts under it; False = first pattern wins
    # owner 2026-10-09 "figure out how the bot can play every move way faster. it runs out of time
    # almost always when we have a timer (usually 100s)" -- live 3: median 79 s, p90 115 s, 7/22
    # turns over 100 s. One switch per change (SPEED_PLAN 1-8); each flips back alone.
    "probe_async": True,            # 1. probe PNG/JSON and data/unread writes on a worker thread; False = written inside the turn
    "probe_unknown_on_turn": False,  # 1. screen_unknown probes on our own turn / within 5 s of our roll (dice frames); True = old
    "chat_once_per_turn": True,     # 2. chat opened at most once per own turn (End turn still polls); False = every chat_poll_s
    "trade_wait_floor_s": 0,        # 3. floor under trade_wait_s for player offers (15 = old)
    "offer_cutoff_s": 40,           # 3. no new player offer this far into our turn (the city-save ask aside); 0 = off
    "offers_per_turn": 1,           # 3. player offers per turn (max_trade_offers still caps); 0 = max_trade_offers only
    "bank_settle_poll": True,       # 4. bank dialog: poll for the screen to settle (hand guard kept), ratios once a game; False = fixed sleeps
    "click_delay_fast_s": 0.15,     # 5. pause after a click the next screen read checks anyway; confirms/drags/trade OK keep click_delay_s (0.35 = old)
    "poll_s_turn": 0.4,             # 6. loop wait on our own turn (poll_s off-turn); 0 = poll_s
    "fast_transitions": True,       # 7. unknown frame < 3 s after our roll / trade OK / End turn: no seat read, no unread write, 0.25 s poll
    "scan_after_drag": "check",     # 8. after a progress drag: compare 3 hexes with the last board frame, rescan only on a move; "full" = old
    "scan_check_max_diff": 0.5,     # 8. mean grey difference per hex patch that still counts as "board did not move"
    "log_timing": True,             # per step 't: snap= ocr= classify=' (ms) + a 'turn done:' line per own turn
    # bot 0.4.3 "fast turn" (SPEED_TURN_DESIGN steps 1-8). Owner 2026-10-09: "turn can be planned
    # before the dice is rolled but has to be redone once the dice is rolled, it gets its resources,
    # it uses the cards that can steal more resources, the plan should be made once all resources
    # are in hand, and then all resources used quickly". Owner 2026-10-10: "the timer is a color
    # edge around the next button. bot can also see what the timer is in the game, the timer does
    # not change throughout a game and bot can time how much time it has". One switch per change.
    "turn_clock": True,             # 1+7. phase cutoffs scaled to the game's timer (False = fixed turn_budget_s / offer_cutoff_s)
    "turn_timer_default_s": 100,    # 7. timer when the info panel's "Round Timer" wasn't read (people games 2026-10-09: 100 s)
    "turn_late_frac": 0.45,         # 1. past this share of the timer: late plan (trades only for builds the hand pays for)
    "turn_offer_frac": 0.45,        # 1. no new player offer past this share
    "turn_chat_frac": 0.55,         # 1. no chat on our own turn past this share
    "turn_end_frac": 0.70,          # 1. past this share: builds only, then End turn
    "turn_hard_margin_s": 6,        # 1. this close to the timer's end: End turn now (a missed End turn = lost turn)
    "turn_ring_read": True,         # 7. the coloured border round the button moves our clock forward when it shows more time used
    "steal_first": True,            # 2. robber chase / steal cards before the other cards, one re-read after them
    "build_batch": True,            # 3. a landed build whose hand read matches hand - cost: next build without a re-plan
    "preplan": True,                # 4. on rivals' turns: threat rank + a plan per dice total, used when the hand matches
    "robber_fast": True,            # 5. robber / steal screens: wait for the screen to settle, not fixed 0.6-1.0 s sleeps
    "offer_poll_s": 0.4,            # 6. loop wait while our player offer is out (0 = poll_s_turn)
    "log_phases": True,             # 8. a 'phase:' line per own turn: seconds in roll/steal/plan/trade/build/end
}


def load() -> dict:
    cfg = dict(DEFAULTS)
    if PATH.exists():
        cfg.update(json.loads(PATH.read_text()))
    return cfg


def save(cfg: dict) -> None:
    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.write_text(json.dumps(cfg, indent=1))
