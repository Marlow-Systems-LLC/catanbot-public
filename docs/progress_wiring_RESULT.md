# Progress-card wiring: result

`PYTHONPATH=. nice -n 15 .venv/bin/python -m pytest -q tests` passes: 51 tests, including 12 new ones in `tests/test_progress_wiring.py`. Nothing was committed.

## What changed
- `catanbot/state.py:PlayerState.progress_hand`: a new `list[str]` of `progress_cards.card_key` names. It defaults to an empty list. The existing `progress_cards: int` count is kept, and the runner sets both.
- `catanbot/state.py:GameState.barbarians_attacked`: a new `bool = False`.
- `catanbot/brain.py:Brain.progress_play(gs, phase, hand_cards=None)`: wraps `pc.choose_play`. The hand defaults to `gs.me().progress_hand`, and it returns the first `(card_key, args)` or None.
  - **Bishop** is dropped while `gs.barbarians_attacked` is False.
  - In `end_turn`, if dropping Bishop leaves nothing to play and the hand is over the limit of 4, it falls back to `("discard", {...})` with the lowest `KEEP_VALUE`.
- `catanbot/brain.py:Brain.start_fleet / end_fleet / fleet_resource`: `bank_ratio` returns 2 for `fleet_resource`, and this check comes first.
- `catanbot/runner.py:Runner.play_progress(phase, img=None, ocr=None)`, with the helpers `_read_progress_hand` and `_progress_choices`. It:
  1. OCRs the `progress_hand` box and maps names through `card_key`, skipping names that raise KeyError.
  2. Builds the state, sets `progress_hand`, and copies `barbarians_attacked` from `self.barbarians_attacked` (default False) into the state.
  3. Calls `brain.progress_play` and saves a screenshot to `data/popups/<time>_progress_<phase>_<key>_<card>.png`.
  4. Clicks the card name, then `progress_play` (or `progress_discard`), answers the card's choices, and clicks confirm if it appears.
  5. Re-snapshots and repeats, up to 4 times. Every step is logged with `self.note`.

  Other behaviour:
  - It does nothing unless the mode is C&K.
  - Each card is tried at most once per turn. The tried set clears at the end of `end_turn` and when `pre_roll` follows an earlier `main` or `end_turn`.
  - Merchant Fleet calls `brain.start_fleet(resource)`. `end_turn` and the next turn's `pre_roll` call `end_fleet()`.
- Runner call sites:
  - `on_roll`: `play_progress("pre_roll", img, ocr)` runs before the dice click. If Alchemist was played, it returns without rolling, so the next step re-reads the screen.
  - `on_main`: `play_progress("main", img, ocr)` runs after `ensure_state` and before `plan_turn` and the build loop. If anything was played, it returns so the next step re-reads and plans with the new hand.
  - `on_main`: `play_progress("end_turn")` runs before the end-turn click. It takes its own screenshot.
- `data/calib/ui.json`: new keys `progress_hand` [x0,y0,x1,y1], `progress_play`, `progress_discard`, `progress_live: false` and `_progress_UNCONFIRMED`.

## Safety gate
`progress_live` is **false**. In a live run that is not a dry run, `play_progress` reads the cards, decides, saves the screenshot and logs `would <card> ...`, but **clicks nothing**. Dry runs and tests do record the clicks.

The runner process that is running now still has the old code loaded. These changes take effect only when the owner restarts it.

## UNCONFIRMED live
- None of the progress-card UI is calibrated: where card names appear, whether a panel must be opened first, the play and discard buttons, and the confirm step. Every coordinate in ui.json for this is a guess.
- Choice flows:
  - Resource, commodity and player choices are found by OCR text and clicked.
  - Hex, vertex and edge choices are clicked through the board lattice.
  - Alchemist `dice`, Inventor `swap` and Commercial Harbor `give` only log "no UI flow yet".
- Alchemist: it is unknown whether the app rolls automatically after the dice are chosen, or whether the roll button must still be pressed. The bot returns and re-reads the screen.
- Card names are assumed to be the English 5th-edition names, read by OCR. Two-line names or names cut by OCR will not match.
- **`barbarians_attacked` is never set from the screen.** Nothing detects the first attack yet, so the runner always passes False and Bishop is never played. A barbarian-attack detector should set `runner.barbarians_attacked = True`.
- **Robber lock: not applied.** The codebase has no robber-lock concept. `friendly_robber` is a different house rule. So, as briefed, `Brain.choose_robber` was not changed.

## What the owner must calibrate
1. On the bot's turn, take a screenshot with progress cards in hand. Record where the card names are, as the `progress_hand` box, and whether a button opens the card panel. If one does, a `progress_open` key and a click are needed.
2. Record the play button that appears after selecting a card (`progress_play`) and the discard button used when over 4 cards (`progress_discard`).
3. Record how the choices are made for Alchemist dice, Inventor, Commercial Harbor, Bishop, Merchant and the player pickers.
4. Then set `"progress_live": true`. The `data/popups/*_progress_*.png` screenshots that the bot saves while the gate is off are the calibration material.
