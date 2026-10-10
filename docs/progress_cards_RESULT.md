# Progress cards: result

## Files created
- `docs/progress_cards.md`: all 54 cards with deck, count, effect, timing, heuristic and required choices, plus sources.
- `catanbot/progress_cards.py`: `CARDS` (25 card types, 18/18/18 per deck), `card_key()` (name normalizer, including German-edition aliases), one `choose_<card>()` per card, and `choose_play(gs, me, hand_cards, phase)`.
- `tests/test_progress_cards.py`: 15 tests, which pass in 0.04 s (`PYTHONPATH=. nice -n 15 .venv/bin/python -m pytest -q tests/test_progress_cards.py`).

No existing file was edited.

## Brain hook to add (not applied; another agent owns brain.py)

```python
# catanbot/brain.py, top
from . import progress_cards as pc

# in class Brain
    def progress_play(self, gs: GameState, hand_cards: list[str], phase: str):
        """Next progress card to play now, or None. Call again after each card resolves:
        the hand and board change (Irrigation adds grain, Wedding adds cards...)."""
        plays = pc.choose_play(gs, gs.me(), hand_cards, phase)
        return plays[0] if plays else None
```

Runner and UI flow (for whoever drives the live UI):
1. Before the roll: `progress_play(gs, cards, "pre_roll")`. If the result is `("alchemist", {"dice": (w, r)})`, play it and set white = w, red = r.
2. After the roll, before `plan_turn`: loop `progress_play(gs, cards, "main")`. Play each card, re-read the state, and repeat until the result is None. Then call `plan_turn`. Merchant Fleet means 2:1 for `args["resource"]` this turn; `bank_ratio()` should return 2 for that resource. That is a further brain.py hook: keep a `self.fleet_resource` that `bank_ratio` checks.
3. After building, before ending the turn: loop with `"end_turn"`. `("discard", {"card": c})` means discard c to get back to 4 progress cards.
4. Off-turn (`"on_seven"` / `"opp_turn"`): only VP cards are returned. They reveal automatically in the app.

Optional brain.py touch: in `action_value`, count a held Medicine as making a city cost 2 ore + 1 grain, and a held Crane as making an improvement cost 1 commodity less.

## Unconfirmed
- Which edition Catan Universe uses for **Saboteur**. The bot assumes the 5th edition: players with VP ≥ yours discard half. The German original puts a city on its side instead.
- In Seafarers + C&K, whether Diplomat and Intrigue can target ships, and whether Bishop can move the pirate. Only roads and the robber are handled.
- Whether Bishop and the robber are blocked before the first barbarian attack. The state has no flag for this, so the caller must skip Bishop then.
- Catan Universe card names. They are assumed to be the English 5th-edition names, and `card_key` raises KeyError on unknown names.
- Opponent hands are estimated from `hand_count` when `hand` is empty, so monopoly and Saboteur thresholds are rough.
- `state.PlayerState` has no "holds the merchant" field, so Merchant is always played. That is correct under the rules, since it moves the merchant to you.
- Progress draws (the red die and level table) are not modelled beyond picking a low red die for Alchemist.
