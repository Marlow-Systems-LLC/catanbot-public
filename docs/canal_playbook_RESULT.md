# Canal playbook: result (2026-10-02)

## Changed
- `docs/canal_playbook.md`: new. Opening, mid-game, end-game, and a 13-row event table, with every owner rule tagged [O].
- `catanbot/brain.py`. Line numbers are for the new file. runner, vision and mac were not touched.
  - :8 imports `hex_corners`.
  - :73-75 `resource_value`: wool ×1.3 and ore ×1.1 while a knight is owed (knights < cities, or the defence gap is open).
  - :321-327 `candidate_actions` (canal mode): knight candidates on gold and harbour corners reached by roads or ships.
  - :367-368 a city built without knight cover now costs 0.02 + 0.04 × barbarian danger (was a flat 0.02).
  - :374-376 the canal knight bonus is now `_canal_pair_value`, plus `_prospector_value` for knight placement.
  - :386-434 new `_canal_pair_value`: second corner of my own pair +0.05 (knight) or +0.02 (activate); next to the leader's active knight -0.03 (activate) or -0.01 (knight); next to a weaker player's +0.01; first corner +0.02. New `_prospector_value`: a gold corner is worth 0.006 × pips, only once per gold hex; a 3:1 harbour corner +0.04 if I don't have one yet; a 2:1 corner is income-weighted.
  - :473-476 `bank_ratio`: in canal mode, knights on harbour corners count as owning that harbour (trader).
  - :583-584 `choose_robber`: barren farmland before the canal is finished scores -0.5, so the robber doesn't waste a 7 there.
  - :645-647 `evaluate_offer`: refuses any trade with a player at target − 2 VP or more.
  - :686-690 `propose_trade`: asks for knight and activation cards first while a knight is owed.
- `tests/test_canal_playbook.py`: new, 11 tests. 9 of them fail on the old brain.py, and 2 are regression guards (own-pair dig = 2 VP; spending commodities near the limit).

## Tests
`nice -n 15 .venv/bin/python -m pytest -q tests` gives **24 passed**. The 5 test_runner.py failures from the start of this session now pass too; those were fixed outside this task. A 60-turn great_canal smoke sim (1.2 s) ran without errors.

## Unconfirmed
- The weights (0.05, 0.04 and so on) are hand-set and haven't been tuned by simulation, because long sims are off-limits during the live game.
- The board model doesn't tell small islands from the home island. The runner's legal-spot highlighting is relied on to keep knights legal.
- The trader-knight harbour in the app's bank UI hasn't been checked. If the board read misses a port, the runner's `harbor_3to1` config still overrides it.
- Robber-chase with a knight, and the progress-card play, are in the playbook only (progress cards are being handled by another agent).
