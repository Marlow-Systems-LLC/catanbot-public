"""Brain play profiles a model file can carry ("brain" in the model json, applied in Brain.__init__).

V1 = the play the last model (the server data/model.json, 2026-10-04 02:05) was trained with:
     every owner rule of 2026-10-04 off. It runs on today's engine (rules fixes, keep_value,
     Deserter target), so it is the old *play*, not the old code.
V2 = every learning of the 2026-10-02..04 session that held up (scripts/ab_policy.py results,
     docs/analysis/session_learnings_2026-10-04.md); filled in by the v2 program.
"""
from __future__ import annotations

V1 = {"wall_late": True, "set_for_grain": False, "knight_always": 0.0, "defender_vp": 0.0,
      "spend_grain": 0.0, "gold_route": 0.0, "limit_lookahead": False, "city_wall_slot": 0.0,
      "spend_exposed": 0.0, "improve_ladder": 0.0, "road_cuts": False, "knight_bounce": False,
      "barb_sabotage": False}

# v0.2 strategy round 2026-10-04 (docs/analysis/strategy_{haiku,sonnet,opus,fable}.md), the points
# all four agreed on: owner rules kept but at the dose that tested best (knight_always 0.003,
# defender_vp 0.01, gold_route 0.01: each "off" scored 0.28-0.29, the full dose bundle 0.167);
# opus fleet grain_gate on (0.325 on 2 seeds); walls before the 1st attack (0.30), improve ladder
# (0.275), road cuts kept. Losing rules stay in code, off: knight_bounce, barb_sabotage, displace,
# situational (situational1004 0.221). spend_grain split 2/2 -> v2 (0.03) vs v2b (0.0) battled.
V2: dict = {"wall_late": False, "set_for_grain": True, "knight_always": 0.003, "defender_vp": 0.01,
            "spend_grain": 0.03, "gold_route": 0.01, "limit_lookahead": True, "city_wall_slot": 0.01,
            "spend_exposed": 0.004, "improve_ladder": 0.02, "road_cuts": True, "knight_bounce": False,
            "barb_sabotage": False, "displace": False, "situational": False, "grain_gate": True,
            "wall_boost": 2.0, "grain_first": True, "spend_turn": True, "spend_need_cap": 2}
V2B: dict = dict(V2, spend_grain=0.0)

# v1 play + only the walls set (walls1004 0.30, the clearest single winner)
V2W: dict = dict(V1, wall_late=False, city_wall_slot=0.01, spend_exposed=0.004)

# overnight: v0.2 (v2w) + the opus fleet's grain_gate (it gates set_for_grain, so that is on too)
V2WG: dict = dict(V2W, set_for_grain=True, grain_gate=True)

# overnight: v0.2 + the relative barbarian rule (take us out of the weakest-defender spot)
V2WR: dict = dict(V2W, barb_relative=0.06)

PROFILES = {"v1": V1, "v2": V2, "v2b": V2B, "v2w": V2W, "v2wg": V2WG, "v2wr": V2WR}
