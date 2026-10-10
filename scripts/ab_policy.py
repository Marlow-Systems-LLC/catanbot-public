"""A/B a policy switch: one seat with Brain.<flag>=True, three with False, same trained model,
seat rotates. Prints the flagged seat's win rate (fair = 0.25).

    python scripts/ab_policy.py GAMES FLAG [MODE] [MODEL_PATH]
"""
import json
import os
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

from catanbot.brain import Brain, set_flag
from catanbot.learning import Model
from catanbot.sim import Sim

GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 200
FLAG = sys.argv[2] if len(sys.argv) > 2 else "spend_turn"
MODE = sys.argv[3] if len(sys.argv) > 3 else "great_canal"
PATH = Path(sys.argv[4]) if len(sys.argv) > 4 else Path("data/train/best.json")
N = 4
# flag sets: (values for the tested seat, values for the other seats = the old play)
SETS = {"owner1004": ({"set_for_grain": True, "knight_always": 0.01, "defender_vp": 0.04,
                       "spend_grain": 0.03, "gold_route": 0.03, "limit_lookahead": True},
                      {"set_for_grain": False, "knight_always": 0.0, "defender_vp": 0.0,
                       "spend_grain": 0.0, "gold_route": 0.0, "limit_lookahead": False}),
        # owner 2026-10-04 (2nd message): walls before the 1st attack, city wall slots, spend exposed cards
        "walls1004": ({"wall_late": False, "city_wall_slot": 0.01, "spend_exposed": 0.004},
                      {"wall_late": True, "city_wall_slot": 0.0, "spend_exposed": 0.0}),
        # owner 2026-10-04 (3rd message): developments to 3 on every track, then 4 and 5 (5 locks the metropolis)
        "ladder1004": ({"improve_ladder": 0.02}, {"improve_ladder": 0.0}),
        # owner 2026-10-06: no spare knight while one is idle; road up to a rival canal knight + push it off
        "owner1006": ({"spare_knight": 0.05, "canal_contest": 0.015}, {"spare_knight": 0.0, "canal_contest": 0.0}),
        # owner 2026-10-04 (4th message): knights that cut a rival's road (moves, pushes, spots)
        "roads1004": ({"road_cuts": True}, {"road_cuts": False}),
        # stage 2026-10-09 game 0959 (round 3): threat ranking weights the climber
        "threat_rise": ({"threat_rise_w": 0.08}, {"threat_rise_w": 0.0}),
        # owner1004 lost (0.158 vs 0.25): one rule at a time, off in the test seat
        "no_set_for_grain": ({"set_for_grain": False}, {"set_for_grain": True}),
        "no_knight_always": ({"knight_always": 0.0}, {"knight_always": 0.01}),
        "no_defender_vp": ({"defender_vp": 0.0}, {"defender_vp": 0.04}),
        "no_spend_grain": ({"spend_grain": 0.0}, {"spend_grain": 0.03}),
        "no_gold_route": ({"gold_route": 0.0}, {"gold_route": 0.03}),
        "no_limit_lookahead": ({"limit_lookahead": False}, {"limit_lookahead": True}),
        # owner 2026-10-04 (6th message): knights steal + bounce the robber; barbarian sabotage
        "bounce1004": ({"knight_bounce": True, "barb_sabotage": True},
                       {"knight_bounce": False, "barb_sabotage": False}),
        # owner 2026-10-04: per-turn selector (catanbot/situational.py) -- BLOCK the advancing rival, else RACE
        "situational1004": ({"situational": True}, {}),
        # owner 2026-10-04: every idea of the day together vs none of them
        "all1004": ({"set_for_grain": True, "knight_always": 0.01, "defender_vp": 0.04, "spend_grain": 0.03,
                     "gold_route": 0.03, "limit_lookahead": True, "wall_late": False, "city_wall_slot": 0.01,
                     "spend_exposed": 0.004, "improve_ladder": 0.02, "road_cuts": True},
                    {"set_for_grain": False, "knight_always": 0.0, "defender_vp": 0.0, "spend_grain": 0.0,
                     "gold_route": 0.0, "limit_lookahead": False, "wall_late": True, "city_wall_slot": 0.0,
                     "spend_exposed": 0.0, "improve_ladder": 0.0, "road_cuts": False})}


def play(g):
    seat = g % N
    brains = []
    for i in range(N):
        b = Brain(Model(PATH))
        if FLAG in SETS:
            for k, v in SETS[FLAG][0 if i == seat else 1].items():
                set_flag(b, k, v)
            brains.append(b)
            continue
        name, _, val = FLAG.partition("=")      # FLAG or attr=value (value set on the seat only)
        if val:
            b.spend_turn = i == seat
            if i == seat:
                set_flag(b, name, type(getattr(b, name))(val))
        else:
            set_flag(b, name, i == seat)
        brains.append(b)
    winner, turns, _ = Sim(MODE, N, seed=int(os.environ.get("AB_SEED", 7_000_000)) + g, brains=brains).run()
    return winner == f"P{seat}", winner is None, turns


if __name__ == "__main__":
    with Pool(int(os.environ.get("CATAN_WORKERS") or max(1, cpu_count() // 2))) as pool:
        res = pool.map(play, range(GAMES))
    wins = sum(w for w, _, _ in res)
    print(json.dumps({"flag": FLAG, "mode": MODE, "games": GAMES, "wins": wins,
                      "win_rate": round(wins / GAMES, 3), "fair": 0.25,
                      "no_winner": sum(n for _, n, _ in res),
                      "avg_turns": round(sum(t for *_, t in res) / GAMES, 1)}), flush=True)
