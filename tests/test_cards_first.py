"""Owner 2026-10-06: "use all cards first and then make the plan with what it has in its hand".
Progress cards play before the turn is planned (live runner and sim); held cards are offered
again after the plan's builds, before End turn."""
import numpy as np
import pytest

import catanbot.runner as R
from catanbot import config
from catanbot.board import canal_board, hex_corners
from catanbot.brain import Action
from catanbot.rules import MODES
from catanbot.runner import ReplayIO, Runner
from catanbot.sim import Sim
from catanbot.state import GameState, PlayerState, new_hand

IMG = np.zeros((888, 1372, 3), np.uint8)


@pytest.fixture
def gs():
    b = canal_board(seed=3)
    me = PlayerState("marlow.systems", is_me=True, hand=new_hand(), vp_visible=4)
    a = PlayerState("Hildegard", vp_visible=5, hand_count=6)
    g = GameState(board=b, mode=MODES["great_canal"], players=[me, a])
    land = [h for h in b.hexes.values() if h.is_land and h.number]
    me.settlements.add(hex_corners(*land[0].coord)[0])
    a.cities.add(hex_corners(*land[-1].coord)[3])
    return g


def live_runner(gs, tmp_path):
    cfg = config.load(); cfg["dry_run"] = True
    r = Runner(ReplayIO([]), cfg)
    r.mode, r.gs, r.popup_dir = gs.mode, gs, tmp_path
    r.ensure_state = lambda img, texts: gs
    r.bar_hidden = lambda img: False
    r._confirm_accept = lambda *a: False
    r._counter_offer = lambda *a: False
    r._handle_responses = lambda *a: False
    r._want_trade = lambda gs: None
    return r


def test_live_card_before_plan_and_build_same_turn(gs, tmp_path):
    """Resource Monopoly brings the 2 ore the city lacks: it plays first, the plan then sees the
    full city and builds it in the same turn."""
    me = gs.me()
    me.hand.update(ore=1, grain=2)
    r = live_runner(gs, tmp_path)
    order, seen = [], []
    cards = ["resource_monopoly"]

    def play(phase, img=None, ocr=None):
        order.append(f"card:{phase}")
        if phase == "main" and cards:
            cards.pop()
            me.hand["ore"] += 2                     # 2 ore taken from the table
            return ["resource_monopoly"]
        return []

    plan_turn = r.brain.plan_turn

    def plan(g):
        order.append("plan")
        seen.append(dict(g.me().hand))
        return plan_turn(g)

    r.play_progress, r.brain.plan_turn = play, plan
    r._run_builds = lambda g, p: order.append("build:" + ",".join(a.kind for a in p))
    r.on_main(IMG, [], [], {})                      # pass 1: the card, nothing planned
    assert order == ["card:main"]
    r.on_main(IMG, [], [], {})                      # pass 2: plan on the new hand, build
    assert order[1:3] == ["card:main", "plan"]
    assert seen[0]["ore"] == 3 and seen[0]["grain"] == 2
    assert any(o.startswith("build:") and "city" in o for o in order)


def test_live_card_at_end_turn_replans_instead_of_ending(gs, tmp_path):
    r = live_runner(gs, tmp_path)
    r.brain.plan_turn = lambda g: []
    r._shed_hand = lambda g: False
    r._build_offered = lambda g: False
    r._bank_into_build = lambda g: False
    r.play_progress = lambda phase, img=None, ocr=None: ["medicine"] if phase == "end_turn" else []
    r.on_main(IMG, [], [], {})
    assert not any(w == "end turn" for *_, w in r.io.clicks)
    r.play_progress = lambda phase, img=None, ocr=None: []
    r.on_main(IMG, [], [], {})
    assert any(w == "end turn" for *_, w in r.io.clicks)


def quiet_sim(monkeypatch):
    sim = Sim("great_canal", seed=5)
    sim.setup()
    for name in ("produce", "seven", "barbarians", "progress_draw", "chase_robber", "player_trade"):
        monkeypatch.setattr(sim, name, lambda *a, **k: None)
    return sim


def test_sim_card_before_plan_and_build_same_turn(monkeypatch):
    sim = quiet_sim(monkeypatch)
    me = sim.gs.players[0]
    for p in sim.gs.players:
        p.hand = new_hand()
        p.progress_hand = []
    for q in sim.gs.players[1:]:
        q.hand["ore"] = 2
    me.hand.update(ore=1, grain=2)
    me.progress_hand = ["resource_monopoly"]
    cities = len(me.cities)
    brain = sim.brains[0]
    seen = []
    plan_turn = brain.plan_turn
    monkeypatch.setattr(brain, "plan_turn", lambda g: (seen.append((list(g.me().progress_hand), dict(g.me().hand))),
                                                       plan_turn(g))[1])
    sim.turn(0)
    assert sim.progress_played.get("resource_monopoly") == 1
    hand0, cards0 = seen[0][0], seen[0][1]
    assert hand0 == [] and cards0["ore"] >= 3       # the plan saw the monopoly's ore
    assert len(me.cities) == cities + 1


def test_sim_offers_cards_again_after_the_plan(monkeypatch):
    """Medicine has no use at the start (no ore); a build/trade in the plan brings 2 ore + grain,
    then the card is offered again before the turn ends and plays."""
    sim = quiet_sim(monkeypatch)
    me = sim.gs.players[0]
    me.hand = new_hand()
    me.progress_hand = ["medicine"]
    assert me.settlements
    cities = len(me.cities)
    plans = [[Action("bank_trade")], []]
    monkeypatch.setattr(sim.brains[0], "plan_turn", lambda g: plans.pop(0) if plans else [])

    def execute(i, a):
        sim.gs.players[i].hand.update(ore=2, grain=1)
    monkeypatch.setattr(sim, "execute", execute)
    sim.turn(0)
    assert me.progress_hand == [] and sim.progress_played.get("medicine") == 1
    assert len(me.cities) == cities + 1
