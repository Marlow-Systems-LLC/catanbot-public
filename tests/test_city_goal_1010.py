"""Owner 2026-10-10 "the bot has to keep trying to build cities throughout the game, this is
still one of the goals". Solo game 20261009_225719 (g3): the city was lost at 23:00:40, the
outlook goals never held a city again (23:01:08 goals=[settlement, settlement, canal_chip],
keep without ore), every banked grain went to activations (23:02:53, 23:06:14, 23:07:01) and
ore to promotions, and cities=0 to the end of the game."""
import copy

from catanbot import config
from catanbot.board import hex_corners, real_canal_board
from catanbot.brain import Brain
from catanbot.rules import MODES
from catanbot.state import GameState, Knight, PlayerState, new_hand
from tests.test_g3_loss_1009 import _runner


def _gs(hand, city=False, attacked=True, knights=True):
    b = real_canal_board(7)
    ps = [PlayerState("marlow.systems", is_me=True), PlayerState("b"), PlayerState("c")]
    for p in ps:
        p.hand = new_hand()
    gs = GameState(board=b, mode=copy.copy(MODES["great_canal"]), players=ps, vp_target=18)
    me = gs.me()
    canal = [h for h in b.hexes.values() if b.is_canal(h)]
    cs = [v for v in hex_corners(*canal[0].coord) if v in b.vertices]
    spots = [v for v in sorted(b.vertices) if gs.can_settle(v) and b.vertex_pips(v) > 6]
    me.settlements = {spots[0]}
    if city:
        me.cities = {spots[2]}
    gs.players[1].settlements = {spots[5]}
    if knights:                               # g3 after the attack: one idle, one active knight
        me.knights = {cs[0]: Knight(active=False), cs[3]: Knight(active=True)}
    me.hand.update(hand)
    gs.barbarian_pos = 1
    gs.barbarians_attacked = attacked
    return gs


def _brain(lost=1, **attrs):
    br = Brain(setup_weights={})
    br.note = lambda *a, **k: None
    br.bank_ratio = lambda g, r: 3
    for k, v in attrs.items():
        setattr(br, k, v)
    return br


def _plan(br, gs, lost=1):
    br.update_outlook(gs, lost)
    return br.plan_turn(gs)


def _banks_grain(plan):
    return [a for a in plan if a.kind == "bank_trade" and a.get == {"grain": 1}]


# ---- the city stays a goal --------------------------------------------------------------

def test_lost_city_is_the_first_goal_and_keeps_ore_grain():
    br = _brain()
    gs = _gs({"wool": 4, "ore": 1})
    br.update_outlook(gs, 1)
    assert br.outlook.goals[0].kind == "city"
    assert {"ore", "grain"} <= set(br.keep) and {"ore", "grain"} <= set(br.outlook.keep)


def test_city_goal_off_switch_restores_the_old_goals():
    br = _brain(city_always=False, city_rebuild_w=0.0)
    gs = _gs({"wool": 4, "ore": 1})
    br.update_outlook(gs, 1)
    assert br.outlook.goals[0].kind != "city"          # the old order: a canal chip first


def test_city_goal_kept_with_a_city_standing_before_the_canal():
    br = _brain()
    gs = _gs({"wool": 1}, city=True, attacked=False, knights=False)
    br.update_outlook(gs, 0)
    assert any(g.kind == "city" for g in br.outlook.goals)
    assert not br.city_rebuild(gs)


# ---- after the loss: the rebuild beats digs and ships -----------------------------------

def test_2340_wool4_ore1_banks_wool_for_grain_not_a_ship_move():
    # 23:40:50 hand wool 4 ore 1, plan=['move_ship']
    plan = _plan(_brain(), _gs({"wool": 4, "ore": 1}))
    assert plan and plan[0].kind == "bank_trade" and plan[0].give == {"wool": 3}
    assert "move_ship" not in [a.kind for a in plan]


def test_lumber_set_goes_to_the_city_not_a_ship():
    plan = _plan(_brain(), _gs({"lumber": 3, "wool": 2, "ore": 1}))
    assert plan[0].kind == "bank_trade" and plan[0].give == {"lumber": 3}
    assert "ship" not in [a.kind for a in plan]


def test_2316_ore_is_not_banked_for_an_activation():
    # 23:16:47 plan=['bank_trade','activate','move_ship'] paid with 3 ore
    plan = _plan(_brain(), _gs({"ore": 3, "wool": 1, "lumber": 1}))
    assert not any(a.kind == "bank_trade" and a.give.get("ore") for a in plan)
    assert "activate" not in [a.kind for a in plan]


def test_rebuild_off_switch_lets_the_old_spend_happen():
    plan = _plan(_brain(city_rebuild_w=0.0), _gs({"lumber": 3, "wool": 2, "ore": 1}))
    kinds = [a.kind for a in plan]
    assert "promote" in kinds and "activate" in kinds     # g3: ore to promotions, grain to digs


def test_city_still_built_when_affordable():
    plan = _plan(_brain(), _gs({"ore": 3, "grain": 2}))
    assert plan and plan[0].kind == "city"


def test_set_need_cap_whole_city_in_rebuild():
    br = _brain()
    gs = _gs({"wool": 2})
    br.update_outlook(gs, 1)
    assert br.set_need_cap(gs, "city") == 5
    assert br.set_need_cap(gs, "settlement") == 3
    assert br.set_need_cap(gs, "science 4") == 3


def test_plan_missing_asks_for_the_city_cards_in_rebuild():
    br = _brain()
    gs = _gs({"wool": 4, "ore": 1})
    br.update_outlook(gs, 1)
    assert set(br.plan_missing(gs)) <= {"ore", "grain"}


# ---- turn 1 of g3: wool set into grain before the knight --------------------------------

def _turn1():
    # 22:58:26 brick 1 lumber 1 wool 3 ore 1 paper 1 coin 1, knights 0, 3:1 harbour
    gs = _gs({"brick": 1, "lumber": 1, "wool": 3, "ore": 1, "paper": 1, "coin": 1},
             city=True, attacked=False, knights=False)
    return gs


def test_turn1_wool_set_banks_for_grain():
    br = _brain()
    a = br._grain_first_trade(_turn1())
    assert a is not None and a.give == {"wool": 3} and a.get == {"grain": 1}


def test_turn1_switch_off_keeps_the_old_hold():
    br = _brain(opening_set_first=False)
    assert br._grain_first_trade(_turn1()) is None


def test_config_switches_reach_the_brain():
    r = _runner(city_always_goal=False, city_rebuild_first=False, opening_set_first=False)
    assert r.brain.city_always is False
    assert r.brain.city_rebuild_w == 0.0
    assert r.brain.opening_set_first is False
    d = config.DEFAULTS
    assert d["city_always_goal"] and d["city_rebuild_first"] and d["opening_set_first"]
