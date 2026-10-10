"""Owner 2026-10-05 game 15: the city fell with 0 active knights; a rival can switch a knight off
before the attack to tip the total, so enough knights now is not safe."""
from catanbot.brain import Action, Brain
from catanbot.state import Knight
from tests.test_canal_playbook import C, _gs, brain


def test_rival_can_switch_off_a_knight_to_sink_us():
    gs = _gs(barbarian_pos=6)
    me, leader, weak = gs.players
    me.cities.add(C[0]); leader.cities.add(C[2]); weak.cities.add(C[4])
    leader.knights[C[1]] = Knight(1, True); leader.knights[C[3]] = Knight(1, True)
    weak.knights[C[5]] = Knight(1, True)
    assert gs.defence_strength(me) == 0
    assert Brain.barb_loss_risk(gs)          # 3 knights vs 3 cities now, but the leader can drop one


def test_promoting_an_idle_knight_gets_no_danger_bonus():
    gs = _gs(barbarian_pos=6)
    me = gs.players[0]
    me.cities.add(C[0]); me.knights[C[2]] = Knight(1, False)
    me.hand.update({"wool": 1, "ore": 1})
    br = brain()
    on = br.action_value(gs, Action("promote", C[2]))
    br.idle_knight_first = False
    off = br.action_value(gs, Action("promote", C[2]))
    assert on < off
