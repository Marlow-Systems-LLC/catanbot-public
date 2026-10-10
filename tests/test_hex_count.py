"""Owner 2026-10-09: settle on corners that pay from 3 hexes, then 2, rarely 1 -- pips still order
corners with the same count, and a must-have (the canal 3:1 we lack) still wins on one hex."""
from catanbot.board import PIPS
from catanbot.brain import Brain, producing_hexes
from catanbot.learning import Model
from tests.test_live_bugs_1008 import _live_board_gs


def _brains():
    on = Brain(Model(path=None), setup_weights={})
    off = Brain(Model(path=None), setup_weights={})
    off.setup_w.update({f"hex_count_{n}": 0.0 for n in range(4)})
    return on, off


def _pips(gs, v):
    return sum(PIPS[h.number] for h in gs.board.vertex_hexes(v)
               if h.number and h.terrain not in ("gold", "fields"))


def test_producing_hexes_skips_gold_canal_farmland_robber_and_fog():
    gs, _ = _live_board_gs()
    b = gs.board
    for v in b.vertices:
        want = sum(1 for h in b.vertex_hexes(v) if h.number and h.terrain not in ("gold", "fields"))
        assert producing_hexes(gs, v) == want
    v = next(v for v in b.vertices if producing_hexes(gs, v) == 3)
    h = next(h for h in b.vertex_hexes(v) if h.number and h.terrain not in ("gold", "fields"))
    b.robber = h.coord
    assert producing_hexes(gs, v) == 2
    b.robber = None
    h.fog = True
    assert producing_hexes(gs, v) == 2


def test_three_hex_corner_beats_a_two_hex_corner_one_pip_better():
    gs, me = _live_board_gs()
    three, two = (1, 1), (5, -5)
    assert (producing_hexes(gs, three), producing_hexes(gs, two)) == (3, 2)
    assert _pips(gs, two) == _pips(gs, three) + 1 and not gs.board.harbor_at(two)
    on, off = _brains()
    assert off.choose_setup_settlement(gs, [three, two]) == two        # the old pick
    assert on.choose_setup_settlement(gs, [three, two]) == three


def test_one_hex_corner_loses_to_a_two_hex_corner():
    gs, me = _live_board_gs()
    two, one = (-2, 4), (6, 2)
    assert (producing_hexes(gs, two), producing_hexes(gs, one)) == (2, 1)
    assert not gs.board.harbor_at(one)
    on, off = _brains()
    assert off.choose_setup_settlement(gs, [two, one]) == one
    assert on.choose_setup_settlement(gs, [two, one]) == two


def test_last_canal_pick_still_takes_the_one_hex_3to1():
    from catanbot.sim import Sim
    sim = Sim("great_canal", seed=5)
    sim.setup()
    gs, me = sim.gs, sim.gs.players[0]
    v = (6, -4)
    assert v in me.settlements and gs.board.harbor_at(v) == "3:1"
    assert producing_hexes(gs, v) == 1


def test_one_hex_corner_is_no_road_goal():
    gs, me = _live_board_gs()
    br, _ = _brains()
    b = gs.board
    ones = [v for v in b.vertices if gs.can_settle(v) and producing_hexes(gs, v) <= 1
            and br._raw_spot(gs, v) >= br.ROAD_SPOT_MIN and not br._hex_must_have(gs, v, me)]
    assert ones                                                     # the fixture has such corners
    assert not any(br._road_target(gs, me, v) for v in ones)


def test_selfplay_smoke():
    from catanbot.sim import Sim
    winner, turns, _ = Sim("great_canal", 4, seed=11).run(max_turns=120)
    assert turns > 0
