"""Owner 2026-10-05 game 15: city and settlement swapped -- a phantom scanned piece shifted the
brain's pick number. The runner makes our pieces match the prompt before the brain picks."""
from types import SimpleNamespace

from catanbot.runner import Runner
from catanbot.state import PlayerState


def _run(picks, settlements=(), cities=(), kind="settlement"):
    me = PlayerState(name="marlow.systems")
    me.settlements, me.cities = set(settlements), set(cities)
    gs = SimpleNamespace(mode=SimpleNamespace(start_with_city=True), me=lambda: me)
    notes = []
    fake = SimpleNamespace(_setup_picks=picks, note=notes.append)
    Runner._reconcile_setup_pieces(fake, gs, kind)
    return me, notes


def test_phantom_piece_dropped_before_first_settlement():
    me, notes = _run([], settlements=[(-6, -8)])
    assert me.buildings() == [] and notes


def test_city_prompt_keeps_only_the_placed_settlement():
    picks = [{"kind": "settlement", "v": (0, 2)}]
    me, _ = _run(picks, settlements=[(-6, -8), (0, 2)], kind="city")
    assert me.buildings() == [((0, 2), "settlement")]


def test_city_prompt_adds_a_missed_settlement():
    picks = [{"kind": "settlement", "v": (9, 9)}, {"kind": "settlement", "v": (0, 2)}]
    me, _ = _run(picks, kind="city")
    assert me.buildings() == [((0, 2), "settlement")]


def test_last_settlement_after_city():
    picks = [{"kind": "settlement", "v": (0, 2)}, {"kind": "city", "v": (5, -5)}]
    me, _ = _run(picks, settlements=[(-6, -8), (0, 2)], cities=[(5, -5)])
    assert sorted(me.buildings()) == [((0, 2), "settlement"), ((5, -5), "city")]


def test_rejoin_trusts_the_scan():
    me, notes = _run([], settlements=[(0, 2)], cities=[(5, -5)])
    assert len(me.buildings()) == 2 and not notes
