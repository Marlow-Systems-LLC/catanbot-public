"""owner 2026-10-08: Marlow never throws grain away; grain is the last card given up."""
from catanbot.runner import Runner

from test_canal_playbook import _gs, brain


def _hand(gs, **cards):
    me = gs.me()
    me.hand.clear()
    me.hand.update(cards)
    me.hand_count = sum(cards.values())


def test_grain_kept_while_anything_else_is_left():
    br, gs = brain(), _gs()
    _hand(gs, grain=6, ore=1, wool=1, coin=1)
    assert br.choose_discard(gs, 3).get("grain", 0) == 0


def test_grain_goes_only_after_everything_else():
    br, gs = brain(), _gs()
    _hand(gs, grain=4, brick=2)
    assert br.choose_discard(gs, 3) == {"brick": 2, "grain": 1}


def test_blind_discard_clicks_grain_last():
    assert Runner.BLIND_DISCARD[-1] == "grain"
