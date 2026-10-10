"""Stage 2026-10-09 07:22: Wedding held all game while we led ("no opponent has more VP than us")."""
from catanbot import progress_cards as pc
from catanbot.sim import Sim
from catanbot.state import new_hand


def _gs(lead):
    sim = Sim("great_canal", seed=1)
    sim.setup()
    g = sim.gs
    for p in g.players:
        p.hand, p.progress_hand = new_hand(), []
        p.hand.update(wool=2)
        p.vp_visible = 6
    g.me().vp_visible = 6 + lead
    return sim, g


def _drop(g, dead):
    hand = ["wedding", "bishop", "saboteur"]
    return min(hand, key=lambda c: pc.keep_value(g, g.me(), c, dead))


def test_leading_drops_wedding_first():
    _, g = _gs(2)
    assert _drop(g, True) == "wedding"


def test_trailing_or_flag_off_unchanged():
    _, g = _gs(-2)
    assert _drop(g, True) == _drop(g, False) != "wedding"
    _, g = _gs(2)
    assert _drop(g, False) != "wedding"


def test_sim_limit_uses_the_brain_flag():
    sim, g = _gs(2)
    me = g.me()
    me.progress_hand = ["wedding", "bishop", "saboteur", "warlord", "spy"]
    sim._progress_limit(me)
    assert "wedding" not in me.progress_hand and len(me.progress_hand) == pc.PROGRESS_HAND_LIMIT
