"""owner 2026-10-08: "bot gave away grain for a trade when asked by a player with more points" --
grain never goes to another player: not on an offer, not on a wildcard counter, not in our offers."""
from catanbot import moves as mv
from catanbot.brain import TurnCtx

from test_canal_playbook import _gs, brain


def _hand(gs):
    gs.me().hand = {"grain": 3, "wool": 1, "lumber": 1}
    gs.me().hand_count = 5
    return gs


def test_offer_asking_grain_is_refused_from_anyone():
    br = brain()
    gs = _hand(_gs())
    for partner in ("weak", "leader"):
        assert not br.evaluate_offer(gs, partner, {"grain": 1}, {"ore": 1})
        assert not br.evaluate_offer(gs, partner, {"grain": 1}, {"ore": 2})


def test_wildcard_asking_grain_is_declined_not_countered():
    br = brain()
    gs = _hand(_gs())
    counters = [mv.Move("counter", give=(("grain", 1),), get=((r, 1),)) for r in ("lumber", "ore", "brick")]
    ans = br.choose_answer(gs, counters + [mv.Move("decline")], "weak")
    assert ans.kind == "decline"


def test_grain_is_never_in_our_offers():
    br = brain()
    assert "grain" in br._protected(TurnCtx(), _gs())
