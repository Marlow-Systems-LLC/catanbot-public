"""Owner 2026-10-02: "giving out hey for cley when it needs to build a knight (sheep and stone)"."""
from catanbot.brain import TurnCtx
from catanbot import moves as mv
from tests.test_progress_wiring import gs, runner  # noqa: F401 (fixture)


def _hand(gs):
    me = gs.me()
    me.hand.update(lumber=2, grain=2)
    return ["knight", "settlement", "city", "road"]


def test_runner_never_offers_grain_in_canal(gs, tmp_path):
    kinds = _hand(gs)
    r = runner(gs, [], tmp_path)
    r.brain._trade_targets = lambda g: kinds
    t = r._want_trade(gs)
    assert t is None or "grain" not in t.give
    assert t is None or set(t.get) <= {"wool", "ore"}      # asks toward the knight


def test_brain_offer_never_gives_grain_in_canal(gs, tmp_path):
    kinds = _hand(gs)
    r = runner(gs, [], tmp_path)
    b = r.brain
    b._offer_kinds = lambda g, ctx: kinds
    L = set(mv.turn_moves(gs, offers_left=True))
    m = b._offer_move(gs, L, TurnCtx())
    assert m is None or "grain" not in m.give_d
