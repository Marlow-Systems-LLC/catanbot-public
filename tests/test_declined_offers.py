"""Stage 2026-10-09: ore->lumber offered 4 times and declined by all -- remember it for the game."""
from catanbot.hands import HandLedger
from tests.test_progress_wiring import gs, runner  # noqa: F401 (fixture)


def _setup(gs, tmp_path):
    me = gs.me()
    me.hand.clear()
    me.hand.update(lumber=1, brick=1, wool=1, ore=2)      # settlement: 1 grain short
    r = runner(gs, [], tmp_path)
    r.brain._trade_targets = lambda g: ["settlement"]
    led = HandLedger(); led.me = me.name
    r._hands = gs.hands = led
    r._turn_n = 1
    return r


def _decline(r, t):
    r._pending_offer = (dict(t.give), dict(t.get))
    r._remember_decline(*r._pending_offer)


def test_declined_offer_not_sent_next_turn(gs, tmp_path):
    r = _setup(gs, tmp_path)
    t = r._want_trade(gs)
    assert t is not None
    _decline(r, t)
    r._offers_made = set(); r._turn_n = 2                  # next own turn
    t2 = r._want_trade(gs)
    assert t2 is None or (t2.give, t2.get) != (t.give, t.get)


def test_opponent_gain_of_wanted_card_allows_it_again(gs, tmp_path):
    r = _setup(gs, tmp_path)
    t = r._want_trade(gs)
    _decline(r, t)
    want = next(iter(t.get))
    r._hands.p(gs.opponents()[0].name).add({want: 2})
    r._offers_made = set(); r._turn_n = 2
    t2 = r._want_trade(gs)
    assert t2 is not None and (t2.give, t2.get) == (t.give, t.get)


def test_decline_expires_and_cfg_off(gs, tmp_path):
    r = _setup(gs, tmp_path)
    t = r._want_trade(gs)
    _decline(r, t)
    r._turn_n = 7                                          # 6 own turns later
    assert not r._was_declined(t.give, t.get)
    r.cfg["offer_repeat_after_turns"] = 0
    _decline(r, t)
    assert not r._was_declined(t.give, t.get)


def test_declines_are_per_game():
    from catanbot.runner import Runner
    assert "_declined" in Runner._PER_GAME_LAZY and "_turn_n" in Runner._PER_GAME_LAZY
