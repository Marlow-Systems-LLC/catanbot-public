"""Owner 2026-10-06: when every player has said no, move on at once; wait up to 15 s only while
someone has not answered."""
import time

from catanbot import runner as R


def _runner(states, sent_ago):
    r = R.Runner.__new__(R.Runner)
    r.note = lambda *a, **k: None
    r.cfg = {}
    r.gs = None
    r.offer_sent = time.time() - sent_ago
    r.clicks = []
    r.click = lambda xy, why="": r.clicks.append(why)
    r.offer_responses = lambda img: [(316 + 48 * i, k) for i, k in enumerate(states)]
    r._resp_log_t = time.time()
    return r


def test_all_declined_closes_at_once():
    r = _runner(["decline", "decline", "decline"], 1)
    assert r._handle_responses(None) and r.clicks == ["close declined trade"] and r.offer_sent == 0


def test_one_still_thinking_waits():
    r = _runner(["decline", "thinking", "decline"], 5)
    assert not r._handle_responses(None) and r.clicks == []


def test_still_thinking_after_15_s_gives_up():
    r = _runner(["decline", "thinking", "none"], 16)
    assert r._handle_responses(None) and r.clicks == ["close declined trade"]
