"""Solo 2026-10-05 23:27: a runner restarted on the end screen recorded + shared the game twice."""
import time

from catanbot import runner as R

TEXTS = ["1 Jean", "2 Marianne", "3 marlow.systems", "4 Lin", "QUIT"]


def _runner(tmp_path, monkeypatch):
    monkeypatch.setattr(R.Runner, "GAME_OVER_MARK", tmp_path / "last_game_over.json")
    r = R.Runner.__new__(R.Runner)
    r.note = lambda *a, **k: None
    return r


def test_same_result_after_restart_is_not_handled_again(tmp_path, monkeypatch):
    r = _runner(tmp_path, monkeypatch)
    assert not r._game_over_seen(TEXTS)
    r.GAME_OVER_MARK.write_text('{"ranking": ["Jean", "Marianne", "marlow.systems", "Lin"], "at": %f}' % time.time())
    r2 = _runner(tmp_path, monkeypatch)            # a fresh runner process
    assert r2._game_over_seen(TEXTS)
    assert not r2._game_over_seen(["1 Lin", "2 Jean", "3 marlow.systems", "4 Marianne"])


def test_mark_older_than_the_backstop_does_not_block(tmp_path, monkeypatch):
    r = _runner(tmp_path, monkeypatch)
    r.GAME_OVER_MARK.write_text('{"ranking": ["Jean", "Marianne", "marlow.systems", "Lin"], "at": %f}' % (time.time() - 13 * 3600))
    assert not r._game_over_seen(TEXTS)


def test_end_screen_left_up_for_hours_is_still_the_same_game(tmp_path, monkeypatch):
    # owner 2026-10-06: stop_after_game.flag leaves the end screen up; a restart after 30 min re-posted
    r = _runner(tmp_path, monkeypatch)
    r.GAME_OVER_MARK.write_text('{"ranking": ["Jean", "Marianne", "marlow.systems", "Lin"], "at": %f}' % (time.time() - 3 * 3600))
    assert r._game_over_seen(TEXTS)


def test_first_roll_of_a_new_game_clears_the_mark(tmp_path, monkeypatch):
    r = _runner(tmp_path, monkeypatch)
    r.GAME_OVER_MARK.write_text('{"ranking": ["Jean", "Marianne", "marlow.systems", "Lin"], "at": %f}' % time.time())
    r.io = object()                                  # a live io, not a replay
    try:
        r.on_roll(None, [], [], {})
    except Exception:                                # the rest of on_roll needs a live game
        pass
    assert not r.GAME_OVER_MARK.exists()
    assert not r._game_over_seen(TEXTS)              # same ranking in the new game counts
