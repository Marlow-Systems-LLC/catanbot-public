"""Owner 2026-10-09: every game counts in marlow's tally -- win, loss or failed -- once."""
from catanbot import chat


def test_win_loss_and_streak_per_bucket():
    chat.record_game_open("g1", solo=True)
    chat.record_result(True, solo=True, game_id="g1", place=1, players=4, vp=18)
    chat.record_game_open("g2", solo=True)
    r = chat.record_result(True, solo=True, game_id="g2", place=1, players=4, vp=18)
    s = r["solo"]
    assert (s["games"], s["wins"], s["losses"], s["failed"], s["streak"], s["best_streak"]) == (2, 2, 0, 0, 2, 2)
    assert chat.record().get("games", 0) == 0          # people bucket untouched
    assert "open" not in chat.record()


def test_second_game_over_screen_is_not_counted_again():
    chat.record_game_open("g1")
    chat.record_result(False, game_id="g1", place=3, players=4, vp=9)
    r = chat.record_result(False, game_id="g1", place=3, players=4, vp=9)
    assert r.get("duplicate") and chat.record()["games"] == 1


def test_left_game_counts_as_failed_once_the_next_game_opens():
    chat.record_game_open("g1", solo=True, seats=["marlow.systems", "Lin"], mode="great_canal")
    assert chat.record_left("g1", "left the game (home)")
    assert not chat.record_left("g1", "left the game (home)")
    assert chat.record().get("solo", {}).get("failed", 0) == 0          # not yet: we may be pulled back in
    chat.record_game_open("g2", solo=True, seats=["marlow.systems", "Jean"], mode="great_canal")
    s = chat.record()["solo"]
    assert (s["games"], s["failed"], s["streak"]) == (1, 1, 0)
    assert chat.record()["log"][-1]["outcome"] == "failed"


def test_pulled_back_into_the_same_match_counts_once():
    seats = ["marlow.systems", "Lin", "Marianne", "Jean"]
    chat.record_game_open("A", solo=True, seats=seats, mode="great_canal")
    chat.record_left("A", "left the game (lobby)")
    chat.record_game_open("B", solo=True, seats=seats, mode="great_canal")   # same match, new log id
    chat.record_result(True, solo=True, game_id="B", place=1, players=4, vp=18)
    s = chat.record()["solo"]
    assert (s["games"], s["wins"], s["failed"]) == (1, 1, 0)


def test_crash_fails_the_open_game_when_the_next_one_opens():
    chat.record_game_open("g1", solo=True)        # runner died, no end screen, no leave
    chat.record_game_open("g2", solo=True)
    r = chat.record()
    assert r["solo"]["failed"] == 1 and r["open"]["id"] == "g2"


def test_reopening_the_same_game_after_a_restart_is_not_a_failure():
    chat.record_game_open("g1")
    chat.record_game_open("g1")
    assert chat.record().get("failed", 0) == 0
