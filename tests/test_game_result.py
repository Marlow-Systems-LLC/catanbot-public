"""End of game (live game 13 18:10): the "You've won!" banner sat 4 min unread before the rescue."""
from catanbot.vision.screens import classify


def test_won_banner():
    assert classify(["You've won!", "50", "• Experience", "Ranking", "VIEW PROGRESS", "CONTINUE"])[0] == "game_result"


def test_lost_banner():
    assert classify(["You've lost", "20", "Experience", "Ranking", "VIEW PROGRESS", "CONTINUE"])[0] == "game_result"


def test_game_over_unchanged():
    assert classify(["Experience gained", "Player statistics", "QUIT"])[0] == "game_over"
