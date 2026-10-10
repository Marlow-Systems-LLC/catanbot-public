"""Engineer free wall (live game 13 17:45): "Select position for City waff" is not the setup city prompt."""
from catanbot.vision.screens import classify


def test_city_wall_is_place_wall():
    assert classify(["marlow.systems", "Select position for City waff"])[0] == "place_wall"
    assert classify(["Select position for City wall"])[0] == "place_wall"


def test_setup_city_unchanged():
    assert classify(["Select position for City"])[0] == "setup_city"
    assert classify(["Select position for Cify"])[0] == "setup_city"
