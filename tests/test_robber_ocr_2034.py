"""Live 2026-10-09 20:34:14-20:35:05: after our own discard the prompt OCR'd as
"Select position for Robbef®" and was read as an unknown screen for 50 s."""
from catanbot.vision.screens import classify

TEXTS_2034 = ["10", "Rhea", "2", "ECE", "marlow.systems", "11.03", "Bramble", "E6S", "Marianne", "5",
              "Select position for Robbef®", "3", "7 batetadн", "6 Labondn Fhengs", "BUILD COSTS",
              "EERE", "Crane", "Ship", "1<"]


def test_misread_robber_prompt_is_place_robber():
    assert classify(TEXTS_2034)[0] == "place_robber"


def test_other_select_position_prompts_unchanged():
    for t, want in [("Select position for Road", "setup_road"), ("Select position for Settlement", "setup_settlement"),
                    ("Select position for Cify", "setup_city"), ("Select position for Ship", "place_ship")]:
        assert classify([t])[0] == want
