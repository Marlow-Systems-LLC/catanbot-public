"""Live 2026-10-08: the Spy card picker title OCR'd as "Select progress cara", the screen
classified as my_turn and marlow sat on the dialog for minutes."""
from catanbot.vision.screens import classify, find_text

TEXTS = ["Rhea", "marlow.systems", "Bramble", "Wren", "Select progress cara",
         "Intrigue", "Merchant Fleet", "Hide overlay", "BUILD COSTS", "Road Building"]


def test_misread_title_is_pick_progress():
    assert classify(TEXTS)[0] == "pick_progress"


def test_header_found_for_card_row():
    ocr = [{"text": t, "x": 600, "y": 200 + 10 * i, "w": 80, "h": 20} for i, t in enumerate(TEXTS)]
    assert find_text(ocr, r"select (a )?progress car[a-z]?")["text"] == "Select progress cara"


def test_spy_pick_with_a_deserter_is_not_choose_player():
    """Live 2026-10-09 21:03: the Deserter's text "Choose an opponent ..." on the Spy's card pick
    made it choose_player; 80 s of "no names read" and the Spy took nothing."""
    from catanbot.vision.screens import classify
    t = ['13', 'Rhea', '[EZ)', 'Ship', 'marlow.systems', 'Wedding', 'Each player who has more',
         'victory points than you must', 'give you 2 resources and/or', 'commodities of his choice.',
         'BUILD COSTS', 'STE', 'Bramble', '668', 'Marianne', '• =', 'Select progress card', 'Crane',
         'Deserter', 'One city improvement (abbey,', 'town hall, etc.) that you build',
         'this turn costs one less', 'commodity than usual.', 'Choose an opponent, who has to',
         'remove one of his knights from the', 'board. You may place one of your',
         'own knights of equal strength on', 'the board.', 'Hide overlay', 'Mining', '5']
    assert classify(t)[0] == "pick_progress"
    assert classify(["Choose an opponent", "Rhea", "Marianne", "Hide overlay"])[0] == "choose_player"
