

def test_chat_names_the_scenario_and_answers_questions_addressed_by_name():
    """2026-10-03: the chat called the game Greater Catan, and every "marlow <question>" got
    the who-am-I line because "marlow" matched the about-me pattern."""
    from catanbot import chat
    from catanbot.rules import MODES

    class G:
        mode = MODES["great_canal"]
    c = chat.Chat(clock=lambda: 1e9)
    r = c.respond("Lin", "marlow what game is this?", G())
    assert "The Great Canal" in r and "18 VP" in r and chat.safe(r)
    assert "I'm marlow" not in chat.Chat(clock=lambda: 1e9).respond("Lin", "marlow how do knights work", G())
    assert chat._other_scenario("In Greater Catan the islands matter.", MODES["great_canal"])
    assert not chat._other_scenario("The Great Canal needs knights.", MODES["great_canal"])
