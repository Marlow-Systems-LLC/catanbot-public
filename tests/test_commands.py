from catanbot import commands


def test_leave_words():
    for t in ["marlow exit the game", "Marlow, please leave the game now", "quit the match marlow", "exit"]:
        assert commands.intent({"text": t}) == "leave", t
    for t in ["marlow nice move", "who is winning", "I will leave you a sheep"]:
        assert commands.intent({"text": t}) is None, t
    assert commands.intent({"command": "leave"}) == "leave"


def test_apply_writes_leave_flag(tmp_path):
    (tmp_path / "data").mkdir()
    done = commands.apply([{"id": 7, "command": "leave"}, {"id": 8, "text": "hello"}], note=lambda s: None, root=tmp_path)
    assert done == ["7", "8"] and (tmp_path / "data" / "leave.flag").exists()
