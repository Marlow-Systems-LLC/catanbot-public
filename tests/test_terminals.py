from catanbot import terminals


def test_script_only_targets_idle_bot_windows():
    s = terminals.applescript()
    assert "busy of every tab" in s and "does not contain true" in s
    assert '"start_bot.command"' in s and '"runner_only.command"' in s
    assert "claude" not in s.lower()


def test_close_returns_count_and_survives_errors(monkeypatch):
    monkeypatch.setattr(terminals.sys, "platform", "darwin")

    class R:
        stdout = "3\n"
    assert terminals.close_idle_bot_terminals(run=lambda *a, **k: R()) == 3

    def boom(*a, **k):
        raise OSError("no osascript")
    assert terminals.close_idle_bot_terminals(run=boom) == 0
