"""Owner 2026-10-03: step 1 -- never type a password; on a sign-in prompt or a game that won't
come back, stop (data/STOP) and alert instead of looping."""
import time
from catanbot import runner as R


class FakeIO:
    def paused(self): return False


def _runner(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "STOP", tmp_path / "STOP")
    monkeypatch.setattr(R.Runner, "ALERT", tmp_path / "ALERT")
    monkeypatch.setattr("subprocess.run", lambda *a, **k: type("P", (), {"returncode": 1})())
    r = R.Runner.__new__(R.Runner)
    r.io, r.notes = FakeIO(), []
    r.note = r.notes.append
    return r


def test_login_screen_relaunches_once_then_stops_and_alerts(tmp_path, monkeypatch):
    r = _runner(tmp_path, monkeypatch)
    restarts, t = [], [1000.0]
    r.restart_app = lambda why: (restarts.append(why), setattr(r, "_restarted_at", t[0]))
    monkeypatch.setattr(time, "time", lambda: t[0])
    for _ in range(40):                              # 40 frames, 5 s apart = 200 s on the screen
        r.on_login(None, [], [], None); t[0] += 5
    assert restarts == ["stuck on the login screen"] and not R.STOP.exists()
    for _ in range(80):                              # 400 s more: past the 300 s relaunch guard
        r.on_login(None, [], [], None); t[0] += 5
    assert len(restarts) == 1 and R.STOP.exists()
    assert "sign-in" in (tmp_path / "ALERT").read_text()


def test_fourth_restart_in_half_an_hour_stops(tmp_path, monkeypatch):
    r = _runner(tmp_path, monkeypatch)
    r.leave_game = lambda tries=8: True
    r.cfg = {}
    monkeypatch.setattr(time, "sleep", lambda s: None)
    for i in range(3):
        r.restart_app(f"r{i}")
    assert not R.STOP.exists()
    r.restart_app("r3")
    assert R.STOP.exists() and "4 restarts" in (tmp_path / "ALERT").read_text()


def test_missing_window_relaunches_then_alerts(tmp_path, monkeypatch):
    r = _runner(tmp_path, monkeypatch)
    t = [5000.0]
    monkeypatch.setattr(time, "time", lambda: t[0])
    for _ in range(25):                              # 75 s without a window
        r._window_missing(True); t[0] += 3
    assert any("launching it through Steam" in n for n in r.notes) and not R.STOP.exists()
    r._window_missing(False)
    assert r._no_window_since is None
    for _ in range(110):                             # 330 s without a window
        r._window_missing(True); t[0] += 3
    assert R.STOP.exists()


def test_black_game_window_restarts_the_game(tmp_path, monkeypatch):
    """21:30 2026-10-03: after a relaunch the game window stayed black (hung) for 10 min."""
    r = _runner(tmp_path, monkeypatch)
    restarts = []
    r.restart_app = lambda why: restarts.append(why)
    t = [5000.0]
    monkeypatch.setattr(time, "time", lambda: t[0])
    for _ in range(25):                              # 75 s of black frames
        r._window_missing(True, "game window capture is black and the window is not on screen"); t[0] += 3
    assert restarts == ["game window black for 60 s"]


def test_same_click_thirteen_times_is_blocked_then_alerts_on_third_loop(tmp_path, monkeypatch):
    r = _runner(tmp_path, monkeypatch)
    r.cfg = {"click_delay_s": 0}
    clicks = []
    r.io.click = lambda x, y, why: clicks.append(why)
    t = [100.0]
    monkeypatch.setattr(time, "time", lambda: t[0])
    monkeypatch.setattr(time, "sleep", lambda s: None)
    for _ in range(20):
        r.click((654, 646), "end turn"); t[0] += 2
    assert len(clicks) == 12 and any("LOOP" in n for n in r.notes)
    r.click((300, 300), "build road"); assert clicks[-1] == "build road"   # other actions still go
    for _ in range(2):
        t[0] += 100
        for _ in range(13):
            r.click((654, 646), "end turn"); t[0] += 2
    assert "3 click loops" in (tmp_path / "ALERT").read_text() and not R.STOP.exists()


def test_lobby_needs_a_real_ready_button_and_invites_need_invitation_words():
    from catanbot.vision.screens import classify, FORBIDDEN_BUTTONS
    assert classify(["You already have 2 roads"])[0] != "lobby"
    assert classify(["Rhea", "READY", "marlow.systems"])[0] == "lobby"
    assert classify(["marlow.systems", "NOT READY"])[0] == "lobby"
    assert FORBIDDEN_BUTTONS.search("Quick Play") and FORBIDDEN_BUTTONS.search("Tournament")
    assert not FORBIDDEN_BUTTONS.search("Random board")


def test_wedding_gift_dialog_is_a_discard():
    """Live 2026-10-03 game 6: the Wedding dialog read as 'main' and the bot sat on it."""
    from catanbot.vision.screens import classify
    assert classify(["Wedding", "Player Marianne is marrying! Choose 2 more resources you want to gift him or her."])[0] == "discard"


def test_solo_menu_is_home_and_quickplay_is_not_matchmaking_there():
    """21:15 2026-10-03: the offline-games menu (cards say "Quickplay") read as main, and
    FORBIDDEN "quick play" made start_solo give up and turn solo off."""
    import re
    from catanbot.vision.screens import classify, FORBIDDEN_BUTTONS
    texts = ["SOLO: OFFLINE GAMES", "THE GAME", "SCENARIO SELECTION", "Quickplay",
             "The First Island", "Continue", "The Great Canal"]
    assert classify(texts)[0] == "home"
    joined = " ".join(texts)
    assert FORBIDDEN_BUTTONS.search(joined)               # still forbidden elsewhere
    assert not FORBIDDEN_BUTTONS.search(re.sub(r"(?i)quick ?play", "", joined))


def test_progress_card_screen_has_a_handler():
    from catanbot.runner import Runner
    assert callable(getattr(Runner, "on_progress_card", None))


def test_setup_rings_are_read_on_the_native_capture():
    """Game 7 2026-10-03: full screen 1920x1080, the 1372x888 frame showed 1 ring of ~34 and
    every setup settlement click missed."""
    from pathlib import Path
    import cv2
    from catanbot.runner import Runner, FRAME
    from catanbot.vision.layout import Layout
    native = cv2.imread(str(Path(__file__).parent / "data" / "g7_setup_rings_native.png"))
    lay = Layout(native.shape[1], native.shape[0])
    frame = lay.frame(native, FRAME)
    assert len(Runner.ring_spots(frame)) <= 2                      # the old way
    rings = Runner.ring_spots(frame, native, lay)
    assert len(rings) >= 25
    assert all(0 <= x < frame.shape[1] and 190 <= y < 740 for x, y in rings)


def test_result_screen_ranking_gives_the_winner():
    """Game 7 2026-10-03: the result screen ranks players; no "won" text -> winner=None."""
    from catanbot.runner import Runner
    texts = ["RESULT", "RANKING", "PLAYER STATISTICS", "GAME STATISTICS", "1 Lin", "2 Marianne",
             "3 Jean", "4 marlow.systems", "(12", "Experience gained: 0", "Experience gained: +25", "QUIT"]
    assert Runner.read_ranking(texts) == ["Lin", "Marianne", "Jean", "marlow.systems"]
    assert Runner.read_ranking(["Experience gained: 0", "QUIT"]) == []


def test_ranking_keeps_both_names_of_a_tie():
    # solo 2026-10-06 03:54: "2 marlow.systems" and "2 Jean" -- Jean was dropped from the caption
    from catanbot.runner import Runner
    texts = ["RESULT", "1 Lin", "2 marlow.systems", "2 Jean", "3 Marianne", "Experience gained: 0"]
    assert Runner.read_ranking(texts) == ["Lin", "marlow.systems", "Jean", "Marianne"]
