"""Owner 2026-10-08: "if you don't quit right the game will try to bring you back in the old game
and freeze a couple of times". Leaving a match: every dialog variant pressed by its OCR'd label,
bounded presses, the front page verified, and a relaunch that rejoins the old match is caught."""
import numpy as np
import pytest

import catanbot.runner as R
from catanbot import config
from catanbot.runner import ReplayIO, Runner

HOME = ["PLAY", "Leaderboards", "Almanac", "Guilds"]
BOARD = ["marlow.systems", "Hildegard", "It's Hildegard's turn"]
MID_DIALOG = ["Quit game", "Do you really want to quit the game?", "NO", "YES"]   # live 19:22


class ScreensIO:
    """Each snapshot shows the next screen (a list of OCR texts); the last one stays up.
    `after_click` maps a click label to the screen list that follows it."""
    mac = True

    def __init__(self, screens, after_click=None):
        self.screens, self.after, self.i, self.clicks = list(screens), after_click or {}, 0, []

    def snapshot(self):
        img = np.zeros((888, 1372, 3), np.uint8)
        self.cur = self.screens[min(self.i, len(self.screens) - 1)]
        self.i += 1
        return img

    def ocr(self, img):
        return [{"text": t, "x": 600 + 10 * k, "y": 300 + 30 * k, "w": 80, "h": 20} for k, t in enumerate(self.cur)]

    def click(self, x, y, why=""):
        self.clicks.append((int(x), int(y), why))
        for label, nxt in self.after.items():
            if why == f"leave game: {label}":
                self.screens, self.i = list(nxt), 0

    def paused(self):
        return False


@pytest.fixture
def mk(tmp_path, monkeypatch):
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    monkeypatch.setattr(Runner, "LEAVE_FLAG", tmp_path / "leave.flag")
    monkeypatch.setattr(R, "STOP", tmp_path / "STOP")
    saved = []
    monkeypatch.setattr(Runner, "_save_leave_frame", lambda self, img, texts, kind: saved.append(kind))

    def make(io):
        cfg = config.load()
        cfg["dry_run"] = False
        r = Runner(ReplayIO([]), cfg)
        r.io = io
        r.saved = saved
        return r
    return make


def labels(io):
    return [w for *_, w in io.clicks]


def test_mid_game_gear_then_yes_then_waits_for_the_front_page(mk):
    io = ScreensIO([BOARD], {"quit gear": [MID_DIALOG], "YES": [[], [], ["Loading"], HOME]})
    r = mk(io)
    assert r.leave_game()
    assert labels(io) == ["leave game: quit gear", "leave game: YES"]   # no gear presses while loading


def test_yes_pressed_at_most_twice_when_the_dialog_stays(mk):
    io = ScreensIO([MID_DIALOG])
    r = mk(io)
    r.LEAVE_SETTLE_S = 0
    assert not r.leave_game()
    assert labels(io) == ["leave game: YES", "leave game: YES"] and r.saved[-1] == "fail"


def test_quit_this_game_with_a_green_check(mk):
    io = ScreensIO([["Do you really want to quit this game?"]], {"green check": [HOME]})
    r = mk(io)
    assert r.leave_game()
    assert labels(io) == ["leave game: green check"]


def test_quit_game_menu_item_then_the_check(mk):
    # 2026-10-06: the confirm leaves QUIT GAME readable behind it; QUIT GAME was clicked 13x
    io = ScreensIO([["QUIT GAME"]], {"green check": [HOME]})
    r = mk(io)
    assert r.leave_game()
    assert labels(io) == ["leave game: QUIT GAME", "leave game: green check"]


def test_end_screens_continue_then_quit(mk):
    io = ScreensIO([["You've won!", "Experience", "Ranking", "CONTINUE"]],
                   {"CONTINUE": [["Ranking", "marlow.systems 18", "QUIT"]], "QUIT": [HOME]})
    r = mk(io)
    assert r.leave_game()
    assert labels(io) == ["leave game: CONTINUE", "leave game: QUIT"]


def test_matchmaking_words_are_never_clicked(mk):
    io = ScreensIO([["Quick match", "Ranked"], ["Quick match"], HOME])
    r = mk(io)
    assert r.leave_game()
    assert labels(io) == []


def test_a_board_that_never_opens_the_dialog_gets_two_gear_presses(mk):
    io = ScreensIO([BOARD])
    r = mk(io)
    assert not r.leave_game(tries=8)
    assert labels(io) == ["leave game: quit gear"] * 2 and r.saved[-1] == "fail"


def test_already_home_presses_nothing(mk):
    io = ScreensIO([HOME])
    r = mk(io)
    assert r.leave_game() and labels(io) == [] and r.saved == ["out_home"]


def test_failed_leave_waits_before_walking_out_again(mk, monkeypatch):
    io = ScreensIO([BOARD])
    r = mk(io)
    r._probe = lambda *a, **k: None
    r.LEAVE_FLAG.touch()
    assert r.leave_on_request() == "leaving"
    assert r.LEAVE_FLAG.exists() and r._leave_retry_at > R.time.time() + 20


def test_restart_leaves_first_and_watches_the_relaunch(mk, monkeypatch):
    import subprocess
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda a, **k: calls.append(a) or type("P", (), {"returncode": 1})())
    io = ScreensIO([BOARD], {"quit gear": [MID_DIALOG], "YES": [HOME]})
    r = mk(io)
    r.restart_app("restart requested")
    assert labels(io)[:2] == ["leave game: quit gear", "leave game: YES"]
    assert r._relaunch_watch[1] is True
    assert any("rungameid" in " ".join(c) for c in calls)
    # the relaunched app shows the front page first: clean
    assert not r._relaunch_rejoined(None, HOME, "home") and r._relaunch_watch is None


def test_a_relaunch_back_in_the_left_match_leaves_it_again(mk):
    r = mk(ScreensIO([BOARD]))
    r._relaunch_watch = (R.time.time(), True)
    assert not r._relaunch_rejoined(None, [], "unknown")          # still starting
    assert r._relaunch_rejoined(None, BOARD, "other_turn")
    assert r.LEAVE_FLAG.exists() and r.saved[-1] == "rejoined"


def test_cyrillic_ok_is_an_ok_button():
    # live 20:09: Commercial Harbor "No player has any commodity cards" + OCR "ОK" (Cyrillic O)
    from catanbot.vision.screens import find_text
    box = {"text": "ОK", "x": 620, "y": 460, "w": 30, "h": 18}
    assert find_text([box], r"^(ok|confirm|continue|close|done|accept)$") is box


FIX = __import__("pathlib").Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("name, kind, xy", [
    # live 19:47: bottom-left "QUIT GAME / Do you want to go back to the main menu?" + check at (198, 701)
    ("leave_main_menu_194746.jpg", "check", (198, 701)),
    # live 19:22: centre "Quit game / Do you really want to quit the game? / NO / YES"
    ("leave_yes_no_192315.jpg", "yes", None),
])
def test_live_quit_dialogs_are_read_from_the_frame(mk, name, kind, xy):
    import cv2
    from catanbot import mac
    img = cv2.imread(str(FIX / name))
    if img is None:
        pytest.skip("live frame not in this checkout")
    ocr = mac.ocr(img)
    texts = [o["text"] for o in ocr]
    r = mk(ScreensIO([BOARD]))
    got, at, label = r._leave_action(img, ocr, texts, None)
    assert got == kind, (texts, label)
    if xy:
        assert abs(at[0] - xy[0]) < 12 and abs(at[1] - xy[1]) < 12, at   # not the green Crane card


def test_main_menu_panel_is_confirmed_by_its_check_not_its_title(mk):
    io = ScreensIO([BOARD], {"quit gear": [["QUIT GAME", "Do you want to go back to the main menu?"]],
                             "green check": [["Did you know that ..."], HOME]})
    r = mk(io)
    assert r.leave_game()
    assert labels(io) == ["leave game: quit gear", "leave game: green check"]


def test_a_match_we_could_not_leave_is_played_on_after_the_relaunch(mk):
    r = mk(ScreensIO([BOARD]))
    r._relaunch_watch = (R.time.time(), False)
    assert not r._relaunch_rejoined(None, BOARD, "other_turn")
    assert not r.LEAVE_FLAG.exists()
