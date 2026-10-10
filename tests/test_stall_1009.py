"""Stage 2026-10-09, two solo games left after a 10-15 min stall each.

Game 212844 (22:00): Jean's Deserter on us rang our two knights (white ring round each piece, no
text). ring_spots found no blob ring, the turn marker over Jean made every frame other_turn and
nobody moved until the relaunch. Game 221251 (22:46): our Spy's "Select progress card" showed
Marianne's Bishop, whose text "Move the robber" read as place_robber for 10 min."""
from pathlib import Path

import cv2
import pytest

from catanbot import config
from catanbot.rules import MODES
from catanbot.runner import ReplayIO, Runner
from catanbot.state import GameState, Knight, PlayerState, new_hand
from catanbot.vision import screens
from catanbot.vision.board_reader import read_board
from catanbot.vision.screens import classify

FIX = Path(__file__).parent / "fixtures"
OUR_KNIGHTS = [(2, -2), (3, 1)]

# probe 224606_064_noring_place_robber: OCR of the Spy's card pick (Diplomat, Intrigue, Wedding, Bishop)
SPY_PICK = ["marlow.systems", "4", ".1:", "OLE", "668", "Lin", "Diplomat", "= 2", "You may remove anyiopent road (a",
            "road with nothing attached at one", "end). If you remove one of your", "roads, you may place it in another",
            "location.", "...", "BUILD COSTS", "Marianne", "Jean", "•", "•", "A E", "Select progress card", "Intrigue",
            "Wedding", "Bishop", "You may displace one of your", "opponent's knights. The knight",
            "must be on an intersection", "connected to one of your roads or", "shipping lines.",
            "Each player who has more", "victory points than you must", "esources and/or", "The sea",
            "es of his choice.", "Move the robber. You may draw 1", "random resource or commodity",
            "from each player who has a", "settlement or a city adjacent to the", "robber's new hex.",
            "Hide overlay", "Merchant Fleet"]


@pytest.fixture(autouse=True)
def _restore():
    yield
    screens.PICK_PROGRESS_FIRST = True


def test_spy_pick_showing_a_bishop_is_pick_progress():
    assert classify(SPY_PICK)[0] == "pick_progress"


def test_real_robber_prompts_still_read_as_place_robber():
    assert classify(["Select position for Robber"])[0] == "place_robber"
    assert classify(["Bishop", "Move the robber. You may draw 1", "Hide overlay"])[0] == "place_robber"


def test_pick_progress_first_switch_off_is_the_old_read():
    screens.PICK_PROGRESS_FIRST = False
    assert classify(SPY_PICK)[0] == "place_robber"


def test_runner_sets_the_switch_from_config():
    cfg = config.load(); cfg["pick_progress_first"] = False
    r = Runner(ReplayIO([]), cfg)
    r._brain_cfg()
    assert screens.PICK_PROGRESS_FIRST is False


def _runner(img, cfg_over=None):
    br = read_board(img)
    me = PlayerState("marlow.systems", is_me=True, hand=new_hand(), vp_visible=8)
    others = [PlayerState(n, vp_visible=v) for n, v in (("Lin", 6), ("Marianne", 5), ("Jean", 8))]
    for v in OUR_KNIGHTS:
        me.knights[v] = Knight(active=False)
    gs = GameState(board=br.board, mode=MODES["great_canal"], players=[me] + others)
    cfg = config.load(); cfg.update(cfg_over or {})
    r = Runner(ReplayIO([]), cfg)
    r.mode, r.gs, r.br = gs.mode, gs, br
    r.ensure_state = lambda img, texts: gs
    r._slot_names = {1: "Lin", 2: "Marianne", 3: "Jean"}
    return r


def test_deserter_rings_on_our_knights_are_found_per_corner():
    img = cv2.imread(str(FIX / "deserter_on_us_220040.png"))
    r = _runner(img)
    assert sorted(v for v, _ in r._our_knights_ringed(img)) == sorted(OUR_KNIGHTS)
    assert r._rings_only_on_our_knights(img, [])


def test_deserter_on_us_on_jeans_turn_stays_our_prompt():
    """220040: the frame read as Jean's turn; with our knights ringed it stays unknown, so the
    ring fallback (give up the cheapest knight) can fire."""
    img = cv2.imread(str(FIX / "deserter_on_us_220040.png"))
    r = _runner(img)
    assert r._other_turn_by_marker(img, {}, 3) == "unknown"


def test_no_rings_is_still_jeans_turn():
    img = cv2.imread(str(FIX / "deserter_card_215635.png"))     # the Deserter card shown, no rings yet
    r = _runner(img)
    assert r._our_knights_ringed(img) == []
    assert r._other_turn_by_marker(img, {}, 3) == "other_turn"


def test_deserter_rings_switch_off_is_the_old_read():
    img = cv2.imread(str(FIX / "deserter_on_us_220040.png"))
    r = _runner(img, {"deserter_rings_by_corner": False})
    assert r._our_knights_ringed(img) == []
    assert r._other_turn_by_marker(img, {}, 3) == "other_turn"
