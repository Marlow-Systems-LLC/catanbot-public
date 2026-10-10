"""Banner card counts + dice wired into the runner (vision stubbed: real OCR is minutes)."""
import catanbot.vision.banner as B
from catanbot import config
from catanbot.board import canal_board, hex_corners
from catanbot.rules import MODES
from catanbot.state import GameState, PlayerState, new_hand
from tests.test_progress_wiring import FakeIO, _land
from catanbot.runner import Runner


def _gs():
    b = canal_board(seed=3)
    me = PlayerState("marlow.systems", is_me=True, hand=new_hand())
    a = PlayerState("Hildegard")
    g = GameState(board=b, mode=MODES["great_canal"], players=[me, a])
    h = _land(b, "mountains")[0]
    a.cities.add(hex_corners(*h.coord)[0])
    return g, h


def _runner():
    cfg = config.load(); cfg["dry_run"] = True
    return Runner(FakeIO([]), cfg)


def test_banner_counts_set_opponent_hands(monkeypatch):
    calls = []
    monkeypatch.setattr(B, "read_banner_counts",
                        lambda img, ocr=None: calls.append(1) or {"Hildegard": (7, 2), "marlow.systems": (3, 1)})
    r, (g, _) = _runner(), _gs()
    r._apply_banners(None, g)
    a = g.players[1]
    assert a.hand_count == 7 and a.progress_cards == 2
    assert g.me().hand_count != 3                      # our own hand comes from the hand bar
    r._apply_banners(None, g)
    assert len(calls) == 1                             # cached: OCR only after a roll or 45 s


def test_unread_count_keeps_last_value(monkeypatch):
    seq = iter([{"Hildegard": (5, 1)}, {"Hildegard": (None, None)}])
    monkeypatch.setattr(B, "read_banner_counts", lambda img, ocr=None: next(seq))
    r, (g, _) = _runner(), _gs()
    r._apply_banners(None, g)
    r._banner_stale = True
    r._apply_banners(None, g)
    assert g.players[1].hand_count == 5


def test_new_dice_feed_income_tracker(monkeypatch):
    r, (g, h) = _runner(), _gs()
    r.gs = g
    a, b = (h.number - 1, 1) if h.number > 2 else (1, 1)
    dice = iter([(a, b), (a, b), (3, 4)])
    monkeypatch.setattr(B, "read_dice", lambda img: next(dice))
    r._watch_dice(None)
    got = sum(r._seen_gain.get("Hildegard", {}).values())
    assert got >= 2                                    # city on the rolled mountains
    assert r._banner_stale
    r._watch_dice(None)                                # same dice still showing: not counted twice
    assert sum(r._seen_gain["Hildegard"].values()) == got
    r._watch_dice(None)                                # a 7 produces nothing
    assert sum(r._seen_gain["Hildegard"].values()) == got
