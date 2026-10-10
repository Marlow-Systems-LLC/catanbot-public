"""Solo game 20261009_225719 (4th, 9 VP): ship-move loop, the city lost at the first attack with
a wool pair given away, and the feed line "X discards 1 progress card(s)" read as our discard."""
import copy

from catanbot import config
from catanbot.board import real_canal_board
from catanbot.brain import Action, Brain
from catanbot.rules import MODES
from catanbot.runner import Runner
from catanbot.state import GameState, Knight, PlayerState, new_hand
from catanbot.vision.screens import classify
from tests.test_progress_wiring import FakeIO
from tests.test_ship_move import _line


def _gs():
    b = real_canal_board(7)
    ps = [PlayerState("marlow.systems", is_me=True), PlayerState("b"), PlayerState("c")]
    for p in ps:
        p.hand = new_hand()
    return GameState(board=b, mode=copy.copy(MODES["great_canal"]), players=ps, vp_target=18)


def _runner(screens=(), **over):
    cfg = config.load()
    cfg["dry_run"] = True
    cfg.update(over)
    return Runner(FakeIO(list(screens)), cfg)


# ---- A. ship moves ------------------------------------------------------------------------

class _Lat:
    def edge_px(self, e):
        return (100, 100)


def _ship_runner(screens, **over):
    r = _runner(screens, **over)
    r.br = type("B", (), {"lattice": _Lat()})()
    r._close_leftover = lambda *a, **k: None
    r._probe = lambda *a, **k: None
    r.clicks = []
    r.click = lambda xy, what="": r.clicks.append(what)
    return r


PLACE = [{"text": "Select position for Ship", "x": 560, "y": 230, "w": 200, "h": 20}]
MAIN = [{"text": "BUILD COSTS", "x": 370, "y": 700, "w": 90, "h": 12}]


def test_tap_going_straight_to_placement_taps_the_destination(monkeypatch):
    import catanbot.runner as rn
    monkeypatch.setattr(rn.time, "sleep", lambda s: None)
    r = _ship_runner([PLACE, MAIN])
    a = Action("move_ship", (((0, 0), (1, 0)), ((1, 0), (2, 0))))
    assert r.tap_move_ship(a) is True
    assert not any("popup" in c for c in r.clicks)            # no popup in this flow
    assert any(c.startswith("move ship to") for c in r.clicks)
    assert getattr(r, "_ship_move_misses", 0) == 0


def test_two_misses_block_ship_moves_for_the_game(monkeypatch):
    import catanbot.runner as rn
    monkeypatch.setattr(rn.time, "sleep", lambda s: None)
    r = _ship_runner([PLACE, PLACE, PLACE, PLACE])
    assert r.brain.ship_moves is True
    for i in range(2):
        a = Action("move_ship", (((0, 0), (1, 0)), ((1, 0), (2, i))))
        assert r.tap_move_ship(a) is False
        assert "cancel ship move placement" in r.clicks
        r._turn_n = i + 1
    assert r._ship_move_misses == 2
    assert r.brain.ship_moves is False                        # out of the plan, no more spins
    gs = _gs()
    _line(gs)
    assert not any(a.kind == "move_ship" for a in r.brain.candidate_actions(gs))
    n = len(r.clicks)
    assert r.tap_move_ship(Action("move_ship", (((0, 0), (1, 0)), ((1, 0), (2, 9))))) is False
    assert len(r.clicks) == n                                 # blocked: no click at all
    r._brain_cfg()
    assert r.brain.ship_moves is False                        # a config re-read keeps the block
    r._reset_game_memory("lobby")
    assert r.brain.ship_moves is True                         # next game may move ships again


def test_max_misses_switch():
    assert config.DEFAULTS["ship_move_max_misses"] == 2
    r = _runner(ship_move_max_misses=0)
    r._ship_move_misses = 5
    assert r._ship_move_blocked() is False                    # 0 = never blocked
    r = _runner(ship_move_max_misses=3)
    r._ship_move_misses = 2
    assert r._ship_move_blocked() is False
    r._ship_move_misses = 3
    assert r._ship_move_blocked() is True


# ---- B. the 3:1 wool pair while the knight waits for grain -----------------------------------

def _danger_gs():
    """22:58:51: one city, one inactive knight, no grain, wool 2 at a 3:1, ship at 3/7."""
    gs = _gs()
    me = gs.me()
    vs = [v for v in sorted(gs.board.vertices) if gs.can_settle(v)]
    me.cities = {vs[0]}
    me.knights = {vs[-1]: Knight(active=False)}
    me.hand.update({"brick": 1, "lumber": 1, "wool": 2, "paper": 1, "coin": 1})
    gs.barbarian_pos = 3
    return gs


def _brain():
    br = Brain(setup_weights={})
    br.bank_ratio = lambda gs, r: 3
    return br


def test_grain_set_cards_keeps_the_pair():
    gs = _danger_gs()
    br = _brain()
    assert br.grain_set_cards(gs) == {"wool"}
    gs.me().hand["grain"] = 1
    assert br.grain_set_cards(gs) == set()                    # grain in hand: nothing to keep
    gs.me().hand["grain"] = 0
    next(iter(gs.me().knights.values())).active = True
    assert br.grain_set_cards(gs) == set()                    # the knight is active
    gs = _danger_gs()
    br.keep_grain_set = False
    assert br.grain_set_cards(gs) == set()                    # switch off


def test_player_offer_never_gives_a_pair_card():
    gs = _danger_gs()
    r = _runner()
    r.brain.bank_ratio = lambda g, x: 3
    assert r._grain_set_keep(gs) == {"wool"}
    # the 22:58:51 offer: 1 of the 2 wool for ore toward a 2nd knight
    r.brain._trade_targets = lambda g: ["knight", "city", "settlement"]
    r.brain.table_has = lambda g, x: True
    t = r._want_trade1(gs)
    assert t is None or not t.give.get("wool")
    r2 = _runner(grain_set_keep=False)
    assert r2.brain.keep_grain_set is False
    r2.brain.bank_ratio = lambda g, x: 3
    assert r2._grain_set_keep(gs) == set()


def test_save_city_offer_keeps_the_pair_for_a_knight_card():
    gs = _danger_gs()
    br = _brain()
    br._knight_need = lambda g: ("knight", {"wool": 1, "ore": 1})
    gs.me().knights = {}
    gs.me().hand.update({"lumber": 2, "wool": 1})
    br.grain_set_cards = lambda g: {"lumber"}
    br.city_in_danger = lambda g: True
    br.bank_trades_for = lambda g, c: None
    a = br.save_city_offer(gs)
    assert a is None or not a.give.get("lumber")


# ---- the feed line is not our discard -------------------------------------------------------

def test_feed_discards_line_is_not_our_discard():
    texts = ["marlow.systems", "Lin", "Marianne", "Jean", "BUILD COSTS", "Marianne discards 1 progress card(s)."]
    assert classify(texts)[0] != "discard"
    real = ["Choose 5 more resources to hand over to the robber", "Lin discards 4 card(s)."]
    assert classify(real)[0] == "discard"
