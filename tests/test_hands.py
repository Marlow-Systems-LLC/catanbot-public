"""Hand ledger (owner 2026-10-05): know what everyone holds; ask only for cards someone has."""
import pytest

from catanbot.board import hex_corners
from catanbot.hands import HOLD_MIN, HandLedger
from catanbot.state import Knight
from tests.test_canal_playbook import _gs, brain
from tests.test_progress_wiring import gs, runner  # noqa: F401 (fixture)


def _approx(a, b):
    return abs(a - b) < 1e-6


def _led(gs):
    led = HandLedger()
    led.me = gs.me().name
    gs.hands = led
    return led


def _corner(gs, terrain, i=0):
    h = next(h for h in gs.board.hexes.values() if h.terrain == terrain)
    return hex_corners(*h.coord)[i]


# ---- production ------------------------------------------------------------------------
def test_roll_adds_production_like_roll_yield():
    gs = _gs()
    weak = gs.players[2]
    weak.settlements.add(_corner(gs, "forest"))           # forest = 6
    led = _led(gs)
    out = led.on_roll(gs, 6, now=0)
    assert out["weak"] == {"lumber": 1}
    assert _approx(led.p("weak").known["lumber"], 1.0)
    assert led.rolls == 1


def test_seven_produces_nothing_but_counts_the_roll():
    gs = _gs()
    gs.players[2].settlements.add(_corner(gs, "forest"))
    led = _led(gs)
    assert led.on_roll(gs, 7, now=0) == {}
    assert led.rolls == 1 and led.p("weak").total() == 0


def test_robber_hex_and_barren_fields_pay_nothing():
    gs = _gs()
    weak = gs.players[2]
    weak.settlements.add(_corner(gs, "forest"))
    weak.settlements.add(_corner(gs, "fields", 3))         # barren canal farmland (4)
    gs.board.robber = next(h.coord for h in gs.board.hexes.values() if h.terrain == "forest")
    led = _led(gs)
    led.on_roll(gs, 6, now=0)
    led.on_roll(gs, 4, now=0)
    assert led.p("weak").total() == 0


def test_gold_knight_on_eight_is_an_unknown_card_leaning_grain():
    gs = _gs()
    weak = gs.players[2]
    weak.knights[_corner(gs, "gold", 1)] = Knight()
    led = _led(gs)
    led.on_roll(gs, 8, now=0)
    pl = led.p("weak")
    assert _approx(pl.unknown, 1.0)
    est = led.est("weak", gs)
    assert _approx(sum(est.values()), 1.0) and est["grain"] >= HOLD_MIN


# ---- builds ----------------------------------------------------------------------------
def test_pieces_before_first_roll_are_free():
    gs = _gs()
    weak = gs.players[2]
    led = _led(gs)
    led.p("weak").add({"brick": 1, "lumber": 1})
    weak.roads.add((_corner(gs, "hills", 0), _corner(gs, "hills", 1)))
    assert led.observe_pieces(gs, now=0) == []
    assert _approx(led.p("weak").total(), 2.0)


def test_new_road_settlement_and_knight_are_paid():
    gs = _gs()
    weak = gs.players[2]
    led = _led(gs)
    led.observe_pieces(gs, now=0)                       # baseline
    led.rolls = 1
    led.p("weak").add({"brick": 2, "lumber": 2, "wool": 2, "grain": 1, "ore": 1})
    weak.roads.add((_corner(gs, "hills", 0), _corner(gs, "hills", 1)))
    weak.settlements.add(_corner(gs, "hills", 0))
    assert sorted(k for _, k in led.observe_pieces(gs, now=1)) == ["road", "settlement"]
    pl = led.p("weak")
    assert pl.known["brick"] == 0 and pl.known["lumber"] == 0 and pl.known["grain"] == 0
    weak.knights[_corner(gs, "pasture", 2)] = Knight()
    assert led.observe_pieces(gs, now=2) == []          # first sight: wait for a repeat
    assert led.observe_pieces(gs, now=3.5) == [("weak", "knight")]
    assert pl.known["wool"] == 0 and pl.known["ore"] == 0


def test_knight_flicker_is_not_a_build():
    gs = _gs()
    weak = gs.players[2]
    led = _led(gs)
    led.observe_pieces(gs, now=0)
    led.rolls = 1
    led.p("weak").add({"wool": 1, "ore": 1})
    weak.knights[_corner(gs, "pasture", 2)] = Knight()
    led.observe_pieces(gs, now=1)
    weak.knights.clear()                                # gone again on the next frame
    led.observe_pieces(gs, now=2.5)
    assert _approx(led.p("weak").total(), 2.0)


def test_knight_activation_costs_grain():
    gs = _gs()
    weak = gs.players[2]
    v = _corner(gs, "pasture", 2)
    weak.knights[v] = Knight()
    led = _led(gs)
    led.observe_pieces(gs, now=0)
    led.rolls = 1
    led.p("weak").add({"grain": 1, "wool": 1})
    weak.knights[v] = Knight(active=True)
    led.observe_pieces(gs, now=1)
    assert led.observe_pieces(gs, now=2.5) == [("weak", "activate")]
    assert led.p("weak").known["grain"] == 0 and _approx(led.p("weak").total(), 1.0)


def test_pay_takes_shortfall_from_unknown_never_negative():
    led = HandLedger()
    pl = led.p("x")
    pl.add({"wool": 1})
    pl.unknown = 1.0
    led.pay("x", {"wool": 1, "ore": 1})
    assert pl.known["wool"] == 0 and pl.unknown == 0
    led.pay("x", {"brick": 2})
    assert pl.total() == 0 and min(pl.known.values()) >= 0


# ---- trades, offers, steals, monopoly ---------------------------------------------------
def test_seen_trade_moves_cards_between_partner_and_us():
    led = HandLedger()
    led.me = "bot"
    led.p("weak").add({"ore": 2})
    led.trade("weak", {"ore": 1}, "bot", {"brick": 1})
    pl = led.p("weak")
    assert _approx(pl.known["ore"], 1) and _approx(pl.known["brick"], 1)
    assert "bot" not in led.players                   # our hand is read exactly


def test_offer_raises_the_known_floor_out_of_unknown():
    led = HandLedger()
    pl = led.p("weak")
    pl.unknown = 3.0
    led.saw_holding("weak", {"grain": 1})
    assert _approx(pl.known["grain"], 1) and _approx(pl.unknown, 2) and _approx(pl.total(), 3)


def test_steal_moves_one_card_of_mass_not_certainty():
    led = HandLedger()
    led.p("weak").add({"ore": 3, "wool": 1})
    share = led.steal("leader", "weak")
    assert _approx(sum(share.values()), 1.0) and _approx(share["ore"], 0.75)
    assert _approx(led.p("weak").total(), 3.0) and _approx(led.p("leader").total(), 1.0)
    assert 0 < led.p("leader").known["ore"] < 1


def test_steal_from_us_uses_our_exact_hand():
    led = HandLedger()
    led.me = "bot"
    led.steal("leader", "bot", victim_est={"grain": 1, "brick": 3})
    assert _approx(led.holds("leader", "grain"), 0.25)
    assert "bot" not in led.players


def test_our_monopoly_takes_up_to_two_each():
    led = HandLedger()
    led.me = "bot"
    led.p("a").add({"ore": 3})
    led.p("b").add({"ore": 1})
    took = led.monopoly("bot", "ore", 2, ["bot", "a", "b"])
    assert _approx(took, 3) and _approx(led.p("a").known["ore"], 1) and led.p("b").known["ore"] == 0


def test_discard_removes_spread_over_the_hand():
    led = HandLedger()
    led.p("a").add({"ore": 4, "wool": 4})
    led.discard("a", 4)
    assert _approx(led.p("a").known["ore"], 2) and _approx(led.p("a").known["wool"], 2)


# ---- reconcile with the visible count ---------------------------------------------------
def test_reconcile_gap_up_is_unknown_gap_down_spread():
    led = HandLedger()
    pl = led.p("a")
    pl.add({"ore": 2})
    led.reconcile("a", 5, read_at=100)
    assert _approx(pl.unknown, 3) and _approx(pl.known["ore"], 2)
    led.reconcile("a", 1, read_at=100)
    assert _approx(pl.total(), 1) and min(pl.known.values()) >= 0 and pl.unknown >= 0


def test_reconcile_skips_a_count_read_before_the_last_event():
    led = HandLedger()
    led.gain("a", {"ore": 2}, now=50)
    led.reconcile("a", 0, read_at=49)                   # banner read before the roll
    led.reconcile("a", 0, read_at=51)                   # too soon after: cards may still fly
    assert _approx(led.p("a").total(), 2)
    led.reconcile("a", 0, read_at=60)
    assert _approx(led.p("a").total(), 0)


# ---- est_hand ---------------------------------------------------------------------------
def test_est_hand_uses_the_ledger_and_falls_back_without_it():
    gs = _gs()
    weak = gs.players[2]
    weak.hand_count = 4
    gs.seen_gain["weak"] = {"wool": 1}
    assert gs.est_hand(weak) == {"wool": 4.0}             # old behaviour: no ledger
    led = _led(gs)
    led.p("weak").add({"ore": 1})
    led.p("weak").unknown = 3.0
    est = gs.est_hand(weak)
    assert _approx(sum(est.values()), 4) and _approx(est["ore"], 4.0)   # unknown split like its gains


# ---- trade filter -----------------------------------------------------------------------
def _grain_short(gs):
    me = gs.me()
    C = hex_corners(0, 0)
    me.cities.add(C[0]); me.knights[C[2]] = Knight()
    me.hand.update({"wool": 2, "ore": 1, "brick": 1})


def test_no_grain_anywhere_means_no_grain_ask():
    gs = _gs(barbarian_pos=1)
    _grain_short(gs)
    led = _led(gs)
    for p in gs.opponents():
        led.p(p.name).add({"ore": 2, "wool": 1})
    t = brain().propose_trade(gs)
    assert t is None or t[1] != {"grain": 1}
    assert not brain().table_has(gs, "grain")


def test_grain_asked_only_from_players_who_hold_it():
    gs = _gs(barbarian_pos=1)
    _grain_short(gs)
    led = _led(gs)
    led.p("leader").add({"ore": 2})
    led.p("weak").add({"grain": 1, "ore": 1})
    give, get, partners = brain().propose_trade(gs)
    assert get == {"grain": 1} and partners == ["weak"]


def test_without_ledger_trades_are_unchanged():
    gs = _gs(barbarian_pos=1)
    _grain_short(gs)
    give, get, partners = brain().propose_trade(gs)
    assert get == {"grain": 1} and partners == ["weak"]   # leader is the threat, as before
    assert brain().table_has(gs, "grain")


def test_table_has_ignores_players_we_would_not_trade_with():
    gs = _gs()
    led = _led(gs)
    led.p("leader").add({"grain": 3})                   # 12 VP vs our 5: never a partner
    led.reconcile("weak", 0, read_at=10)                # weak: seen with an empty hand
    assert not brain().table_has(gs, "grain")
    led.p("weak").add({"grain": 1})
    assert brain().table_has(gs, "grain")


def test_runner_want_trade_skips_a_card_nobody_holds(gs, tmp_path):
    me = gs.me()
    me.hand.update(lumber=1, brick=1, wool=1, ore=2)      # settlement: 1 grain short
    r = runner(gs, [], tmp_path)
    r.brain._trade_targets = lambda g: ["settlement"]
    before = r._want_trade(gs)
    assert before is not None and before.get == {"grain": 1}   # without a ledger: asks the table
    r._offers_made = set()
    led = _led(gs)
    for p in gs.opponents():
        led.p(p.name).add({"ore": 3})
    t = r._want_trade(gs)
    assert t is None or "grain" not in t.get
    led.p("Nassir").add({"grain": 1})                     # 5 VP: a partner we'd take
    r._offers_made = set()
    assert r._want_trade(gs).get == {"grain": 1}


def test_a_player_we_know_nothing_about_is_still_asked():
    """Banner count never read and no event yet: don't drop the player from trades."""
    gs = _gs()
    led = _led(gs)
    led.p("weak")                                       # baseline entry only
    assert brain().table_has(gs, "grain")
    led.reconcile("weak", 0, read_at=10)                # now we know: empty hand
    assert not brain().table_has(gs, "grain")
