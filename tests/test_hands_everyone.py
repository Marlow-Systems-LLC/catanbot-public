"""Owner 2026-10-06: "make the bot always know what everyone holds at all times, and use that in
its advantage." Ledger events added on top of tests/test_hands.py, and the decisions that read them."""
import copy

from catanbot import progress_cards as pc
from catanbot.board import hex_corners
from catanbot.hands import HandLedger
from catanbot.state import Knight
from tests.test_canal_playbook import _gs, brain
from tests.test_hands import _approx, _corner, _led
from tests.test_progress_wiring import gs, runner  # noqa: F401 (fixture)


# ---- production the board read can't see ------------------------------------------------
def test_rise_after_a_roll_is_the_hidden_city_commodity():
    gs = _gs()
    weak = gs.players[2]
    weak.settlements.add(_corner(gs, "forest"))           # really a city: the read can't tell
    led = _led(gs)
    led.reconcile("weak", 0, read_at=0)
    led.on_roll(gs, 6, now=10)                              # +1 lumber seen
    led.reconcile("weak", 2, read_at=20)                    # the count says 2
    pl = led.p("weak")
    assert _approx(pl.known["lumber"], 1) and _approx(pl.known["paper"], 1) and pl.unknown == 0


def test_gold_field_pays_unknown_cards_outside_the_canal():
    gs = _gs()
    gs.mode = copy.copy(gs.mode)
    gs.mode.canal = False
    weak = gs.players[2]
    weak.settlements.add(_corner(gs, "gold"))               # gold = 8
    led = _led(gs)
    out = led.on_roll(gs, 8, now=0)
    assert out["weak"]["gold"] == 1 and _approx(led.p("weak").unknown, 1)


# ---- 7s ---------------------------------------------------------------------------------
def test_seven_makes_players_over_the_limit_discard_half():
    gs = _gs()
    led = _led(gs)
    led.p("weak").add({"ore": 6, "wool": 3})               # 9 > 7: throws 4
    led.p("leader").add({"ore": 7})                         # at the limit: keeps all
    assert led.discard_risk(gs) == {"weak": 4}
    led.on_roll(gs, 7, now=0)
    assert _approx(led.p("weak").total(), 5) and _approx(led.p("leader").total(), 7)
    gs.players[2].walls.add(_corner(gs, "hills"))           # a wall: limit 9
    assert led.discard_risk(gs) == {}


# ---- the visible counts -----------------------------------------------------------------
def test_one_up_others_down_is_a_steal_between_opponents():
    gs = _gs()
    led = _led(gs)
    led.reconcile_all({"leader": 0, "weak": 3}, {"leader": 10, "weak": 10})
    led.p("weak").known["ore"] = 3.0; led.p("weak").unknown = 0.0
    led.reconcile_all({"leader": 1, "weak": 2}, {"leader": 20, "weak": 20})
    assert _approx(led.p("leader").known["ore"], 1) and led.p("leader").unknown == 0
    assert _approx(led.p("weak").known["ore"], 2)


def test_count_drop_matching_a_city_pays_the_city():
    led = HandLedger()
    led.p("a").add({"ore": 3, "grain": 2, "wool": 1})
    led.reconcile("a", 1, read_at=100)                      # -5 = city (cities look like settlements)
    pl = led.p("a")
    assert _approx(pl.known["wool"], 1) and pl.known["ore"] == 0 and pl.known["grain"] == 0


def test_count_drop_with_no_clear_build_is_spread():
    led = HandLedger()
    led.p("a").add({"ore": 2, "wool": 2})
    led.reconcile("a", 1, read_at=100)                      # -3: no dev card (no grain) -> spread
    pl = led.p("a")
    assert _approx(pl.known["ore"], 0.5) and _approx(pl.known["wool"], 0.5)


def test_ck_drop_of_three_coins_is_an_improvement():
    led = HandLedger()
    led.ck = True
    led.p("a").add({"coin": 3, "ore": 1})
    led.reconcile("a", 1, read_at=100)
    assert led.p("a").known["coin"] == 0 and _approx(led.p("a").known["ore"], 1)


def test_total_always_ends_at_the_visible_count():
    led = HandLedger()
    led.p("a").add({"ore": 1.4, "wool": 0.3})
    for c in (4, 0, 7, 2):
        led.reconcile("a", c, read_at=100)
        assert _approx(led.p("a").total(), c)


def test_recount_wanted_after_an_event_settles():
    led = HandLedger()
    led.me = "bot"
    led.gain("a", {"ore": 1}, now=100)
    assert not led.needs_recount(99, now=101)               # cards may still fly
    assert led.needs_recount(99, now=104)
    assert not led.needs_recount(104, now=110)              # read since


# ---- cards we take: exact through our hand ----------------------------------------------
def test_our_steal_becomes_exact_when_our_hand_shows_the_card():
    led = HandLedger()
    led.me = "bot"
    led.p("weak").add({"ore": 1, "wool": 1})
    led.my_hand({"ore": 0, "wool": 0}, now=0)
    led.expect_take(["weak"], 1, now=1)
    assert _approx(led.p("weak").known["wool"], 0.5)       # spread until we see it
    assert led.my_hand({"ore": 0, "wool": 1}, now=2) == {"wool": 1}
    pl = led.p("weak")
    assert _approx(pl.known["wool"], 0) and _approx(pl.known["ore"], 1) and _approx(pl.total(), 1)


def test_expectation_expires_and_keeps_the_spread():
    led = HandLedger()
    led.me = "bot"
    led.p("weak").add({"ore": 2})
    led.my_hand({"ore": 0}, now=0)
    led.expect_take(["weak"], 1, now=1)
    assert led.my_hand({"ore": 0}, now=30) is None
    assert led.my_hand({"ore": 1}, now=31) is None          # too late: not attributed
    assert _approx(led.p("weak").known["ore"], 1)


def test_master_merchant_takes_two_exact_cards(gs):
    led = HandLedger()
    led.me = gs.me().name
    led.p("Hildegard").add({"paper": 1, "ore": 2, "wool": 1})
    led.my_hand(dict(gs.me().hand), now=0)
    led.on_my_card("master_merchant", {"player": "Hildegard"}, gs, now=1)
    after = dict(gs.me().hand); after["paper"] = after.get("paper", 0) + 1; after["ore"] = after.get("ore", 0) + 1
    led.my_hand(after, now=2)
    pl = led.p("Hildegard")
    assert pl.known["paper"] == 0 and _approx(pl.known["ore"], 1) and _approx(pl.known["wool"], 1)


def test_commercial_harbor_swaps_our_resource_for_their_commodity(gs):
    led = HandLedger()
    led.me = gs.me().name
    gs.players[2].hand_count = 0                             # Nassir: nothing to swap
    led.p("Hildegard").add({"cloth": 1, "ore": 1})
    led.my_hand({"wool": 1, "cloth": 0}, now=0)
    led.on_my_card("commercial_harbor", {"give": ["wool"]}, gs, now=1)
    led.my_hand({"wool": 0, "cloth": 1}, now=2)
    pl = led.p("Hildegard")
    assert pl.known["cloth"] == 0 and _approx(pl.known["wool"], 1) and _approx(pl.known["ore"], 1)


# ---- monopoly ---------------------------------------------------------------------------
def test_monopoly_take_counts_the_unknown_pool():
    led = HandLedger()
    pl = led.p("a")
    pl.gained["ore"] = 1.0                                  # everything seen coming in was ore
    pl.unknown = 3.0
    assert _approx(led.expected_take("a", "ore", 2), 2)
    pl.gained["wool"] = 1.0                                 # now half ore: E[min(2, Bin(3, .5))]
    assert _approx(led.expected_take("a", "ore", 2), 0 * 1 / 8 + 1 * 3 / 8 + 2 * 4 / 8)


def test_monopoly_leaves_none_and_the_count_corrects_it():
    led = HandLedger()
    led.me = "bot"
    led.p("a").add({"ore": 1, "wool": 2})
    led.p("a").unknown = 1.0
    led.monopoly("bot", "ore", 2, ["bot", "a"], now=0)
    pl = led.p("a")
    assert pl.known["ore"] == 0 and "ore" in pl.excl        # unlikely to hold a 3rd
    assert led.monopoly("bot", "ore", 2, ["bot", "a"], now=5) == 0   # same play reported twice
    led.reconcile("a", 3, read_at=100)                      # it gave only 1
    assert _approx(pl.total(), 3) and pl.known["ore"] == 0 and led.chance("a", "ore") == 0


def test_monopoly_drop_bigger_than_expected_is_the_named_kind():
    led = HandLedger()
    led.me = "bot"
    led.p("a").add({"wool": 2})
    led.p("a").gained["ore"] = 2.0
    led.p("a").unknown = 2.0
    took = led.monopoly("bot", "ore", 2, ["bot", "a"], now=0)
    assert 0 < took < 2
    led.reconcile("a", 2, read_at=100)                      # it gave 2 ore
    assert _approx(led.p("a").known["wool"], 2) and _approx(led.p("a").total(), 2)


def test_resource_monopoly_names_what_the_ledger_sees():
    gs = _gs()
    for p in gs.opponents():
        p.hand_count = 4
    gs.seen_gain = {"leader": {"wool": 4}, "weak": {"wool": 4}}
    without = pc.choose_resource_monopoly(gs, gs.me())
    assert without == {"resource": "wool"}                  # old guess: what they were seen getting
    led = _led(gs)
    for p in gs.opponents():
        led.p(p.name).add({"ore": 3, "wool": 1})           # wool traded away / spent, ore came in
    assert pc.choose_resource_monopoly(gs, gs.me()) == {"resource": "ore"}


# ---- steal ------------------------------------------------------------------------------
def test_steal_victim_is_the_hand_with_what_we_need():
    gs = _gs()
    gs.players[1].vp_visible = gs.players[2].vp_visible = 8  # same win chance, not protected
    led = _led(gs)
    led.p("leader").add({"wool": 3})
    led.p("weak").add({"brick": 3})                         # the plan misses brick
    assert brain().choose_victim(gs, ["leader", "weak"]) == "weak"
    led.p("weak").known["brick"] = 0.0
    led.reconcile("weak", 0, read_at=100)                   # empty hand now
    assert brain().choose_victim(gs, ["leader", "weak"]) == "leader"


def test_robber_victim_on_a_shared_hex_follows_the_ledger():
    gs = _gs()
    gs.mode = copy.copy(gs.mode)
    gs.mode.friendly_robber = False
    leader, weak = gs.players[1], gs.players[2]
    h = next(x for x in gs.board.hexes.values() if x.terrain == "mountains")
    C = hex_corners(*h.coord)
    leader.settlements.add(C[0]); weak.settlements.add(C[3])
    leader.hand_count = weak.hand_count = 3
    leader.vp_visible = weak.vp_visible = 8
    assert brain().choose_robber(gs, [h.coord])[1] is leader   # without a ledger: the likely winner
    led = _led(gs)
    led.reconcile("leader", 0, read_at=100)                 # the leader just spent it all (stale banner)
    led.p("weak").add({"brick": 3})
    hexc, victim = brain().choose_robber(gs, [h.coord])
    assert hexc == h.coord and victim is weak


# ---- trades -----------------------------------------------------------------------------
def test_trade_ask_skips_a_player_our_monopoly_emptied():
    gs = _gs()
    gs.players[2].vp_visible = 5
    led = _led(gs)
    led.p("weak").add({"grain": 1, "ore": 1})
    assert brain().table_has(gs, "grain")
    led.monopoly(gs.me().name, "grain", 2, [p.name for p in gs.players], now=0)
    assert not brain().table_has(gs, "grain")


def test_never_feed_a_rival_the_last_card_of_a_city():
    gs = _gs()
    weak = gs.players[2]
    weak.vp_visible = 6                                     # ahead of us (5), under 1.5x
    me = gs.me()
    me.hand.update({"ore": 3, "wool": 3})
    br = brain()
    offer = ({"ore": 1}, {"brick": 1})                      # we give ore, get brick
    assert br.evaluate_offer(gs, "weak", *offer)           # a fine trade without the ledger
    led = _led(gs)
    led.p("weak").add({"ore": 2, "grain": 2, "brick": 1})
    assert led.feeds_build("weak", {"ore": 1}, gs=gs) == "city"
    assert not br.evaluate_offer(gs, "weak", *offer)
    led.p("weak").known["ore"] = 0.0                        # two ore short: one card is no city
    led.p("weak").known["wool"] = 2.0
    assert led.feeds_build("weak", {"ore": 1}, gs=gs) is None
    weak.vp_visible = 4                                     # behind us: feeding it is fine
    led.p("weak").known["ore"] = 2.0
    assert led.feeds_build("weak", {"ore": 1}, gs=gs) == "city"   # (the rule only blocks rivals)


def test_master_merchant_target_holds_the_best_cards():
    gs = _gs()
    me = gs.me()
    a, b = gs.players[1], gs.players[2]
    a.vp_visible, b.vp_visible = 13, 12
    a.hand_count = b.hand_count = 4
    assert pc.choose_master_merchant(gs, me) == {"player": "leader"}   # without a ledger
    led = _led(gs)
    led.p("leader").add({"wool": 4})
    led.p("weak").add({"paper": 2, "wool": 2})
    assert pc.choose_master_merchant(gs, me) == {"player": "weak"}


# ---- runner wiring ----------------------------------------------------------------------
def test_runner_reconciles_all_counts_and_sets_hand_count(gs, tmp_path):
    from catanbot.runner import _norm
    r = runner(gs, [], tmp_path)
    led = r._ledger()
    led.me = gs.me().name
    r._banner_counts = {_norm("Hildegard"): (5, None), _norm("Nassir"): (2, None)}
    r._banner_read_at = {_norm("Hildegard"): 1000.0, _norm("Nassir"): 1000.0}
    r._ledger_observe(gs)
    assert led.count("Hildegard") == 5 and gs.players[1].hand_count == 5
    led.gain("Nassir", {"ore": 2}, now=2000)                # seen production, banner not read since
    r._ledger_observe(gs)
    assert gs.players[2].hand_count == 4                    # the ledger, not the stale banner
