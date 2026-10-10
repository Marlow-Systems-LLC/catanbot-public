"""Progress cards: data completeness and play heuristics (fast, no simulation)."""
import pytest

from catanbot import progress_cards as pc
from catanbot.board import canal_board, hex_corners
from catanbot.rules import MODES
from catanbot.state import GameState, Knight, PlayerState, new_hand


def _land(b, terrain=None, number=True):
    return [h for h in b.hexes.values() if h.is_land and h.terrain != "canal"
            and (terrain is None or h.terrain == terrain) and (h.number or not number)]


@pytest.fixture
def gs():
    b = canal_board(seed=3)
    me = PlayerState("marlow", is_me=True, hand=new_hand(), vp_visible=6)
    a = PlayerState("Hildegard", vp_visible=10, hand_count=6)
    c = PlayerState("Nassir", vp_visible=5, hand_count=2)
    g = GameState(board=b, mode=MODES["great_canal"], players=[me, a, c])
    land = _land(b)
    # me: city on a mountain and settlement on fields; leader: city elsewhere
    mtn = _land(b, "mountains")[0]
    fld = _land(b, "fields")[0]
    me.cities.add(hex_corners(*mtn.coord)[0])
    me.settlements.add(hex_corners(*fld.coord)[3])
    other = next(h for h in land if h.coord not in (mtn.coord, fld.coord)
                 and not set(hex_corners(*h.coord)) & (me.cities | me.settlements))
    a.cities.add(hex_corners(*other.coord)[2])
    return g


def test_deck_counts():
    by = {}
    for c in pc.CARDS.values():
        by[c["deck"]] = by.get(c["deck"], 0) + c["count"]
    assert by == {"science": 18, "trade": 18, "politics": 18}


def test_aliases():
    assert pc.card_key("Printing Press") == "printer"
    assert pc.card_key("Merchant Fleet") == "merchant_fleet"
    assert pc.card_key("Field Marshall") == "warlord"
    with pytest.raises(KeyError):
        pc.card_key("Monopoly of Sheep")


def test_vp_cards_never_played(gs):
    """Owner 2026-10-06: VP cards are revealed when drawn, never played (no drag)."""
    me = gs.me()
    for phase in ("pre_roll", "main", "end_turn", "on_seven", "opp_turn"):
        why = {}
        assert pc.choose_play(gs, me, ["Printer", "Constitution"], phase, why) == [], phase
        assert "never played" in why["printer"], phase


def test_not_our_turn_plays_nothing(gs):
    assert pc.choose_play(gs, gs.me(), ["Alchemist", "Wedding", "Warlord"], "on_seven") == []


def test_alchemist_pre_roll_only(gs):
    me = gs.me()
    pre = pc.choose_play(gs, me, ["Alchemist"], "pre_roll")
    assert pre and pre[0][0] == "alchemist"
    w, r = pre[0][1]["dice"]
    assert 1 <= w <= 6 and 1 <= r <= 6 and w + r != 7
    assert w + r in {h.number for h in pc._my_hexes(gs, me)}
    assert not any(c == "alchemist" for c, _ in pc.choose_play(gs, me, ["Alchemist"], "main"))


def test_wedding_and_master_merchant_target_ahead(gs):
    me = gs.me()
    plays = dict(pc.choose_play(gs, me, ["Wedding", "Master Merchant"], "main"))
    assert plays["wedding"]["from"] == ["Hildegard"]
    assert plays["master_merchant"]["player"] == "Hildegard"
    me.vp_visible = 12
    assert pc.choose_play(gs, me, ["Wedding", "Master Merchant"], "main") == []


def test_bishop_hits_leader_not_me(gs):
    me = gs.me()
    gs.barbarians_attacked = True                  # owner rule: robber locked before the 1st attack
    args = dict(pc.choose_play(gs, me, ["Bishop"], "main"))["bishop"]
    owners = {gs.owner_at(v) for v in hex_corners(*args["hex"])} - {None}
    assert me not in owners and gs.players[1] in owners


def test_defence_first_when_barbarians_near(gs):
    me = gs.me()
    v = next(iter(me.cities))
    me.knights[v] = Knight(1, False)
    gs.barbarian_pos = 6
    order = [c for c, _ in pc.choose_play(gs, me, ["Merchant Fleet", "Wedding", "Warlord"], "main")]
    assert order[0] == "warlord"


def test_spend_before_robber(gs):
    me = gs.me()
    me.hand.update({"wool": 9})
    plays = dict(pc.choose_play(gs, me, ["Merchant Fleet"], "main"))
    assert plays["merchant_fleet"] == {"resource": "wool"}
    # end of turn over the limit: don't take Mining cards that would sit through a 7
    assert "mining" not in dict(pc.choose_play(gs, me, ["Mining"], "end_turn"))
    me.hand["wool"] = 0
    assert dict(pc.choose_play(gs, me, ["Mining"], "main"))["mining"] == {"gain": {"ore": 2}}


def test_irrigation_counts_barren_fields(gs):
    me = gs.me()
    for h in pc._hexes_of(gs, me, "fields"):
        h.barren = True
    assert dict(pc.choose_play(gs, me, ["Irrigation"], "main"))["irrigation"]["gain"]["grain"] >= 2


def test_crane_prefers_paper(gs):
    me = gs.me()
    me.improvements.update({"science": 2, "politics": 2, "trade": 2})
    me.hand.update({"paper": 2, "coin": 2, "cloth": 2})
    assert dict(pc.choose_play(gs, me, ["Crane"], "main"))["crane"] == {"track": "science"}


def test_engineer_and_medicine(gs):
    me = gs.me()
    me.hand.update({"ore": 2, "grain": 1})
    plays = dict(pc.choose_play(gs, me, ["Engineer", "Medicine"], "main"))
    assert plays["engineer"]["city"] in me.cities
    assert plays["medicine"]["settlement"] in me.settlements


def test_smith_respects_level_cap(gs):
    me = gs.me()
    v = next(iter(me.cities))
    me.knights[v] = Knight(2, True)
    assert "smith" not in dict(pc.choose_play(gs, me, ["Smith"], "main"))
    me.improvements["politics"] = 3
    assert dict(pc.choose_play(gs, me, ["Smith"], "main"))["smith"]["knights"] == [v]


def test_inventor_helps_me(gs):
    me = gs.me()
    before = sum(gs.income(me).values())
    args = pc.choose_inventor(gs, me)
    assert args is not None
    if args:
        a, b = (gs.board.hexes[c] for c in args["swap"])
        assert a.number not in (2, 6, 8, 12) and b.number not in (2, 6, 8, 12)
        a.number, b.number = b.number, a.number
        assert sum(gs.income(me).values()) > before


def test_inventor_takes_the_leaders_number(gs):
    # owner 2026-10-06: take the leader's best chip for ourselves, never just shuffle our own
    me, lead = gs.players[0], gs.players[1]
    b = gs.board
    mine = [h for h in b.hexes.values() if any(v in me.cities | me.settlements for v in hex_corners(*h.coord))]
    theirs = [h for h in b.hexes.values() if any(v in lead.cities for v in hex_corners(*h.coord))]
    for h in mine + theirs:
        h.number = 6 if h.terrain == "gold" else h.number
    ours = next(h for h in mine if pc.inventor_movable(gs, h) and h not in theirs)
    best = next(h for h in theirs if pc.inventor_movable(gs, h) and h not in mine)
    for h in b.hexes.values():                  # every other movable chip: a dead 3/11
        if h.number in (3, 4, 5, 9, 10, 11) and h not in mine + theirs:
            h.number = 11
    for h in mine + theirs:
        if pc.inventor_movable(gs, h):
            h.number = 3
    ours.number, best.number = 3, 5
    lead_before = sum(gs.income(lead).values())
    me_before = sum(gs.income(me).values())
    args = pc.choose_inventor(gs, me)
    assert args and set(args["swap"]) == {ours.coord, best.coord}
    ours.number, best.number = best.number, ours.number
    assert sum(gs.income(me).values()) > me_before and sum(gs.income(lead).values()) < lead_before


def test_progress_hand_limit_at_end_turn(gs):
    me = gs.me()
    gs.players[1].vp_visible = 0
    gs.players[2].vp_visible = 0
    out = pc.choose_play(gs, me, ["Wedding", "Master Merchant", "Diplomat", "Intrigue", "Spy"], "end_turn")
    assert out and out[-1][0] in ("discard",) + tuple(pc.CARDS)


def test_dead_cards_are_discarded_first():
    """Owner 2026-10-04: Medicine with no grain / Smith with nothing to promote go before live cards."""
    from catanbot import progress_cards as pc
    from catanbot.state import GameState, PlayerState
    from catanbot.board import Board
    import copy
    from catanbot.rules import MODES
    me = PlayerState("bot", is_me=True)
    gs = GameState(board=Board(), mode=copy.copy(MODES["great_canal"]), players=[me, PlayerState("x")])
    assert pc.keep_value(gs, me, "smith") == pc.KEEP_VALUE.get("smith", 1) * 0.5
    assert pc.keep_value(None, None, "smith") == pc.KEEP_VALUE.get("smith", 1)
    assert pc.keep_value(gs, me, "warlord") == pc.KEEP_VALUE.get("warlord", 1)


# ---- owner 2026-10-06: never hold a card it can use; Spy picks by real value ---------------
def test_alchemist_plays_when_any_roll_pays_us(gs):
    """Old rule held it unless the roll beat 2 cards; a chosen non-7 roll that pays us more than
    the leader beats a random roll."""
    me = gs.me()
    a = pc.choose_alchemist(gs, me)
    assert a is not None and a["sum"] != 7
    me.cities.clear()
    me.settlements.clear()                             # nothing produces for us: held, with reason
    why = {}
    assert pc.choose_play(gs, me, ["Alchemist"], "pre_roll", why) == []
    assert why["alchemist"] == pc.NO_USE["alchemist"]


def test_hold_reasons_logged(gs):
    me = gs.me()
    why = {}
    out = pc.choose_play(gs, me, ["Medicine", "Alchemist", "Mining"], "main", why)
    assert [c for c, _ in out] == ["mining"]
    assert why["medicine"].startswith("can't play — ")       # can't pay 2 ore + 1 grain
    assert "before the roll" in why["alchemist"] and "mining" not in why


def test_diplomat_takes_any_open_road(gs):
    """Old rule played it only on the leader with Longest Road or a road touching ours, so it
    sat in hand all game."""
    me, lead, other = gs.players
    assert pc.choose_diplomat(gs, me) is None
    taken = {v for p in gs.players for v, _ in p.buildings()}
    v0 = next(v for v in gs.board.vertices if gs.can_settle(v) and v not in taken)
    v1 = min(gs.board.neighbors(v0))
    e = next(x for x in gs.board.vertex_edges(v0) if v1 in x)
    other.settlements.add(v0)
    other.roads.add(e)
    assert e in pc._open_roads(gs, other)
    assert pc.choose_diplomat(gs, me) == {"edge": e}


def test_saboteur_on_two_cards(gs):
    me, a, c = gs.players
    a.hand_count, c.hand_count = 4, 1
    assert pc.choose_saboteur(gs, me) == {}            # the leader loses 2
    a.hand_count = 3
    assert pc.choose_saboteur(gs, me) is None          # only 1 card lost


def test_spy_pick_playable_now_beats_later(gs):
    me = gs.me()
    pick, scores = pc.spy_pick(gs, me, ["alchemist", "mining", "printer"], "Nassir")
    assert pick == "mining" and "printer" not in scores
    assert scores["mining"] > scores["alchemist"]
    # before the roll nothing else is playable: the Alchemist (best kept) wins
    assert pc.spy_pick(gs, me, ["alchemist", "mining"], "Nassir", phase="pre_roll")[0] == "alchemist"


def test_spy_pick_bishop_waits_for_first_attack(gs):
    me = gs.me()
    assert pc.playable_now(gs, me, "bishop", "main", attacked=True)
    assert not pc.playable_now(gs, me, "bishop", "main", attacked=False)
    assert pc.spy_pick(gs, me, ["bishop", "smith"], attacked=False)[1]["bishop"] < pc.PLAY_NOW_BONUS


def test_spy_pick_leader_denial(gs):
    me = gs.me()
    _, from_lead = pc.spy_pick(gs, me, ["warlord", "diplomat"], "Hildegard")
    _, from_last = pc.spy_pick(gs, me, ["warlord", "diplomat"], "Nassir")
    assert from_lead["warlord"] - from_last["warlord"] == 0.5 * pc.KEEP_VALUE["warlord"]



def test_diplomat_moves_our_own_road_when_no_rival_road_is_open(gs):
    """Owner 2026-10-06 "never hold cards it can use": with no rival open road, the Diplomat
    moves one of OUR open roads where it lengthens the road or opens a settlement corner."""
    me = gs.players[0]
    c = next(iter(me.cities))
    v1 = min(v for v in gs.board.neighbors(c) if gs.owner_at(v) is None)
    v2 = min(v for v in gs.board.neighbors(v1) if v != c and gs.owner_at(v) is None)
    e1 = next(x for x in gs.board.vertex_edges(c) if v1 in x)
    e2 = next(x for x in gs.board.vertex_edges(v1) if v2 in x)
    me.roads |= {e1}
    assert pc.choose_diplomat(gs, me) is None          # one road: moving it gains nothing
    me.roads |= {e2}
    s = next(iter(me.settlements))                     # a stray road off the settlement
    e3 = min(gs.board.vertex_edges(s))
    me.roads.add(e3)
    spots0, len0 = len(gs.legal_settlement_spots(me)), gs.longest_road_len(me)
    a = pc.choose_diplomat(gs, me)
    assert a is not None and a["own"]
    old, new = a["edges"]
    assert old in pc._open_roads(gs, me)
    me.roads.discard(old)
    assert new in gs.legal_road_edges(me)
    me.roads.add(new)
    assert len(gs.legal_settlement_spots(me)) > spots0 or gs.longest_road_len(me) > len0


def test_alchemist_chooses_seven_only_to_clear_our_hex_and_never_over_the_limit(gs):
    """Owner 2026-10-06: the Alchemist may pick 7 when the robber blocks our hex; a producing
    roll usually pays more; never a 7 with too many cards."""
    me = gs.players[0]
    c = next(iter(me.cities))
    h = max((x for x in gs.board.vertex_hexes(c) if x.number), key=lambda x: pc.PIPS[x.number])
    gs.board.robber = h.coord
    gs.barbarians_attacked = False
    assert pc.choose_alchemist(gs, me) is None or pc.choose_alchemist(gs, me)["sum"] != 7   # robber locked
    gs.barbarians_attacked = True
    s7 = pc._alchemist_seven(gs, me)
    assert s7 is not None and s7 > 1.0
    me.hand = {r: 0 for r in me.hand}
    a = pc.choose_alchemist(gs, me)
    best_other = max(pc._value(gs, pc._roll_gain(gs, me, t)) for t in range(2, 13) if t != 7)
    assert (a["sum"] == 7) == (s7 > best_other)            # the 7 only when it beats every roll
    me.cities.add(next(v for v in hex_corners(*h.coord) if v != c and not gs.occupied(v)
                       and all(not gs.occupied(n) for n in gs.board.neighbors(v))))  # 2nd city on it
    assert pc.choose_alchemist(gs, me)["sum"] == 7          # a big blocked hex: clear it
    me.hand = {"ore": 10}                                  # over the limit: never a 7
    a = pc.choose_alchemist(gs, me)
    assert a is None or a["sum"] != 7
    gs.board.robber = None
    assert pc._alchemist_seven(gs, me) is None
