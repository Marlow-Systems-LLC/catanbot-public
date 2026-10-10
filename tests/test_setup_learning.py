"""Setup-weight learning loop: named weights, feature rows, fit helpers (scripts/fit_setup.py)."""
import json
import random
from pathlib import Path

import pytest

from catanbot import brain as brain_mod
from catanbot.board import PIPS, ekey
from catanbot.brain import SETUP_DEFAULTS, Brain, load_setup_weights
from catanbot.sim import Sim, setup_features, write_setup_rows
from catanbot.state import Knight, PlayerState
from scripts import fit_setup

GOLDEN = json.loads((Path(__file__).parent / "data" / "setup_golden.json").read_text())["games"]


def _placements(sim):
    return [[sorted(map(list, p.settlements)), sorted(map(list, p.cities)),
             sorted(map(lambda e: [list(x) for x in e], p.roads | p.ships))] for p in sim.gs.players]


def _old(w=None):
    """The per-spot scorer these goldens and terms were written for (Brain.setup_set off)."""
    b = Brain(setup_weights={} if w is None else w)
    b.setup_set = False
    b.canal_road_front = 0.0        # the goldens predate the 2026-10-09 canal-front road rule
    if w is None:   # the goldens predate the owner's 2026-10-09 hex-count rule
        b.setup_w.update({f"hex_count_{n}": 0.0 for n in range(4)})
    return b


@pytest.mark.parametrize("key", sorted(GOLDEN))
def test_default_weights_reproduce_pre_refactor_choices(key):
    mode, seed, n = key.split("/")
    sim = Sim(mode, int(n), seed=int(seed), brains=[_old() for _ in range(int(n))])
    sim.setup()
    assert _placements(sim) == GOLDEN[key]


def test_no_config_key_means_defaults(monkeypatch):
    monkeypatch.setattr("catanbot.config.load", lambda: {"bot_name": "x"})
    assert load_setup_weights() == SETUP_DEFAULTS
    b = Brain()
    assert b.setup_w == SETUP_DEFAULTS and b.third_harbour == 0.0 and b.first_harbour == 0.0


def test_config_overrides_known_keys_only(monkeypatch):
    monkeypatch.setattr("catanbot.config.load",
                        lambda: {"setup_weights": {"dup": 0.3, "third_harbour": 5, "bogus": 9}})
    w = load_setup_weights()
    assert w["dup"] == 0.3 and w["third_harbour"] == 5.0 and "bogus" not in w
    assert Brain().third_harbour == 5.0
    assert Brain(setup_weights={}).third_harbour == 0.0       # explicit dict: config not read


def test_owner_terms_off_by_default():
    sim = Sim("great_canal", 4, seed=3)
    gs = sim.as_me(0)
    b = Brain(setup_weights={})
    assert all(b._owner_terms(gs, v, "settlement", gs.me()) == 0.0
               for v in gs.legal_settlement_spots(gs.me(), setup=True))


def test_setup_features_match_the_board():
    sim = Sim("great_canal", 4, seed=11, record_setup=True)
    sim.setup()
    b = sim.gs.board
    for i, p in enumerate(sim.gs.players):
        row = sim.setup_feats[i]
        f = row["features"]
        assert [k for k, *_ in row["pieces"]] == ["settlement", "city", "settlement"]
        assert f["seat"] == i and f["n_players"] == 4
        raw = sum(PIPS[h.number] for v, _ in p.buildings() for h in b.vertex_hexes(v) if h.number)
        assert f["pips_raw"] == raw
        assert f["harb_3to1"] == any(b.harbor_at(v) == "3:1" for v, _ in p.buildings())
        assert f["start_cards"] == {r: c for r, c in p.hand.items() if c}
        assert f["start_total"] == sum(p.hand.values())
        last = tuple(row["pieces"][-1][1])
        assert f["last_coastal"] and b.is_coastal_vertex(last)
        assert f["grain_hexes_last"] == sum(1 for h in b.vertex_hexes(last) if h.terrain == "fields" and h.number)
        assert not f["on_gold_corner"]
        assert f["distinct"] == sum(1 for r in ("brick", "lumber", "wool", "grain", "ore")
                                    if f["pips_by_res"].get(r, 0) > 0)


def test_setup_rows_join_outcome(tmp_path):
    sim = Sim("great_canal", 4, seed=5, record_setup=True)
    winner, _, _ = sim.run(max_turns=500)
    rows = sim.setup_rows({"phase": "t"})
    assert len(rows) == 4 and all(r["phase"] == "t" for r in rows)
    assert sum(r["outcome"]["rank"] for r in rows) == pytest.approx(10.0)       # 1+2+3+4, ties averaged
    assert all(0.0 <= r["outcome"]["rank_norm"] <= 1.0 for r in rows)
    assert [r["outcome"]["won"] for r in rows] == [p.name == winner for p in sim.gs.players]
    best = max(rows, key=lambda r: r["outcome"]["final_vp"])
    assert best["outcome"]["rank"] <= 1.5
    path = write_setup_rows(rows, tmp_path / "x.jsonl")
    back = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(back) == 4 and back[0]["features"]["seat"] == 0


def test_gold_route_len():
    sim = Sim("great_canal", 4, seed=2)
    gs = sim.as_me(0)
    b = gs.board
    br = Brain(setup_weights={})
    gold = br._gold_corners(gs)
    assert gold
    g = next(iter(gold))
    assert br.gold_route_len(gs, g) == 0
    one = [w for w in b.neighbors(g) if w not in gold and b.is_coastal_vertex(w) and b.is_sea_edge(ekey(w, g))]
    if one:
        assert br.gold_route_len(gs, one[0]) == 1
    for v in gold:                       # every gold corner taken by knights -> no free route
        gs.players[1].knights[v] = Knight(1, False)
    assert all(br.gold_route_len(gs, v) is None for v in b.vertices)


def test_last_free_3to1():
    sim = Sim("great_canal", 4, seed=7)
    gs = sim.as_me(0)
    b = gs.board
    br = Brain(setup_weights={})
    me, opp = gs.players[0], gs.players[1]
    spots = [v for v in gs.legal_settlement_spots(me, setup=True) if b.harbor_at(v) == "3:1"]
    assert len(spots) >= 2
    keep = spots[0]
    for v in spots[1:]:                  # opponents sit on every other 3:1 corner
        if v not in b.neighbors(keep):
            opp.settlements.add(v)
    left = [v for v in gs.legal_settlement_spots(me, setup=True) if b.harbor_at(v) == "3:1"]
    if set(left) - ({keep} | b.neighbors(keep)):
        pytest.skip("board leaves a second 3:1 next to the kept one")
    assert br.last_free_3to1(gs, keep, me)
    for p in gs.players[1:]:             # nobody left to place -> no denial value
        p.settlements |= {("x", i) for i in range(5)}
    assert not br.last_free_3to1(gs, keep, me)


def test_extreme_weights_keep_owner_rules():
    """Whatever the fit draws, the rules hold: no building on a gold corner, 3rd settlement coastal."""
    base = dict(SETUP_DEFAULTS)
    for seed in range(4):
        rng = random.Random(seed)
        ws = [fit_setup.sample_weights(base, rng, 1.0) for _ in range(4)]
        sim = Sim("great_canal", 4, seed=100 + seed, brains=[Brain(setup_weights=w) for w in ws],
                  record_setup=True)
        sim.setup()
        for f in sim.setup_feats:
            assert not f["features"]["on_gold_corner"]
            assert f["features"]["last_coastal"]


def test_new_terms_change_choices_when_on():
    """grain2/gold_route/deny_3to1 are live: large weights move at least one pick on some board."""
    moved = False
    for seed in range(6):
        a = Sim("great_canal", 4, seed=seed, brains=[_old() for _ in range(4)])
        a.setup()
        on = {"grain2": 12.0, "gold_route": 12.0, "deny_3to1": 12.0}
        c = Sim("great_canal", 4, seed=seed, brains=[_old(on) for _ in range(4)])
        c.setup()
        if _placements(a) != _placements(c):
            moved = True
            break
    assert moved


def test_ridge_fe_recovers_a_within_game_effect():
    rng = random.Random(0)
    rows, X, y = [], [], []
    for g in range(300):
        board = rng.gauss(0, 1)                      # board luck shared by the game
        for s in range(4):
            x0, x1 = rng.gauss(0, 1), rng.gauss(0, 1)
            rows.append({"seed": g})
            X.append([x0, x1])
            y.append(board + 0.3 * x0 + rng.gauss(0, 0.5))
    beta, se, _, _ = fit_setup.ridge_fe(rows, X, y, lam=1.0)
    assert abs(beta[0] - 0.3) < 4 * se[0] and beta[0] / se[0] > 5
    assert abs(beta[1] / se[1]) < 3


def test_design_and_sampling():
    base = {"a": 2.0, "third_harbour": 0.0}
    rng = random.Random(1)
    w = fit_setup.sample_weights(base, rng, 0.35)
    assert w["a"] > 0 and 0 <= w["third_harbour"] <= fit_setup.SCALE["third_harbour"]
    x = fit_setup.design(w, base)
    assert x[0] == pytest.approx(__import__("math").log(w["a"] / 2.0))
    assert x[1] == pytest.approx(w["third_harbour"] / fit_setup.SCALE["third_harbour"])
