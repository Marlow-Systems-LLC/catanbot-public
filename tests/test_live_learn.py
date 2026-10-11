"""Owner 2026-10-06: live games train the model, and the model only changes when it gets better."""
import json

import pytest

from catanbot import live_learn as L
from catanbot.brain import Brain
from catanbot.learning import Model
from catanbot.sim import Sim


@pytest.fixture
def paths(tmp_path, monkeypatch):
    model = tmp_path / "model.json"
    m = Model(path=None)
    m.path = model
    m.save()
    monkeypatch.setattr(L, "MODEL", model)
    monkeypatch.setattr(L, "GLOG", tmp_path / "live_glog.json")
    monkeypatch.setattr(L, "TRAIN", tmp_path / "train")
    monkeypatch.setattr(L, "PENDING", tmp_path / "train" / "live_pending")
    monkeypatch.setattr(L, "USED", tmp_path / "train" / "live_used")
    monkeypatch.setattr(L, "CAND", tmp_path / "train" / "cand.json")
    monkeypatch.setattr(L, "PREV", tmp_path / "train" / "prev.json")
    monkeypatch.setattr(L, "LEDGER", tmp_path / "train" / "stages.jsonl")
    monkeypatch.setattr(L, "LOCK", tmp_path / "train" / "gate.lock")
    return tmp_path


def _game():
    sim = Sim("great_canal", 4, seed=3)
    sim.setup()
    return sim.gs, Brain(Model(path=None))


def _play_logged_game(paths, monkeypatch):
    gs, brain = _game()
    t = [1000.0]
    monkeypatch.setattr(L.time, "time", lambda: t[0])
    for _ in range(5):
        L.snapshot(gs, brain, "great_canal", "P0")
        L.snapshot(gs, brain, "great_canal", "P0")        # a repeated roll frame: same turn
        t[0] += 60
    return L.finish("p2", "great_canal")


def test_one_snapshot_per_turn_and_finish_makes_a_pending_game(paths, monkeypatch):
    f = _play_logged_game(paths, monkeypatch)
    d = json.loads(f.read_text())
    assert len(d["snapshots"]) == 5 and d["winner"] == "P2" and d["mode"] == "great_canal"
    assert not L.GLOG.exists()
    assert L.finish("P2") is None                       # the same game is never learned twice


def test_unmatched_winner_learns_nothing(paths, monkeypatch):
    gs, brain = _game()
    L.snapshot(gs, brain, "great_canal", "P0")
    assert L.finish("Somebody") is None


def test_match_name_prefix_both_ways():
    assert L.match_name("marlow.systems", ["Lin", "marlow.sys"]) == "marlow.sys"
    assert L.match_name("Marianne", ["marianne_92", "Jean"]) == "marianne_92"
    assert L.match_name(None, ["Lin"]) is None


def test_gate_promotes_only_a_better_candidate(paths, monkeypatch):
    _play_logged_game(paths, monkeypatch)
    before = L.MODEL.read_text()
    monkeypatch.setattr(L, "strategy_ok", lambda cand: (True, "39 passed"))
    monkeypatch.setattr(L, "battle", lambda *a: {"games": 800, "A_share": 0.53, "se": 0.0177})
    row = L.gate()                                      # stage 19's margin: under 2 se, not promoted
    assert row["promoted"] is False and L.MODEL.read_text() == before
    assert len(L.pending()) == 1                        # kept: learned again with the next game
    monkeypatch.setattr(L, "battle", lambda *a: {"games": 800, "A_share": 0.56, "se": 0.0177})
    row = L.gate()
    assert row["promoted"] is True and row["stage"] == 2 and row["live_games"] == 1
    assert L.MODEL.read_text() != before and L.PREV.read_text() == before
    assert L.pending() == [] and not L.LOCK.exists()
    assert Model(L.MODEL).games == 1


def test_gate_never_promotes_a_candidate_that_breaks_an_owner_rule(paths, monkeypatch):
    _play_logged_game(paths, monkeypatch)
    before = L.MODEL.read_text()
    monkeypatch.setattr(L, "strategy_ok", lambda cand: (False, "5 failed, 34 passed"))
    monkeypatch.setattr(L, "battle", lambda *a: pytest.fail("no battle for a candidate that fails the tests"))
    row = L.gate()
    assert row["promoted"] is False and row["tests"] == "5 failed, 34 passed"
    assert L.MODEL.read_text() == before and len(L.pending()) == 1


def test_passes_needs_two_standard_errors():
    assert not L.passes(0.53, 0.0177) and L.passes(0.54, 0.0177)
    assert not L.passes(0.27, 0.0153, 0.25) and L.passes(0.29, 0.0153, 0.25)


def test_gate_without_live_games_changes_nothing(paths):
    assert L.gate() == {"skipped": "no pending live games"}
    assert not L.LEDGER.exists()
