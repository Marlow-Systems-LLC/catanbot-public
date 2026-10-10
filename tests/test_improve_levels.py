"""Game 17 (2026-10-05): live improvement levels were never set, so the planner priced every
improvement at 1 commodity (114 refused tries) and politics never reached 3 for promotions."""
from catanbot import runner as R


def _runner(tmp_path, monkeypatch):
    monkeypatch.setattr(R.Runner, "FLAGS", tmp_path / "game_flags.json")
    r = R.Runner.__new__(R.Runner)
    r.note = lambda *a, **k: None
    r.mode = object()                     # rules known: _load_game_flags won't re-read them
    return r


def test_levels_start_at_zero(tmp_path, monkeypatch):
    r = _runner(tmp_path, monkeypatch)
    assert r._improve_levels() == {"trade": 0, "politics": 0, "science": 0}


def test_landed_improvements_count_and_cap_at_5(tmp_path, monkeypatch):
    r = _runner(tmp_path, monkeypatch)
    r._improved("trade"); r._improved("trade"); r._improved("politics")
    assert r._improve_levels() == {"trade": 2, "politics": 1, "science": 0}
    for _ in range(6):
        r._improved("science")
    assert r._improve_levels()["science"] == 5


def test_levels_survive_a_runner_restart(tmp_path, monkeypatch):
    r = _runner(tmp_path, monkeypatch)
    r._improved("politics"); r._improved("politics"); r._improved("politics")
    r2 = _runner(tmp_path, monkeypatch)   # a fresh runner process, same game
    r2._load_game_flags()
    assert r2._improve_levels()["politics"] == 3
