"""Owner 2026-10-09: "every failure must be recorded in the logs" (live: Crane played, the
improvement never landed, the game log said nothing)."""
import json

from catanbot.game_log import GameLog
from catanbot.runner import Runner


def _gl(tmp_path):
    gl = GameLog(tmp_path)
    gl.game_id, gl.turn = "g1", 5
    return gl


def test_failure_goes_into_the_record_once_per_turn(tmp_path):
    gl = _gl(tmp_path)
    gl.failure("improve science: not seen")
    gl.failure("improve science: not seen")
    gl.failure("coin unreadable at slot 3")
    gl.failure("coin unreadable at slot 4")          # same shape, numbers aside
    rows = [json.loads(x) for x in (tmp_path / "current.jsonl").read_text().splitlines()]
    fails = [r for r in rows if r["kind"] == "failure"]
    assert [r["note"] for r in fails] == ["FAILED: improve science: not seen", "FAILED: coin unreadable at slot 3"]
    st = json.loads((tmp_path / "current.json").read_text())
    assert st["failCount"] == 2 and st["failures"][0] == {"turn": 5, "what": "improve science: not seen"}
    gl.turn = 6
    gl.failure("improve science: not seen")           # a new turn logs it again
    assert gl.fail_count == 3


def test_runner_note_routes_failures(tmp_path):
    gl = _gl(tmp_path)
    r = Runner.__new__(Runner)
    r.log = []
    r._glog = lambda start=True, ending=False: gl
    for m in ("progress failed: crane played, improve science not completed (level 3 unchanged)",
              "build city at (3, 7) not seen (hand unchanged, spot not ours)",
              "click improve science (city ring) at (779, 355)",     # a click, not an outcome
              "smith was refused earlier this game",                 # a skip of an old failure
              "improve science: landed"):
        Runner.note(r, m)
    assert [f["what"][:20] for f in gl.failures] == ["progress failed: cra", "build city at (3, 7)"]
