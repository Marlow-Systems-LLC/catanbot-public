"""owner 2026-10-08: one log per game with every turn; a new game archives the old log; the chat
and the API (voice) know where we are."""
import json

from catanbot import chat, game_log
from catanbot.game_log import GameLog, summary

SEATS = ["Rhea", "marlow.systems", "Bramble", "Marianne"]


def test_turns_follow_rolls_and_our_turn_reanchors(tmp_path):
    gl = GameLog(tmp_path)
    gl.start("great_canal", SEATS, "marlow.systems", 19, game_id="g1")
    gl.setup("setup settlement (0, -4)")
    gl.roll((3, 4))                              # first roll: seat 1
    assert (gl.turn, gl.current) == (1, "Rhea")
    gl.our_turn({"marlow.systems": 3}, {"wool": 2})
    gl.our_turn()                                # same roll screen again: no new turn
    gl.roll((2, 2))                              # our own roll belongs to our turn
    assert (gl.turn, gl.current, gl.last_roll) == (2, "marlow.systems", 4)
    gl.roll((6, 5))
    assert (gl.turn, gl.current) == (3, "Bramble")
    gl.roll((1, 1)); gl.roll((5, 5))             # Marianne, then round to Rhea
    assert gl.current == "Rhea"
    gl.our_turn()
    assert (gl.turn, gl.our_turns) == (6, 2)
    s = json.loads((tmp_path / "current.json").read_text())
    assert s["currentPlayer"] == "marlow.systems" and s["turn"] == 6
    assert "hand" not in json.dumps(s)
    assert any(r["kind"] == "setup" for r in map(json.loads, (tmp_path / "current.jsonl").read_text().splitlines()))


def test_new_game_archives_the_old_log(tmp_path):
    gl = GameLog(tmp_path)
    gl.start("great_canal", SEATS, "marlow.systems", 19, game_id="g1")
    gl.roll((3, 4))
    gl.end("Bramble", ["Bramble"], {"Bramble": 19})
    assert "Bramble won" in summary(game_log.current_status(tmp_path))
    gl2 = GameLog(tmp_path)
    assert not gl2.resume()                      # ended: nothing to resume
    gl2.start("great_canal", SEATS, "marlow.systems", 19, game_id="g2")
    assert (tmp_path / "archive" / "g1.jsonl").exists() and (tmp_path / "archive" / "g1.json").exists()
    assert json.loads((tmp_path / "current.jsonl").read_text().splitlines()[0])["kind"] == "start"


def test_runner_restart_resumes_the_open_game(tmp_path):
    gl = GameLog(tmp_path)
    gl.start("great_canal", SEATS, "marlow.systems", 19, game_id="g1")
    gl.roll((3, 4)); gl.roll((2, 3))
    gl2 = GameLog(tmp_path)
    assert gl2.resume() and (gl2.turn, gl2.current) == (2, "marlow.systems")
    gl2.roll((6, 6))
    assert gl2.current == "Bramble"


def test_private_fields_stay_out_of_the_api(tmp_path):
    gl = GameLog(tmp_path)
    gl.start("great_canal", SEATS, "marlow.systems", 19, game_id="g1")
    gl.our_turn({}, {"grain": 1})
    rows = game_log.turns(tmp_path)
    assert rows and all("hand" not in r for r in rows)


def test_chat_answers_where_we_are(tmp_path, monkeypatch):
    gl = GameLog(tmp_path)
    gl.start("great_canal", SEATS, "marlow.systems", 19, game_id="g1")
    gl.roll((3, 4), {"Rhea": 4, "marlow.systems": 5})
    monkeypatch.setattr(game_log, "DIR", tmp_path)
    out = chat.answer("what turn is it?")
    assert "Turn 1" in out and "Rhea" in out
    assert "Points" in chat.answer("who is winning")


def test_api_live_and_game_routes(tmp_path, monkeypatch):
    from catanbot import api
    gl = GameLog(tmp_path)
    gl.start("great_canal", SEATS, "marlow.systems", 19, game_id="g1")
    gl.our_turn({}, {"grain": 1})
    monkeypatch.setattr(game_log, "DIR", tmp_path)
    code, body = api.route("/game", {})
    assert code == 200 and body["status"]["turn"] == 1 and "hand" not in json.dumps(body)
    code, body = api.route("/live", {})
    assert code == 200 and body["game"]["summary"].startswith("Turn 1")


def test_runner_opens_the_log_on_its_first_click_and_archives_at_the_next_game(tmp_path):
    from test_canal_playbook import _gs
    from test_probe_controls import make
    r = make(tmp_path)
    r.gs = _gs()
    r._glog_force = True                         # FakeIO is a replay: the live bot has a real screen
    r.game_colors = [("leader", "red"), ("bot", "magenta"), ("weak", "orange")]
    r._glog_new = True
    r.click((10, 10), "setup settlement (0, -4)")
    gl = r._glog_obj
    assert gl.active() and gl.seats == ["leader", "bot", "weak"] and gl.me == "bot"
    r.note("offer from leader: give {} get {}")
    rows = game_log.turns()
    assert [x["kind"] for x in rows][:3] == ["start", "setup", "note"]
    first = gl.game_id
    r._glog_new = True                           # lobby seen: the next game
    r.game_colors = [("weak", "red"), ("bot", "magenta")]
    r._game_dir = None
    r.click((10, 10), "setup settlement (1, 1)")
    assert r._glog_obj.seats == ["weak", "bot"]
    assert list((game_log.DIR / "archive").glob("*.jsonl")), f"{first} not archived"


# ---- owner 2026-10-09: the strategy in the ledger, for the owner's channels only -------------
def _game_with_plan(tmp_path):
    gl = GameLog(tmp_path)
    gl.start("great_canal", SEATS, "marlow.systems", 19, game_id="g1")
    gl.our_turn({"marlow.systems": 5}, {"grain": 1, "ore": 3})
    gl.plan(["bank 3 ore -> 1 grain", "knight"], "race; next goal city+1@1.0",
            ["city+1@1.0", "settlement+1@2.0"], {"Rhea": ["city+1@0.5"], "Marianne": []},
            hand={"grain": 1, "ore": 3})
    gl.plan(["bank 3 ore -> 1 grain", "knight"], "race; next goal city+1@1.0")   # same plan: no event
    gl.decision("knight first: bank {'ore': 3} -> {'grain': 1} + knight")
    gl.decision("knight first: bank {'ore': 3} -> {'grain': 1} + knight")        # re-noted: once
    gl.action("robber on (2, -2) (victim Rhea)")
    return gl


def test_plan_reason_outlook_and_decisions_are_in_the_ledger(tmp_path):
    _game_with_plan(tmp_path)
    rows = [json.loads(x) for x in (tmp_path / "current.jsonl").read_text().splitlines()]
    plans = [r for r in rows if r["kind"] == "plan"]
    assert len(plans) == 1 and plans[0]["plan"] == ["bank 3 ore -> 1 grain", "knight"]
    assert plans[0]["reason"].startswith("race") and plans[0]["goals"][0] == "city+1@1.0"
    assert plans[0]["hand"] == {"grain": 1, "ore": 3}             # the event file keeps the hand
    assert [r["decision"] for r in rows if r["kind"] == "decision"] == [
        "knight first: bank ore 3 -> grain 1 + knight"]
    st = game_log.current_status(tmp_path)["strategy"]
    assert st["turn"] == 1 and st["threats"] == {"Rhea": ["city+1@0.5"]}
    assert st["decisions"] == ["T1 plan: bank 3 ore -> 1 grain, knight because race; next goal city+1@1.0",
                               "T1 knight first: bank ore 3 -> grain 1 + knight",
                               "T1 robber on (2, -2) (victim Rhea) (goal city+1)"]


def test_strategy_never_carries_the_hand_and_survives_a_restart(tmp_path, monkeypatch):
    from catanbot import api
    _game_with_plan(tmp_path)
    monkeypatch.setattr(game_log, "DIR", tmp_path)
    code, body = api.route("/game", {})
    assert code == 200 and "hand" not in json.dumps(body)
    assert body["strategy"].startswith("Plan (turn 1): bank 3 ore -> 1 grain, knight -- race")
    assert "Threats: Rhea city+1@0.5" in body["strategy"]
    gl2 = GameLog(tmp_path)
    assert gl2.resume() and gl2.strategy["plan"] == ["bank 3 ore -> 1 grain", "knight"]


def test_in_game_chat_never_reveals_the_plan(tmp_path, monkeypatch):
    _game_with_plan(tmp_path)
    monkeypatch.setattr(game_log, "DIR", tmp_path)
    for q in ("what turn is it?", "who is winning", "where are we", "what's your plan marlow?",
              "marlow what is your strategy", "what will you build next"):
        out = chat.answer(q).lower()
        assert not any(w in out for w in ("bank 3 ore", "next goal", "victim rhea", "city+1@", "{'ore'"))
    assert "plan" not in summary(game_log.current_status(tmp_path)).lower()


def test_strategy_line_is_none_without_a_game_or_after_it_ended(tmp_path):
    assert game_log.strategy_line(None) is None
    gl = _game_with_plan(tmp_path)
    gl.end("Rhea", ["Rhea"], {"Rhea": 19})
    assert game_log.strategy_line(game_log.current_status(tmp_path)) is None


def test_runner_logs_plan_decisions_and_ends_the_game_after_leaving_it(tmp_path):
    from test_canal_playbook import _gs
    from test_probe_controls import make
    r = make(tmp_path)
    r.gs = _gs()
    r._glog_force = True
    r.game_colors = [("leader", "red"), ("bot", "magenta"), ("weak", "orange")]
    r._glog_new = True
    r.click((10, 10), "setup settlement (0, -4)")
    gl = r._glog_obj
    r.note("knight first: bank {'ore': 3} -> {'grain': 1} + knight")
    r.note("bank set: 2 wool, asking for 1 more to bank 3 wool -> grain")      # the hand: not logged
    r.note("hand 12 > limit 9: bank 3 ore -> 1 grain")
    assert [d.split(" ", 1)[1] for d in gl.strategy["decisions"]] == [
        "knight first: bank ore 3 -> grain 1 + knight", "hand 12 > limit 9: bank 3 ore -> 1 grain"]
    # stage 2026-10-09 01:07:15: the game-over screen marks the next game before the result is read
    r._glog_new = True
    r._record_game_over(["1 leader", "2 bot", "3 weak"])
    st = game_log.current_status()
    assert st["ended"] and st["winner"] == "leader"


# ---- owner 2026-10-09: "it can tell us why it makes a decision" ------------------------------
def _next(gl):
    if gl.current == gl.me:
        gl.roll((1, 2))                          # our roll closes our open turn
    gl.our_turn()


def _turn(gl, plan, reason="race; next goal settlement+1@1.0", goals=("settlement+1@1.0",), stance=None):
    _next(gl)
    gl.plan(plan, reason, list(goals), {}, hand={"ore": 3, "grain": 2}, stance=stance)


def test_each_decision_carries_its_why(tmp_path):
    gl = GameLog(tmp_path)
    gl.start("great_canal", SEATS, "marlow.systems", 19, game_id="g1")
    _turn(gl, ["bank 3 ore -> 1 brick", "settlement (1, -2)"], stance="block")
    gl.action("bank trade {'ore': 3}->{'brick': 1}")
    gl.decision("save the city: offer {'lumber': 1} -> {'ore': 1}")
    gl.action("send offer {'lumber': 1}->{'ore': 1}")
    gl.action("accept trade with player who said yes")
    gl.action("robber on (2, -2) (victim Rhea)")
    ds = gl.strategy["decisions"]
    assert ds[1] == "T1 bank trade ore 3 -> brick 1 for settlement (1, -2) (goal settlement+1)"
    assert ds[3] == "T1 send offer lumber 1 -> ore 1 (save the city; goal settlement+1)"
    assert ds[4] == "T1 accept trade with player who said yes (save the city; goal settlement+1)"
    assert ds[5] == "T1 robber on (2, -2) (victim Rhea) (stance block; goal settlement+1)"
    assert all(len(d) <= game_log.LINE_MAX for d in ds)
    # the context of an earlier turn is not this turn's why; no plan, no goal: no tail at all
    _next(gl)
    gl.strategy["goals"] = []
    gl.action("bank trade {'lumber': 1}->{'ore': 1}")
    assert gl.strategy["decisions"][-1] == "T2 bank trade lumber 1 -> ore 1"
    # a long reason still fits one line
    _turn(gl, ["knight"], reason="x" * 400)
    assert len(gl.strategy["decisions"][-1]) == game_log.LINE_MAX


def test_plan_changes_are_lines_and_repeats_collapse(tmp_path):
    gl = GameLog(tmp_path)
    gl.start("great_canal", SEATS, "marlow.systems", 19, game_id="g1")
    _turn(gl, ["bank 3 ore -> 1 grain", "knight"])
    for _ in range(5):                                   # re-planned every second: one line
        gl.plan(["bank 3 ore -> 1 grain", "knight"], "race")
    gl.plan(["city (1, 1)"], "race; next goal city+1@1.0")
    gl.plan(["bank 3 ore -> 1 grain", "knight"], "race")  # back to a plan logged this turn
    gl.plan([], "nothing affordable or worth it: hold")
    _next(gl)
    gl.plan([], "nothing affordable or worth it: hold")  # same as the last line: not again
    gl.plan(["knight"], "race")
    plans = [d for d in gl.strategy["decisions"] if " plan: " in d]
    assert plans == ["T1 plan: bank 3 ore -> 1 grain, knight because race; next goal settlement+1@1.0",
                     "T1 plan: city (1, 1) because race; next goal city+1@1.0",
                     "T1 plan: hold because nothing affordable or worth it: hold",
                     "T2 plan: knight because race"]


def test_decisions_cap_keeps_plans_and_trades_over_robber_taps(tmp_path):
    gl = GameLog(tmp_path)
    gl.start("great_canal", SEATS, "marlow.systems", 19, game_id="g1")
    for t in range(80):
        _turn(gl, ["bank 3 ore -> 1 brick", f"road ({t}, 0)"])
        gl.action("bank trade {'ore': 3}->{'brick': 1}")
        gl.action(f"robber on ({t}, 1) (victim Rhea)")
        gl.action("steal from Rhea")
    ds = game_log.current_status(tmp_path)["strategy"]["decisions"]
    assert len(ds) <= game_log.KEEP_DECISIONS
    assert len(json.dumps(ds)) <= game_log.DECISIONS_BYTES
    robber = [d for d in ds if " robber on " in d or " steal " in d]
    assert len(robber) <= 2                               # the newest only: plans and trades stay
    assert ds[-1].startswith("T80 steal from Rhea")
    assert any(d.startswith("T80 bank trade") for d in ds) and any(d.startswith("T60 plan:") for d in ds)
    # the whole status stays far under the 64 KB marlow.systems clips at
    assert len((tmp_path / "current.json").read_text()) < 16_000


def test_hand_never_reaches_the_status_or_the_api(tmp_path, monkeypatch):
    from catanbot import api
    gl = GameLog(tmp_path)
    gl.start("great_canal", SEATS, "marlow.systems", 19, game_id="g1")
    _turn(gl, ["knight"])
    gl.note("offer from Rhea: ['ore'] -> give {'ore': 1} get {'wool': 1}: decline "
            "hand={'ore': 1, 'grain': 2} vp={'Rhea': 5}")
    gl.note("counter offer from Rhea: give {} get {}: decline hand={'wool': 4}")
    gl.decision("save the city: offer {'ore': 1} -> {'grain': 1} hand={'ore': 1}")
    gl.action("send offer {'ore': 1}->{'grain': 1} hand={'ore': 1}")
    st = game_log.current_status(tmp_path)
    blob = json.dumps(st)
    assert "hand" not in blob and "'grain': 2" not in blob and "'wool': 4" not in blob
    assert any("decline vp={'Rhea': 5}" in x for x in st["recent"])
    # the event file keeps the hand for training, the API's turns never show it
    notes = [json.loads(x) for x in (tmp_path / "current.jsonl").read_text().splitlines()
             if json.loads(x)["kind"] == "note"]
    assert notes[0]["hand"] == "{'ore': 1, 'grain': 2}" and "hand=" not in notes[0]["note"]
    monkeypatch.setattr(game_log, "DIR", tmp_path)
    for route in ("/game", "/live"):
        code, body = api.route(route, {})
        assert code == 200 and "hand=" not in json.dumps(body) and "'grain': 2" not in json.dumps(body)
    # a status written before the scrub (the old leak) is cleaned when read
    raw = json.loads((tmp_path / "current.json").read_text())
    raw["recent"].append("T1 offer from Rhea: decline hand={'brick': 9}")
    (tmp_path / "current.json").write_text(json.dumps(raw))
    assert "hand" not in json.dumps(game_log.current_status(tmp_path))


def test_live_tail_drops_the_hand(tmp_path, monkeypatch):
    from catanbot import api
    log = tmp_path / "runner_stage.log"
    log.write_text("02:00:00 me: hand={'ore': 3} cities=1 knights=0\n02:00:01 click end turn\n")
    monkeypatch.setattr(api, "DATA", tmp_path)
    monkeypatch.setattr(game_log, "DIR", tmp_path)
    code, body = api.route("/live", {})
    assert code == 200 and body["tail"][0].endswith("me: cities=1 knights=0")
    assert "hand" not in json.dumps(body["tail"])
