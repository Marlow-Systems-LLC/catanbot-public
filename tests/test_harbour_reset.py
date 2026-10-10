"""Live 2026-10-08 21:41-21:54: the new game kept the old game's harbours (no 'harbours read'
after 21:45:40), the bot settled on the old 3:1 edge (5,1), and the city built there at 21:53:34
wiped the bank dialog's ratios, so 21:54:12 planned a lumber 3:1 it never had."""
import types

import catanbot.runner as R
from catanbot.board import Board, Harbor, ekey
from catanbot.state import PlayerState

V, W = (5, -1), (5, 1)
OLD = [Harbor(ekey(V, W), "3:1"), Harbor(ekey((-7, 5), (-7, 7)), "brick")]
NEW = [Harbor(ekey((3, -7), (4, -8)), "3:1")]


def _runner(gs=None):
    r = R.Runner.__new__(R.Runner)
    r.notes = []
    r.note = r.notes.append
    r._save_game_flags = lambda: None
    r.brain = types.SimpleNamespace(ratio_seen={})
    r.gs = gs
    r._harbors_best, r._harbors_tpl, r._harbors_seen, r._harbors_denied = None, False, None, set()
    return r


def _board(harbors, template=False):
    b = Board()
    b.harbors = list(harbors)
    if template:
        b.template_offset = (0, 0)
    return b


def test_new_game_reads_and_logs_harbours_again():
    r = _runner()
    r._pick_harbors(_board(OLD, template=True))            # old game: template-locked
    assert any("harbours read" in n for n in r.notes)
    b = _board(NEW)
    r._pick_harbors(b)
    assert b.harbors == OLD                                 # same game: the lock holds
    r._reset_harbors("game 2")
    r.notes.clear()
    b = _board(NEW)
    r._pick_harbors(b)
    assert b.harbors == NEW                                 # the old game's harbours never carry over
    assert any("harbours read" in n for n in r.notes)
    r.notes.clear()
    r._reset_harbors("game 3")
    r._pick_harbors(_board(NEW))                            # the same list as last game is logged too
    assert any("harbours read" in n for n in r.notes)


def test_bank_dialog_drops_a_wrong_harbour_for_the_game():
    me = PlayerState("marlow.systems", is_me=True)
    me.cities.add(W)
    b = _board(OLD, template=True)
    gs = types.SimpleNamespace(me=lambda: me, board=b)
    r = _runner(gs)
    r._pick_harbors(b)
    r._check_harbors({"lumber": 4, "brick": 4, "wool": 2, "ore": 4, "grain": 4})
    assert b.harbor_at(W) is None
    b2 = _board(OLD, template=True)                         # the next scan reads the template again
    r._pick_harbors(b2)
    assert b2.harbor_at(W) is None and b2.harbor_at((-7, 5)) == "brick"


def _gs_on_3to1():
    me = PlayerState("marlow.systems", is_me=True)
    me.settlements.add(W)
    return types.SimpleNamespace(me=lambda: me, board=_board(OLD))


def test_city_build_keeps_the_seen_ratios():
    r = _runner(_gs_on_3to1())
    seen = {"lumber": 4, "brick": 4, "wool": 2, "ore": 4, "grain": 4}
    r.brain.ratio_seen = dict(seen)
    r._record_build("city", W)                              # a city on the 3:1 corner we already hold
    assert r.brain.ratio_seen == seen


def test_new_settlement_on_a_harbour_drops_only_what_it_can_lower():
    r = _runner(_gs_on_3to1())
    r.brain.ratio_seen = {"lumber": 4, "brick": 4, "wool": 2, "ore": 3}
    r._record_build("settlement", W)                        # new 3:1: only the 4s may change
    assert r.brain.ratio_seen == {"wool": 2, "ore": 3}
    r.brain.ratio_seen = {"lumber": 4, "brick": 4}
    r._record_build("settlement", (-7, 5))                  # new brick 2:1: only brick may change
    assert r.brain.ratio_seen == {"lumber": 4}
    r._record_build("settlement", (0, 0))                   # no harbour: nothing changes
    assert r.brain.ratio_seen == {"lumber": 4}


def test_clearing_already_clear_harbours_logs_nothing(monkeypatch, tmp_path):
    # stage 2026-10-09: "harbours: cleared (left the game (home))" every second on the home screen
    from catanbot import config
    from catanbot.runner import ReplayIO, Runner
    cfg = config.load(); cfg["dry_run"] = True
    r = Runner(ReplayIO([]), cfg)
    notes = []
    monkeypatch.setattr(r, "note", lambda m: notes.append(m))
    r._harbors_best = [("x", 3)]
    r._reset_harbors("home")
    r._reset_harbors("home")
    r._reset_harbors("home")
    assert len([n for n in notes if "harbours: cleared" in n]) == 1
