"""Owner 2026-10-02: one knight, a back-up, strong, Politics 3, then mighty and more."""
from catanbot.brain import Brain
from catanbot.state import Knight
from catanbot import moves as mv
from tests.test_progress_wiring import gs  # noqa: F401 (fixture)


def _free(gs, n):
    me = gs.me()
    out = []
    for v in gs.board.vertices:
        if gs.board.is_land_vertex(v) and not gs.occupied(v) and v not in me.knights \
                and not any(h.terrain == "gold" for h in gs.board.vertex_hexes(v)):
            out.append(v)
        if len(out) == n:
            return out


def test_knight_ladder(gs):
    b, me = Brain(), gs.me()
    me.cities.add(next(iter(me.settlements)) if me.settlements else _free(gs, 1)[0])
    me.knights.clear()
    assert b.knight_plan(gs, me) == ("knight", None)
    a, c = _free(gs, 2)
    me.knights[a] = Knight(1, True)
    assert b.knight_plan(gs, me) == ("knight", None)           # back-up knight
    me.knights[c] = Knight(1, False)
    assert b.knight_plan(gs, me) == ("promote", None)
    me.knights[a].level = 2
    me.improvements["politics"] = 1
    assert b.knight_plan(gs, me) == ("improve", "politics")
    me.improvements["politics"] = 3
    assert b.knight_plan(gs, me) == ("promote", None)          # mighty now allowed


def test_two_basic_knights_max(gs):
    me = gs.me()
    me.hand.update(wool=5, ore=5, grain=5)
    me.knights.clear()
    for v in _free(gs, 2):
        me.knights[v] = Knight(1, False)
    assert not [m for m in mv.build_moves(gs) if m.kind == "knight"]
