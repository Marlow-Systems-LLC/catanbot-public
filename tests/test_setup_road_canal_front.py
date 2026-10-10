"""Live 2026-10-09 20:02: the Great Canal setup road went up toward grain. Owner: "it should go on
the canal towards the other players to restrict how much of the canal they can get"."""
from catanbot.board import hex_corners

from test_canal_playbook import _gs, brain


def _spot():
    """We sit on a corner of canal hex (0,0); one edge runs along the canal toward the leader's
    settlement, the other away from the canal."""
    gs = _gs()
    b = gs.board
    canal = [h.coord for h in b.hexes.values() if b.is_canal(h)]
    assert canal
    gs.canal_progress, gs.canal_total = 0, 9
    C = hex_corners(*canal[0])
    for at in C:
        if not b.is_land_vertex(at):
            continue
        edges = [e for e in b.vertex_edges(at) if b.is_land_edge(e) and not b.is_sea_edge(e)]
        fars = [(e, e[0] if e[1] == at else e[1]) for e in edges]
        on = [(e, f) for e, f in fars if f in C and b.is_land_vertex(f)]
        off = [(e, f) for e, f in fars if f not in C]
        if on and off:
            gs.me().settlements.add(at)
            e_on, f_on = on[0]
            # the leader's settlement two steps further along the canal
            rival = next((n for n in b.neighbors(f_on) if n in C and n != at and b.is_land_vertex(n)), None)
            if rival is None:
                continue
            nxt = next(n for n in b.neighbors(rival) if n != f_on)
            gs.players[1].settlements.add(nxt)
            return gs, at, e_on, [e for e, _ in off]
    raise AssertionError("no test spot")


def test_setup_road_runs_along_the_canal_toward_rivals():
    br, (gs, at, e_on, off) = brain(), _spot()
    options = [e_on] + off
    assert br.choose_setup_road(gs, at, options) == e_on


def test_canal_front_scores_closer_rival_higher():
    br, (gs, at, e_on, off) = brain(), _spot()
    far = e_on[0] if e_on[1] == at else e_on[1]
    assert br._canal_front(gs, far) > 0.4
    for e in off:
        f = e[0] if e[1] == at else e[1]
        assert br._canal_front(gs, f) < br._canal_front(gs, far)


def test_canal_done_falls_back_to_spot_reach():
    br, (gs, at, e_on, off) = brain(), _spot()
    gs.canal_progress = 9
    br.canal_road_front = 0
    assert br.choose_setup_road(gs, at, [e_on] + off) in [e_on] + off
