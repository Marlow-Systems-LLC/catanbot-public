"""scripts/situations.py: a tiny run finishes and the summary has the buckets."""
from scripts import situations


def test_two_game_run_has_buckets():
    out = situations.run(2, "roads1004", workers=1)
    assert len(out["rows"]) == 2 and {r["seat"] for r in out["rows"]} == {0, 1}
    s = out["summary"]
    assert s["fair"] == 0.25 and s["overall"]["n"] == 2
    for feat in ("pips_total", "grain_pips", "harbour_3to1_or_2to1", "distinct",
                 "gap_third", "gap_two_thirds", "lost_city"):
        b = s["buckets"][feat]
        assert b and sum(x["n"] for x in b.values()) == 2
        assert all({"n", "win_rate", "se"} <= set(x) for x in b.values())
