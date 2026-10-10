"""Owner rule: the event die moves the barbarians (3 faces) or opens a gate; a gate gives a
progress card to each player whose improvement level on that colour reaches the red die."""
from catanbot.sim import EVENT_DIE, Sim


def test_event_die_faces():
    assert EVENT_DIE.count("ship") == 3 and {"trade", "politics", "science"} <= set(EVENT_DIE)


def test_gate_draws_by_red_die():
    sim = Sim("great_canal", seed=5)
    sim.setup()
    a, b, c, d = sim.gs.players
    a.improvements["science"] = 1       # draws on red 1-2
    b.improvements["science"] = 3       # draws on red 1-4
    c.improvements["trade"] = 5         # wrong colour
    sim.progress_draw("science", 3)
    assert sim.progress_drawn == {b.name: 1}
    sim.progress_draw("science", 2)
    assert sim.progress_drawn == {a.name: 1, b.name: 2}
    n = len(sim.decks["science"])
    sim.progress_draw("science", 6)     # nobody at level 5
    assert len(sim.decks["science"]) == n


def test_fifth_card_off_turn_is_discarded_roller_keeps_it():
    sim = Sim("great_canal", seed=5)
    sim.setup()
    a, b = sim.gs.players[:2]
    sim.gs.current = 0
    for p in (a, b):
        p.progress_hand = ["merchant", "smith", "spy", "crane"]
        p.improvements["trade"] = 5
    sim.decks["trade"] = ["master_merchant", "master_merchant"]
    sim.progress_draw("trade", 1)
    assert len(a.progress_hand) == 5            # the roller may hold 5 during its turn
    assert len(b.progress_hand) == 4 and sim.progress_discarded == 1
    sim.play_held(0)
    assert len(a.progress_hand) <= 4


def test_live_rule_discards_fifth_off_turn():
    from catanbot import progress_cards as pc
    sim = Sim("great_canal", seed=5)
    sim.setup()
    gs = sim.as_me(1)
    plays = pc.choose_play(gs, gs.me(), ["merchant", "smith", "spy", "crane", "warlord"], "opp_turn")
    assert [k for k, _ in plays] == ["discard"]
    assert pc.choose_play(gs, gs.me(), ["merchant", "printer"], "opp_turn") == []   # VP cards never played
