"""A model file carries its own play (brain flags): v1 = old play, v2 = every learning."""
from catanbot.brain import Brain
from catanbot.learning import Model
from catanbot.profiles import V1


def test_model_brain_flags_round_trip_and_apply(tmp_path):
    m = Model(path=None)
    m.brain = dict(V1)
    m.path = tmp_path / "v1.json"
    m.save()
    b = Brain(Model(tmp_path / "v1.json"))
    assert b.set_for_grain is False and b.wall_late is True and b.road_cuts is False
    plain = Brain(Model(path=None))
    assert plain.set_for_grain is True                     # no brain block: code defaults
    m.brain = {"no_such_flag": 1}
    m.save()
    assert not hasattr(Brain(Model(tmp_path / "v1.json")), "no_such_flag")
