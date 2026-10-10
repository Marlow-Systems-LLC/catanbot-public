from catanbot.learning import SIGNS, Model
from catanbot.threat import DEFAULT_WEIGHTS


def test_signs_match_default_weights():
    # a guard that disagrees with the hand-set defaults would zero a real signal
    for f, sign in SIGNS.items():
        assert DEFAULT_WEIGHTS[f] * sign > 0, f


def test_flipped_weight_is_clamped():
    m = Model(path=None)
    m.mode_weights["x"] = {"discard_risk": 5.0, "turns": 1.0}
    w = m.weights_for("x")
    assert w["discard_risk"] == 0.0
    assert w["turns"] > 0
