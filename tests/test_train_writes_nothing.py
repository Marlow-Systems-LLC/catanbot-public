"""Owner 2026-10-06: only a gate changes a model file; training rounds never write one."""
from catanbot.learning import Model
from catanbot.sim import train


def test_train_keeps_the_model_file_untouched(tmp_path):
    f = tmp_path / "best.json"
    m = Model(path=None)
    m.path = f
    m.save()
    before = f.read_text()
    model, st = train(2, "great_canal", seed=11, model=Model(f))
    assert st["games"] == 2 and f.read_text() == before
    assert model.path == f                       # the caller can still save it on purpose
