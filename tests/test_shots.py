"""Owner 2026-10-11: keep a screenshot of each new screen; repeats only for the game being played."""
import numpy as np

from catanbot import shots


def _img(seed):
    rng = np.random.default_rng(seed)
    return (rng.random((90, 160, 3)) * 255).astype(np.uint8)


def test_kind_strips_the_time_stamp_and_ignores_other_folders():
    d = shots.DATA
    assert shots.kind(d / "popups" / "231501_road_after.png") == "popups/road_after"
    assert shots.kind(d / "probe" / "231501_007_screen_unknown_native.png") == "probe/screen_unknown"
    assert shots.kind(d / "games" / "20261011_120000" / "scan.png") is None
    assert shots.kind(d / "endgame" / "game_over.png") is None


def test_new_screens_stay_repeats_go_at_the_next_game():
    d = shots.DATA / "popups"
    d.mkdir(parents=True)
    a, b = _img(1), _img(2)
    assert shots.imwrite(d / "100000_road.png", a)                  # first of its type: kept
    assert shots.imwrite(d / "100001_road.png", a)                  # same screen: this game only
    (d / "100001_road.json").write_text("{}")
    assert shots.imwrite(d / "100001_road_native.png", a)
    assert shots.imwrite(d / "100002_road.png", b)                  # looks different: kept
    assert shots.new_game() == 3
    assert sorted(p.name for p in d.iterdir()) == ["100000_road.png", "100002_road.png"]


def test_repeats_are_capped_per_game_and_natives_follow(monkeypatch):
    monkeypatch.setattr(shots, "GAME_PER_TYPE", 2)
    d = shots.DATA / "unread"
    d.mkdir(parents=True)
    a = _img(3)
    for i in range(4):
        shots.imwrite(d / f"10000{i}_unknown.png", a)
    assert not (d / "100003_unknown.png").exists()                  # 1 kept + 2 repeats, then none
    assert not shots.imwrite(d / "100003_unknown_native.png", a)    # its main frame was skipped


def test_other_paths_write_as_before(tmp_path):
    p = tmp_path / "x.png"
    assert shots.imwrite(p, _img(4)) and p.exists()


def test_prune_keeps_one_of_each_screen(tmp_path):
    import cv2
    d = shots.DATA / "probe"
    d.mkdir(parents=True)
    for i, seed in enumerate([5, 5, 5, 6]):
        cv2.imwrite(str(d / f"10000{i}_gain_pick.png"), _img(seed))
        (d / f"10000{i}_gain_pick.json").write_text("{}")
    dry = shots.prune()
    assert dry["kept"] == 2 and dry["dropped"] == 2 and len(list(d.iterdir())) == 8
    shots._cache = None
    assert shots.prune(apply=True)["dropped"] == 2
    assert sorted(p.name for p in d.iterdir()) == ["100000_gain_pick.json", "100000_gain_pick.png",
                                                   "100003_gain_pick.json", "100003_gain_pick.png"]
