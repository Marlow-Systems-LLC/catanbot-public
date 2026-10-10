"""owner 2026-10-08 "some of the moves take a long time": after a click the runner waits only until
the window changed and settled, never longer than the old fixed sleep."""
import time

import numpy as np

from catanbot import runner as R


class _IO:
    def __init__(self, frames):
        self.f, self.i, self.settle_poll = frames, 0, True

    def snapshot(self):
        f = self.f[min(self.i, len(self.f) - 1)]
        self.i += 1
        return f


def _runner(frames):
    r = R.Runner.__new__(R.Runner)
    r.io = _IO(frames)
    return r


def test_returns_once_the_popup_is_up_and_steady():
    a = np.zeros((888, 1372, 3), np.uint8)
    b = a.copy(); b[200:600, 300:900] = 200
    r = _runner([b, b, b])
    t = time.time()
    assert r._settled_snapshot(a, 0.8) is b
    assert time.time() - t < 0.5


def test_never_waits_longer_than_the_old_sleep():
    a = np.zeros((888, 1372, 3), np.uint8)
    r = _runner([a] * 30)
    t = time.time()
    r._settled_snapshot(a, 0.7)
    assert time.time() - t < 0.85
