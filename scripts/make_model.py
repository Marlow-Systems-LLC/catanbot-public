"""Stamp a brain profile into a model file: weights from SRC, play from catanbot/profiles.py.

    python scripts/make_model.py SRC.json PROFILE OUT.json
"""
import sys
from pathlib import Path

from catanbot.learning import Model
from catanbot.profiles import PROFILES

src, name, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
m = Model(src)
m.brain = dict(PROFILES[name])
m.path = out
m.save()
print(f"{out}: weights of {src.name}, profile {name} ({len(m.brain)} flags)")
