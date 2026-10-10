#!/usr/bin/env python3
"""FaceTime voice add-on (optional, local, off by default). See docs/facetime.md.

    PYTHONPATH=. python scripts/facetime_voice.py                 # the daemon (reads data/config.json "facetime")
    PYTHONPATH=. python scripts/facetime_voice.py --check         # print the parsed config (no secrets), exit
    PYTHONPATH=. python scripts/facetime_voice.py --mic-check     # is the microphone allowed? (not during a call)
    PYTHONPATH=. python scripts/facetime_voice.py --level 10      # input level of BlackHole 16ch for 10 s
    PYTHONPATH=. python scripts/facetime_voice.py --in talk.wav [--out reply.wav] [--kind dm|group] [--brain stub|local]
    PYTHONPATH=. python scripts/facetime_voice.py --ask "marlow, what's the score?"   # one brain reply, no audio

No "facetime" block in the config = the add-on is off and the daemon exits.
"answer": false = watch and log rings, never press anything.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from catanbot.facetime import config as C  # noqa: E402

LOG_DIR = Path(os.path.expanduser("~/.catanbot-facetime"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(C.PATH))
    p.add_argument("--check", action="store_true")
    p.add_argument("--mic-check", action="store_true")
    p.add_argument("--level", type=float, metavar="SECONDS")
    p.add_argument("--in", dest="input")
    p.add_argument("--out")
    p.add_argument("--kind", default="dm", choices=["dm", "group"])
    p.add_argument("--brain", default="local", choices=["stub", "local"])
    p.add_argument("--ask")
    a = p.parse_args()
    cfg = C.load(a.config)

    if a.check:
        print(json.dumps({"enabled": cfg.enabled, "answer": cfg.answer, "live": cfg.live, "name": cfg.name,
                          "callers": len(cfg.callers), "groups": sorted(cfg.groups), "stt": bool(cfg.stt_url),
                          "llm": bool(cfg.llm), "tts": "say" if cfg.tts == "say" else "url",
                          "in_device": cfg.in_device, "out_device": cfg.out_device}, indent=1))
        return 0
    if a.ask is not None:
        from catanbot.facetime import lib as L
        from catanbot.facetime.brain import Brain
        L.set_name(cfg.name)
        text, action = Brain(cfg).reply(a.ask, a.kind)
        print(json.dumps({"reply": text, "action": action}))
        return 0

    from catanbot.facetime import daemon as D
    if a.mic_check:
        return D.mic_check(cfg, LOG_DIR)
    lvl = a.level if a.level else float(os.environ.get("FACETIME_LEVEL_S") or 0)
    if lvl > 0:
        return D.level_meter(cfg, lvl, LOG_DIR)
    if a.input:
        from catanbot.facetime.brain import Brain
        brain = D.StubBrain() if a.brain == "stub" else Brain(cfg)
        print(json.dumps(D.synthetic(cfg, a.input, a.out, a.kind, brain), indent=1))
        return 0
    if not cfg.enabled:
        print("facetime: no \"facetime\" block in", a.config, "- the add-on is off")
        return 0
    if sys.platform != "darwin":
        sys.exit("facetime: the daemon needs macOS (FaceTime, BlackHole)")
    D.Daemon(cfg).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
