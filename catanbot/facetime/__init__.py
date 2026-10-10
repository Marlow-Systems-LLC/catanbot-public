"""Optional FaceTime voice add-on: answer allow-listed FaceTime calls and talk about the game.

Off by default. Nothing here runs unless scripts/facetime_voice.py is started and data/config.json
has a "facetime" block with "answer": true. See docs/facetime.md. Only lib.py, config.py and
brain.py are imported by tests; daemon.py imports audio libraries lazily, inside functions.
"""
