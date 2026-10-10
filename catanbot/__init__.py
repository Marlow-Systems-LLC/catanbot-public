"""Marlow Catan bot (Catan Universe, The Great Canal)."""
import re as _re
import subprocess as _sp
from pathlib import Path as _Path

__version__ = "0.4.3"


def _commit() -> str:
    """Short git sha of this checkout, '' when it is not one (an archive copy) or git is missing."""
    try:
        out = _sp.run(["git", "rev-parse", "--short", "HEAD"], cwd=_Path(__file__).resolve().parent.parent,
                      capture_output=True, text=True, timeout=3)
    except Exception:                             # noqa: BLE001 -- the version must never stop a start
        return ""
    sha = out.stdout.strip()
    return sha if out.returncode == 0 and _re.fullmatch(r"[0-9a-f]{4,40}", sha) else ""


# owner 2026-10-09: marlow says what it runs; read once at start, so a later checkout does not
# change what a running process reports
COMMIT = _commit()
