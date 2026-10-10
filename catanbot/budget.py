"""CPU budget for training and evaluation on the server (owner 2026-10-04): use 1/4 to 1/2 of
the cores at low priority and never push the machine to its limits."""
import os
import time
from multiprocessing import cpu_count

NICE = 15
BUSY = 0.6          # wait while the OTHER work alone is above 60% of the cores


def workers() -> int:
    n = cpu_count()
    lo, hi = max(1, n // 4), max(1, n // 2)
    want = int(os.environ.get("CATAN_WORKERS") or round(n * 0.3))
    return min(hi, max(lo, want))


def init_worker():
    os.nice(NICE)


def wait_for_quiet(log=print, max_wait=1800, own=0):
    """Don't start a batch while the server is busy with its other work (VM, Docker, API).
    own: our own workers, still in the 1-min load average right after a batch."""
    waited = 0
    while os.getloadavg()[0] - own > BUSY * cpu_count() and waited < max_wait:
        time.sleep(60)
        waited += 60
    if waited:
        log(f"waited {waited // 60} min for the server load to drop")
