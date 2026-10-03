import time

T0 = time.time()


def log(msg: str) -> None:
    """Print a progress line with the time since the start."""
    m, sec = divmod(int(time.time() - T0), 60)
    print(f"[{m:02d}:{sec:02d}] {msg}", flush=True)
