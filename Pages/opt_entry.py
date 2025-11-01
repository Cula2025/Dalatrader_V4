# pages/opt_entry.py
from datetime import date
from importlib import import_module

class _Slot:
    def markdown(self, *a, **k): pass
    def write(self, *a, **k): pass
    def empty(self, *a, **k): pass

def run_headless(
    ticker: str,
    start: str | date,
    end: str | date,
    sims: int,
    *,
    seed: int | None = None,
    run_tag: str = "batch",
    workers: int = 1,
    chunk_size: int = 500,
    topk: int = 1,
    tri_obv: str = "Allowed",
    tri_chop: str = "Allowed",
    tri_hma: str = "Allowed",
    locks: dict | None = None,
    disable_gates: bool = True,
    resume: bool = False,
):
    """Headless anrop utan Streamlit-slots. Returnerar dict med resultat."""
    if isinstance(start, str): start = date.fromisoformat(start)
    if isinstance(end, str):   end   = date.fromisoformat(end)
    if locks is None: locks = {}

    mod = import_module("pages.10_optimizer")
    fn  = getattr(mod, "run_optimizer_server_for_ticker")

    return fn(
        ticker, start, end, sims, seed, run_tag,
        workers, chunk_size, topk,
        tri_obv, tri_chop, tri_hma,
        locks, disable_gates, resume,
        _Slot(), _Slot(), _Slot()
    )
