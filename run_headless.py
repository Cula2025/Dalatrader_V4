# run_headless.py
import sys, os, datetime, importlib, pathlib

ROOT = pathlib.Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYTHONPATH", str(ROOT))

# gör pages till paket (om någon kör skriptet från annan cwd)
if not (ROOT / "pages" / "__init__.py").exists():
    (ROOT / "pages" / "__init__.py").write_text("", encoding="utf-8")

# Importera modulen som ett riktigt paket → dataclasses blir nöjd
optmod = importlib.import_module("pages.10_optimizer")

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_headless.py <TICKER> [SIMS] [START] [END] [PROCS]")
        sys.exit(1)

    ticker = sys.argv[1]
    sims   = int(sys.argv[2]) if len(sys.argv) > 2 else 50000

    today  = datetime.date.today()
    start_default = (today.replace(day=1) - datetime.timedelta(days=365*5)).isoformat()
    start  = sys.argv[3] if len(sys.argv) > 3 else start_default
    end    = sys.argv[4] if len(sys.argv) > 4 else today.isoformat()
    procs  = int(sys.argv[5]) if len(sys.argv) > 5 else (os.cpu_count() or 1)

    start_d = datetime.date.fromisoformat(start)
    end_d   = datetime.date.fromisoformat(end)

    res = optmod.run_optimizer_server_for_ticker(
        ticker=ticker,
        start=start_d,
        end=end_d,
        sims=int(sims),
        procs=int(procs),
        resume=False,
    )
    print("DONE", ticker, res if res is not None else "")

if __name__ == "__main__":
    main()
