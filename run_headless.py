# run_headless.py
import sys, os, datetime, importlib.util, pathlib

ROOT = pathlib.Path(__file__).resolve().parent
os.environ.setdefault("PYTHONPATH", str(ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- ladda pages/10_optimizer.py som modul och registrera i sys.modules ---
opt_path = ROOT / "pages" / "10_optimizer.py"  # OBS: rätt filnamn (10_optimizer.py)
spec = importlib.util.spec_from_file_location("optmod", opt_path)
optmod = importlib.util.module_from_spec(spec)
# VIKTIGT: registrera innan exec_module så dataclasses hittar modulen
sys.modules[spec.name] = optmod
spec.loader.exec_module(optmod)

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

    # konvertera till date-objekt om din funktion kräver det
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

