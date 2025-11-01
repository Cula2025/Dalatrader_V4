# run_headless.py
import sys
from pages.opt_entry import run_headless

def main(argv):
    if len(argv) < 5:
        print("Usage: python run_headless.py <TICKER> <SIMS> <START> <END> <RESUME:0|1> [RUN_TAG] [WORKERS]")
        sys.exit(1)

    ticker   = argv[0]
    sims     = int(argv[1])
    start    = argv[2]     # ISO: YYYY-MM-DD
    end      = argv[3]     # ISO: YYYY-MM-DD
    resume   = bool(int(argv[4]))
    run_tag  = argv[5] if len(argv) > 5 else "batch"
    workers  = int(argv[6]) if len(argv) > 6 else 1

    res = run_headless(
        ticker=ticker,
        start=start,
        end=end,
        sims=sims,
        run_tag=run_tag,
        workers=workers,
        chunk_size=100,
        topk=1,
        tri_obv="Allowed",
        tri_chop="Allowed",
        tri_hma="Allowed",
        locks={},
        disable_gates=False,
        resume=resume,
    )
    print(f"DONE {ticker}  TR={res.get('best',{}).get('TR')}  Trades={res.get('best',{}).get('Trades')}")
    print(f"Profile: {res.get('profile_path')}")
    print(f"Log:     {res.get('log_path')}")

if __name__ == "__main__":
    main(sys.argv[1:])

