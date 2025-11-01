# pages/10_Optimizer_M1.py
from __future__ import annotations

import os, sys, json, math, time, glob, argparse, hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import concurrent.futures as cf
from importlib.machinery import SourceFileLoader as _SrcLoader
import importlib.util as _imputil
from queue import Queue, Empty  # <-- live-status kö

# =========================
# Konstanter / defaults
# =========================
DEFAULT_TRIALS       = 50_000
DEFAULT_TOPK         = 100
DEFAULT_CHUNK        = 1_000
DEFAULT_PAR_TICKERS  = os.cpu_count() or 8
P_USE_DEFAULT        = {"obv": 0.50, "chop": 0.33, "hma": 0.50}
TRI_STATES           = ["Off", "Allowed", "Forced On"]

# =========================
# CLI filter-override (valfritt)
# =========================
def _parse_cli_filters(argv: List[str]) -> Dict[str, str] | None:
    try:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--filters", type=str, default=None)
        ns, _ = parser.parse_known_args(argv)
        if not ns.filters:
            return None
        raw = ns.filters.strip()
        out: Dict[str, str] = {}
        for part in raw.split(","):
            if not part.strip(): continue
            k, v = part.split("=", 1)
            k, v = k.strip().lower(), v.strip().lower()
            if k not in ("obv", "chop", "hma"): continue
            if v in ("off", "allowed", "forced", "forcedon", "force", "on"):
                if v == "off":
                    out[k] = "Off"
                elif v == "allowed":
                    out[k] = "Allowed"
                else:
                    out[k] = "Forced On"
        return out or None
    except Exception:
        return None

CLI_FILTERS = _parse_cli_filters(sys.argv)

# =========================
# Robust import av data_loader
# =========================
def _import_data_loader():
    """
    Robust importer för data_loader:
    1) försök 'app.data_loader'
    2) försök 'app_v4.data_loader'
    3) ladda direkt från filsystemet om paketvägar saknas
    """
    # 1) package import: app
    try:
        from app import data_loader as dl  # type: ignore
        return dl
    except Exception:
        pass
    # 2) package import: app_v4
    try:
        from app_v4 import data_loader as dl  # type: ignore
        return dl
    except Exception:
        pass
    # 3) filesystem fallbacks
    try:
        here = os.path.abspath(os.path.dirname(__file__))
    except NameError:
        here = os.getcwd()
    proj = os.path.abspath(os.path.join(here, ".."))
    candidates = [
        os.path.join(proj, "app", "data_loader.py"),
        os.path.join(proj, "app_v4", "data_loader.py"),
        os.path.join(os.getcwd(), "app", "data_loader.py"),
        os.path.join(os.getcwd(), "app_v4", "data_loader.py"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            modname = f"_dl_{hash(path) & 0xFFFFFFFF:x}"
            spec = _imputil.spec_from_loader(modname, _SrcLoader(modname, path))
            if spec is None: 
                continue
            mod = _imputil.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(mod)  # type: ignore[attr-defined]
            return mod  # type: ignore[return-value]
    raise ImportError("Could not locate data_loader in app/ or app_v4/ or filesystem.")

# =========================
# Loader
# =========================
def load_ohlcv(ticker: str, dfrom: str, dto: str) -> pd.DataFrame:
    try:
        dl = _import_data_loader()
        df = None
        for name in ("load_ohlcv", "get_ohlcv", "fetch_ohlcv", "get_prices_eod", "load_prices", "get_eod_ohlcv"):
            if hasattr(dl, name):
                fn = getattr(dl, name)
                try:
                    if name == "get_eod_ohlcv":
                        df = fn(ticker=ticker, start=dfrom, end=dto)
                    elif "dfrom" in fn.__code__.co_varnames:
                        df = fn(ticker, dfrom=dfrom, dto=dto)
                    else:
                        df = fn(ticker, dfrom, dto)  # type: ignore[arg-type]
                except TypeError:
                    # sista försök med namngivna argument
                    df = fn(ticker=ticker, dfrom=dfrom, dto=dto)
                break
        if df is None:
            raise RuntimeError("Hittar ingen loader-funktion i data_loader")

        cols = {c.lower(): c for c in df.columns}
        need = ["open", "high", "low", "close", "volume"]
        ren = {cols[c]: c for c in need if c in cols and cols[c] != c}
        if ren:
            df = df.rename(columns=ren)
        df = df[need].sort_index()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df.index = df.index.tz_localize(None)
        if len(df) == 0:
            raise RuntimeError("Tom prislista från Börsdata.")
        return df
    except Exception as e:
        raise RuntimeError(f"Kunde inte ladda data för {ticker}. Fel: {e}")

# =========================
# Indikatorer
# =========================
def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    close = close.astype(float)
    vol = volume.fillna(0).astype(float)
    d = close.diff()
    sign = (d.gt(0).astype(int) - d.lt(0).astype(int))
    return (sign * vol).fillna(0).cumsum()

def _tr(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    a = (high - low).abs()
    b = (high - prev_close).abs()
    c = (low - prev_close).abs()
    return pd.concat([a, b, c], axis=1).max(axis=1)

def _chop(high: pd.Series, low: pd.Series, close: pd.Series, w: int) -> pd.Series:
    tr = _tr(high, low, close)
    sum_tr = tr.rolling(w, min_periods=w).sum()
    hh = high.rolling(w, min_periods=w).max()
    ll = low .rolling(w, min_periods=w).min()
    rng = (hh - ll).replace(0.0, np.nan)
    return 100.0 * (np.log10(sum_tr / rng)) / np.log10(w)

def _wma(s: pd.Series, period: int) -> pd.Series:
    if period <= 1: return s.rolling(1).mean()
    w = np.arange(1, period+1, dtype=float)
    def _calc(x): return float(np.dot(x, w) / w.sum())
    return s.rolling(period, min_periods=period).apply(_calc, raw=True)

def _hma(s: pd.Series, period: int) -> pd.Series:
    p = int(period)
    if p < 2: return s.rolling(1).mean()
    half = max(1, p // 2)
    sqrtp = max(1, int(np.sqrt(p)))
    wma1 = _wma(s, half)
    wma2 = _wma(s, p)
    diff = (2.0 * wma1) - wma2
    return _wma(diff, sqrtp)

# =========================
# Precompute
# =========================
@dataclass
class Precomputed:
    close: pd.Series
    high: pd.Series
    low: pd.Series
    volume: pd.Series
    HH: Dict[int, pd.Series]
    LL: Dict[int, pd.Series]
    CH: Dict[int, pd.Series]
    HMA: Dict[int, pd.Series]
    OBV: pd.Series

def precompute_all(df: pd.DataFrame) -> Precomputed:
    close = df["close"].astype(float)
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    vol   = df["volume"].astype(float)
    HH: Dict[int, pd.Series] = {N: high.rolling(N, min_periods=N).max().shift(1) for N in range(20, 121)}
    LL: Dict[int, pd.Series] = {M: low .rolling(M, min_periods=M).min().shift(1) for M in range(5, 61)}
    CH: Dict[int, pd.Series] = {w: _chop(high, low, close, w=w).shift(1) for w in range(14, 29)}
    HMA: Dict[int, pd.Series] = {w: _hma(close, w).shift(1) for w in range(20, 81)}
    obv = _obv(close, vol)
    return Precomputed(close, high, low, vol, HH, LL, CH, HMA, obv)

# =========================
# Backtest (Trend)
# =========================
@dataclass
class Trade:
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    return_pct: float
    holding_days: int
    reason: str

@dataclass
class BTResult:
    equity: pd.Series
    trades: List[Trade]
    metrics: Dict[str, float]

def run_trend_bt(pc: Precomputed, N: int, M: int,
                 gate_obv: Optional[Tuple[bool, int]],
                 gate_chop: Optional[Tuple[bool, int, float]],
                 gate_hma: Optional[Tuple[bool, int]],
                 force_flat_last: bool = True) -> BTResult:
    close = pc.close
    HH = pc.HH[N]
    LL = pc.LL[M]

    gate_series = pd.Series(True, index=close.index)

    if gate_obv and gate_obv[0]:
        lb = int(gate_obv[1])
        obv_prev = pc.OBV.shift(1)
        obv_prev_lb = pc.OBV.shift(1 + lb)
        ok = (obv_prev.notna() & obv_prev_lb.notna() & (obv_prev > obv_prev_lb))
        gate_series &= ok.fillna(False)

    if gate_chop and gate_chop[0]:
        w, cmax = int(gate_chop[1]), float(gate_chop[2])
        ch = pc.CH[w]
        ok = (ch.notna() & (ch < cmax))
        gate_series &= ok.fillna(False)

    if gate_hma and gate_hma[0]:
        w = int(gate_hma[1])
        h = pc.HMA[w]
        ok = (close.shift(1).notna() & h.notna() & (close.shift(1) > h))
        gate_series &= ok.fillna(False)

    in_pos = False
    entry_px = np.nan
    entry_date: Optional[pd.Timestamp] = None
    trades: List[Trade] = []
    eq = pd.Series(index=close.index, dtype=float); eq.iloc[0] = 1.0

    for i, t in enumerate(close.index):
        px = float(close.iloc[i])
        eq.iloc[i] = eq.iloc[i-1] if i > 0 else 1.0

        buy_sig  = (not in_pos) and (not pd.isna(HH.iloc[i])) and (px > float(HH.iloc[i])) and bool(gate_series.iloc[i])
        sell_sig = in_pos and (not pd.isna(LL.iloc[i])) and (px < float(LL.iloc[i]))

        did_sell = False
        if sell_sig and in_pos and entry_date is not None:
            ll_val = float(LL.iloc[i])
            reason = f"EXIT: Donchian exit (n_exit={M}) | Close={px:g} < LL{M}={ll_val:g}"
            ret = (px/entry_px)-1.0
            trades.append(Trade(entry_date,float(entry_px),t,px,ret*100.0,(t-entry_date).days,reason))
            in_pos, entry_px, entry_date = False, np.nan, None
            did_sell = True

        if (not did_sell) and buy_sig:
            in_pos = True
            entry_px = px
            entry_date = t

        if in_pos and not np.isnan(entry_px) and i > 0:
            eq.iloc[i] = eq.iloc[i-1] * (px / float(close.iloc[i-1]))

    if force_flat_last and in_pos and entry_date is not None:
        last_t, last_px = close.index[-1], float(close.iloc[-1])
        ret = (last_px/entry_px)-1.0
        trades.append(Trade(entry_date,float(entry_px),last_t,last_px,ret*100.0,(last_t-entry_date).days,
                            f"EXIT: FORCE_FLAT | Close={last_px:g} (period end)"))

    eq = eq.ffill().fillna(1.0)
    rets = eq.pct_change().fillna(0.0)
    TR = float(eq.iloc[-1])
    n_days = max(1, (eq.index[-1] - eq.index[0]).days)
    years = n_days / 365.25
    CAGR = float(TR ** (1/years) - 1) if years > 0 and TR > 0 else 0.0
    SharpeD = float((rets.mean() / (rets.std() + 1e-12)) * math.sqrt(252)) if rets.std() > 0 else 0.0
    roll_max = eq.cummax(); maxdd = float(((eq/roll_max)-1.0).min())
    d2f = float("nan")
    if trades: d2f = float((trades[0].entry_date - eq.index[0]).days)
    metrics = {"TR":TR,"CAGR":CAGR,"SharpeD":SharpeD,"MaxDD":maxdd,"Trades":float(len(trades)),"DaysToFirst":d2f}
    return BTResult(eq, trades, metrics)

# =========================
# Sampling (tri-state)
# =========================
@dataclass
class SampledParams:
    N: int
    M: int
    filters: Dict[str, Any]

def _sample_bool_from_tristate(rng: np.random.Generator, tristate: str, p_default: float) -> bool:
    if tristate == "Off": return False
    if tristate == "Forced On": return True
    return bool(rng.random() < float(p_default))  # Allowed

def sample_params(rng: np.random.Generator,
                  tri_obv: str, tri_chop: str, tri_hma: str,
                  locks: Dict[str, Any]) -> SampledParams:
    if int(locks.get("N_lock", 0)) > 0:
        N = int(locks["N_lock"]); N_source = "locked"
    else:
        N = int(rng.integers(20, 121)); N_source = "sampled"

    if int(locks.get("M_lock", 0)) > 0:
        M = int(locks["M_lock"]); M_source = "locked"
    else:
        M = int(rng.integers(5, 61)); M_source = "sampled"

    filters: Dict[str, Any] = {"N_source": N_source, "M_source": M_source}

    obv_on = _sample_bool_from_tristate(rng, tri_obv, P_USE_DEFAULT["obv"])
    if obv_on:
        if int(locks.get("obv_lb_lock", 0)) > 0:
            lb = int(locks["obv_lb_lock"]); src = "locked"
        else:
            lb = int(rng.integers(10, 41)); src = "sampled"
        filters["obv"] = {"on": True, "lb": lb, "source": src}
    else:
        filters["obv"] = {"on": False}

    chop_on = _sample_bool_from_tristate(rng, tri_chop, P_USE_DEFAULT["chop"])
    if chop_on:
        if int(locks.get("chop_w_lock", 0)) > 0:
            w = int(locks["chop_w_lock"]); wsrc = "locked"
        else:
            w = int(rng.integers(14, 29)); wsrc = "sampled"
        if float(locks.get("chop_max_lock", 0.0)) > 0:
            cmax = float(locks["chop_max_lock"]); msrc = "locked"
        else:
            cmax = float(rng.uniform(56.0, 72.0)); msrc = "sampled"
        filters["chop"] = {"on": True, "w": w, "max": cmax, "source": {"w": wsrc, "max": msrc}}
    else:
        filters["chop"] = {"on": False}

    hma_on = _sample_bool_from_tristate(rng, tri_hma, P_USE_DEFAULT["hma"])
    if hma_on:
        if int(locks.get("hma_w_lock", 0)) > 0:
            w = int(locks["hma_w_lock"]); src = "locked"
        else:
            w = int(rng.integers(20, 81)); src = "sampled"
        filters["hma"] = {"on": True, "w": w, "source": src}
    else:
        filters["hma"] = {"on": False}

    return SampledParams(N=N, M=M, filters=filters)

# =========================
# Gates (togglebar; oförändrad logik när OFF, alltid True när ON)
# =========================
def passes_gates(metrics: Dict[str, float], *, disable_gates: bool) -> Tuple[bool, str]:
    if disable_gates:
        return True, "ok"
    # Bakåtkompatibla gates (används bara när toggle=OFF)
    trades = int(metrics.get("Trades", 0))
    d2f = metrics.get("DaysToFirst", float("nan"))
    maxdd = float(metrics.get("MaxDD", -1.0))
    if trades < 6: return False, "trades"
    if not np.isnan(d2f) and d2f > 90: return False, "first"
    if maxdd <= -0.55: return False, "dd"
    return True, "ok"

# =========================
# Kandidater
# =========================
@dataclass
class Candidate:
    TR: float
    SharpeD: float
    MaxDD: float
    Trades: int
    DaysToFirst: float
    N: int
    M: int
    filters: Dict[str, Any]

def candidate_from(bt: BTResult, N: int, M: int, filters: Dict[str, Any]) -> Candidate:
    m = bt.metrics
    return Candidate(
        TR=float(m["TR"]),
        SharpeD=float(m["SharpeD"]),
        MaxDD=float(m["MaxDD"]),
        Trades=int(m["Trades"]),
        DaysToFirst=float(m["DaysToFirst"]),
        N=N, M=M, filters=filters or {}
    )

# =========================
# Worker (med live-best-uppdateringar via Queue)
# =========================
def _simulate_chunk_worker(args) -> Dict[str, Any]:
    (seed_child, chunk_idx, sims_in_chunk, pc, tri_states, locks, disable_gates, live_q, ticker) = args
    rng = np.random.default_rng(seed_child)
    tri_obv, tri_chop, tri_hma = tri_states

    top: List[Candidate] = []
    fail_first = fail_trades = fail_dd = 0
    local_best_tr = -1e9  # för live uppdatering

    for i in range(int(sims_in_chunk)):
        sp = sample_params(rng, tri_obv, tri_chop, tri_hma, locks)
        gate_obv  = (sp.filters["obv"]["on"],  sp.filters["obv"].get("lb", 0))  if sp.filters["obv"]["on"]  else None
        gate_chop = (sp.filters["chop"]["on"], sp.filters["chop"].get("w", 0), sp.filters["chop"].get("max", 0.0)) if sp.filters["chop"]["on"] else None
        gate_hma  = (sp.filters["hma"]["on"],  sp.filters["hma"].get("w", 0))  if sp.filters["hma"]["on"]  else None

        bt = run_trend_bt(pc, sp.N, sp.M, gate_obv, gate_chop, gate_hma, force_flat_last=True)
        ok, reason = passes_gates(bt.metrics, disable_gates=bool(disable_gates))
        if not ok:
            if reason == "first":   fail_first += 1
            elif reason == "trades": fail_trades += 1
            elif reason == "dd":     fail_dd += 1
            continue

        cand = candidate_from(bt, sp.N, sp.M, sp.filters)
        top.append(cand)

        # Live-uppdatering när ny lokal topp hittas (huvudtråd jämför mot global)
        if cand.TR > local_best_tr:
            local_best_tr = cand.TR
            if live_q is not None:
                try:
                    live_q.put_nowait({
                        "type": "best",
                        "ticker": ticker,
                        "TR": cand.TR,
                        "SharpeD": cand.SharpeD,
                        "MaxDD": cand.MaxDD,
                    })
                except Exception:
                    pass

    # Sortering: TR primär (ties → SharpeD → MaxDD)
    top.sort(key=lambda c: (c.TR, c.SharpeD, c.MaxDD), reverse=True)
    top = top[: (DEFAULT_TOPK*3)]
    return {
        "chunk": int(chunk_idx),
        "top_local": top,
        "fails": {"first": int(fail_first), "trades": int(fail_trades), "dd": int(fail_dd)},
    }

# =========================
# Determinism
# =========================
def _locks_digest(tri_obv: str, tri_chop: str, tri_hma: str, locks: Dict[str, Any]) -> int:
    payload = {"tri": {"obv": tri_obv, "chop": tri_chop, "hma": tri_hma}, "locks": locks}
    s = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    h = hashlib.blake2b(s.encode("utf-8"), digest_size=8).hexdigest()
    return int(h, 16)

# =========================
# Optimizer-kärna (med text-status i UI + live-bästa)
# =========================
def run_optimizer_server_for_ticker(
    ticker: str,
    start: date,
    end: date,
    total_sims: int,
    seed: Optional[int],
    run_tag: str,
    workers: int,
    chunk_size: int,
    topk: int,
    tri_obv: str, tri_chop: str, tri_hma: str,
    locks: Dict[str, Any],
    disable_gates: bool,
    resume: bool,
    ui_best_slot, ui_sims_slot, ui_ticker_slot,
) -> Dict[str, Any]:
    base_dir = os.path.join("runs", run_tag)
    ck_dir   = os.path.join(base_dir, "checkpoints")
    log_dir  = os.path.join(base_dir, "logs")
    prof_dir = os.path.join(base_dir, "profiles")
    os.makedirs(ck_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(prof_dir, exist_ok=True)

    period = f"{start.isoformat()}→{end.isoformat()}"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"optimizer_run_{run_id}.log")

    # Körningsheader i logg
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            f"[start] {datetime.now().isoformat()} | ticker={ticker} | period={period} | "
            f"sims={total_sims} | workers={workers} | chunk={chunk_size} | run_tag={run_tag} | "
            f"disable_gates={disable_gates} | seed={seed}\n"
        )

    df = load_ohlcv(ticker, start.isoformat(), end.isoformat())
    pc = precompute_all(df)

    start_chunk = 0
    if resume:
        existing = sorted(glob.glob(os.path.join(ck_dir, f"{ticker.replace(' ','_')}_topk_chunk*.parquet")))
        if not existing:
            existing = sorted(glob.glob(os.path.join(ck_dir, f"{ticker.replace(' ','_')}_topk_chunk*.json")))
        start_chunk = len(existing)

    total_sims  = int(total_sims)
    done_sims   = min(start_chunk * int(chunk_size), total_sims)
    next_sims_tick = ((done_sims // 1000) + 1) * 1000  # text-uppdatering var 1000 sims

    cfg_digest = _locks_digest(tri_obv, tri_chop, tri_hma, locks)
    base_seed = int(seed) if (seed is not None) else int(np.random.SeedSequence().generate_state(1)[0])

    def seed_for(chunk_idx: int, worker_idx: int) -> int:
        ss = np.random.SeedSequence([base_seed, int(cfg_digest & 0xFFFFFFFF), chunk_idx, worker_idx])
        return int(ss.generate_state(1)[0])

    remaining = max(0, int(total_sims) - start_chunk * int(chunk_size))
    n_chunks  = (remaining + chunk_size - 1) // chunk_size if remaining > 0 else 0

    global_top: List[Candidate] = []
    global_best_tr = -1e9  # för snabb jämförelse / UI
    totals = {"pass": 0, "fail_first": 0, "fail_trades": 0, "fail_dd": 0}
    tri_states = (tri_obv, tri_chop, tri_hma)
    t0 = time.time()

    def write_checkpoint(chunk_idx: int, cand_list: List[Candidate]):
        df_ck = pd.DataFrame([{
            "TR": c.TR, "SharpeD": c.SharpeD, "MaxDD": c.MaxDD, "Trades": c.Trades,
            "DaysToFirst": c.DaysToFirst, "N": c.N, "M": c.M, "filters": json.dumps(c.filters or {}, ensure_ascii=False)
        } for c in cand_list])
        base = os.path.join(ck_dir, f"{ticker.replace(' ','_')}_topk_chunk{chunk_idx:03d}")
        try:
            df_ck.to_parquet(base + ".parquet", index=False)
        except Exception:
            df_ck.to_json(base + ".json", orient="records", force_ascii=False, indent=2)

    def append_log(line: str):
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line.rstrip("\n") + "\n")

    # Initiala textytor
    ui_best_slot.markdown("**Bästa TR hittills:** –")
    ui_sims_slot.markdown(f"**Simuleringar körda:** {done_sims}/{total_sims}")
    ui_ticker_slot.markdown(f"**Tickerstatus:** {ticker}: {done_sims}/{total_sims} (best TR –)")

    live_q: Queue = Queue(maxsize=1000)

    # Körning (med live-kö som triggar uppdatering så fort en bättre kandidat hittas)
    for rel_chunk in range(n_chunks):
        chunk_idx = start_chunk + rel_chunk
        sims_this = min(chunk_size, remaining - rel_chunk*chunk_size)
        if sims_this <= 0: break

        per_worker = [sims_this // workers] * workers
        for i in range(sims_this % workers):
            per_worker[i] += 1

        args_list = []
        for w_idx, sims_w in enumerate(per_worker):
            if sims_w == 0: continue
            seed_child = int(seed_for(chunk_idx, w_idx))
            args_list.append((seed_child, chunk_idx, sims_w, pc, tri_states, locks, bool(disable_gates), live_q, ticker))

        pass_count = 0
        fail_first = fail_trades = fail_dd = 0
        local_top_agg: List[Candidate] = []

        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_simulate_chunk_worker, a) for a in args_list]
            # Polla med kort timeout så att vi kan hämta live-bästa från kö
            remaining_futs = set(futs)
            while remaining_futs:
                done, remaining_futs = cf.wait(remaining_futs, timeout=0.1, return_when=cf.FIRST_COMPLETED)
                # Drain queue – uppdatera UI direkt om vi får ett bättre TR
                while True:
                    try:
                        msg = live_q.get_nowait()
                    except Empty:
                        break
                    if msg.get("type") == "best":
                        tr = float(msg["TR"])
                        if tr > global_best_tr:
                            global_best_tr = tr
                            sd = float(msg.get("SharpeD", 0.0))
                            dd = float(msg.get("MaxDD", 0.0))
                            ui_best_slot.markdown(f"**Bästa TR hittills:** {tr:.3f} | SharpeD {sd:.2f} | MaxDD {dd:.2f}")
                            ui_ticker_slot.markdown(f"**Tickerstatus:** {ticker}: {done_sims}/{total_sims} (best TR {tr:.3f})")

            # Samla resultat från workers
            for res in [f.result() for f in futs]:
                local_top_agg.extend(res["top_local"])
                fails = res["fails"]
                fail_first  += int(fails["first"])
                fail_trades += int(fails["trades"])
                fail_dd     += int(fails["dd"])

        pass_count = len(local_top_agg)
        totals["pass"] += pass_count
        totals["fail_first"]  += fail_first
        totals["fail_trades"] += fail_trades
        totals["fail_dd"]     += fail_dd

        global_top.extend(local_top_agg)
        global_top.sort(key=lambda c: (c.TR, c.SharpeD, c.MaxDD), reverse=True)
        if len(global_top) > topk:
            global_top = global_top[:topk]

        # Hämta nuvarande bästa (kan vara uppdaterad via live-kö redan)
        best = global_top[0] if global_top else None
        if best and best.TR > global_best_tr:
            global_best_tr = best.TR
            ui_best_slot.markdown(f"**Bästa TR hittills:** {best.TR:.3f} | SharpeD {best.SharpeD:.2f} | MaxDD {best.MaxDD:.2f}")
            ui_ticker_slot.markdown(f"**Tickerstatus:** {ticker}: {done_sims}/{total_sims} (best TR {best.TR:.3f})")

        write_checkpoint(chunk_idx, global_top)
        append_log(
            f"chunk={chunk_idx}, sims={sims_this}, pass={pass_count}, "
            f"fail_first={fail_first}, fail_trades={fail_trades}, fail_dd={fail_dd}, "
            f"best_TR={(best.TR if best else 'NA')}, "
            f"params={{N={best.N if best else 'NA'},M={best.M if best else 'NA'}}}, "
            f"filters={json.dumps(best.filters if best else {}, ensure_ascii=False)}, "
            f"disable_gates={disable_gates}"
        )

        # Textstatus – uppdatera endast var 1000 sims (globalt räknat per ticker)
        done_sims = min(total_sims, done_sims + sims_this)
        if (done_sims >= next_sims_tick) or (done_sims == total_sims):
            ui_sims_slot.markdown(f"**Simuleringar körda:** {done_sims}/{total_sims}")
            if best:
                ui_ticker_slot.markdown(f"**Tickerstatus:** {ticker}: {done_sims}/{total_sims} (best TR {best.TR:.3f})")
            else:
                ui_ticker_slot.markdown(f"**Tickerstatus:** {ticker}: {done_sims}/{total_sims}")

            next_sims_tick = ((done_sims // 1000) + 1) * 1000

    out_profile = None
    if global_top:
        best = global_top[0]
        profile = {
            "version":"v4",
            "ticker": ticker,
            "model": "M2a-ServerTrend",
            "generated": datetime.now().isoformat(timespec="seconds"),
            "selected":{
                "archetype":"trend",
                "TR":best.TR,
                "params":{"breakout_lookback":best.N,"exit_lookback":best.M},
                "filters": best.filters or {}
            },
            "runner_up": None,
            "run":{
                "seed": (None if seed is None else int(seed)),
                "trials": int(total_sims),
                "period": period,
                "filters_mode":{"obv": tri_obv, "chop": tri_chop, "hma": tri_hma},
                "disable_gates": bool(disable_gates)
            },
            "notes":"MAX-TR; ties→SharpeD→MaxDD; live best updates via queue; tri-state affects entry only"
        }
        outp = os.path.join(prof_dir, f"{ticker.replace(' ','_')}.profile.json")
        with open(outp, "w", encoding="utf-8") as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)
        out_profile = outp

        top_list = [{
            "TR": c.TR, "SharpeD": c.SharpeD, "MaxDD": c.MaxDD, "Trades": c.Trades, "DaysToFirst": c.DaysToFirst,
            "N": c.N, "M": c.M, "filters": c.filters or {}
        } for c in global_top]
        with open(os.path.join(prof_dir, f"{ticker.replace(' ','_')}_topk.json"), "w", encoding="utf-8") as f:
            json.dump(top_list, f, ensure_ascii=False, indent=2)

    elapsed = int(time.time() - t0)
    mm, ss = divmod(elapsed, 60)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[done] {datetime.now().isoformat()} | ticker={ticker} | elapsed={mm:02d}:{ss:02d} | best_TR={(global_top[0].TR if global_top else 'NA')}\n")

    return {
        "run_id": run_id,
        "ticker": ticker,
        "period": period,
        "seed": seed,
        "trials": total_sims,
        "best": (vars(global_top[0]) if global_top else None),
        "log_path": log_path,
        "profile_path": out_profile,
        "totals": totals,
        "topk_len": len(global_top),
        "base_dir": base_dir,
    }

# =========================
# UI
# =========================
st.set_page_config(page_title="V4 Optimizer – Server (Trend, textstatus)", layout="wide")
st.title("V4 Optimizer – Server (Trend)")

today = datetime.now().date()
start_default = today - timedelta(days=365*5)

# --- Körinställningar (Server)
st.subheader("Körinställningar (Server)")
colA, colB, colC, colD = st.columns(4)
with colA:
    trials = st.number_input("Antal simuleringar / instrument", value=DEFAULT_TRIALS, min_value=1, step=1000)
with colB:
    workers = st.number_input("Antal kärnor / workers", value=DEFAULT_PAR_TICKERS, min_value=1, max_value=256, step=1)
with colC:
    chunk_size = st.number_input("Chunk / checkpoint-storlek", value=DEFAULT_CHUNK, min_value=100, step=100)
with colD:
    run_tag = st.text_input("Run tag", value=datetime.now().strftime("srv_%Y%m%d"))

colE, colF, colG = st.columns(3)
with colE:
    start_d = st.date_input("Från-datum (ISO)", value=start_default)
with colF:
    end_d   = st.date_input("Till-datum (ISO)", value=today)
with colG:
    resume  = st.checkbox("Resume (fortsätt avbruten körning)", value=False)

seed_str = st.text_input("Seed (valfritt heltal; tomt = RNG)", value="1234")
seed_val: Optional[int] = None
try:
    seed_val = int(seed_str) if seed_str.strip() else None
except Exception:
    st.warning("Seed måste vara heltal eller tomt – ignorerar och använder RNG.")
    seed_val = None

st.caption(f"Utdata-mappar: runs/{run_tag}/profiles, runs/{run_tag}/logs")

# --- Läsa universe/universe.txt
st.subheader("Universum")
if "batch_raw" not in st.session_state:
    st.session_state["batch_raw"] = "ABB\nAAK\nHM B"

colU1, colU2 = st.columns([1,2])
with colU1:
    if st.button("Läs universumfil"):
        uni_path = os.path.join("universe", "universe.txt")
        try:
            with open(uni_path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f.readlines()]
            lines = [ln for ln in lines if ln]  # ignorera tomma
            # avdubbla men bevara ordning
            seen = set()
            dedup: List[str] = []
            dups = 0
            for t in lines:
                if t in seen:
                    dups += 1
                else:
                    seen.add(t); dedup.append(t)
            st.session_state["batch_raw"] = "\n".join(dedup)
            st.success(f"{len(dedup)} tickers inlästa ({dups} dubbletter ignorerade) från {uni_path}")
        except FileNotFoundError:
            st.error("universumfil saknas: universe/universe.txt")
        except Exception as e:
            st.error(f"Kunde inte läsa universe/universe.txt: {e}")

with colU2:
    st.text_area("Batchlista (en ticker per rad)", key="batch_raw", height=140, help="Du kan manuellt lägga till/ta bort efter inläsning.")

# --- Tri-state
st.subheader("Indikator-kandidater (tri-state)")
def _tri_idx(name: str, default: str) -> int:
    val = default
    if CLI_FILTERS and name in CLI_FILTERS:
        val = CLI_FILTERS[name]
    return TRI_STATES.index(val)

c1, c2, c3, c4 = st.columns(4)
with c1:
    tri_obv = st.selectbox("OBV filter", TRI_STATES, index=_tri_idx("obv", "Allowed"))
with c2:
    tri_chop = st.selectbox("CHOP filter", TRI_STATES, index=_tri_idx("chop", "Allowed"))
with c3:
    tri_hma = st.selectbox("HMA filter", TRI_STATES, index=_tri_idx("hma", "Allowed"))
with c4:
    disable_gates = st.checkbox("Disable gates (allow all)", value=True)

with st.expander("Avancerat / lås parametrar"):
    lc1, lc2, lc3 = st.columns(3)
    with lc1:
        N_lock = st.number_input("Lås N (breakout)", value=0, min_value=0, step=1)
        M_lock = st.number_input("Lås M (exit)",     value=0, min_value=0, step=1)
    with lc2:
        obv_lb_lock  = st.number_input("Lås obv_lb",  value=0, min_value=0, step=1)
        chop_w_lock  = st.number_input("Lås chop_w",  value=0, min_value=0, step=1)
        hma_w_lock   = st.number_input("Lås hma_w",   value=0, min_value=0, step=1)
    with lc3:
        chop_max_lock = st.number_input("Lås chop_max", value=0.0, min_value=0.0, step=0.5, format="%.1f")

locks: Dict[str, Any] = {
    "N_lock": int(N_lock), "M_lock": int(M_lock),
    "obv_lb_lock": int(obv_lb_lock), "chop_w_lock": int(chop_w_lock), "hma_w_lock": int(hma_w_lock),
    "chop_max_lock": float(chop_max_lock),
}

# --- Läge
mode = st.radio("Läge", ["Single", "Batch"], horizontal=True)
ticker_single = st.text_input("Ticker (Single)", value="ABB") if mode == "Single" else None

# --- Statusytor (text-baserade)
st.markdown("---")
st.subheader("Körstatus")
best_slot   = st.empty()     # "Bästa TR hittills"
sims_slot   = st.empty()     # "Simuleringar körda"
ticker_slot = st.empty()     # "Tickerstatus"

# --- Kör-funktioner
def run_single(ticker: str) -> Dict[str, Any]:
    return run_optimizer_server_for_ticker(
        ticker=ticker.strip(),
        start=start_d, end=end_d,
        total_sims=int(trials),
        seed=seed_val,
        run_tag=str(run_tag).strip() or "run",
        workers=int(workers),
        chunk_size=int(chunk_size),
        topk=int(DEFAULT_TOPK),
        tri_obv=str(tri_obv), tri_chop=str(tri_chop), tri_hma=str(tri_hma),
        locks=locks,
        disable_gates=bool(disable_gates),
        resume=bool(resume),
        ui_best_slot=best_slot, ui_sims_slot=sims_slot, ui_ticker_slot=ticker_slot,
    )

def run_batch(tickers: List[str]) -> List[Dict[str, Any]]:
    outs = []
    for t in tickers:
        best_slot.markdown("**Bästa TR hittills:** –")
        sims_slot.markdown(f"**Simuleringar körda:** 0/{int(trials)}")
        ticker_slot.markdown(f"**Tickerstatus:** {t}: 0/{int(trials)} (best TR –)")
        outs.append(run_single(t))
    return outs

# --- Körknappar
st.markdown("---")
if mode == "Single":
    if st.button("Kör Single"):
        with st.spinner("Kör optimizer…"):
            res = run_single(ticker_single or "ABB")
        st.success("Klar.")
        st.json({k: v for k, v in res.items() if k not in ("base_dir",)})
else:
    if st.button("Kör Batch"):
        symbols = [x.strip() for x in st.session_state["batch_raw"].splitlines() if x.strip()]
        with st.spinner("Kör batch…"):
            res_list = run_batch(symbols)
        st.success("Batch klar.")
        st.json([ {k: v for k, v in r.items() if k not in ("base_dir",)} for r in res_list ])
