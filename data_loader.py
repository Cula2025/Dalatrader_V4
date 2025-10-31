# app_v4/data_loader.py
# -----------------------------------------------------------------------------
# Dalatrader V4 – Börsdata loader (hårdkodad API-nyckel, inga env/fallbacks)
# Exponerar:
#   - get_eod_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame
#   - load_ohlcv(ticker: str, dfrom: str, dto: str) -> pd.DataFrame
#   - get_prices_eod(ticker: str, dfrom: str, dto: str) -> pd.DataFrame
#   - fetch_ohlcv(ticker: str, dfrom: str, dto: str) -> pd.DataFrame
#   - get_ohlcv(ticker: str, dfrom: str, dto: str) -> pd.DataFrame
# -----------------------------------------------------------------------------

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

# === Hårdkodad Börsdata-nyckel ===============================================
BORSDATA_API_KEY = "85218870c1744409b0624920db023ba8"

# === Konstanter & cache =======================================================
BD_BASE = "https://apiservice.borsdata.se/v1"
CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
INSTR_CACHE = CACHE_DIR / "instruments.json"


# === Feltyp ==================================================================
class BorsdataError(RuntimeError):
    pass


# === Hjälpare =================================================================
def _auth_key() -> str:
    # Endast den hårdkodade nyckeln ska användas
    return BORSDATA_API_KEY


def _req(url: str, params: Optional[Dict[str, Any]] = None) -> requests.Response:
    p = dict(params or {})
    p.setdefault("authKey", _auth_key())
    r = requests.get(url, params=p, timeout=30)
    if r.status_code != 200:
        raise BorsdataError(f"HTTP {r.status_code}: {r.text[:500]}")
    return r


def _normalize_ticker(s: str) -> str:
    return " ".join(str(s).upper().split())


# === Instrument-lista (cache på disk) ========================================
def _load_instruments_remote() -> List[Dict[str, Any]]:
    js = _req(f"{BD_BASE}/instruments").json()
    # Vanliga nycklar: "instruments" eller "data"
    items = js.get("instruments") or js.get("data")
    if not isinstance(items, list):
        raise BorsdataError("Oväntat instruments-svar från Börsdata.")
    return items


def _load_instruments_cached() -> List[Dict[str, Any]]:
    if INSTR_CACHE.exists():
        try:
            data = json.loads(INSTR_CACHE.read_text(encoding="utf-8"))
            if isinstance(data, list) and data:
                return data
        except Exception:
            pass
    items = _load_instruments_remote()
    try:
        INSTR_CACHE.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    return items


def resolve_ins_id(ticker_or_name: str) -> int:
    """
    Matcha på exakt TICKER först (t.ex. 'SEB A'), därefter exakt namn (bolagsnamn),
    sedan prefix-match på ticker.
    """
    key = _normalize_ticker(ticker_or_name)
    instruments = _load_instruments_cached()

    # 1) exakt ticker
    for it in instruments:
        t = _normalize_ticker(it.get("ticker", ""))
        if t == key:
            return int(it["insId"] if "insId" in it else it["id"])

    # 2) exakt namn
    for it in instruments:
        nm = _normalize_ticker(it.get("name", it.get("insName", "")))
        if nm == key:
            return int(it["insId"] if "insId" in it else it["id"])

    # 3) startswith på ticker (t.ex. "HM" → "HM B")
    for it in instruments:
        t = _normalize_ticker(it.get("ticker", ""))
        if t.startswith(key):
            return int(it["insId"] if "insId" in it else it["id"])

    raise BorsdataError(f"Saknar insId för '{ticker_or_name}'. Finns inte i /instruments.")


# === Prisparser som tål flera API-format =====================================
def _extract_price_rows(js: Any) -> List[Dict[str, Any]]:
    """
    Extrahera prisrader till en lista av dicts med nycklar:
      d (date), o (open), h (high), l (low), c (close), v (volume)
    Stödjer flera kända varianter från Börsdata.
    """
    # Kolumn-orienterat format: {"columns":[...], "data":[[...], ...]}
    if isinstance(js, dict) and "columns" in js and "data" in js and isinstance(js["data"], list):
        cols = [str(c).strip().lower() for c in js["columns"]]
        idx = {name: i for i, name in enumerate(cols)}
        out: List[Dict[str, Any]] = []
        for row in js["data"]:
            if not isinstance(row, (list, tuple)):
                continue
            def _col(k, fallback=None):
                i = idx.get(k)
                return row[i] if i is not None and i < len(row) else fallback
            out.append({
                "d": _col("d", _col("date")),
                "o": _col("o", _col("open")),
                "h": _col("h", _col("high")),
                "l": _col("l", _col("low")),
                "c": _col("c", _col("close")),
                "v": _col("v", _col("volume")),
            })
        return out

    # Dict med listvärden
    if isinstance(js, dict):
        for key in ("stockPricesList", "stockPrices", "prices", "data"):
            val = js.get(key)
            if isinstance(val, list):
                js = val
                break
        else:
            for v in js.values():
                if isinstance(v, list):
                    js = v
                    break

    # Direkt lista
    if isinstance(js, list):
        if js and isinstance(js[0], (list, tuple)):
            # anta ordning [d, o, h, l, c, v]
            out: List[Dict[str, Any]] = []
            for row in js:
                d = row[0] if len(row) > 0 else None
                o = row[1] if len(row) > 1 else None
                h = row[2] if len(row) > 2 else None
                l = row[3] if len(row) > 3 else None
                c = row[4] if len(row) > 4 else None
                v = row[5] if len(row) > 5 else None
                out.append({"d": d, "o": o, "h": h, "l": l, "c": c, "v": v})
            return out

        out: List[Dict[str, Any]] = []
        for r in js:
            if not isinstance(r, dict):
                continue
            out.append({
                "d": r.get("d", r.get("date", r.get("Date"))),
                "o": r.get("o", r.get("open", r.get("Open"))),
                "h": r.get("h", r.get("high", r.get("High"))),
                "l": r.get("l", r.get("low", r.get("Low"))),
                "c": r.get("c", r.get("close", r.get("Close"))),
                "v": r.get("v", r.get("volume", r.get("Volume"))),
            })
        return out

    return []


# === EOD-hämtning ============================================================
def _fetch_by_id(ins_id: int, start: str, end: str) -> pd.DataFrame:
    """
    Hämtar dagliga priser för ett instrument-ID.
    Prova flera endpoints/varianter för kompatibilitet.
    """
    # 1) instruments/{id}/stockprices
    url1 = f"{BD_BASE}/instruments/{ins_id}/stockprices"
    js1 = _req(url1, params={"from": start, "to": end}).json()
    rows = _extract_price_rows(js1)

    # 2) stockprices/period/{id}?resolution=d
    if not rows:
        url2 = f"{BD_BASE}/stockprices/period/{ins_id}"
        js2 = _req(url2, params={"resolution": "d", "from": start, "to": end}).json()
        rows = _extract_price_rows(js2)

    # 3) pris-typ 0/1 på instruments-endpointen
    if not rows:
        for pt in (0, 1):
            js3 = _req(url1, params={"from": start, "to": end, "priceType": pt}).json()
            rows = _extract_price_rows(js3)
            if rows:
                break

    if not rows:
        raise BorsdataError(f"Tom prislista för insId={ins_id} ({start}..{end}).")

    recs = []
    for r in rows:
        d = r.get("d")
        if d is None:
            continue
        ds = str(d)[:10]
        recs.append((ds, r.get("o"), r.get("h"), r.get("l"), r.get("c"), r.get("v")))

    if not recs:
        raise BorsdataError("Inga giltiga datapunkter i prislistan.")

    df = pd.DataFrame(recs, columns=["date", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# === Publika funktioner ======================================================
def get_eod_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    ins_id = resolve_ins_id(ticker)
    return _fetch_by_id(int(ins_id), start, end)


# Alias som Optimizern/Backtrackern använder
def load_ohlcv(ticker: str, dfrom: str, dto: str) -> pd.DataFrame:
    return get_eod_ohlcv(ticker=ticker, start=dfrom, end=dto)

def get_prices_eod(ticker: str, dfrom: str, dto: str) -> pd.DataFrame:
    return get_eod_ohlcv(ticker=ticker, start=dfrom, end=dto)

def fetch_ohlcv(ticker: str, dfrom: str, dto: str) -> pd.DataFrame:
    return get_eod_ohlcv(ticker=ticker, start=dfrom, end=dto)

def get_ohlcv(ticker: str, dfrom: str, dto: str) -> pd.DataFrame:
    return get_eod_ohlcv(ticker=ticker, start=dfrom, end=dto)


__all__ = [
    "BORSDATA_API_KEY",
    "get_eod_ohlcv",
    "load_ohlcv",
    "get_prices_eod",
    "fetch_ohlcv",
    "get_ohlcv",
]

