"""
Binance Vision data loader.

Downloads historical OHLCV data from Binance's public S3 bucket and caches it
locally as parquet for fast re-reads. Free, no API key, deepest history available.

URL pattern:
  https://data.binance.vision/data/spot/monthly/klines/{SYMBOL}/{INTERVAL}/{SYMBOL}-{INTERVAL}-{YYYY-MM}.zip

Each zip contains a single CSV with columns:
  open_time, open, high, low, close, volume, close_time, quote_volume,
  trades, taker_buy_base, taker_buy_quote, ignore
"""

from __future__ import annotations

import io
import logging
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

log = logging.getLogger(__name__)

BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"
DEFAULT_CACHE = Path.home() / ".krypt-sim" / "binance_vision"

# Columns in the raw CSV (Binance schema)
RAW_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]

# Columns we keep after parsing
KEEP_COLS = ["open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_quote"]


@dataclass(frozen=True)
class DataRequest:
    symbol: str          # e.g. "BTCUSDT"
    interval: str        # e.g. "1m", "5m", "15m", "1h"
    start: date          # inclusive month
    end: date            # inclusive month


def _month_iter(start: date, end: date) -> Iterable[date]:
    """Yield first-of-month dates from start to end inclusive."""
    cur = date(start.year, start.month, 1)
    last = date(end.year, end.month, 1)
    while cur <= last:
        yield cur
        if cur.month == 12:
            cur = date(cur.year + 1, 1, 1)
        else:
            cur = date(cur.year, cur.month + 1, 1)


def _cache_path(cache_dir: Path, symbol: str, interval: str, month: date) -> Path:
    return cache_dir / symbol / interval / f"{symbol}-{interval}-{month:%Y-%m}.parquet"


def _download_month(
    symbol: str,
    interval: str,
    month: date,
    cache_dir: Path,
    timeout: int = 60,
) -> pd.DataFrame | None:
    """Download a single month, return DataFrame or None if not available."""
    cache_file = _cache_path(cache_dir, symbol, interval, month)

    if cache_file.exists():
        try:
            return pd.read_parquet(cache_file)
        except Exception as e:
            log.warning("Cache read failed for %s, re-downloading: %s", cache_file, e)
            cache_file.unlink(missing_ok=True)

    url = f"{BASE_URL}/{symbol}/{interval}/{symbol}-{interval}-{month:%Y-%m}.zip"

    try:
        resp = requests.get(url, timeout=timeout)
    except requests.RequestException as e:
        log.warning("Network error %s %s: %s", symbol, month, e)
        return None

    if resp.status_code == 404:
        # Month not available (e.g. before listing, or future month)
        return None
    if resp.status_code != 200:
        log.warning("HTTP %s for %s %s", resp.status_code, symbol, month)
        return None

    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            csv_name = zf.namelist()[0]
            with zf.open(csv_name) as fh:
                # Newer Binance CSVs have a header; older ones don't.
                # Read first byte to decide.
                first = fh.read(1)
                fh.close()

            with zf.open(csv_name) as fh:
                if first.isdigit():
                    df = pd.read_csv(fh, header=None, names=RAW_COLS)
                else:
                    df = pd.read_csv(fh)
                    # Normalize column names just in case
                    df.columns = [c.strip().lower() for c in df.columns]
                    # Some headers use slightly different names
                    rename = {
                        "number_of_trades": "trades",
                        "taker_buy_base_asset_volume": "taker_buy_base",
                        "taker_buy_quote_asset_volume": "taker_buy_quote",
                        "quote_asset_volume": "quote_volume",
                    }
                    df = df.rename(columns=rename)
    except Exception as e:
        log.warning("Failed to parse zip for %s %s: %s", symbol, month, e)
        return None

    # Parse timestamps. Binance ms timestamps; auto-detect us vs ms by magnitude.
    ot = pd.to_numeric(df["open_time"], errors="coerce")
    if ot.iloc[0] > 10**14:  # microseconds
        df["open_time"] = pd.to_datetime(ot, unit="us", utc=True)
    else:
        df["open_time"] = pd.to_datetime(ot, unit="ms", utc=True)

    # Numeric coercion
    for col in ["open", "high", "low", "close", "volume", "quote_volume",
                "trades", "taker_buy_quote"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.set_index("open_time")[KEEP_COLS].sort_index()

    # Cache as parquet
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_file, compression="snappy")

    return df


def load(
    symbol: str,
    interval: str,
    start: str | date,
    end: str | date,
    cache_dir: Path = DEFAULT_CACHE,
    workers: int = 8,
) -> pd.DataFrame:
    """
    Load OHLCV for a single symbol/interval over a date range.
    Downloads missing months in parallel, caches as parquet.

    Returns DataFrame indexed by UTC open_time with columns:
      open, high, low, close, volume, quote_volume, trades, taker_buy_quote
    """
    if isinstance(start, str):
        start = datetime.strptime(start, "%Y-%m-%d").date()
    if isinstance(end, str):
        end = datetime.strptime(end, "%Y-%m-%d").date()

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    months = list(_month_iter(start, end))
    log.info("Loading %s %s: %d months", symbol, interval, len(months))

    frames: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(_download_month, symbol, interval, m, cache_dir): m
            for m in months
        }
        for fut in as_completed(futures):
            df = fut.result()
            if df is not None:
                frames.append(df)

    if not frames:
        raise RuntimeError(f"No data found for {symbol} {interval} {start}..{end}")

    out = pd.concat(frames).sort_index()
    # Drop duplicates on index (can happen at month boundaries)
    out = out[~out.index.duplicated(keep="first")]

    # Trim to requested range (start of first month to end of last month)
    return out


def load_many(
    symbols: list[str],
    interval: str,
    start: str | date,
    end: str | date,
    cache_dir: Path = DEFAULT_CACHE,
    workers: int = 8,
) -> dict[str, pd.DataFrame]:
    """Load multiple symbols. Returns dict of symbol -> DataFrame."""
    out = {}
    for sym in symbols:
        try:
            out[sym] = load(sym, interval, start, end, cache_dir, workers)
        except RuntimeError as e:
            log.warning("Skipping %s: %s", sym, e)
    return out
