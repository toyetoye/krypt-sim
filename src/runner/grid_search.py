"""
Grid search runner with brutal filtering.

Generates a parameter grid for a strategy family, runs each variant against
each (symbol, year) slice, and ranks survivors by *worst-case* performance
across slices, not by average. A strategy that hits the target in 2024 but
loses in 2022 is dead.

Usage:
    python -m runner.grid_search --family donchian \
        --pairs BTCUSDT ETHUSDT SOLUSDT \
        --years 2021 2022 2023 2024 \
        --interval 1h \
        --output results/donchian_grid.csv
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path

# Allow `python -m runner.grid_search` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from data import binance_vision as bv
from engine.backtest import run_backtest, ExecutionConfig
from strategies.base import Strategy
from strategies.donchian import DonchianBreakout
from strategies.zscore_rev import ZScoreReversion
from strategies.squeeze import SqueezeBreakout
from strategies.volume_anomaly import VolumeAnomaly


log = logging.getLogger(__name__)


# Parameter grids per family. Tune these on your box; these are reasonable defaults.
GRIDS = {
    "donchian": {
        "entry_n": [20, 50, 100, 150, 200],
        "exit_n":  [10, 25, 50, 75],
        "long_only": [True, False],
        "regime_ma": [0, 100, 200],
    },
    "zscore_rev": {
        "lookback": [20, 50, 100, 200],
        "entry_z":  [1.5, 2.0, 2.5, 3.0],
        "exit_z":   [0.0, 0.5, 1.0],
        "long_only": [True, False],
        "regime_off_ma": [0, 200],
        "regime_threshold_pct": [3.0, 5.0],
    },
    "squeeze": {
        "atr_n": [14, 20],
        "squeeze_lookback": [50, 100, 200],
        "squeeze_pctile":   [20.0, 30.0, 40.0],
        "breakout_n":       [10, 20, 40],
        "max_hold_bars":    [12, 24, 48],
        "long_only": [True, False],
    },
    "volume_anomaly": {
        "vol_lookback":     [50, 100, 200],
        "vol_z_threshold":  [2.0, 2.5, 3.0, 3.5],
        "min_body_pct":     [0.1, 0.2, 0.3],
        "hold_bars":        [3, 6, 12, 24],
        "long_only": [True, False],
    },
}

FAMILIES = {
    "donchian": DonchianBreakout,
    "zscore_rev": ZScoreReversion,
    "squeeze": SqueezeBreakout,
    "volume_anomaly": VolumeAnomaly,
}


def _generate_param_combos(grid: dict) -> list[dict]:
    """Cartesian product over a parameter grid."""
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _run_one(args: tuple) -> dict:
    """Run a single (family, params, symbol, year) backtest."""
    family, params, symbol, year, interval, fee, slip, pos_size, cache_dir = args

    cls = FAMILIES[family]
    strat = cls(params=params)
    cfg = ExecutionConfig(
        fee_per_side=fee, slippage_bps=slip, position_size=pos_size
    )

    try:
        df = bv.load(
            symbol, interval,
            f"{year}-01-01", f"{year}-12-31",
            cache_dir=cache_dir,
            workers=4,
        )
    except RuntimeError:
        return None

    if len(df) < 50:  # skip slices with too little data
        return None

    sig = strat.signal(df)
    r = run_backtest(df, sig, cfg)

    days = (df.index[-1] - df.index[0]).total_seconds() / 86400
    daily = (1 + r.total_return) ** (1 / days) - 1 if days > 0 else 0

    return {
        "family": family,
        "symbol": symbol,
        "year": year,
        "interval": interval,
        **{f"p_{k}": v for k, v in params.items()},
        "total_return": r.total_return,
        "daily_return": daily,
        "sharpe": r.sharpe,
        "max_drawdown": r.max_drawdown,
        "trades": r.n_trades,
        "win_rate": r.win_rate,
        "bars_in_market": r.bars_in_market,
        "fees_paid": r.total_fees,
        "slippage_paid": r.total_slippage,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--family", required=True, choices=list(FAMILIES.keys()))
    p.add_argument("--pairs", nargs="+", required=True)
    p.add_argument("--years", nargs="+", type=int, required=True)
    p.add_argument("--interval", default="1h")
    p.add_argument("--fee", type=float, default=0.001)
    p.add_argument("--slippage-bps", type=float, default=5.0)
    p.add_argument("--position-size", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--output", required=True)
    p.add_argument("--cache-dir", default=str(Path.home() / ".krypt-sim" / "binance_vision"))
    p.add_argument("--limit", type=int, default=0,
                   help="Limit number of param combos (for quick smoke tests)")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    grid = GRIDS[args.family]
    combos = _generate_param_combos(grid)
    if args.limit:
        combos = combos[:args.limit]

    jobs = []
    for params in combos:
        for sym in args.pairs:
            for yr in args.years:
                jobs.append((
                    args.family, params, sym, yr, args.interval,
                    args.fee, args.slippage_bps, args.position_size,
                    args.cache_dir,
                ))

    log.info(
        "Grid search: family=%s combos=%d pairs=%d years=%d interval=%s -> %d jobs",
        args.family, len(combos), len(args.pairs), len(args.years),
        args.interval, len(jobs),
    )

    results = []
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_run_one, j) for j in jobs]
        for fut in as_completed(futures):
            r = fut.result()
            completed += 1
            if r is not None:
                results.append(r)
            if completed % 100 == 0:
                log.info("  %d / %d done (%d valid)", completed, len(jobs), len(results))

    if not results:
        log.error("No valid results.")
        return

    df = pd.DataFrame(results)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    log.info("Wrote %d rows to %s", len(df), out)

    # Quick summary: aggregate per param-combo across all symbol-year slices
    param_cols = [c for c in df.columns if c.startswith("p_")]
    grouped = df.groupby(param_cols).agg({
        "total_return": ["mean", "min"],
        "sharpe": ["mean", "min"],
        "max_drawdown": "min",
        "trades": "mean",
    })
    grouped.columns = ["_".join(c) for c in grouped.columns]
    grouped = grouped.reset_index()
    # Brutal filter: rank by WORST-case sharpe across slices
    grouped = grouped.sort_values("sharpe_min", ascending=False)
    log.info("Top 10 by worst-case Sharpe across slices:")
    print(grouped.head(10).to_string())


if __name__ == "__main__":
    main()
