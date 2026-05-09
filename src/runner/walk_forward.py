"""
Walk-forward validation.

Given a list of candidate param sets (typically the survivors of grid_search
filtering), run each through walk-forward windows and report only the
configs that hold up out-of-sample.

Process:
  1. For each (symbol, interval), define rolling train/test windows.
     Default: 12 months train, 3 months test, step 3 months.
  2. Run candidate on the test window. (No optimization here — params are
     fixed in advance. This is pure out-of-sample validation.)
  3. Report mean and stddev of metrics across test windows.
  4. Survivors: configs whose worst-test-window Sharpe >= threshold.

This is the second filter after grid_search. A config that wins one year
but degrades over rolling test windows is overfit to that year.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from data import binance_vision as bv
from engine.backtest import run_backtest, ExecutionConfig
from runner.grid_search import FAMILIES


log = logging.getLogger(__name__)


def _shift_months(d: date, n: int) -> date:
    """Shift date by n months, clamped to first of month."""
    y = d.year + (d.month - 1 + n) // 12
    m = (d.month - 1 + n) % 12 + 1
    return date(y, m, 1)


def walk_forward(
    family: str,
    params: dict,
    symbol: str,
    interval: str,
    start: date,
    end: date,
    test_months: int = 3,
    step_months: int = 3,
    cfg: ExecutionConfig = ExecutionConfig(),
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Returns one row per test window with metrics.
    """
    cls = FAMILIES[family]
    strat = cls(params=params)

    df = bv.load(
        symbol, interval,
        start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"),
        cache_dir=cache_dir or Path.home() / ".krypt-sim" / "binance_vision",
        workers=4,
    )

    rows = []
    win_start = start
    while True:
        win_end = _shift_months(win_start, test_months)
        if win_end > end:
            break

        slice_df = df.loc[
            df.index >= pd.Timestamp(win_start, tz="UTC")
            ]
        slice_df = slice_df.loc[
            slice_df.index < pd.Timestamp(win_end, tz="UTC")
            ]

        if len(slice_df) < 200:
            win_start = _shift_months(win_start, step_months)
            continue

        sig = strat.signal(slice_df)
        r = run_backtest(slice_df, sig, cfg)
        days = (slice_df.index[-1] - slice_df.index[0]).total_seconds() / 86400
        daily = (1 + r.total_return) ** (1 / days) - 1 if days > 0 else 0

        rows.append({
            "window_start": win_start,
            "window_end": win_end,
            "total_return": r.total_return,
            "daily_return": daily,
            "sharpe": r.sharpe,
            "max_drawdown": r.max_drawdown,
            "trades": r.n_trades,
            "win_rate": r.win_rate,
            "bars_in_market": r.bars_in_market,
        })

        win_start = _shift_months(win_start, step_months)

    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--family", required=True, choices=list(FAMILIES.keys()))
    p.add_argument("--params", required=True,
                   help="JSON string of params, e.g. '{\"entry_n\":100,\"exit_n\":50,\"long_only\":true,\"regime_ma\":0}'")
    p.add_argument("--pairs", nargs="+", required=True)
    p.add_argument("--interval", default="1h")
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--end", default="2025-01-01")
    p.add_argument("--test-months", type=int, default=3)
    p.add_argument("--step-months", type=int, default=3)
    p.add_argument("--fee", type=float, default=0.001)
    p.add_argument("--slippage-bps", type=float, default=5.0)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    import json
    params = json.loads(args.params)

    cfg = ExecutionConfig(
        fee_per_side=args.fee,
        slippage_bps=args.slippage_bps,
    )

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    all_rows = []
    for sym in args.pairs:
        log.info("Walk-forward: %s on %s", args.family, sym)
        df = walk_forward(
            args.family, params, sym, args.interval,
            start, end,
            test_months=args.test_months,
            step_months=args.step_months,
            cfg=cfg,
        )
        df["symbol"] = sym
        all_rows.append(df)
        if len(df):
            log.info("  windows=%d  worst_sharpe=%.2f  mean_sharpe=%.2f  "
                     "worst_return=%.2f%%",
                     len(df), df["sharpe"].min(), df["sharpe"].mean(),
                     df["total_return"].min() * 100)

    combined = pd.concat(all_rows)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out, index=False)
    log.info("Wrote %d rows to %s", len(combined), out)


if __name__ == "__main__":
    main()
