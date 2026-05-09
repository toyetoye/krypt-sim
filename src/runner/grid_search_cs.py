"""
Cross-sectional strategy grid search runner.

Different from runner.grid_search because:
  - Loads ALL pairs at once into a panel, not one at a time
  - Uses MultiAssetStrategy.signals() and run_portfolio_backtest()
  - Grid is over strategy params (no per-pair iteration; pairs are universe)

Usage:
    python -m runner.grid_search_cs \
        --pairs BTCUSDT ETHUSDT SOLUSDT BNBUSDT XRPUSDT ADAUSDT AVAXUSDT LINKUSDT DOTUSDT MATICUSDT \
        --years 2021 2022 2023 2024 2025 \
        --interval 1d \
        --output results/cs_momentum_1d.csv
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from data import binance_vision as bv
from engine.backtest import ExecutionConfig
from engine.portfolio_backtest import run_portfolio_backtest
from strategies.cs_momentum import CrossSectionalMomentum


log = logging.getLogger(__name__)


# Parameter grid for cross-sectional momentum.
# Mix of long-only and long-short, different ranks, different lookbacks/rebalance freq
GRID_CS_MOMENTUM = {
    "lookback_bars":  [5, 10, 20, 40, 60],
    "rebalance_bars": [1, 5, 10],
    "top_k":          [1, 2, 3, 0],   # 0 means use top_quantile
    "top_quantile":   [0.25, 0.5],
    "long_short":     [True, False],
}


def _generate_param_combos(grid: dict) -> list[dict]:
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    combos = []
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        # Skip redundancy: when top_k > 0, top_quantile is ignored, so dedupe
        if params["top_k"] > 0 and params["top_quantile"] != 0.25:
            continue
        combos.append(params)
    return combos


def _run_one(
    panel_year: dict[str, pd.DataFrame],
    params: dict,
    year: int,
    cfg: ExecutionConfig,
) -> dict | None:
    """Run a single (params, year) backtest on the panel."""
    if not panel_year:
        return None

    strat = CrossSectionalMomentum(params={**params, "min_universe": 4})
    try:
        signals = strat.signals(panel_year)
        r = run_portfolio_backtest(panel_year, signals, cfg)
    except (RuntimeError, ValueError) as e:
        log.debug("Skip %s %d: %s", params, year, e)
        return None

    days = (r.equity.index[-1] - r.equity.index[0]).total_seconds() / 86400
    daily = (1 + r.total_return) ** (1 / days) - 1 if days > 0 else 0

    return {
        "year": year,
        **{f"p_{k}": v for k, v in params.items()},
        "total_return": r.total_return,
        "daily_return": daily,
        "sharpe": r.sharpe,
        "max_drawdown": r.max_drawdown,
        "avg_gross_exposure": r.avg_gross_exposure,
        "avg_net_exposure": r.avg_net_exposure,
        "n_rebalances": r.n_rebalances,
        "bars_in_market": r.bars_in_market,
        "fees_paid": r.total_fees,
        "slippage_paid": r.total_slippage,
    }


def _slice_panel(panel: dict[str, pd.DataFrame], year: int) -> dict[str, pd.DataFrame]:
    """Slice all pairs in panel to a single year."""
    start = pd.Timestamp(f"{year}-01-01", tz="UTC")
    end = pd.Timestamp(f"{year+1}-01-01", tz="UTC")
    out = {}
    for sym, df in panel.items():
        sub = df.loc[(df.index >= start) & (df.index < end)]
        if len(sub) > 30:
            out[sym] = sub
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", nargs="+", required=True)
    p.add_argument("--years", nargs="+", type=int, required=True)
    p.add_argument("--interval", default="1d")
    p.add_argument("--fee", type=float, default=0.001)
    p.add_argument("--slippage-bps", type=float, default=5.0)
    p.add_argument("--position-size", type=float, default=1.0)
    p.add_argument("--output", required=True)
    p.add_argument("--cache-dir", default=str(Path.home() / ".krypt-sim" / "binance_vision"))
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    cfg = ExecutionConfig(
        fee_per_side=args.fee,
        slippage_bps=args.slippage_bps,
        position_size=args.position_size,
    )

    # Load full date range for all pairs once
    min_year = min(args.years)
    max_year = max(args.years)
    log.info("Loading panel: %d pairs %d-%d %s", len(args.pairs), min_year, max_year, args.interval)
    panel = {}
    for sym in args.pairs:
        try:
            df = bv.load(
                sym, args.interval,
                f"{min_year}-01-01", f"{max_year}-12-31",
                cache_dir=Path(args.cache_dir),
                workers=4,
            )
            panel[sym] = df
            log.info("  %s: %d bars", sym, len(df))
        except RuntimeError as e:
            log.warning("Skipping %s: %s", sym, e)

    if len(panel) < 4:
        log.error("Universe too small after loading: %d pairs", len(panel))
        return

    combos = _generate_param_combos(GRID_CS_MOMENTUM)
    if args.limit:
        combos = combos[:args.limit]
    log.info("Param combos: %d   Years: %d   Total jobs: %d",
             len(combos), len(args.years), len(combos) * len(args.years))

    # Pre-slice the panel by year to avoid repeated filtering
    panels_by_year = {y: _slice_panel(panel, y) for y in args.years}
    for y, p_y in panels_by_year.items():
        log.info("  year %d: %d pairs in universe", y, len(p_y))

    results = []
    total = len(combos) * len(args.years)
    done = 0
    for params in combos:
        for year in args.years:
            r = _run_one(panels_by_year[year], params, year, cfg)
            if r is not None:
                results.append(r)
            done += 1
            if done % 50 == 0:
                log.info("  %d / %d done (%d valid)", done, total, len(results))

    if not results:
        log.error("No valid results.")
        return

    df = pd.DataFrame(results)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    log.info("Wrote %d rows to %s", len(df), out)

    # Summary: aggregate per param-combo across years
    param_cols = [c for c in df.columns if c.startswith("p_")]
    grouped = df.groupby(param_cols).agg(
        sharpe_mean=("sharpe", "mean"),
        sharpe_min=("sharpe", "min"),
        return_mean=("total_return", "mean"),
        return_min=("total_return", "min"),
        max_dd=("max_drawdown", "min"),
        rebalances_mean=("n_rebalances", "mean"),
    ).reset_index()
    grouped = grouped.sort_values("sharpe_min", ascending=False)
    log.info("\nTop 10 by worst-year Sharpe:")
    print(grouped.head(10).to_string())

    log.info("\nTop 10 by mean Sharpe:")
    print(grouped.sort_values("sharpe_mean", ascending=False).head(10).to_string())


if __name__ == "__main__":
    main()
