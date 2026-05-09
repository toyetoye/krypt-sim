"""
Per-(strategy, pair) survivor analyzer.

Where analyze.py asks "is this config good across all pairs?", this asks
"on which specific pairs does this config have edge?". The portfolio thesis
is that we run different strategies on different pairs — so a Donchian config
that's brilliant on AVAX and useless on LINK is still a real candidate, as
long as we deploy it only on AVAX.

For each (config, pair):
  - Compute Sharpe min/mean across years
  - Compute total return min/mean across years
  - Filter: did it produce positive return in every year? Sharpe above a floor?
  - Output a leaderboard of (config, pair) survivors

This is the right granularity for a portfolio approach.

Usage:
    python -m runner.per_pair --input results/donchian_full.csv \
        --min-sharpe 0.5 --min-return 0.0 --top 30
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

log = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Grid search CSV")
    p.add_argument("--min-sharpe", type=float, default=0.5,
                   help="Minimum worst-year Sharpe to qualify as survivor")
    p.add_argument("--min-return", type=float, default=0.0,
                   help="Minimum worst-year total return to qualify (e.g. 0.0 = profitable)")
    p.add_argument("--min-trades", type=float, default=5.0,
                   help="Minimum mean trades per year to avoid statistical noise")
    p.add_argument("--top", type=int, default=30,
                   help="How many top (config, pair) combos to display")
    p.add_argument("--rank-by", default="sharpe_min",
                   choices=["sharpe_min", "sharpe_mean", "return_mean", "return_min"])
    p.add_argument("--exclude-years", nargs="*", type=int, default=[],
                   help="Years to exclude from the analysis (e.g. --exclude-years 2025)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    df = pd.read_csv(args.input)
    log.info("Loaded %d rows from %s", len(df), args.input)

    if args.exclude_years:
        before = len(df)
        df = df[~df["year"].isin(args.exclude_years)]
        log.info("Excluded years %s: %d -> %d rows",
                 args.exclude_years, before, len(df))

    param_cols = [c for c in df.columns if c.startswith("p_")]

    # Aggregate per (config, pair) across years
    group_cols = param_cols + ["symbol"]
    agg = df.groupby(group_cols).agg(
        sharpe_mean=("sharpe", "mean"),
        sharpe_min=("sharpe", "min"),
        sharpe_max=("sharpe", "max"),
        return_mean=("total_return", "mean"),
        return_min=("total_return", "min"),
        return_max=("total_return", "max"),
        dd_min=("max_drawdown", "min"),
        trades_mean=("trades", "mean"),
        n_years=("sharpe", "count"),
    ).reset_index()

    log.info("\nTotal (config, pair) combinations: %d", len(agg))

    # Apply filters
    survivors = agg[
        (agg["sharpe_min"] >= args.min_sharpe) &
        (agg["return_min"] >= args.min_return) &
        (agg["trades_mean"] >= args.min_trades)
    ].copy()

    log.info("\n=== SURVIVORS ===")
    log.info("Filters: sharpe_min >= %.2f  return_min >= %.2f  trades_mean >= %.0f",
             args.min_sharpe, args.min_return, args.min_trades)
    log.info("Survivors: %d / %d (%.1f%%)",
             len(survivors), len(agg), 100 * len(survivors) / len(agg))

    if len(survivors) == 0:
        log.info("\nNo survivors at this threshold. Lower --min-sharpe or --min-return.")
        # Show what would survive at progressively easier floors
        log.info("\nAt easier floors:")
        for floor in [0.3, 0.0, -0.2, -0.5]:
            n = ((agg["sharpe_min"] >= floor) &
                 (agg["return_min"] >= args.min_return) &
                 (agg["trades_mean"] >= args.min_trades)).sum()
            log.info("  sharpe_min >= %+.1f: %d combos", floor, n)
        return

    # Rank
    rank_col_map = {
        "sharpe_min": "sharpe_min",
        "sharpe_mean": "sharpe_mean",
        "return_mean": "return_mean",
        "return_min": "return_min",
    }
    rank_col = rank_col_map[args.rank_by]
    survivors = survivors.sort_values(rank_col, ascending=False)

    # Display
    log.info("\n=== Top %d survivors by %s ===\n", args.top, args.rank_by)

    display_cols = (
        param_cols + ["symbol"] +
        ["sharpe_min", "sharpe_mean", "return_min", "return_mean",
         "dd_min", "trades_mean", "n_years"]
    )

    top = survivors.head(args.top)[display_cols]

    # Pretty print
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    formatters = {
        "sharpe_min": lambda x: f"{x:+.2f}",
        "sharpe_mean": lambda x: f"{x:+.2f}",
        "return_min": lambda x: f"{x:+.1%}",
        "return_mean": lambda x: f"{x:+.1%}",
        "dd_min": lambda x: f"{x:.1%}",
        "trades_mean": lambda x: f"{x:.0f}",
        "n_years": lambda x: f"{x:.0f}",
    }
    log.info(top.to_string(index=False, formatters=formatters))

    # Pair coverage: which pairs have ANY survivor?
    log.info("\n=== Pair coverage ===")
    pair_counts = survivors.groupby("symbol").size().sort_values(ascending=False)
    for sym, n in pair_counts.items():
        log.info("  %s: %d surviving configs", sym, n)

    pairs_no_survivor = set(df["symbol"].unique()) - set(pair_counts.index)
    if pairs_no_survivor:
        log.info("\n  Pairs with NO surviving config: %s",
                 ", ".join(sorted(pairs_no_survivor)))

    # Best config per pair
    log.info("\n=== Best surviving config PER PAIR ===\n")
    best_per_pair = (
        survivors.sort_values(rank_col, ascending=False)
        .groupby("symbol")
        .head(1)
        .sort_values(rank_col, ascending=False)
    )
    log.info(best_per_pair[display_cols].to_string(index=False, formatters=formatters))

    # Save survivors as CSV for downstream walk-forward feeding
    out_path = Path(args.input).with_name(
        Path(args.input).stem + "_survivors.csv"
    )
    survivors.to_csv(out_path, index=False)
    log.info("\nWrote %d survivors to %s", len(survivors), out_path)


if __name__ == "__main__":
    main()
