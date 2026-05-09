"""
Per-slice breakdown of grid search results.

Given a grid_search CSV, takes the top N configs (ranked by worst-case sharpe
across slices) and shows their per-(symbol, year) performance as a matrix.

Purpose: diagnose WHICH slices kill a candidate. If all top configs die in
the same year, that's a regime problem (and a regime filter might rescue
them). If they die in different slices, the family is just unstable.

Usage:
    python -m runner.analyze --input results/donchian_full.csv --top 5
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
    p.add_argument("--top", type=int, default=5, help="How many top configs to break down")
    p.add_argument("--rank-by", default="sharpe_min",
                   choices=["sharpe_min", "sharpe_mean", "total_return_mean"])
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    df = pd.read_csv(args.input)
    log.info("Loaded %d rows from %s", len(df), args.input)

    param_cols = [c for c in df.columns if c.startswith("p_")]

    # Aggregate per param-combo across slices
    agg = df.groupby(param_cols).agg(
        sharpe_mean=("sharpe", "mean"),
        sharpe_min=("sharpe", "min"),
        sharpe_max=("sharpe", "max"),
        return_mean=("total_return", "mean"),
        return_min=("total_return", "min"),
        return_max=("total_return", "max"),
        dd_min=("max_drawdown", "min"),
        trades_mean=("trades", "mean"),
        n_slices=("sharpe", "count"),
    ).reset_index()

    # Pick ranking metric
    rank_col_map = {
        "sharpe_min": "sharpe_min",
        "sharpe_mean": "sharpe_mean",
        "total_return_mean": "return_mean",
    }
    rank_col = rank_col_map[args.rank_by]
    agg = agg.sort_values(rank_col, ascending=False)

    log.info("\n=== HEADLINE: How many configs survived various Sharpe floors? ===")
    for floor in [2.0, 1.5, 1.0, 0.5, 0.0, -0.5]:
        n = (agg["sharpe_min"] >= floor).sum()
        log.info("  worst-case Sharpe >= %+.1f: %d / %d configs (%.1f%%)",
                 floor, n, len(agg), 100 * n / len(agg))

    log.info("\n=== HEADLINE: How many configs were profitable in EVERY slice? ===")
    profitable_in_all = (agg["return_min"] > 0).sum()
    log.info("  positive total_return in worst slice: %d / %d (%.1f%%)",
             profitable_in_all, len(agg), 100 * profitable_in_all / len(agg))

    log.info("\n=== Top %d configs by %s ===\n", args.top, args.rank_by)

    top = agg.head(args.top)

    for i, (_, row) in enumerate(top.iterrows()):
        params = {c: row[c] for c in param_cols}
        param_str = "  ".join(f"{c[2:]}={v}" for c, v in params.items())
        log.info("--- Config #%d ---", i + 1)
        log.info("  Params: %s", param_str)
        log.info("  Sharpe: mean=%.2f  min=%.2f  max=%.2f",
                 row["sharpe_mean"], row["sharpe_min"], row["sharpe_max"])
        log.info("  Return: mean=%+.1f%%  min=%+.1f%%  max=%+.1f%%",
                 row["return_mean"] * 100, row["return_min"] * 100, row["return_max"] * 100)
        log.info("  Worst DD: %.1f%%   Avg trades: %.0f   Slices: %d",
                 row["dd_min"] * 100, row["trades_mean"], row["n_slices"])

        # Pull this config's per-slice rows
        mask = pd.Series(True, index=df.index)
        for c in param_cols:
            mask &= (df[c] == row[c])
        slice_df = df[mask]

        # Build pivot tables
        pivot_ret = slice_df.pivot_table(
            index="symbol", columns="year", values="total_return"
        ) * 100
        pivot_sharpe = slice_df.pivot_table(
            index="symbol", columns="year", values="sharpe"
        )

        log.info("\n  Total return %% per (symbol, year):")
        log.info(pivot_ret.to_string(float_format=lambda x: f"{x:+7.1f}"))
        log.info("\n  Sharpe per (symbol, year):")
        log.info(pivot_sharpe.to_string(float_format=lambda x: f"{x:+5.2f}"))

        # Year averages and pair averages — find the worst
        log.info("\n  Average Sharpe by YEAR (across pairs):")
        year_avg = slice_df.groupby("year")["sharpe"].mean().sort_values()
        for yr, sh in year_avg.items():
            marker = "  <-- worst" if yr == year_avg.index[0] else ""
            log.info("    %d: %+.2f%s", yr, sh, marker)

        log.info("\n  Average Sharpe by PAIR (across years):")
        pair_avg = slice_df.groupby("symbol")["sharpe"].mean().sort_values()
        for sym, sh in pair_avg.items():
            marker = "  <-- worst" if sym == pair_avg.index[0] else ""
            log.info("    %s: %+.2f%s", sym, sh, marker)

        log.info("")

    # Cross-config slice analysis: are the SAME slices killing every config?
    log.info("=== Cross-config slice stress: average Sharpe of TOP %d configs per slice ===\n",
             args.top)

    top_param_tuples = set()
    for _, row in top.iterrows():
        top_param_tuples.add(tuple(row[c] for c in param_cols))

    df["param_tuple"] = df[param_cols].apply(tuple, axis=1)
    top_df = df[df["param_tuple"].isin(top_param_tuples)]

    cross = top_df.pivot_table(
        index="symbol", columns="year", values="sharpe", aggfunc="mean"
    )
    log.info("Avg Sharpe across top %d configs:", args.top)
    log.info(cross.to_string(float_format=lambda x: f"{x:+5.2f}"))

    log.info("\nWorst (symbol, year) cells across top configs:")
    flat = cross.stack().sort_values()
    for (sym, yr), sh in flat.head(5).items():
        log.info("  %s %d: avg Sharpe = %+.2f", sym, int(yr), sh)


if __name__ == "__main__":
    main()
