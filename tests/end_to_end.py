"""
End-to-end smoke test: Donchian breakout, multiple parameters, single pair.

This is what 'minimal end-to-end' looks like. Once the numbers here are
believable, we scale to grid search and walk-forward.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from data import binance_vision as bv
from engine.backtest import run_backtest, ExecutionConfig
from strategies.donchian import DonchianBreakout


def fmt_result(name: str, r) -> str:
    days = (r.equity.index[-1] - r.equity.index[0]).total_seconds() / 86400
    daily_ret = (1 + r.total_return) ** (1 / days) - 1 if days > 0 else 0
    return (
        f"{name:30s}  "
        f"ret={r.total_return:+8.2%}  "
        f"daily={daily_ret:+.3%}  "
        f"sharpe={r.sharpe:5.2f}  "
        f"dd={r.max_drawdown:+.1%}  "
        f"trades={r.n_trades:4d}  "
        f"win={r.win_rate:.1%}  "
        f"in_mkt={r.bars_in_market:.1%}  "
        f"fees={r.total_fees:.1%}"
    )


def main():
    print("Loading 1 year of BTC 1h...")
    df = bv.load("BTCUSDT", "1h", "2024-01-01", "2024-12-31")
    print(f"  {len(df)} bars  {df.index.min().date()} -> {df.index.max().date()}")
    print(f"  underlying B&H: {(df['open'].iloc[-1] / df['open'].iloc[0] - 1):+.1%}")

    cfg = ExecutionConfig(fee_per_side=0.001, slippage_bps=5.0, position_size=1.0)

    print("\n=== Donchian breakout parameter sweep on BTCUSDT 1h, 2024 ===\n")

    configs = [
        # (entry_n, exit_n, long_only, regime_ma, label)
        (20, 10, False, 0,   "20/10 long-short"),
        (20, 10, True,  0,   "20/10 long-only"),
        (50, 25, False, 0,   "50/25 long-short"),
        (50, 25, True,  0,   "50/25 long-only"),
        (100, 50, False, 0,  "100/50 long-short"),
        (100, 50, True,  0,  "100/50 long-only"),
        (50, 25, False, 200, "50/25 + MA200 regime"),
        (50, 25, True,  200, "50/25 long-only + MA200"),
        (200, 50, True, 0,   "200/50 long-only (slow)"),
    ]

    results = []
    for entry, exit_, lo, rm, label in configs:
        strat = DonchianBreakout(params={
            "entry_n": entry, "exit_n": exit_,
            "long_only": lo, "regime_ma": rm,
        })
        sig = strat.signal(df)
        r = run_backtest(df, sig, cfg)
        results.append((label, r))
        print(fmt_result(label, r))

    print("\n=== Same configs, on 1m (very different time horizon) ===")
    print("Loading 1 year of BTC 1m... (this will cache ~12 months)")
    df1m = bv.load("BTCUSDT", "1m", "2024-01-01", "2024-03-31")
    print(f"  {len(df1m)} bars (3 months only to keep this fast)")

    for entry, exit_, lo, rm, label in [
        (60, 30, True, 0, "60/30 long-only 1m"),
        (240, 120, True, 0, "240/120 long-only 1m"),
        (60, 30, True, 1440, "60/30 + MA1440 (1d) 1m"),
    ]:
        strat = DonchianBreakout(params={
            "entry_n": entry, "exit_n": exit_,
            "long_only": lo, "regime_ma": rm,
        })
        sig = strat.signal(df1m)
        r = run_backtest(df1m, sig, cfg)
        print(fmt_result(label, r))


if __name__ == "__main__":
    main()
