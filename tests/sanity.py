"""
Engine sanity tests. These should pass before any real strategy is trusted.

Tests:
  1. Always-flat strategy: returns = 0, no trades, no fees.
  2. Always-long strategy: matches buy-and-hold (minus 2x entry/exit cost).
  3. Random strategy: should average to ~0 minus fees, definitely not magic.
  4. Perfect-foresight strategy: should be wildly profitable (proves engine
     can detect a real edge when one exists).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from data import binance_vision as bv
from engine.backtest import run_backtest, ExecutionConfig


def main():
    print("Loading 6 months of BTC 1h data...")
    df = bv.load("BTCUSDT", "1h", "2024-01-01", "2024-06-30")
    print(f"  {len(df)} bars, {df.index.min()} -> {df.index.max()}")
    print(f"  BTC: {df['open'].iloc[0]:.0f} -> {df['open'].iloc[-1]:.0f} "
          f"({(df['open'].iloc[-1] / df['open'].iloc[0] - 1) * 100:+.1f}%)")

    cfg = ExecutionConfig(fee_per_side=0.001, slippage_bps=5.0, position_size=1.0)

    # Test 1: always flat
    print("\n[1] Always-flat strategy")
    sig = pd.Series(0, index=df.index)
    r = run_backtest(df, sig, cfg)
    print(f"  total_return={r.total_return:+.4%}  trades={r.n_trades}  "
          f"fees={r.total_fees:.4%}  bars_in_market={r.bars_in_market:.1%}")
    assert r.total_return == 0 and r.n_trades == 0, "Always-flat should be zero"

    # Test 2: always long (buy and hold)
    print("\n[2] Always-long strategy (buy and hold)")
    sig = pd.Series(1, index=df.index)
    r = run_backtest(df, sig, cfg)
    bh = df["open"].iloc[-1] / df["open"].iloc[0] - 1
    print(f"  total_return={r.total_return:+.4%}  trades={r.n_trades}  "
          f"fees={r.total_fees:.4%}  bars_in_market={r.bars_in_market:.1%}")
    print(f"  underlying B&H: {bh:+.4%}  (expect ~equal minus 1 entry cost)")

    # Test 3: random
    print("\n[3] Random strategy (50/50 long/short)")
    rng = np.random.default_rng(42)
    sig = pd.Series(rng.choice([-1, 0, 1], size=len(df), p=[0.3, 0.4, 0.3]), index=df.index)
    r = run_backtest(df, sig, cfg)
    print(f"  total_return={r.total_return:+.4%}  trades={r.n_trades}  "
          f"sharpe={r.sharpe:.2f}  fees={r.total_fees:.4%}  win={r.win_rate:.1%}")

    # Test 4: perfect foresight (cheating: position = sign of next bar's return)
    print("\n[4] Perfect foresight (UPPER BOUND, cheating)")
    next_open = df["open"].shift(-1)
    perfect = np.sign((next_open - df["open"]).fillna(0)).astype(int)
    # The signal is "what to be in for the next bar" = perfect
    r = run_backtest(df, perfect, cfg)
    print(f"  total_return={r.total_return:+.4%}  trades={r.n_trades}  "
          f"sharpe={r.sharpe:.2f}  fees={r.total_fees:.4%}  win={r.win_rate:.1%}")
    print("  ^ this is the theoretical ceiling. Any real strategy < this.")

    print("\nAll sanity checks complete.")


if __name__ == "__main__":
    main()
