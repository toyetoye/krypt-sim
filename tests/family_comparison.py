"""
Multi-family, multi-pair comparison.

Run the four strategy families across BTC/ETH/SOL on 1h, 2024.
Show side-by-side honest results.

This is still 'minimal end-to-end' — it's the proof that all four families
work through the pipeline. Real grid search comes after.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from data import binance_vision as bv
from engine.backtest import run_backtest, ExecutionConfig
from strategies.donchian import DonchianBreakout
from strategies.zscore_rev import ZScoreReversion
from strategies.squeeze import SqueezeBreakout
from strategies.volume_anomaly import VolumeAnomaly


def fmt(label: str, sym: str, r) -> str:
    days = (r.equity.index[-1] - r.equity.index[0]).total_seconds() / 86400
    daily = (1 + r.total_return) ** (1 / days) - 1 if days > 0 else 0
    return (
        f"{label:24s} {sym:8s}  "
        f"ret={r.total_return:+8.2%}  "
        f"daily={daily:+.3%}  "
        f"sharpe={r.sharpe:5.2f}  "
        f"dd={r.max_drawdown:+.1%}  "
        f"trades={r.n_trades:4d}  "
        f"win={r.win_rate:.1%}  "
        f"in_mkt={r.bars_in_market:.1%}"
    )


def main():
    cfg = ExecutionConfig(fee_per_side=0.001, slippage_bps=5.0, position_size=1.0)
    pairs = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    # One reasonable config per family. We're not optimizing yet — this is
    # just to see if each family is alive on real data.
    strategies = [
        ("Donchian 100/50 LO",   DonchianBreakout(params={
            "entry_n": 100, "exit_n": 50, "long_only": True, "regime_ma": 0
        })),
        ("Donchian 50/25 LO",    DonchianBreakout(params={
            "entry_n": 50, "exit_n": 25, "long_only": True, "regime_ma": 0
        })),
        ("ZScoreRev 50/2.0 LO",  ZScoreReversion(params={
            "lookback": 50, "entry_z": 2.0, "exit_z": 0.5, "long_only": True
        })),
        ("ZScoreRev 50/2.5 L/S", ZScoreReversion(params={
            "lookback": 50, "entry_z": 2.5, "exit_z": 0.0, "long_only": False
        })),
        ("Squeeze 100/25/20 LO", SqueezeBreakout(params={
            "atr_n": 14, "squeeze_lookback": 100, "squeeze_pctile": 25.0,
            "breakout_n": 20, "max_hold_bars": 24, "long_only": True
        })),
        ("Squeeze 200/30/30 LO", SqueezeBreakout(params={
            "atr_n": 14, "squeeze_lookback": 200, "squeeze_pctile": 30.0,
            "breakout_n": 30, "max_hold_bars": 48, "long_only": True
        })),
        ("VolAnom 50/2.5/6 LO",  VolumeAnomaly(params={
            "vol_lookback": 50, "vol_z_threshold": 2.5, "min_body_pct": 0.2,
            "hold_bars": 6, "long_only": True
        })),
        ("VolAnom 100/3.0/12 LO", VolumeAnomaly(params={
            "vol_lookback": 100, "vol_z_threshold": 3.0, "min_body_pct": 0.3,
            "hold_bars": 12, "long_only": True
        })),
    ]

    print(f"Loading {len(pairs)} pairs, 1h, 2024...")
    data = {}
    for sym in pairs:
        data[sym] = bv.load(sym, "1h", "2024-01-01", "2024-12-31")
        bh = data[sym]['open'].iloc[-1] / data[sym]['open'].iloc[0] - 1
        print(f"  {sym}: {len(data[sym])} bars  B&H={bh:+.1%}")

    print("\n=== Strategy x Pair grid ===\n")
    rows = []
    for label, strat in strategies:
        for sym in pairs:
            df = data[sym]
            sig = strat.signal(df)
            r = run_backtest(df, sig, cfg)
            print(fmt(label, sym, r))
            rows.append({
                "strategy": label,
                "symbol": sym,
                "total_return": r.total_return,
                "sharpe": r.sharpe,
                "max_dd": r.max_drawdown,
                "trades": r.n_trades,
                "win_rate": r.win_rate,
                "bars_in_market": r.bars_in_market,
            })
        print()

    # Also show buy-and-hold for reference
    print("=== Buy-and-hold reference ===")
    for sym in pairs:
        df = data[sym]
        bh = df['open'].iloc[-1] / df['open'].iloc[0] - 1
        days = (df.index[-1] - df.index[0]).total_seconds() / 86400
        daily = (1 + bh) ** (1 / days) - 1
        print(f"  {sym}: total={bh:+.2%}  daily={daily:+.3%}")

    # Summary by strategy (averaged across pairs)
    print("\n=== Average across 3 pairs ===")
    summary = pd.DataFrame(rows)
    avg = summary.groupby("strategy").agg({
        "total_return": "mean",
        "sharpe": "mean",
        "max_dd": "mean",
        "trades": "mean",
        "win_rate": "mean",
        "bars_in_market": "mean",
    }).sort_values("sharpe", ascending=False)
    print(avg.to_string(float_format=lambda x: f"{x:+.3f}"))


if __name__ == "__main__":
    main()
