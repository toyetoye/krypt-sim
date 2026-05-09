"""
Multi-asset / portfolio backtest engine.

Where engine.backtest runs ONE strategy on ONE pair, this runs ONE multi-asset
strategy across MANY pairs simultaneously, with a single shared equity curve.

Inputs:
  panel: dict of {symbol: OHLCV DataFrame}, all aligned to same index
  signals: DataFrame from MultiAssetStrategy.signals() - rows = timestamp,
           cols = symbol, values in [-1, +1]

The engine:
  1. Shifts signals by 1 bar (signal at t -> position during bar t+1)
  2. Computes per-symbol per-bar returns from open-to-open
  3. Multiplies position[symbol, t] * return[symbol, t] for portfolio return
  4. Subtracts fees + slippage on every position change per symbol
  5. Aggregates to single equity curve

This is intentionally separate code from engine.backtest. It would be cleaner
to refactor both onto a shared base, but separation now means the working
single-asset path can't break while we develop the multi-asset path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import ExecutionConfig


@dataclass
class PortfolioBacktestResult:
    equity: pd.Series
    returns: pd.Series              # net portfolio returns per bar
    positions: pd.DataFrame         # position matrix
    per_symbol_returns: pd.DataFrame  # gross per-symbol contributions

    total_return: float
    sharpe: float
    max_drawdown: float
    avg_gross_exposure: float       # mean |position| sum per bar
    avg_net_exposure: float         # mean signed position sum per bar
    n_rebalances: int               # bars where ANY position changed
    bars_in_market: float

    total_fees: float
    total_slippage: float
    bars_per_year: float


def _bars_per_year(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 365.0
    median_delta = pd.Series(index).diff().median()
    seconds = median_delta.total_seconds()
    return (365.25 * 24 * 3600) / seconds


def run_portfolio_backtest(
    panel: dict[str, pd.DataFrame],
    signals: pd.DataFrame,
    cfg: ExecutionConfig = ExecutionConfig(),
) -> PortfolioBacktestResult:
    """
    Args:
        panel: dict of {symbol: OHLCV DataFrame} - all must share the same index
        signals: position matrix from a MultiAssetStrategy
        cfg: execution costs

    Returns:
        PortfolioBacktestResult with equity curve and metrics.
    """
    # Align everything to a common index (intersection of all panels and signals)
    common_idx = signals.index
    for df in panel.values():
        common_idx = common_idx.intersection(df.index)
    common_idx = common_idx.sort_values()

    if len(common_idx) < 10:
        raise RuntimeError(f"Aligned index too short: {len(common_idx)} bars")

    # Build aligned open-price matrix
    opens = pd.DataFrame({
        sym: df.loc[common_idx, "open"].astype(float)
        for sym, df in panel.items()
    })

    # Align signals to the same columns (drop any signal columns not in panel)
    signal_cols = [c for c in signals.columns if c in opens.columns]
    sig = signals.loc[common_idx, signal_cols].fillna(0).astype(float)
    opens = opens[signal_cols]

    # Position[t] = signal[t-1]: we decide at close of t-1, hold during bar t
    positions = sig.shift(1).fillna(0)

    # Per-symbol bar return: (open[t+1] - open[t]) / open[t]
    open_ret = opens.pct_change().shift(-1).fillna(0)

    # Per-symbol gross contribution to portfolio return
    per_symbol_returns = positions * open_ret

    # Portfolio gross return per bar = sum across symbols
    gross_portfolio_ret = per_symbol_returns.sum(axis=1)

    # Costs: per-symbol turnover -> per-symbol fee
    turnover = positions.diff().abs().fillna(positions.abs())
    cost_per_unit = cfg.fee_per_side + (cfg.slippage_bps / 10_000.0)
    per_symbol_costs = turnover * cost_per_unit
    total_costs = per_symbol_costs.sum(axis=1)

    net_returns = gross_portfolio_ret - total_costs

    # Equity curve
    equity = (1.0 + net_returns).cumprod()

    # Metrics
    bpy = _bars_per_year(pd.DatetimeIndex(common_idx))
    total_ret = float(equity.iloc[-1] - 1.0) if len(equity) else 0.0
    sharpe = _sharpe(net_returns, bpy)
    max_dd = _max_drawdown(equity)

    gross_exposure = positions.abs().sum(axis=1)
    net_exposure = positions.sum(axis=1)
    n_rebalances = int((turnover.sum(axis=1) > 0).sum())
    bars_in_mkt = float((gross_exposure > 0).mean())

    return PortfolioBacktestResult(
        equity=equity,
        returns=net_returns,
        positions=positions,
        per_symbol_returns=per_symbol_returns,
        total_return=total_ret,
        sharpe=sharpe,
        max_drawdown=max_dd,
        avg_gross_exposure=float(gross_exposure.mean()),
        avg_net_exposure=float(net_exposure.mean()),
        n_rebalances=n_rebalances,
        bars_in_market=bars_in_mkt,
        total_fees=float((turnover * cfg.fee_per_side).sum().sum()),
        total_slippage=float((turnover * cfg.slippage_bps / 10_000.0).sum().sum()),
        bars_per_year=bpy,
    )


def _sharpe(returns: pd.Series, bars_per_year: float) -> float:
    r = returns.dropna()
    if len(r) < 2 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(bars_per_year))


def _max_drawdown(equity: pd.Series) -> float:
    if len(equity) == 0:
        return 0.0
    running_max = equity.cummax()
    dd = (equity - running_max) / running_max
    return float(dd.min())
