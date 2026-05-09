"""
Vectorized backtest engine.

Design principles:
  - Position is determined at bar close, executed at next bar's open
    (no look-ahead, no in-bar fills against the same bar that produced the signal)
  - Fees and slippage subtracted on every position change, not just round-trips
  - Returns are simple bar-to-bar; equity curve compounds them
  - One position per symbol at a time (sized to a fraction of equity, default 100%)

This is a single-asset backtester. Multi-asset is a runner-level concern
(run each symbol independently, then aggregate).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExecutionConfig:
    """Realistic-ish execution costs for retail crypto."""
    # Fees per side. MEXC default: 0% maker, 0.10% taker. We assume taker.
    # (Maker-only would require more complex order modeling.)
    fee_per_side: float = 0.001

    # Slippage in bps applied on every fill, additive to fees.
    # Calibrated pessimistically: small caps will lie about fillable size.
    slippage_bps: float = 5.0  # 0.05%

    # Fraction of equity deployed per trade. 1.0 = full, 0.5 = half, etc.
    # Lower than 1.0 leaves cash buffer, useful for multi-strategy portfolios.
    position_size: float = 1.0


@dataclass
class BacktestResult:
    """Output of a backtest run. All series indexed by bar timestamp."""
    equity: pd.Series           # equity curve in account currency units (start = 1.0)
    returns: pd.Series          # net per-bar returns
    position: pd.Series         # position held during each bar (-1, 0, 1)
    trades: pd.DataFrame        # rows: entry_time, exit_time, side, gross, fees, net, bars_held

    # Headline metrics
    total_return: float
    sharpe: float               # annualized, assumes bars are equal-spaced
    max_drawdown: float         # negative number
    n_trades: int
    win_rate: float
    avg_trade_return: float
    bars_in_market: float       # fraction of bars with a position open

    # Execution costs we paid
    total_fees: float
    total_slippage: float

    # Metadata
    bars_per_year: float


def _bars_per_year(index: pd.DatetimeIndex) -> float:
    """Estimate bars per year from index spacing."""
    if len(index) < 2:
        return 365.0
    median_delta = pd.Series(index).diff().median()
    seconds = median_delta.total_seconds()
    return (365.25 * 24 * 3600) / seconds


def run_backtest(
    df: pd.DataFrame,
    signal: pd.Series,
    cfg: ExecutionConfig = ExecutionConfig(),
) -> BacktestResult:
    """
    Execute a backtest on bar-close OHLCV data.

    Args:
        df: DataFrame with columns at least ['open', 'close'], indexed by bar time.
        signal: Series aligned to df.index with values in {-1, 0, +1}
                indicating desired position for the NEXT bar.
                A signal value at time t triggers execution at the open of bar t+1.

    Returns:
        BacktestResult with equity curve, trades, and metrics.
    """
    if not df.index.equals(signal.index):
        signal = signal.reindex(df.index).fillna(0)

    # Shift signal: signal at time t -> position held during bar t+1
    # Position is "what we hold during this bar"
    position = signal.shift(1).fillna(0).astype(float)
    position *= cfg.position_size

    # Bar return: open-to-open is more honest than close-to-close because we
    # execute at the open. So the return earned during bar t while holding
    # `position[t]` is (open[t+1] - open[t]) / open[t].
    open_px = df["open"].astype(float)
    open_ret = open_px.pct_change().shift(-1).fillna(0)
    # ^ open_ret[t] = (open[t+1] - open[t]) / open[t]
    # The last bar has no t+1, so its return is 0.

    gross_returns = position * open_ret

    # Costs: every change in position incurs fee + slippage on the size of the change.
    # E.g. flipping from +1 to -1 is a 2.0 turnover -> 2x fees.
    turnover = position.diff().abs().fillna(position.abs())
    cost_per_unit = cfg.fee_per_side + (cfg.slippage_bps / 10_000.0)
    costs = turnover * cost_per_unit

    net_returns = gross_returns - costs

    # Equity curve
    equity = (1.0 + net_returns).cumprod()

    # Trade ledger: identify entry/exit pairs
    trades = _build_trade_ledger(position, open_px, net_returns, costs)

    # Metrics
    bpy = _bars_per_year(df.index)
    total_ret = float(equity.iloc[-1] - 1.0) if len(equity) else 0.0
    sharpe = _sharpe(net_returns, bpy)
    max_dd = _max_drawdown(equity)
    win_rate = float((trades["net"] > 0).mean()) if len(trades) else 0.0
    avg_trade = float(trades["net"].mean()) if len(trades) else 0.0
    bars_in_mkt = float((position != 0).mean())

    return BacktestResult(
        equity=equity,
        returns=net_returns,
        position=position,
        trades=trades,
        total_return=total_ret,
        sharpe=sharpe,
        max_drawdown=max_dd,
        n_trades=len(trades),
        win_rate=win_rate,
        avg_trade_return=avg_trade,
        bars_in_market=bars_in_mkt,
        total_fees=float((turnover * cfg.fee_per_side).sum()),
        total_slippage=float((turnover * cfg.slippage_bps / 10_000.0).sum()),
        bars_per_year=bpy,
    )


def _build_trade_ledger(
    position: pd.Series,
    open_px: pd.Series,
    net_returns: pd.Series,
    costs: pd.Series,
) -> pd.DataFrame:
    """Build a ledger of round-trip trades from a position series."""
    pos = position.values
    idx = position.index

    trades = []
    in_trade = False
    entry_i = 0
    entry_side = 0

    for i in range(len(pos)):
        cur = pos[i]
        prev = pos[i - 1] if i > 0 else 0

        # Trade open: position changes from 0 to non-zero, OR flips sign
        if not in_trade and cur != 0:
            in_trade = True
            entry_i = i
            entry_side = np.sign(cur)
        elif in_trade and (cur == 0 or np.sign(cur) != entry_side):
            # Trade close
            exit_i = i
            trade_returns = net_returns.iloc[entry_i:exit_i]
            trade_costs = costs.iloc[entry_i:exit_i]
            trades.append({
                "entry_time": idx[entry_i],
                "exit_time": idx[exit_i],
                "side": int(entry_side),
                "bars_held": exit_i - entry_i,
                "gross": float((trade_returns + trade_costs).sum()),
                "fees_slip": float(trade_costs.sum()),
                "net": float(trade_returns.sum()),
            })
            # If flipped, we're now in a new trade
            if cur != 0:
                in_trade = True
                entry_i = i
                entry_side = np.sign(cur)
            else:
                in_trade = False

    # Close any open trade at the end of data
    if in_trade:
        trade_returns = net_returns.iloc[entry_i:]
        trade_costs = costs.iloc[entry_i:]
        trades.append({
            "entry_time": idx[entry_i],
            "exit_time": idx[-1],
            "side": int(entry_side),
            "bars_held": len(pos) - entry_i,
            "gross": float((trade_returns + trade_costs).sum()),
            "fees_slip": float(trade_costs.sum()),
            "net": float(trade_returns.sum()),
        })

    return pd.DataFrame(trades)


def _sharpe(returns: pd.Series, bars_per_year: float) -> float:
    """Annualized Sharpe ratio. Zero-mean assumption (no risk-free rate)."""
    r = returns.dropna()
    if len(r) < 2 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(bars_per_year))


def _max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough drawdown as a negative fraction."""
    if len(equity) == 0:
        return 0.0
    running_max = equity.cummax()
    dd = (equity - running_max) / running_max
    return float(dd.min())
