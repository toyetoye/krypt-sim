"""
Donchian channel breakout momentum strategy.

Logic:
  - Long entry: close breaks above rolling N-bar high (excluding current bar)
  - Short entry: close breaks below rolling N-bar low (excluding current bar)
  - Exit: opposite breakout, OR price crosses M-bar exit channel (M < N)
  - Optional: ATR-based trailing stop, regime filter via slow MA

Why this primitive:
  - The simplest, most-studied momentum signal.
  - Single parameter (channel length) controls behavior cleanly.
  - Survivorship in academic literature is mixed but real on trending instruments.

Critically: this is ONE momentum primitive. We will sweep parameters and run it
across many pairs. If it doesn't survive brutal filtering, it dies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .base import Strategy


@dataclass
class DonchianBreakout(Strategy):
    """
    Parameters:
        entry_n: lookback for entry channel (longer = stronger trend req'd)
        exit_n:  lookback for exit channel (shorter = quicker exit)
        long_only: if True, only take long entries (no shorts)
        regime_ma: optional moving-average length for regime filter.
                   If set, longs only when close > MA, shorts only when close < MA.
                   Set to 0 to disable.
    """
    name: str = "donchian"
    params: dict[str, Any] = field(default_factory=lambda: {
        "entry_n": 20,
        "exit_n": 10,
        "long_only": False,
        "regime_ma": 0,
    })

    def signal(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        entry_n = int(p["entry_n"])
        exit_n = int(p["exit_n"])
        long_only = bool(p.get("long_only", False))
        regime_ma = int(p.get("regime_ma", 0))

        close = df["close"]
        high = df["high"]
        low = df["low"]

        # Channels (exclude current bar to avoid trivial self-breakout)
        entry_hi = high.rolling(entry_n).max().shift(1)
        entry_lo = low.rolling(entry_n).min().shift(1)
        exit_hi = high.rolling(exit_n).max().shift(1)
        exit_lo = low.rolling(exit_n).min().shift(1)

        # Regime filter
        if regime_ma > 0:
            ma = close.rolling(regime_ma).mean()
            long_ok = close > ma
            short_ok = close < ma
        else:
            long_ok = pd.Series(True, index=close.index)
            short_ok = pd.Series(True, index=close.index)

        # State machine: vectorized via forward-fill of entry/exit events
        long_entry = (close > entry_hi) & long_ok
        long_exit = close < exit_lo
        short_entry = (close < entry_lo) & short_ok & (not long_only)
        short_exit = close > exit_hi

        # Build position by walking events. Vectorized approach:
        #   raw_signal = +1 on long_entry, -1 on short_entry, 0 elsewhere
        #   then forward-fill until an exit event resets to 0
        raw = pd.Series(0, index=close.index, dtype=float)
        raw[long_entry] = 1
        if not long_only:
            raw[short_entry] = -1

        # Manual loop for state transitions. This is the part that's hard to
        # truly vectorize without a state machine. For thousands of variants
        # we'd JIT this with numba; for now it's clear Python.
        pos = np.zeros(len(close), dtype=np.int8)
        cur = 0
        le = long_entry.values
        ls = long_exit.values
        se = short_entry.values
        ss = short_exit.values

        for i in range(len(close)):
            if cur == 0:
                if le[i]:
                    cur = 1
                elif se[i]:
                    cur = -1
            elif cur == 1:
                if ls[i]:
                    cur = 0
                # Allow flip-on-opposite-breakout for momentum chasing
                if se[i] and not long_only:
                    cur = -1
            elif cur == -1:
                if ss[i]:
                    cur = 0
                if le[i]:
                    cur = 1
            pos[i] = cur

        return pd.Series(pos, index=close.index, dtype=float)
