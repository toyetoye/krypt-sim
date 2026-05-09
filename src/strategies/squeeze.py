"""
Volatility expansion / "squeeze" breakout.

Logic:
  - Detect a *quiet period*: rolling ATR or rolling range is below its own
    historical percentile (range compression).
  - When price then breaks out of the recent N-bar range, take the breakout
    direction. Premise: compressed volatility tends to expand directionally.
  - Exit on time stop (held N bars) OR when ATR drops back to compression.

Why this primitive:
  - Different from Donchian: requires a *prior quiet period*, not just any
    breakout. Filters out chop-breakouts that immediately fail.
  - Different from mean-reversion: trades WITH the breakout.
  - Captures regime shifts where the market wakes up from a sleepy range.

Notes:
  - ATR is computed as Wilder smoothing-free (simple rolling mean of true range).
  - We avoid look-ahead: the "is it quiet?" check at bar t uses only data up to t.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .base import Strategy


def _atr(df: pd.DataFrame, n: int) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()


@dataclass
class SqueezeBreakout(Strategy):
    """
    Parameters:
        atr_n: lookback for ATR
        squeeze_lookback: window over which we measure "is current ATR low?"
        squeeze_pctile: ATR must be below this percentile of squeeze_lookback
                        window to count as squeezed (0..100). e.g. 25 = bottom quartile.
        breakout_n: bars used for the breakout high/low channel
        max_hold_bars: time stop. Strategy exits after this many bars regardless.
        long_only: skip shorts
    """
    name: str = "squeeze"
    params: dict[str, Any] = field(default_factory=lambda: {
        "atr_n": 14,
        "squeeze_lookback": 100,
        "squeeze_pctile": 25.0,
        "breakout_n": 20,
        "max_hold_bars": 24,
        "long_only": False,
    })

    def signal(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        atr_n = int(p["atr_n"])
        sq_lb = int(p["squeeze_lookback"])
        sq_pct = float(p["squeeze_pctile"]) / 100.0
        brk_n = int(p["breakout_n"])
        max_hold = int(p["max_hold_bars"])
        long_only = bool(p.get("long_only", False))

        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)

        atr = _atr(df, atr_n)
        # Rank current ATR within its trailing squeeze_lookback window
        atr_rank = atr.rolling(sq_lb).apply(
            lambda x: (x <= x[-1]).mean(), raw=True
        )
        # atr_rank is in [0, 1]. Low = current ATR is among the lowest in window.
        squeezed = atr_rank <= sq_pct

        # Breakout channel
        brk_hi = high.rolling(brk_n).max().shift(1)
        brk_lo = low.rolling(brk_n).min().shift(1)

        # Entry: was squeezed in the recent past AND price breaks out.
        # "Recently squeezed" = squeezed at any point in the past few bars
        # (use a small look-back to allow for the breakout bar itself to no longer
        # qualify as squeezed).
        recent_squeeze = squeezed.rolling(5).max().fillna(0).astype(bool)

        long_entry = (close > brk_hi) & recent_squeeze
        short_entry = (close < brk_lo) & recent_squeeze & (not long_only)

        pos = np.zeros(len(close), dtype=np.int8)
        cur = 0
        bars_held = 0
        le = long_entry.fillna(False).values
        se = short_entry.fillna(False).values

        for i in range(len(close)):
            if cur == 0:
                if le[i]:
                    cur = 1
                    bars_held = 0
                elif se[i]:
                    cur = -1
                    bars_held = 0
            else:
                bars_held += 1
                # Time stop
                if bars_held >= max_hold:
                    cur = 0
                    bars_held = 0
                # Allow flip if opposite squeeze-breakout fires
                elif cur == 1 and se[i]:
                    cur = -1
                    bars_held = 0
                elif cur == -1 and le[i]:
                    cur = 1
                    bars_held = 0
            pos[i] = cur

        return pd.Series(pos, index=close.index, dtype=float)
