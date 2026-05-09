"""
Z-score mean reversion on price extremes.

Logic:
  - Compute rolling mean and std over `lookback` bars.
  - Z-score = (close - mean) / std.
  - Long when z < -entry_z (price stretched far below mean).
  - Short when z > +entry_z (stretched above).
  - Exit when z crosses back to ±exit_z (closer to mean).

Why this primitive:
  - Genuinely orthogonal to momentum. If Donchian wins on trend, this should
    win on chop / range-bound regimes.
  - Single core parameter (entry_z) controls aggressiveness.
  - Optional regime filter can suppress trades during strong trends
    (mean reversion fights trends and loses; we want to *only* trade chop).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .base import Strategy


@dataclass
class ZScoreReversion(Strategy):
    """
    Parameters:
        lookback: bars for mean/std calculation
        entry_z: |z| threshold to enter (e.g. 2.0)
        exit_z: |z| threshold to exit (e.g. 0.5). Smaller = held longer.
        long_only: skip shorts
        regime_off_ma: if set, disable trades when |close - MA| / MA > threshold,
                       i.e. when price is far from a longer-term MA (trending hard).
                       0 disables this filter.
        regime_threshold_pct: percent deviation from MA above which we sit out.
    """
    name: str = "zscore_rev"
    params: dict[str, Any] = field(default_factory=lambda: {
        "lookback": 50,
        "entry_z": 2.0,
        "exit_z": 0.5,
        "long_only": False,
        "regime_off_ma": 0,
        "regime_threshold_pct": 5.0,
    })

    def signal(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        lookback = int(p["lookback"])
        entry_z = float(p["entry_z"])
        exit_z = float(p["exit_z"])
        long_only = bool(p.get("long_only", False))
        regime_off_ma = int(p.get("regime_off_ma", 0))
        regime_threshold = float(p.get("regime_threshold_pct", 5.0)) / 100.0

        close = df["close"].astype(float)
        mean = close.rolling(lookback).mean()
        std = close.rolling(lookback).std()
        z = (close - mean) / std

        # Regime filter: skip when far from a longer MA
        if regime_off_ma > 0:
            ma = close.rolling(regime_off_ma).mean()
            deviation = (close - ma).abs() / ma
            quiet = deviation < regime_threshold
        else:
            quiet = pd.Series(True, index=close.index)

        # State machine
        long_entry = (z < -entry_z) & quiet
        long_exit = z > -exit_z
        short_entry = (z > entry_z) & quiet & (not long_only)
        short_exit = z < exit_z

        pos = np.zeros(len(close), dtype=np.int8)
        cur = 0
        le = long_entry.fillna(False).values
        ls = long_exit.fillna(False).values
        se = short_entry.fillna(False).values
        ss = short_exit.fillna(False).values

        for i in range(len(close)):
            if cur == 0:
                if le[i]:
                    cur = 1
                elif se[i]:
                    cur = -1
            elif cur == 1:
                if ls[i]:
                    cur = 0
                if se[i] and not long_only:
                    cur = -1
            elif cur == -1:
                if ss[i]:
                    cur = 0
                if le[i]:
                    cur = 1
            pos[i] = cur

        return pd.Series(pos, index=close.index, dtype=float)
