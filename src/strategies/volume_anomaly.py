"""
Volume anomaly directional follow-through.

Logic:
  - Detect bars where volume is significantly above its rolling mean
    (z-score > threshold).
  - Take a position in the direction of the volume bar's price move
    (long if close > open with volume spike; short if close < open).
  - Hold for N bars, then exit. (Optional: trail-out on volume normalization.)

Why this primitive:
  - Volume spikes tend to mark genuine information arrival: news, listings,
    liquidations, large flow. Pure-price strategies can't see this directly.
  - This is the closest thing to "event-driven" we can do from candle data.
  - Different from breakout: a high-volume bar can occur INSIDE a range and
    still signal continuation.

Notes:
  - We use quote_volume (USD-denominated) for stability across pairs of
    different price levels. Falls back to base volume if not present.
  - Signal is generated at bar close, executed at next bar's open (engine
    handles the shift).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .base import Strategy


@dataclass
class VolumeAnomaly(Strategy):
    """
    Parameters:
        vol_lookback: bars for rolling mean/std of volume
        vol_z_threshold: how many std above mean qualifies as "anomaly"
        min_body_pct: minimum |close-open|/open to consider direction reliable
                      (filters out doji-style high-volume bars with no direction)
        hold_bars: bars to hold after entry
        long_only: skip shorts
    """
    name: str = "volume_anomaly"
    params: dict[str, Any] = field(default_factory=lambda: {
        "vol_lookback": 50,
        "vol_z_threshold": 2.5,
        "min_body_pct": 0.2,   # 0.2% of price
        "hold_bars": 6,
        "long_only": False,
    })

    def signal(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        vol_lb = int(p["vol_lookback"])
        vol_z = float(p["vol_z_threshold"])
        min_body = float(p["min_body_pct"]) / 100.0
        hold = int(p["hold_bars"])
        long_only = bool(p.get("long_only", False))

        close = df["close"].astype(float)
        open_ = df["open"].astype(float)

        # Prefer USD volume; fall back to base
        if "quote_volume" in df.columns and df["quote_volume"].sum() > 0:
            vol = df["quote_volume"].astype(float)
        else:
            vol = df["volume"].astype(float)

        # Z-score of volume vs rolling mean/std (use ln(vol) so large spikes
        # don't distort std; volume distribution is heavy-tailed)
        log_vol = np.log(vol.replace(0, np.nan))
        vol_mean = log_vol.rolling(vol_lb).mean()
        vol_std = log_vol.rolling(vol_lb).std()
        vol_z_score = (log_vol - vol_mean) / vol_std

        # Bar body direction
        body_pct = (close - open_) / open_

        anomaly = vol_z_score > vol_z
        bullish = body_pct > min_body
        bearish = body_pct < -min_body

        long_entry = anomaly & bullish
        short_entry = anomaly & bearish & (not long_only)

        # State machine with fixed-bar hold
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
                if bars_held >= hold:
                    cur = 0
                    bars_held = 0
                # Allow flip on opposite-direction anomaly
                elif cur == 1 and se[i]:
                    cur = -1
                    bars_held = 0
                elif cur == -1 and le[i]:
                    cur = 1
                    bars_held = 0
            pos[i] = cur

        return pd.Series(pos, index=close.index, dtype=float)
