"""
Cross-sectional momentum / relative strength rotation.

Logic:
  At each rebalance time:
    1. Compute past-N-bar return for each pair in the universe
    2. Rank pairs by return
    3. Long the top K (or top quartile/half), optionally short the bottom K
    4. Hold positions until next rebalance

Why this is fundamentally different from single-asset strategies:
  - Single-asset momentum bets that a pair's recent direction will continue
  - Cross-sectional bets that the *strongest pairs* will keep outperforming
    the *weakest pairs*, regardless of overall market direction
  - In a market crash, single-asset goes all short; cross-sectional just rotates
    to whichever pair is "least bad". This reduces market beta and can keep
    producing returns in regimes that kill directional strategies

Variants:
  - Long-only top-K vs. long-short top/bottom K
  - Different ranks (top 1 vs top quartile vs top half)
  - Different lookback (1 week, 4 weeks, 12 weeks)
  - Different rebalance frequency (daily, weekly, monthly)

Notes on risk normalization:
  - Long-only top K: each long position = 1/K of capital
  - Long-short: each long = 1/K of capital, each short = -1/K of capital
    (so gross exposure = 2*K*(1/K) = 200%; net = 0; we don't lever, so
     position size in the engine should be set to 0.5 to keep gross at 100%)
  - The engine handles per-bar return × position; we just output the matrix
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .multi_asset_base import MultiAssetStrategy


@dataclass
class CrossSectionalMomentum(MultiAssetStrategy):
    """
    Parameters:
        lookback_bars:    bars to compute past return for ranking
        rebalance_bars:   how often to recompute rankings (must be >= 1)
        top_k:            how many top performers to go long.
                          If 0, use top_quantile instead.
        top_quantile:     fraction of universe to go long (e.g. 0.25 = top quartile).
                          Ignored if top_k > 0.
        long_short:       if True, also short the bottom symmetrically.
        min_universe:     minimum pairs needed in universe to take any positions.
                          Below this, sit flat (e.g. early in data when some pairs
                          haven't listed yet).
    """
    name: str = "cs_momentum"
    params: dict[str, Any] = field(default_factory=lambda: {
        "lookback_bars": 20,
        "rebalance_bars": 5,
        "top_k": 0,
        "top_quantile": 0.25,
        "long_short": False,
        "min_universe": 4,
    })

    def signals(self, panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
        p = self.params
        lookback = int(p["lookback_bars"])
        rebal = int(p["rebalance_bars"])
        top_k = int(p["top_k"])
        top_q = float(p["top_quantile"])
        long_short = bool(p["long_short"])
        min_uni = int(p["min_universe"])

        # Build aligned close-price matrix: rows = timestamp, cols = symbols
        closes = pd.DataFrame({sym: df["close"] for sym, df in panel.items()})
        closes = closes.sort_index()

        # Compute past-N-bar return for each pair at each bar
        past_ret = closes.pct_change(lookback)

        # Initialize position matrix (same shape as closes)
        positions = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)

        # Walk forward, recompute ranks every `rebal` bars, hold between
        last_rebal_idx = -10**9  # so first eligible bar triggers rebalance
        current_positions = pd.Series(0.0, index=closes.columns)

        ret_values = past_ret.values
        idx = closes.index

        for i in range(len(idx)):
            if i - last_rebal_idx >= rebal:
                # Rebalance time
                row = past_ret.iloc[i]
                # Drop NaN (pairs not yet listed or insufficient lookback)
                valid = row.dropna()

                if len(valid) >= min_uni:
                    # Determine number to long
                    if top_k > 0:
                        n_long = min(top_k, len(valid))
                    else:
                        n_long = max(1, int(round(len(valid) * top_q)))

                    sorted_syms = valid.sort_values(ascending=False).index
                    longs = sorted_syms[:n_long]

                    new_positions = pd.Series(0.0, index=closes.columns)

                    if long_short:
                        shorts = sorted_syms[-n_long:]
                        # 50% gross long, 50% gross short = 100% gross, 0 net
                        long_size = 0.5 / n_long
                        short_size = 0.5 / n_long
                        new_positions[longs] = long_size
                        new_positions[shorts] = -short_size
                    else:
                        # Long-only: equal-weight across top_k
                        long_size = 1.0 / n_long
                        new_positions[longs] = long_size

                    current_positions = new_positions
                else:
                    current_positions = pd.Series(0.0, index=closes.columns)

                last_rebal_idx = i

            positions.iloc[i] = current_positions.values

        return positions
