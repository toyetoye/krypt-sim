"""Multi-asset strategy base. Sees ALL pairs at once, outputs per-pair signals."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class MultiAssetStrategy(ABC):
    """
    A multi-asset strategy generates a position matrix across many pairs simultaneously.

    Where Strategy.signal(df) takes one pair and returns a single signal series,
    MultiAssetStrategy.signals(panel) takes a panel (dict of {symbol: DataFrame})
    and returns a DataFrame with columns per symbol and rows per timestamp.

    Each cell is the desired position for that (symbol, time) cell, in [-1, +1].
    Cross-sectional strategies typically use fractional positions (e.g. +0.25 if
    long the top quartile of a 4-pair universe).

    Signal at time t = desired position for bar t+1 (engine handles the shift).
    """

    name: str = "multi_asset_strategy"
    params: dict[str, Any] = field(default_factory=dict)

    @abstractmethod
    def signals(self, panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Args:
            panel: dict mapping symbol -> OHLCV DataFrame, all aligned to same index.

        Returns:
            DataFrame indexed by timestamp, columns = symbols, values in [-1, +1].
            Sum of |row| across columns should be <= 1.0 for proper risk normalization.
        """
        ...

    def describe(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({params_str})"
