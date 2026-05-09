"""Strategy base class. A strategy is a function: OHLCV -> signal series."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class Strategy(ABC):
    """
    A strategy generates a signal series in {-1, 0, +1} from an OHLCV DataFrame.

    Signal at time t = desired position for bar t+1 (the engine handles the shift).
    """

    name: str = "strategy"
    params: dict[str, Any] = field(default_factory=dict)

    @abstractmethod
    def signal(self, df: pd.DataFrame) -> pd.Series:
        """Return signal series indexed identically to df, values in {-1, 0, +1}."""
        ...

    def describe(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({params_str})"
