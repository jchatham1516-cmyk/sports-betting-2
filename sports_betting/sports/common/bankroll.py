from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BankrollState:
    initial: float
    current: float
    peak: float
    max_drawdown: float = 0.0

    def apply_pnl(self, pnl: float) -> None:
        self.current += pnl
        self.peak = max(self.peak, self.current)
        dd = (self.peak - self.current) / self.peak if self.peak else 0.0
        self.max_drawdown = max(self.max_drawdown, dd)
