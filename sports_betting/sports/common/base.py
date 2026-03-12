from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from sports_betting.models.types import GameContext, MarketPrediction


@dataclass
class MarketThresholds:
    min_edge: float
    min_ev: float
    min_confidence: float


class SportModel(ABC):
    sport: str

    @abstractmethod
    def predict_game(self, game: GameContext) -> list[MarketPrediction]:
        raise NotImplementedError
