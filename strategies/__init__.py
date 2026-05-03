"""Signal generation strategies and the aggregating engine."""
from .base import BaseStrategy, Signal, SIDE_BUY, SIDE_WAIT, SIDE_AVOID
from .triple_confirmation import TripleConfirmation
from .mean_reversion import MeanReversion
from .breakout import DonchianBreakout
from .engine import SignalEngine

REGISTRY: dict[str, type[BaseStrategy]] = {
    "triple_confirmation": TripleConfirmation,
    "mean_reversion": MeanReversion,
    "breakout": DonchianBreakout,
}

STRATEGY_LABELS: dict[str, str] = {
    "triple_confirmation": "🎯 Triple Confirmation",
    "mean_reversion":      "🔄 RSI-2 Mean Reversion",
    "breakout":            "🚀 Donchian Breakout",
}

__all__ = [
    "BaseStrategy", "Signal", "SIDE_BUY", "SIDE_WAIT", "SIDE_AVOID",
    "TripleConfirmation", "MeanReversion", "DonchianBreakout",
    "SignalEngine", "REGISTRY", "STRATEGY_LABELS",
]
