"""Signal generation strategies and the aggregating engine."""
from .base import BaseStrategy, Signal
from .triple_confirmation import TripleConfirmation
from .mean_reversion import MeanReversion
from .breakout import DonchianBreakout
from .engine import SignalEngine

REGISTRY: dict[str, type[BaseStrategy]] = {
    "triple_confirmation": TripleConfirmation,
    "mean_reversion": MeanReversion,
    "breakout": DonchianBreakout,
}

__all__ = [
    "BaseStrategy", "Signal", "TripleConfirmation", "MeanReversion",
    "DonchianBreakout", "SignalEngine", "REGISTRY",
]
