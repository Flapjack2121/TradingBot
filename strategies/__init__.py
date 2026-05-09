"""Signal generation strategies and the aggregating engine.

Two families of named, well-documented systems are shipped:

Swing (daily bars):
    - Mark Minervini Trend Template (SEPA-lite)
    - Larry Connors RSI-2 mean reversion
    - Turtle Trading System 1 (20-bar Donchian)

Day (hourly bars):
    - Opening Range Breakout (Crabel)
    - Session-anchored VWAP mean reversion
    - Inside-Bar / NR4 compression breakout (Crabel)

Each strategy advertises its supported modes via the ``modes`` class
attribute. Use :func:`registry_for_mode` to obtain just the strategies
appropriate for one mode.
"""
from .base import BaseStrategy, Signal, SIDE_BUY, SIDE_WAIT, SIDE_AVOID

# Swing strategies
from .minervini import MinerviniTrendTemplate
from .mean_reversion import ConnorsRSI2, MeanReversion          # alias kept
from .breakout import TurtleSystem, DonchianBreakout            # alias kept

# Day strategies
from .orb import OpeningRangeBreakout
from .vwap_reversion import VWAPReversion
from .inside_bar import InsideBarBreakout

from .engine import SignalEngine

REGISTRY: dict[str, type[BaseStrategy]] = {
    # Swing
    MinerviniTrendTemplate.name:  MinerviniTrendTemplate,
    ConnorsRSI2.name:             ConnorsRSI2,
    TurtleSystem.name:            TurtleSystem,
    # Day
    OpeningRangeBreakout.name:    OpeningRangeBreakout,
    VWAPReversion.name:           VWAPReversion,
    InsideBarBreakout.name:       InsideBarBreakout,
}

# Display labels — pulled from each class so they stay in one place.
STRATEGY_LABELS: dict[str, str] = {name: cls.label for name, cls in REGISTRY.items()}


def registry_for_mode(mode_key: str) -> dict[str, type[BaseStrategy]]:
    """Return only the strategies suitable for the given trading mode."""
    return {name: cls for name, cls in REGISTRY.items() if mode_key in cls.modes}


__all__ = [
    "BaseStrategy", "Signal", "SIDE_BUY", "SIDE_WAIT", "SIDE_AVOID",
    "MinerviniTrendTemplate", "ConnorsRSI2", "MeanReversion",
    "TurtleSystem", "DonchianBreakout",
    "OpeningRangeBreakout", "VWAPReversion", "InsideBarBreakout",
    "SignalEngine", "REGISTRY", "STRATEGY_LABELS", "registry_for_mode",
]
