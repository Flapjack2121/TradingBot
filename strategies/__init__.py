"""Signal generation strategies and the aggregating engine.

Two families of named, well-documented systems are shipped:

Swing (daily bars):
    - 🏆 Minervini Trend Template (SEPA-lite)         [equities]
    - 🔄 Connors RSI-2 mean reversion                  [equities]
    - 🐢 Turtle Donchian Trading System 1              [universal]
    - 💥 Bollinger Band Squeeze Breakout (Bollinger)   [universal]

Day (hourly bars):
    - 🌅 Opening Range Breakout (Crabel)               [universal]
    - 📊 Session VWAP Mean Reversion                   [stocks/FX/crypto]
    - 🪤 Inside-Bar / NR4 Breakout (Crabel)            [universal]
    - 🧭 Pivot Point R1/S1 Breakout                    [universal]

Each strategy advertises:
    * ``modes``         → which trading horizons it belongs in.
    * ``asset_classes`` → which asset classes the literature has documented
                          its edge on (or ``["all"]`` for universal systems).
    * ``source``        → the book / paper / author it's documented in.

Use :func:`registry_for_mode` to obtain just the strategies for one mode,
and :func:`registry_for_asset_class` to filter further.
"""
from .base import BaseStrategy, Signal, SIDE_BUY, SIDE_WAIT, SIDE_AVOID

# Swing strategies
from .minervini import MinerviniTrendTemplate
from .mean_reversion import ConnorsRSI2, MeanReversion          # alias kept
from .breakout import TurtleSystem, DonchianBreakout            # alias kept
from .bb_breakout import BollingerBandBreakout

# Day strategies
from .orb import OpeningRangeBreakout
from .vwap_reversion import VWAPReversion
from .inside_bar import InsideBarBreakout
from .pivot_breakout import PivotPointBreakout

from .engine import SignalEngine

REGISTRY: dict[str, type[BaseStrategy]] = {
    # Swing
    MinerviniTrendTemplate.name:  MinerviniTrendTemplate,
    ConnorsRSI2.name:             ConnorsRSI2,
    TurtleSystem.name:            TurtleSystem,
    BollingerBandBreakout.name:   BollingerBandBreakout,
    # Day
    OpeningRangeBreakout.name:    OpeningRangeBreakout,
    VWAPReversion.name:           VWAPReversion,
    InsideBarBreakout.name:       InsideBarBreakout,
    PivotPointBreakout.name:      PivotPointBreakout,
}

# Display labels — pulled from each class so they stay in one place.
STRATEGY_LABELS: dict[str, str] = {name: cls.label for name, cls in REGISTRY.items()}


def registry_for_mode(mode_key: str) -> dict[str, type[BaseStrategy]]:
    """Return only the strategies suitable for the given trading mode."""
    return {name: cls for name, cls in REGISTRY.items() if mode_key in cls.modes}


def registry_for_asset_class(asset_class: str,
                              mode_key: str | None = None
                              ) -> dict[str, type[BaseStrategy]]:
    """Return strategies whose literature documents an edge on this asset class.

    Pass ``mode_key`` to also constrain by trading horizon.
    """
    base = registry_for_mode(mode_key) if mode_key else REGISTRY
    return {name: cls for name, cls in base.items() if cls.suits(asset_class)}


__all__ = [
    "BaseStrategy", "Signal", "SIDE_BUY", "SIDE_WAIT", "SIDE_AVOID",
    "MinerviniTrendTemplate", "ConnorsRSI2", "MeanReversion",
    "TurtleSystem", "DonchianBreakout", "BollingerBandBreakout",
    "OpeningRangeBreakout", "VWAPReversion", "InsideBarBreakout",
    "PivotPointBreakout",
    "SignalEngine", "REGISTRY", "STRATEGY_LABELS",
    "registry_for_mode", "registry_for_asset_class",
]
