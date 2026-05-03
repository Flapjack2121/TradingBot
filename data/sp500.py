"""Compatibility shim — the canonical universe registry now lives in
:mod:`data.universes`. Kept so older imports continue to work.
"""
from .universes import _sp500_live as get_sp500_tickers  # noqa: F401
