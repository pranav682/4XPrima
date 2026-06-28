"""Deterministic, read-only analysis utilities for the fast loop.

Nothing here places trades, calls an LLM, or contains strategy logic. These are
structural characterisation tools used to *narrow* the search space before any
strategy work begins. See :mod:`core.analysis.pair_screener`.
"""

from core.analysis.pair_screener import (
    CorrelationMatrix,
    ExclusionEntry,
    PairScreener,
    ScreenConfig,
    ScreeningReport,
    ShortlistEntry,
)

__all__ = [
    "CorrelationMatrix",
    "ExclusionEntry",
    "PairScreener",
    "ScreenConfig",
    "ScreeningReport",
    "ShortlistEntry",
]
