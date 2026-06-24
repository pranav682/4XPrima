"""Market-data interface for the fast loop.

This module defines only the *interface*. A real market-data feed (vendor SDK,
broker websocket, CSV replayer for backtests) plugs in by implementing
:class:`PriceProvider`. The :class:`StubPriceProvider` here is for tests.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Protocol

from core.models import Quote


class PriceProvider(Protocol):
    """Source of current bid/ask for a pair.

    Implementations are responsible for freshness (the caller can read the
    :attr:`Quote.timestamp` to decide whether a quote is stale enough to
    refuse to trade on it — that policy lives upstream, not here).
    """

    def get_quote(self, pair: str) -> Quote:
        """Return the most recent quote for ``pair``.

        Raises:
            KeyError: if the provider has no quote for ``pair``.
        """
        ...


class StubPriceProvider:
    """Trivial in-memory provider, for tests.

    Holds a single quote per pair. ``set_quote`` overwrites; ``get_quote``
    raises ``KeyError`` if the pair was never set — implementations of the
    real feed will surface a similar error when an unknown pair is requested.
    """

    def __init__(self, quotes: dict[str, Quote] | None = None) -> None:
        self._quotes: dict[str, Quote] = {}
        if quotes:
            for pair, quote in quotes.items():
                self.set_quote(pair, quote)

    def set_quote(self, pair: str, quote: Quote) -> None:
        if pair.upper() != quote.pair:
            raise ValueError(
                f"quote pair {quote.pair!r} does not match key {pair!r}"
            )
        self._quotes[quote.pair] = quote

    def set_price(
        self,
        pair: str,
        *,
        bid: Decimal,
        ask: Decimal,
        timestamp: datetime | None = None,
    ) -> None:
        """Convenience: build the Quote and stash it in one call."""
        self.set_quote(
            pair,
            Quote(
                pair=pair,
                bid=bid,
                ask=ask,
                timestamp=timestamp or datetime.now(UTC),
            ),
        )

    def get_quote(self, pair: str) -> Quote:
        return self._quotes[pair.upper()]
