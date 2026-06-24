"""The I/O boundary between the fast loop and a brokerage account.

The :class:`Broker` Protocol is the only place in the codebase where order
execution may cross a process boundary in production. All implementations
(paper, real brokers, replay) must conform to it.

**Contract: callers must go through** :class:`core.execution.ExecutionEngine`,
**not call broker methods directly.** ExecutionEngine is the single place that
runs the :class:`core.risk_manager.RiskManager` before any order. Direct calls
to :meth:`Broker.place_order` would bypass the risk gates and break the hard
invariant from ``CLAUDE.md``: *no order reaches the market without first
clearing the risk manager*.

This module defines only the Protocol. Implementations live in their own
modules — :class:`core.paper_broker.PaperBroker` for in-process paper trading;
real-broker adapters land in later stages.
"""

from __future__ import annotations

from typing import Protocol

from core.models import AccountState, Fill, OrderRequest, Position


class Broker(Protocol):
    """Minimal broker interface.

    Implementations may raise on any operational failure (network, auth,
    rejection by the venue). The :class:`ExecutionEngine` catches those,
    trips the kill switch, and audits the failure — implementations
    themselves should not silently swallow errors.
    """

    def get_account_state(self) -> AccountState:
        """Return a fresh, frozen snapshot of the account.

        The returned ``AccountState`` is what the risk manager evaluates
        against. It must reflect the broker's authoritative view *right now*
        (equity, peak_equity, day_start_equity, open_positions).
        """
        ...

    def get_open_positions(self) -> list[Position]:
        """List current open positions. May be ordered by entry time or not —
        ``ExecutionEngine`` does not rely on the order."""
        ...

    def place_order(self, order: OrderRequest) -> Fill:
        """Place ``order`` and return the resulting fill.

        Implementations:
        - Must NOT be called directly — see the module docstring.
        - Must raise on rejection by the venue; do not return a sentinel.
        - Must report the *actual* fill price and commission, not the
          requested entry price.
        """
        ...

    def close_position(self, position: Position) -> Fill:
        """Close ``position`` and return the resulting (closing-side) fill.

        The returned ``Fill.direction`` is the side of the fill itself (i.e.
        opposite to ``position.direction``).

        Raises:
            ValueError: if ``position`` is not currently open.
        """
        ...
