"""Cost model for backtest fills.

Maps to ``skills/forex-cost-modeling``: spread, commission, slippage, and
swap/rollover. A backtest that ignores these is a lie. Costs are reported
broken-down in the :class:`CostBreakdown` accumulator so we can attribute
exactly where simulated edge went.

Convention used by the engine:

- The strategy decides on **mid** prices at bar ``t``'s close.
- The fill happens at bar ``t+1``'s open price, adjusted as follows:
  - **BUY** (going long or closing short): pay ``ref + half_spread + slippage``.
  - **SELL** (going short or closing long): receive ``ref - half_spread - slippage``.
- Every fill pays ``size * commission_per_unit``.
- Open positions held over the daily rollover hour pay/receive swap; the
  Wednesday rollover charges ``weekend_swap_multiplier`` (default 3) to
  cover the weekend.

Defaults below are conservative round numbers for EURUSD-class spot pairs.
Per-pair overrides are the right answer in production; the dev CLI
exposes a single pair config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from core.models import Direction


@dataclass(frozen=True, slots=True)
class CostModel:
    """Per-pair cost parameters in price units.

    All values are absolute price-unit quantities (NOT pips). For a typical
    EURUSD pair priced like ``1.0850``, 1 pip = ``0.0001``.

    - ``half_spread``: added to the buy fill, subtracted from the sell
      fill. A spread of 2 pips is modelled as ``half_spread=0.0001``.
    - ``commission_per_unit``: charged on every fill as
      ``size * commission_per_unit``.
    - ``slippage_per_unit``: applied adversely (against the trader) on
      every fill, in the same direction as the spread.
    - ``swap_long_per_unit_per_day``, ``swap_short_per_unit_per_day``:
      daily rollover charge or rebate per position unit. By convention,
      negative = cost to the trader, positive = rebate.
    - ``weekend_swap_multiplier``: applied on the Wednesday rollover to
      cover the upcoming weekend. Default 3.
    """

    half_spread: Decimal = Decimal("0.0001")
    commission_per_unit: Decimal = Decimal("0.00001")
    slippage_per_unit: Decimal = Decimal("0.00005")
    swap_long_per_unit_per_day: Decimal = Decimal("-0.00001")
    swap_short_per_unit_per_day: Decimal = Decimal("0.00001")
    weekend_swap_multiplier: int = 3


@dataclass(slots=True)
class CostBreakdown:
    """Mutable accumulator for cost attribution over a backtest run.

    Reported in :class:`core.backtest.types.BacktestResult` so we can see
    where simulated edge actually went.
    """

    spread_cost: Decimal = field(default_factory=lambda: Decimal("0"))
    commission: Decimal = field(default_factory=lambda: Decimal("0"))
    slippage: Decimal = field(default_factory=lambda: Decimal("0"))
    swap: Decimal = field(default_factory=lambda: Decimal("0"))

    @property
    def total(self) -> Decimal:
        return self.spread_cost + self.commission + self.slippage + self.swap

    def to_dict(self) -> dict[str, str]:
        return {
            "spread_cost": str(self.spread_cost),
            "commission": str(self.commission),
            "slippage": str(self.slippage),
            "swap": str(self.swap),
            "total": str(self.total),
        }


# ---------------------------------------------------------------------------
# Fill-price helpers
# ---------------------------------------------------------------------------


def buy_fill_price(reference_mid: Decimal, model: CostModel) -> Decimal:
    """Fill price for a buy (entering long, OR closing short).

    The trader pays the ask plus modelled slippage. Both move the fill in
    the adverse direction compared to the mid.
    """
    return reference_mid + model.half_spread + model.slippage_per_unit


def sell_fill_price(reference_mid: Decimal, model: CostModel) -> Decimal:
    """Fill price for a sell (entering short, OR closing long).

    The trader receives the bid minus modelled slippage. Both move the
    fill in the adverse direction compared to the mid.
    """
    price = reference_mid - model.half_spread - model.slippage_per_unit
    # Defence in depth: refuse a non-positive fill price (the cost knobs
    # are too aggressive for this market). Surfacing this loudly is the
    # right call — silent zero-priced fills would be worse.
    if price <= 0:
        raise ValueError(
            f"sell fill price {price} is non-positive — cost model is wider "
            f"than the price level {reference_mid}"
        )
    return price


def fill_price_for_direction(
    *,
    reference_mid: Decimal,
    side_is_buy: bool,
    model: CostModel,
) -> Decimal:
    """Convenience: pick buy/sell helper based on the executed side."""
    if side_is_buy:
        return buy_fill_price(reference_mid, model)
    return sell_fill_price(reference_mid, model)


def spread_cost_per_unit(model: CostModel) -> Decimal:
    """Per-unit cost the trader pays at fill time, from spread + slippage."""
    return model.half_spread + model.slippage_per_unit


def commission_for(size: Decimal, model: CostModel) -> Decimal:
    return size * model.commission_per_unit


def swap_per_unit_per_day(direction: Direction, model: CostModel) -> Decimal:
    """Per-unit daily swap for an open position. Sign convention: negative
    = cost to the trader; positive = rebate."""
    if direction is Direction.LONG:
        return model.swap_long_per_unit_per_day
    return model.swap_short_per_unit_per_day


__all__ = [
    "CostBreakdown",
    "CostModel",
    "buy_fill_price",
    "commission_for",
    "fill_price_for_direction",
    "sell_fill_price",
    "spread_cost_per_unit",
    "swap_per_unit_per_day",
]
