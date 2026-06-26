"""USD cost estimation for OpenAI calls.

Cost table is the single source of truth for the runner's budget enforcement.
Update when OpenAI pricing changes — verified against
developers.openai.com/api/docs (2026-06-26).

OpenAI bills cached input at a discount from base input. The exact rate has
varied across models; we use a 50% multiplier as a conservative default that
matches OpenAI's published rates on the GPT-5 family. Re-check on each price
update.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Final

# (input USD per MTok, output USD per MTok). Verify on price changes.
_COSTS_PER_MTOK: Final[dict[str, tuple[Decimal, Decimal]]] = {
    # nano pricing is not explicitly published in the docs page we verified;
    # this placeholder is the conservative figure used in our budget config
    # and treated as an upper bound. Bump if the docs ever expose a higher
    # number.
    "gpt-5.4-nano": (Decimal("0.10"), Decimal("0.40")),
    "gpt-5.4": (Decimal("2.50"), Decimal("15.00")),
    "gpt-5.5": (Decimal("5.00"), Decimal("30.00")),
}

CACHED_INPUT_MULTIPLIER: Final[Decimal] = Decimal("0.50")


def estimate_cost(
    model: str,
    *,
    prompt_tokens: int,
    cached_tokens: int,
    completion_tokens: int,
) -> Decimal:
    """USD cost estimate for a single call.

    ``cached_tokens`` is the subset of ``prompt_tokens`` served from cache
    (per OpenAI's usage shape). Non-cached prompt tokens cost the base rate;
    cached pay :data:`CACHED_INPUT_MULTIPLIER` of base. Output is straight.

    Returns ``Decimal("0")`` for unknown models — budget enforcement is the
    safety net there, not the estimator.
    """
    rates = _COSTS_PER_MTOK.get(model)
    if rates is None:
        return Decimal("0")
    input_rate, output_rate = rates
    non_cached = max(0, prompt_tokens - cached_tokens)
    cost = (
        Decimal(non_cached) * input_rate / Decimal("1000000")
        + Decimal(cached_tokens) * input_rate * CACHED_INPUT_MULTIPLIER / Decimal("1000000")
        + Decimal(completion_tokens) * output_rate / Decimal("1000000")
    )
    return cost.quantize(Decimal("0.000001"))


__all__ = ["CACHED_INPUT_MULTIPLIER", "estimate_cost"]
