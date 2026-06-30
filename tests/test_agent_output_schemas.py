"""Guard: agent OUTPUT models must produce OpenAI-strict-mode-safe JSON schemas.

OpenAI Structured Outputs (and especially the HEAVY tier) reject regex
*lookaround* in a field `pattern`. pydantic's default `Decimal` schema emits
exactly that (a `(?!...)` negative lookahead), so any agent output model with a
plain `Decimal` field gets its structured-output call rejected at request time.

This bug is invisible to the mocked-LLM agent tests (the schema is never sent to
OpenAI) — it only shows up on a live call. These hermetic checks catch it: they
assert no agent output schema contains a lookaround pattern. Decimal fields in
output models must use ``core.models.JsonDecimal`` (a string-schema Decimal).
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from core.models import (
    BacktestVerdictSet,
    CriticVerdictSet,
    CycleReport,
    MarketContextReport,
    StrategyProposal,
)

_LOOKAROUND = re.compile(r"\(\?[=!<]")  # (?= (?! (?<= (?<!


def _lookaround_patterns(schema: dict[str, Any]) -> list[str]:
    """Every regex ``pattern`` in the schema that uses a lookaround."""
    found: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            pat = node.get("pattern")
            if isinstance(pat, str) and _LOOKAROUND.search(pat):
                found.append(pat)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(schema)
    return found


@pytest.mark.parametrize(
    "model",
    [BacktestVerdictSet, CriticVerdictSet, StrategyProposal, CycleReport],
    ids=lambda m: m.__name__,
)
def test_output_schema_has_no_lookaround_pattern(model: type) -> None:
    patterns = _lookaround_patterns(model.model_json_schema())
    assert not patterns, f"{model.__name__} emits OpenAI-incompatible lookaround: {patterns}"


@pytest.mark.xfail(
    reason="market_context runs on the DEFAULT tier, which tolerates the Decimal "
    "lookaround; its constrained-Decimal scores still need the JsonDecimal "
    "treatment before it could move to the HEAVY tier.",
    strict=True,
)
def test_market_context_schema_lookaround_is_known_debt() -> None:
    assert not _lookaround_patterns(MarketContextReport.model_json_schema())
