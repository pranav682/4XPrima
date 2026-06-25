"""Tests for core/llm_client.py — fully hermetic (mocked OpenAI SDK).

We verify:

- Tier → model mapping is pinned and correct.
- Stable system / volatile user message structure is what reaches the SDK.
- Structured outputs flow: the SDK's pydantic instance is returned.
- Token accounting captures prompt / cached / completion correctly.
- Schema-validation failure / refusal / missing parsed → LlmClientError.
- The API key never appears in error messages.
- Live smokes (cheap nano + DEFAULT-tier caching check) are gated on
  RUN_LIVE_TESTS=1 + OPENAI_API_KEY; skipped in CI.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import openai
import pytest
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from core.config import OpenAISettings
from core.llm_client import (
    MODEL_CHEAP,
    MODEL_DEFAULT,
    MODEL_HEAVY,
    AgentResponse,
    LlmClientError,
    ModelTier,
    OpenAIProvider,
    TokenUsage,
    model_for_tier,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class TinyReport(BaseModel):
    """Small structured-output model for the LLM client tests."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    title: str
    score: float = Field(ge=0.0, le=1.0)


@dataclass
class FakePromptTokensDetails:
    cached_tokens: int = 0


@dataclass
class FakeUsage:
    prompt_tokens: int = 1500
    completion_tokens: int = 60
    prompt_tokens_details: FakePromptTokensDetails = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.prompt_tokens_details is None:
            self.prompt_tokens_details = FakePromptTokensDetails()


@dataclass
class FakeMessage:
    parsed: Any
    refusal: str | None = None


@dataclass
class FakeChoice:
    message: FakeMessage
    finish_reason: str = "stop"


@dataclass
class FakeCompletion:
    choices: list[FakeChoice]
    usage: FakeUsage


def _ok_completion(
    parsed: TinyReport | None = None,
    *,
    cached_tokens: int = 0,
) -> FakeCompletion:
    parsed = parsed if parsed is not None else TinyReport(title="ok", score=0.5)
    return FakeCompletion(
        choices=[FakeChoice(message=FakeMessage(parsed=parsed))],
        usage=FakeUsage(
            prompt_tokens_details=FakePromptTokensDetails(cached_tokens=cached_tokens)
        ),
    )


def _make_provider(
    sdk: MagicMock | None = None, recorder: MagicMock | None = None
) -> tuple[OpenAIProvider, MagicMock, MagicMock]:
    sdk = sdk or MagicMock()
    recorder = recorder or MagicMock()
    settings = OpenAISettings(api_key=SecretStr("test-key-not-real"))
    provider = OpenAIProvider(
        settings, openai_client=sdk, usage_recorder=recorder
    )
    return provider, sdk, recorder


def _call(
    provider: OpenAIProvider,
    tier: ModelTier = ModelTier.DEFAULT,
    stable: str = "STABLE INSTRUCTIONS — pretend this is long and identical across calls.",
    volatile: str = "VOLATILE DATA for this call.",
) -> tuple[TinyReport, AgentResponse]:
    return provider.generate_structured(
        agent_name="t",
        run_id="r1",
        tier=tier,
        stable_system=stable,
        volatile_user=volatile,
        output_model=TinyReport,
    )


# ---------------------------------------------------------------------------
# Pure constants
# ---------------------------------------------------------------------------


def test_tier_to_model_mapping_is_pinned() -> None:
    # Bumping these is a deliberate change — see docs/llm-conventions.md.
    assert MODEL_CHEAP == "gpt-5.4-nano"
    assert MODEL_DEFAULT == "gpt-5.4"
    assert MODEL_HEAVY == "gpt-5.5"
    assert model_for_tier(ModelTier.CHEAP) == MODEL_CHEAP
    assert model_for_tier(ModelTier.DEFAULT) == MODEL_DEFAULT
    assert model_for_tier(ModelTier.HEAVY) == MODEL_HEAVY


# ---------------------------------------------------------------------------
# Message structure (stable system + volatile user)
# ---------------------------------------------------------------------------


def test_message_structure_is_system_then_user() -> None:
    sdk = MagicMock()
    sdk.chat.completions.parse.return_value = _ok_completion()
    provider, _, _ = _make_provider(sdk=sdk)
    _call(provider)
    kwargs = sdk.chat.completions.parse.call_args.kwargs
    msgs = kwargs["messages"]
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"].startswith("STABLE INSTRUCTIONS")
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"].startswith("VOLATILE DATA")


def test_response_format_passes_the_pydantic_model() -> None:
    sdk = MagicMock()
    sdk.chat.completions.parse.return_value = _ok_completion()
    provider, _, _ = _make_provider(sdk=sdk)
    _call(provider)
    kwargs = sdk.chat.completions.parse.call_args.kwargs
    # The SDK's pydantic-native API takes the model class itself.
    assert kwargs["response_format"] is TinyReport


def test_call_uses_correct_model_per_tier() -> None:
    for tier, expected in [
        (ModelTier.CHEAP, MODEL_CHEAP),
        (ModelTier.DEFAULT, MODEL_DEFAULT),
        (ModelTier.HEAVY, MODEL_HEAVY),
    ]:
        sdk = MagicMock()
        sdk.chat.completions.parse.return_value = _ok_completion()
        provider, _, _ = _make_provider(sdk=sdk)
        _call(provider, tier=tier)
        kwargs = sdk.chat.completions.parse.call_args.kwargs
        assert kwargs["model"] == expected


def test_empty_stable_or_volatile_is_rejected() -> None:
    provider, _, _ = _make_provider()
    with pytest.raises(ValueError, match="stable_system"):
        provider.generate_structured(
            agent_name="t",
            run_id="r1",
            tier=ModelTier.DEFAULT,
            stable_system="",
            volatile_user="x",
            output_model=TinyReport,
        )
    with pytest.raises(ValueError, match="volatile_user"):
        provider.generate_structured(
            agent_name="t",
            run_id="r1",
            tier=ModelTier.DEFAULT,
            stable_system="x",
            volatile_user="",
            output_model=TinyReport,
        )


# ---------------------------------------------------------------------------
# Structured output parsing
# ---------------------------------------------------------------------------


def test_returns_parsed_pydantic_instance() -> None:
    canned = TinyReport(title="hello", score=0.42)
    sdk = MagicMock()
    sdk.chat.completions.parse.return_value = _ok_completion(parsed=canned)
    provider, _, _ = _make_provider(sdk=sdk)
    parsed, meta = _call(provider)
    assert isinstance(parsed, TinyReport)
    assert parsed.title == "hello"
    assert parsed.score == 0.42
    assert meta.model == MODEL_DEFAULT
    assert meta.tier == ModelTier.DEFAULT


def test_schema_validation_failure_at_boundary_raises() -> None:
    """The SDK occasionally returns a parsed object via a path that
    bypasses our strict schema — we re-validate with model_validate at the
    boundary and surface failures as LlmClientError."""

    class BypassingFake(BaseModel):
        # Carries an extra field that TinyReport.extra='forbid' rejects.
        model_config = ConfigDict(extra="allow")

        title: str
        score: float

    # Construct a BypassingFake with extra field; model_dump preserves it,
    # so when we round-trip through TinyReport.model_validate(...), it raises.
    bad = BypassingFake.model_construct(title="x", score=0.5, extra_field="leak")
    sdk = MagicMock()
    sdk.chat.completions.parse.return_value = _ok_completion(parsed=bad)
    provider, _, _ = _make_provider(sdk=sdk)
    with pytest.raises(LlmClientError, match="schema validation"):
        _call(provider)


def test_missing_parsed_raises() -> None:
    sdk = MagicMock()
    sdk.chat.completions.parse.return_value = FakeCompletion(
        choices=[FakeChoice(message=FakeMessage(parsed=None))],
        usage=FakeUsage(),
    )
    provider, _, _ = _make_provider(sdk=sdk)
    with pytest.raises(LlmClientError, match="did not include a parsed"):
        _call(provider)


def test_refusal_is_surfaced_as_llm_client_error() -> None:
    sdk = MagicMock()
    sdk.chat.completions.parse.return_value = FakeCompletion(
        choices=[
            FakeChoice(
                message=FakeMessage(parsed=None, refusal="I can't help with that."),
            )
        ],
        usage=FakeUsage(),
    )
    provider, _, _ = _make_provider(sdk=sdk)
    with pytest.raises(LlmClientError, match="refused"):
        _call(provider)


def test_sdk_exception_wraps_into_llm_client_error_without_key() -> None:
    sdk = MagicMock()
    sdk.chat.completions.parse.side_effect = openai.APIError(
        message="boom", request=MagicMock(), body=None
    )
    provider, _, _ = _make_provider(sdk=sdk)
    with pytest.raises(LlmClientError) as excinfo:
        _call(provider)
    assert "test-key-not-real" not in str(excinfo.value)


def test_openai_settings_repr_does_not_leak_key() -> None:
    s = OpenAISettings(api_key=SecretStr("supersecret"))
    assert "supersecret" not in repr(s)


# ---------------------------------------------------------------------------
# Token accounting
# ---------------------------------------------------------------------------


def test_accounting_records_prompt_cached_completion() -> None:
    sdk = MagicMock()
    sdk.chat.completions.parse.return_value = _ok_completion(cached_tokens=900)
    provider, _, recorder = _make_provider(sdk=sdk)
    _call(provider)
    recorder.record.assert_called_once()
    kwargs = recorder.record.call_args.kwargs
    assert kwargs["agent_name"] == "t"
    assert kwargs["run_id"] == "r1"
    assert kwargs["model"] == MODEL_DEFAULT
    usage: TokenUsage = kwargs["usage"]
    assert usage.prompt_tokens == 1500
    assert usage.cached_tokens == 900
    assert usage.completion_tokens == 60
    assert kwargs["extra"].get("tier") == "default"


def test_cache_hit_ratio_derived_correctly() -> None:
    usage = TokenUsage(prompt_tokens=1500, cached_tokens=900, completion_tokens=60)
    # 900 / 1500 = 0.6
    assert abs(usage.cache_hit_ratio - 0.6) < 1e-9


def test_zero_prompt_tokens_gives_zero_ratio() -> None:
    assert TokenUsage(0, 0, 0).cache_hit_ratio == 0.0


def test_missing_prompt_tokens_details_falls_back_to_zero() -> None:
    sdk = MagicMock()
    # No prompt_tokens_details on the usage.
    completion = _ok_completion()
    completion.usage.prompt_tokens_details = None
    sdk.chat.completions.parse.return_value = completion
    provider, _, recorder = _make_provider(sdk=sdk)
    _call(provider)
    usage = recorder.record.call_args.kwargs["usage"]
    assert usage.cached_tokens == 0


# ---------------------------------------------------------------------------
# Optional live smokes
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("RUN_LIVE_TESTS") != "1" or not os.environ.get("OPENAI_API_KEY"),
    reason="RUN_LIVE_TESTS=1 and OPENAI_API_KEY must be set",
)
def test_live_cheap_single_call() -> None:
    """Cheapest real call confirming wiring + key + structured output."""
    settings = OpenAISettings()
    provider = OpenAIProvider(settings)
    parsed, meta = provider.generate_structured(
        agent_name="smoke",
        run_id="live-smoke",
        tier=ModelTier.CHEAP,
        stable_system=(
            "You are a structured-output tester. Always emit a TinyReport "
            "with title='hello' and score=0.5. No commentary; the structured "
            "schema is your only output."
        ),
        volatile_user="Confirm the wiring.",
        output_model=TinyReport,
        max_output_tokens=256,
    )
    assert isinstance(parsed, TinyReport)
    assert meta.model == MODEL_CHEAP


@pytest.mark.skipif(
    os.environ.get("RUN_LIVE_TESTS") != "1" or not os.environ.get("OPENAI_API_KEY"),
    reason="RUN_LIVE_TESTS=1 and OPENAI_API_KEY must be set",
)
def test_live_default_caching_two_calls() -> None:
    """Two identical-prefix DEFAULT-tier calls; second should show
    cached_tokens > 0. The stable prefix is built to cross the 1024-token
    OpenAI threshold."""
    long_stable = (
        "You are the 4xPrima market_context_agent caching smoke tester. "
        * 250
        + "\nWhen called, emit a TinyReport with title='cached' and score=0.42."
    )
    settings = OpenAISettings()
    provider = OpenAIProvider(settings)
    _, first = provider.generate_structured(
        agent_name="cache_smoke",
        run_id="cache-test-1",
        tier=ModelTier.DEFAULT,
        stable_system=long_stable,
        volatile_user="First call.",
        output_model=TinyReport,
        max_output_tokens=256,
    )
    _, second = provider.generate_structured(
        agent_name="cache_smoke",
        run_id="cache-test-2",
        tier=ModelTier.DEFAULT,
        stable_system=long_stable,
        volatile_user="Second call.",
        output_model=TinyReport,
        max_output_tokens=256,
    )
    # The cache is best-effort; we don't assert exact numbers, only that the
    # second call shows SOMETHING cached when the prefix re-occurs identically.
    assert first.usage.prompt_tokens > 0
    assert second.usage.cached_tokens > 0, (
        f"second call should hit cache; got {second.usage}"
    )
