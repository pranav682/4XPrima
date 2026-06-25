"""Tests for core/llm_client.py — fully hermetic (mocked SDK).

These tests verify the conventions and the tier-aware capability guard the
user flagged: a Haiku call must NEVER carry the `compact-2026-01-12` beta,
even when the caller requested compaction.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from core.config import AnthropicSettings
from core.llm_client import (
    COMPACTION_BETA,
    CONTEXT_MANAGEMENT_BETA,
    MODEL_HAIKU,
    MODEL_OPUS,
    MODEL_SONNET,
    STRUCTURED_OUTPUT_TOOL_NAME,
    AgentRequest,
    ContextManagementConfig,
    LLMClient,
    LlmClientError,
    ModelTier,
    StableBlock,
    TokenUsage,
    build_system_blocks,
    resolve_context_management,
    tier_supports_compaction,
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
class FakeToolUseBlock:
    """Mimics the anthropic SDK's tool_use content block."""

    type: str
    name: str
    input: dict[str, Any]
    id: str = "blk_1"


@dataclass
class FakeUsage:
    input_tokens: int = 50
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 200
    output_tokens: int = 80


@dataclass
class FakeResponse:
    content: list[Any]
    usage: FakeUsage
    stop_reason: str = "end_turn"
    context_management: Any = None


def _ok_response(payload: dict[str, Any] | None = None) -> FakeResponse:
    payload = payload or {"title": "ok", "score": 0.5}
    block = FakeToolUseBlock(
        type="tool_use", name=STRUCTURED_OUTPUT_TOOL_NAME, input=payload
    )
    return FakeResponse(content=[block], usage=FakeUsage())


def _make_client(
    sdk: MagicMock | None = None, recorder: MagicMock | None = None
) -> tuple[LLMClient, MagicMock, MagicMock]:
    sdk = sdk or MagicMock()
    recorder = recorder or MagicMock()
    settings = AnthropicSettings(api_key=SecretStr("test-key-not-real"))
    client = LLMClient(settings, usage_recorder=recorder, anthropic_client=sdk)
    return client, sdk, recorder


def _basic_request(
    tier: ModelTier = ModelTier.SONNET,
    ctx_mgmt: ContextManagementConfig | None = None,
) -> AgentRequest:
    return AgentRequest(
        agent_name="t",
        run_id="r1",
        tier=tier,
        stable_system_blocks=(
            StableBlock(kind="spec", name="agent_spec", text="instructions go here..."),
            StableBlock(kind="skill", name="format", text="schema notes..."),
        ),
        messages=({"role": "user", "content": "go"},),
        context_management=ctx_mgmt,
    )


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_tier_to_model_mapping_is_pinned() -> None:
    # Pinned values — bumping these is a deliberate change that resets caches.
    assert MODEL_HAIKU == "claude-haiku-4-5-20251001"
    assert MODEL_SONNET == "claude-sonnet-4-6"
    assert MODEL_OPUS == "claude-opus-4-8"


def test_tier_supports_compaction_matrix() -> None:
    """The user's flagged guard — verified against current docs.claude.com."""
    assert tier_supports_compaction(ModelTier.HAIKU) is False
    assert tier_supports_compaction(ModelTier.SONNET) is True
    assert tier_supports_compaction(ModelTier.OPUS) is True


def test_build_system_blocks_puts_cache_control_on_last_stable() -> None:
    blocks = (
        StableBlock(kind="spec", name="a", text="A"),
        StableBlock(kind="skill", name="b", text="B"),
        StableBlock(kind="skill", name="c", text="C"),
    )
    out = build_system_blocks(blocks, volatile_policy=None, cache_ttl="5m")
    assert len(out) == 3
    assert "cache_control" not in out[0]
    assert "cache_control" not in out[1]
    assert out[2]["cache_control"] == {"type": "ephemeral"}


def test_build_system_blocks_volatile_policy_is_after_breakpoint() -> None:
    from core.llm_client import VolatilePolicy

    blocks = (StableBlock(kind="spec", name="a", text="A"),)
    out = build_system_blocks(
        blocks, volatile_policy=VolatilePolicy(text="dynamic"), cache_ttl="5m"
    )
    # Stable block has cache_control; volatile (appended after) does NOT.
    assert out[0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in out[1]
    assert out[1]["text"] == "dynamic"


def test_build_system_blocks_empty_rejects() -> None:
    with pytest.raises(ValueError, match="cache the agent spec"):
        build_system_blocks((), volatile_policy=None, cache_ttl="5m")


def test_resolve_context_management_haiku_drops_compaction() -> None:
    """THE GUARD: Haiku call with enable_compaction=True must drop both the
    edit AND the compaction beta header."""
    body, betas = resolve_context_management(
        ContextManagementConfig(enable_compaction=True), ModelTier.HAIKU
    )
    types = [e["type"] for e in body["edits"]]
    assert "compact_20260112" not in types
    assert COMPACTION_BETA not in betas
    # Clear edits still apply on Haiku.
    assert "clear_thinking_20251015" in types
    assert "clear_tool_uses_20250919" in types
    assert CONTEXT_MANAGEMENT_BETA in betas


def test_resolve_context_management_sonnet_keeps_compaction() -> None:
    body, betas = resolve_context_management(
        ContextManagementConfig(enable_compaction=True), ModelTier.SONNET
    )
    types = [e["type"] for e in body["edits"]]
    assert types[0] == "compact_20260112"  # compaction first per the canonical example
    assert COMPACTION_BETA in betas
    assert CONTEXT_MANAGEMENT_BETA in betas


def test_resolve_context_management_opus_keeps_compaction() -> None:
    body, betas = resolve_context_management(
        ContextManagementConfig(enable_compaction=True), ModelTier.OPUS
    )
    assert any(e["type"] == "compact_20260112" for e in body["edits"])
    assert COMPACTION_BETA in betas


def test_resolve_context_management_none() -> None:
    body, betas = resolve_context_management(None, ModelTier.HAIKU)
    assert body is None
    assert betas == []


# ---------------------------------------------------------------------------
# End-to-end client behaviour (mocked SDK)
# ---------------------------------------------------------------------------


def test_call_structured_uses_correct_model_per_tier() -> None:
    for tier, expected in [
        (ModelTier.HAIKU, MODEL_HAIKU),
        (ModelTier.SONNET, MODEL_SONNET),
        (ModelTier.OPUS, MODEL_OPUS),
    ]:
        sdk = MagicMock()
        sdk.beta.messages.create.return_value = _ok_response()
        client, _, _ = _make_client(sdk=sdk)
        client.call_structured(_basic_request(tier=tier), output_model=TinyReport)
        kwargs = sdk.beta.messages.create.call_args.kwargs
        assert kwargs["model"] == expected


def test_call_structured_forces_tool_choice() -> None:
    sdk = MagicMock()
    sdk.beta.messages.create.return_value = _ok_response()
    client, _, _ = _make_client(sdk=sdk)
    client.call_structured(_basic_request(), output_model=TinyReport)
    kwargs = sdk.beta.messages.create.call_args.kwargs
    assert kwargs["tool_choice"] == {
        "type": "tool",
        "name": STRUCTURED_OUTPUT_TOOL_NAME,
    }
    # The structured-output tool is present in `tools` and is NOT deferred.
    tool_names = [t["name"] for t in kwargs["tools"]]
    assert STRUCTURED_OUTPUT_TOOL_NAME in tool_names


def test_call_structured_cache_control_on_last_stable_block_in_request() -> None:
    sdk = MagicMock()
    sdk.beta.messages.create.return_value = _ok_response()
    client, _, _ = _make_client(sdk=sdk)
    client.call_structured(_basic_request(), output_model=TinyReport)
    kwargs = sdk.beta.messages.create.call_args.kwargs
    system = kwargs["system"]
    assert "cache_control" not in system[0]
    assert system[-1]["cache_control"] == {"type": "ephemeral"}


def test_haiku_call_carries_no_compaction_beta_when_caller_asked_for_it() -> None:
    """The required test from the user brief, asserting the guard end-to-end."""
    sdk = MagicMock()
    sdk.beta.messages.create.return_value = _ok_response()
    client, _, _ = _make_client(sdk=sdk)
    client.call_structured(
        _basic_request(
            tier=ModelTier.HAIKU,
            ctx_mgmt=ContextManagementConfig(enable_compaction=True),
        ),
        output_model=TinyReport,
    )
    kwargs = sdk.beta.messages.create.call_args.kwargs
    betas = list(kwargs.get("betas") or [])
    assert COMPACTION_BETA not in betas, (
        f"Haiku call leaked the compaction beta: {betas}"
    )
    # context-management beta still rides since clear_* edits are enabled.
    assert CONTEXT_MANAGEMENT_BETA in betas
    # And the context_management payload likewise lacks compact.
    cm = kwargs.get("context_management") or {}
    types = [e["type"] for e in cm.get("edits", [])]
    assert "compact_20260112" not in types


def test_sonnet_call_carries_compaction_beta_when_requested() -> None:
    sdk = MagicMock()
    sdk.beta.messages.create.return_value = _ok_response()
    client, _, _ = _make_client(sdk=sdk)
    client.call_structured(
        _basic_request(
            tier=ModelTier.SONNET,
            ctx_mgmt=ContextManagementConfig(enable_compaction=True),
        ),
        output_model=TinyReport,
    )
    kwargs = sdk.beta.messages.create.call_args.kwargs
    betas = list(kwargs.get("betas") or [])
    assert COMPACTION_BETA in betas
    assert CONTEXT_MANAGEMENT_BETA in betas


def test_call_structured_parses_tool_input_into_model() -> None:
    sdk = MagicMock()
    sdk.beta.messages.create.return_value = _ok_response(
        {"title": "regime check", "score": 0.7}
    )
    client, _, _ = _make_client(sdk=sdk)
    parsed, meta = client.call_structured(_basic_request(), output_model=TinyReport)
    assert isinstance(parsed, TinyReport)
    assert parsed.title == "regime check"
    assert parsed.score == 0.7
    assert meta.model == MODEL_SONNET
    assert meta.tier == ModelTier.SONNET


def test_accounting_records_token_fields() -> None:
    sdk = MagicMock()
    sdk.beta.messages.create.return_value = _ok_response()
    client, _, recorder = _make_client(sdk=sdk)
    client.call_structured(_basic_request(), output_model=TinyReport)
    recorder.record.assert_called_once()
    kwargs = recorder.record.call_args.kwargs
    assert kwargs["agent_name"] == "t"
    assert kwargs["run_id"] == "r1"
    assert kwargs["model"] == MODEL_SONNET
    usage: TokenUsage = kwargs["usage"]
    assert usage.input_tokens == 50
    assert usage.cache_creation_input_tokens == 200
    assert usage.output_tokens == 80
    assert kwargs["extra"].get("tier") == "sonnet"


def test_schema_validation_failure_raises_llm_client_error() -> None:
    sdk = MagicMock()
    # Returned input doesn't satisfy TinyReport: score > 1.0.
    sdk.beta.messages.create.return_value = _ok_response(
        {"title": "x", "score": 5.0}
    )
    client, _, _ = _make_client(sdk=sdk)
    with pytest.raises(LlmClientError, match="schema validation"):
        client.call_structured(_basic_request(), output_model=TinyReport)


def test_missing_tool_use_block_raises_llm_client_error() -> None:
    sdk = MagicMock()
    sdk.beta.messages.create.return_value = FakeResponse(content=[], usage=FakeUsage())
    client, _, _ = _make_client(sdk=sdk)
    with pytest.raises(LlmClientError, match="did not include a tool_use"):
        client.call_structured(_basic_request(), output_model=TinyReport)


def test_sdk_exception_wraps_into_llm_client_error_without_key() -> None:
    import anthropic

    sdk = MagicMock()
    sdk.beta.messages.create.side_effect = anthropic.APIError(
        message="boom", request=MagicMock(), body=None
    )
    client, _, _ = _make_client(sdk=sdk)
    with pytest.raises(LlmClientError) as excinfo:
        client.call_structured(_basic_request(), output_model=TinyReport)
    assert "test-key-not-real" not in str(excinfo.value)


def test_anthropic_settings_repr_does_not_leak_api_key() -> None:
    s = AnthropicSettings(api_key=SecretStr("supersecret"))
    assert "supersecret" not in repr(s)


# ---------------------------------------------------------------------------
# Optional live smoke
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("RUN_LIVE_TESTS") != "1" or not os.environ.get("ANTHROPIC_API_KEY"),
    reason="RUN_LIVE_TESTS=1 and ANTHROPIC_API_KEY must be set",
)
def test_live_haiku_single_call() -> None:
    """One cheap real Haiku call confirming wiring + key + structured output."""
    settings = AnthropicSettings()  # reads from env / .env
    client = LLMClient(settings)
    request = AgentRequest(
        agent_name="smoke",
        run_id="live-smoke",
        tier=ModelTier.HAIKU,
        stable_system_blocks=(
            StableBlock(
                kind="spec",
                name="haiku_spec",
                text=(
                    "You are a structured-output tester. Always call the "
                    "submit_structured_output tool. Set title to 'hello' and "
                    "score to 0.5."
                ),
            ),
        ),
        messages=({"role": "user", "content": "Confirm the wiring."},),
        max_output_tokens=512,
    )
    parsed, meta = client.call_structured(request, output_model=TinyReport)
    assert isinstance(parsed, TinyReport)
    assert meta.model == MODEL_HAIKU
    # Haiku NEVER carries the compaction beta — assert at the live boundary
    # too (we didn't request compaction; this still confirms no surprise).
    assert COMPACTION_BETA not in meta.betas_sent


@pytest.mark.skipif(
    os.environ.get("RUN_LIVE_TESTS") != "1" or not os.environ.get("ANTHROPIC_API_KEY"),
    reason="RUN_LIVE_TESTS=1 and ANTHROPIC_API_KEY must be set",
)
def test_live_sonnet_caching_two_calls() -> None:
    """Make two identical Sonnet calls and assert the second shows cache_read > 0.

    The stable prefix is the long agent_spec block — well above the 1024-token
    Sonnet cache minimum.
    """
    long_spec = (
        "You are the 4xPrima market_context_agent smoke tester. " * 200
        + "\nWhen called, return title='cached' and score=0.42."
    )
    settings = AnthropicSettings()
    client = LLMClient(settings)
    request = AgentRequest(
        agent_name="cache_smoke",
        run_id="cache-test",
        tier=ModelTier.SONNET,
        stable_system_blocks=(
            StableBlock(kind="spec", name="long_spec", text=long_spec),
        ),
        messages=({"role": "user", "content": "First call."},),
        max_output_tokens=256,
    )
    _, first = client.call_structured(request, output_model=TinyReport)
    # Second call — identical prefix, different volatile user turn.
    request2 = AgentRequest(
        agent_name="cache_smoke",
        run_id="cache-test",
        tier=ModelTier.SONNET,
        stable_system_blocks=(
            StableBlock(kind="spec", name="long_spec", text=long_spec),
        ),
        messages=({"role": "user", "content": "Second call."},),
        max_output_tokens=256,
    )
    _, second = client.call_structured(request2, output_model=TinyReport)
    assert first.usage.cache_creation_input_tokens > 0
    assert second.usage.cache_read_input_tokens > 0
