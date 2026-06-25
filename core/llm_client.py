"""Shared LLM client for the slow loop.

**All runtime LLM calls in 4xPrima must go through this module.** Direct calls
to :func:`anthropic.Anthropic().messages.create` from anywhere else are
forbidden by project convention (``CLAUDE.md``) — they bypass cache layout,
model routing, capability resolution, and usage accounting.

This module encodes the conventions in ``docs/llm-conventions.md`` and the
hard guard the user flagged: server-side compaction (``compact_20260112``) is
only valid on Opus 4.6+ and Sonnet 4.6+. Haiku 4.5 calls **must not** carry
the ``compact-2026-01-12`` beta header. Capability resolution happens
per-call against the chosen model, never against a global default.

Verified against ``platform.claude.com/docs`` (2026-06-24):
- Compaction supported: Opus 4.6 / 4.7 / 4.8, Sonnet 4.6, Fable 5,
  Mythos 5 / Mythos Preview.
- Compaction NOT supported: Haiku 4.5 (and older Sonnets / Opuses).
- ``clear_thinking_20251015`` and ``clear_tool_uses_20250919`` work on all
  supported models with beta ``context-management-2025-06-27``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final, Literal, Protocol, TypeVar

import anthropic
import structlog
from pydantic import BaseModel, ValidationError

from core.config import AnthropicSettings

# ---------------------------------------------------------------------------
# Pinned model strings (single source of truth — see docs/llm-conventions.md §4).
# Switching models invalidates the prompt cache, so pin and bump explicitly.
# ---------------------------------------------------------------------------

MODEL_HAIKU: str = "claude-haiku-4-5-20251001"
MODEL_SONNET: str = "claude-sonnet-4-6"
MODEL_OPUS: str = "claude-opus-4-8"

# Beta headers
CONTEXT_MANAGEMENT_BETA: str = "context-management-2025-06-27"
COMPACTION_BETA: str = "compact-2026-01-12"

# Tool Search Tool identifiers
TOOL_SEARCH_REGEX: str = "tool_search_tool_regex_20251119"
TOOL_SEARCH_BM25: str = "tool_search_tool_bm25_20251119"

# Context-management edit identifiers
CONTEXT_EDIT_COMPACT: str = "compact_20260112"
CONTEXT_EDIT_CLEAR_THINKING: str = "clear_thinking_20251015"
CONTEXT_EDIT_CLEAR_TOOL_USES: str = "clear_tool_uses_20250919"

# Memory tool identifier
MEMORY_TOOL_TYPE: str = "memory_20250818"
MEMORY_TOOL_NAME: str = "memory"

# Name of the synthetic tool the wrapper adds for structured output.
STRUCTURED_OUTPUT_TOOL_NAME: Final[str] = "submit_structured_output"


class ModelTier(StrEnum):
    """Which model class to route to. Picked per agent in its spec."""

    HAIKU = "haiku"
    SONNET = "sonnet"
    OPUS = "opus"


def model_for_tier(tier: ModelTier) -> str:
    return {
        ModelTier.HAIKU: MODEL_HAIKU,
        ModelTier.SONNET: MODEL_SONNET,
        ModelTier.OPUS: MODEL_OPUS,
    }[tier]


# Capability matrix — which beta features each tier supports. The user
# flagged this as the guard for the project: compaction MUST NEVER ride a
# Haiku call. Resolution happens at call time against the CHOSEN tier.
_TIER_SUPPORTS_COMPACTION: Final[dict[ModelTier, bool]] = {
    ModelTier.HAIKU: False,
    ModelTier.SONNET: True,
    ModelTier.OPUS: True,
}


def tier_supports_compaction(tier: ModelTier) -> bool:
    """Public predicate: may a call on this tier carry compaction?

    Documented separately so tests and callers can both read it.
    """
    return _TIER_SUPPORTS_COMPACTION[tier]


# ---------------------------------------------------------------------------
# Message-assembly types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StableBlock:
    """A piece of the stable prefix (agent spec, a skill, taxonomy, etc.)."""

    kind: Literal["spec", "skill", "taxonomy", "schema", "policy_stable"]
    name: str
    text: str


@dataclass(frozen=True, slots=True)
class VolatilePolicy:
    """Per-call policy overrides that go AFTER the cache breakpoint inside
    ``system``. Use sparingly; anything that changes per call belongs here."""

    text: str


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """One tool definition. ``defer_loading=True`` keeps it out of the eager
    prefix; the Tool Search Tool is added automatically when any tool is
    deferred."""

    name: str
    description: str
    input_schema: dict[str, Any]
    defer_loading: bool = True


@dataclass(frozen=True, slots=True)
class ContextManagementConfig:
    """Server-side context-management config for long-running agents.

    Caller's *desired* configuration. The wrapper resolves what is actually
    valid for the chosen tier (see :func:`resolve_context_management`). A
    Haiku call with ``enable_compaction=True`` SILENTLY drops compaction and
    its beta; the other edits still apply.
    """

    enable_compaction: bool = False
    compact_trigger_input_tokens: int = 120_000
    compact_pause_after: bool = False
    compact_instructions: str | None = None

    enable_clear_thinking: bool = True
    clear_thinking_keep_turns: int = 2

    enable_clear_tool_uses: bool = True
    clear_tool_uses_trigger_input_tokens: int = 60_000
    clear_tool_uses_keep: int = 5
    clear_tool_uses_clear_at_least_input_tokens: int = 10_000
    exclude_tools_from_clear: tuple[str, ...] = ("memory",)


@dataclass(frozen=True, slots=True)
class AgentRequest:
    """Everything :meth:`LLMClient.call_structured` needs."""

    agent_name: str
    run_id: str
    tier: ModelTier

    tools: tuple[ToolSpec, ...] = ()
    stable_system_blocks: tuple[StableBlock, ...] = ()
    volatile_policy: VolatilePolicy | None = None
    messages: tuple[dict[str, Any], ...] = ()

    max_output_tokens: int = 2048
    context_management: ContextManagementConfig | None = None
    enable_memory_tool: bool = False
    cache_ttl: Literal["5m", "1h"] = "5m"
    tool_search_variant: Literal["bm25", "regex"] = "bm25"

    extra_metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Token accounting for a single call."""

    input_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int
    output_tokens: int

    @property
    def total_input_tokens(self) -> int:
        return (
            self.input_tokens
            + self.cache_read_input_tokens
            + self.cache_creation_input_tokens
        )

    @property
    def cache_hit_ratio(self) -> float:
        denom = self.total_input_tokens
        return (self.cache_read_input_tokens / denom) if denom else 0.0


@dataclass(frozen=True, slots=True)
class AgentResponse:
    """Metadata about one LLM call. Decoupled from the parsed output so
    structured-output callers can return ``(parsed_model, AgentResponse)``."""

    agent_name: str
    run_id: str
    model: str
    tier: ModelTier
    stop_reason: str
    usage: TokenUsage
    betas_sent: tuple[str, ...]
    applied_context_edits: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class LlmClientError(Exception):
    """Unrecoverable LLM call failure.

    Wraps anthropic SDK errors and schema-validation failures. Error messages
    never include the API key (the SDK error already redacts it; we also do
    not log SecretStr.get_secret_value()).
    """


# ---------------------------------------------------------------------------
# Wire-format assembly
# ---------------------------------------------------------------------------


def build_system_blocks(
    stable_blocks: tuple[StableBlock, ...],
    volatile_policy: VolatilePolicy | None,
    cache_ttl: Literal["5m", "1h"],
) -> list[dict[str, Any]]:
    """Build the ``system`` array.

    The single explicit ``cache_control`` marker is placed on the LAST stable
    block. Volatile policy (if any) is appended after with no cache_control —
    so it never enters the cache and never invalidates it.
    """
    if not stable_blocks:
        raise ValueError("stable_blocks must not be empty — cache the agent spec")

    system: list[dict[str, Any]] = []
    last_index = len(stable_blocks) - 1
    for i, block in enumerate(stable_blocks):
        item: dict[str, Any] = {"type": "text", "text": block.text}
        if i == last_index:
            cache_control: dict[str, Any] = {"type": "ephemeral"}
            if cache_ttl == "1h":
                cache_control["ttl"] = "1h"
            item["cache_control"] = cache_control
        system.append(item)

    if volatile_policy is not None:
        system.append({"type": "text", "text": volatile_policy.text})

    return system


def build_tools(
    tools: tuple[ToolSpec, ...],
    tool_search_variant: Literal["bm25", "regex"],
    enable_memory_tool: bool,
    structured_output_tool: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Assemble the ``tools`` array.

    Adds a Tool Search Tool when any user tool is deferred; adds the memory
    tool when ``enable_memory_tool=True``. The optional structured-output
    tool is appended last (eager — never deferred), so callers forcing a
    tool call (structured output) can find it.
    """
    out: list[dict[str, Any]] = []

    has_any_deferred = any(t.defer_loading for t in tools)
    if has_any_deferred:
        ts_type = TOOL_SEARCH_BM25 if tool_search_variant == "bm25" else TOOL_SEARCH_REGEX
        out.append({"type": ts_type, "name": ts_type})

    if enable_memory_tool:
        out.append({"type": MEMORY_TOOL_TYPE, "name": MEMORY_TOOL_NAME})

    for t in tools:
        out.append(
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
                "defer_loading": t.defer_loading,
            }
        )

    if structured_output_tool is not None:
        out.append(structured_output_tool)

    if out and all(item.get("defer_loading") for item in out):
        raise ValueError(
            "all tools deferred — keep the Tool Search Tool / structured-output "
            "tool eager or mark one tool non-deferred"
        )

    return out


# ---------------------------------------------------------------------------
# Tier-aware context-management resolution (THE compaction guard)
# ---------------------------------------------------------------------------


def resolve_context_management(
    config: ContextManagementConfig | None,
    tier: ModelTier,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Resolve the ``context_management`` payload AND required beta headers
    for the chosen tier.

    The compaction guard lives here: if ``tier`` is Haiku and the caller
    asked for compaction, compaction is silently dropped. The
    ``compact-2026-01-12`` beta is also dropped. Clearing edits still apply.

    Returns ``(context_management_body | None, list[str] of beta headers)``.
    """
    if config is None:
        return None, []

    edits: list[dict[str, Any]] = []
    betas: list[str] = []

    # 1. Compaction (capability-gated).
    if config.enable_compaction:
        if tier_supports_compaction(tier):
            compact_edit: dict[str, Any] = {
                "type": CONTEXT_EDIT_COMPACT,
                "trigger": {
                    "type": "input_tokens",
                    "value": config.compact_trigger_input_tokens,
                },
                "pause_after_compaction": config.compact_pause_after,
            }
            if config.compact_instructions is not None:
                compact_edit["instructions"] = config.compact_instructions
            edits.append(compact_edit)
            betas.append(COMPACTION_BETA)
        # else: silently drop. The user's stated guard — Haiku must never
        # carry the compaction beta — is enforced by NOT appending here.

    # 2. Thinking clearing.
    if config.enable_clear_thinking:
        edits.append(
            {
                "type": CONTEXT_EDIT_CLEAR_THINKING,
                "keep": {"type": "thinking_turns", "value": config.clear_thinking_keep_turns},
            }
        )

    # 3. Tool-use clearing.
    if config.enable_clear_tool_uses:
        edits.append(
            {
                "type": CONTEXT_EDIT_CLEAR_TOOL_USES,
                "trigger": {
                    "type": "input_tokens",
                    "value": config.clear_tool_uses_trigger_input_tokens,
                },
                "keep": {"type": "tool_uses", "value": config.clear_tool_uses_keep},
                "clear_at_least": {
                    "type": "input_tokens",
                    "value": config.clear_tool_uses_clear_at_least_input_tokens,
                },
                "exclude_tools": list(config.exclude_tools_from_clear),
            }
        )

    # 4. The clear_* edits ride the context-management beta.
    if config.enable_clear_thinking or config.enable_clear_tool_uses:
        betas.append(CONTEXT_MANAGEMENT_BETA)

    return ({"edits": edits} if edits else None), betas


# ---------------------------------------------------------------------------
# Usage accounting interop
# ---------------------------------------------------------------------------


class UsageRecorder(Protocol):
    """Anything that can persist a usage row. See :mod:`core.usage_accounting`."""

    def record(
        self,
        *,
        agent_name: str,
        run_id: str,
        model: str,
        usage: TokenUsage,
        extra: dict[str, str],
    ) -> None: ...


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


T = TypeVar("T", bound=BaseModel)


class LLMClient:
    """Thin Anthropic SDK wrapper enforcing the project's conventions.

    Construction:

        >>> client = LLMClient(settings, usage_recorder=recorder)

    Calls:

        >>> parsed, meta = client.call_structured(
        ...     request=AgentRequest(...),
        ...     output_model=MarketContextReport,
        ... )

    The wrapper:

      1. Resolves the tier to a pinned model string.
      2. Builds the ``tools``, ``system``, and ``messages`` arrays per the
         caching conventions (one ``cache_control`` on the last stable block).
      3. Resolves ``context_management`` and required betas FOR THE CHOSEN
         TIER (Haiku never carries the compaction beta).
      4. Forces the model to call a structured-output tool whose schema is
         derived from ``output_model``.
      5. Parses the tool call into ``output_model``, raising
         :class:`LlmClientError` on schema-validation failure.
      6. Records token usage via :class:`UsageRecorder`.
    """

    def __init__(
        self,
        settings: AnthropicSettings,
        *,
        usage_recorder: UsageRecorder | None = None,
        anthropic_client: Any | None = None,
        logger: structlog.stdlib.BoundLogger | None = None,
    ) -> None:
        self._settings = settings
        self._usage_recorder = usage_recorder
        # `anthropic_client` is injectable so tests don't have to monkeypatch
        # the SDK module-level constructor.
        if anthropic_client is None:
            self._client = anthropic.Anthropic(api_key=settings.api_key.get_secret_value())
        else:
            self._client = anthropic_client
        self._logger = (
            logger if logger is not None else _default_logger()
        ).bind(component="llm_client")

    def call_structured(
        self,
        request: AgentRequest,
        *,
        output_model: type[T],
    ) -> tuple[T, AgentResponse]:
        """Issue one agent call and return the parsed structured output.

        The wrapper adds a forced tool call whose input_schema is
        ``output_model.model_json_schema()``. The model's response must be
        a single tool_use block invoking that tool; the input dict is
        validated into ``output_model``.

        Raises:
            LlmClientError: on SDK failure, missing tool call, or
                schema validation failure. Never includes the API key.
        """
        # 1. Resolve tier → model.
        model = model_for_tier(request.tier)

        # 2. Build system / tools / messages.
        system = build_system_blocks(
            request.stable_system_blocks,
            request.volatile_policy,
            request.cache_ttl,
        )
        structured_tool = _build_structured_output_tool(output_model)
        tools = build_tools(
            request.tools,
            request.tool_search_variant,
            request.enable_memory_tool,
            structured_output_tool=structured_tool,
        )

        # 3. Resolve context_management + betas for THIS tier (the guard).
        ctx_mgmt, betas = resolve_context_management(request.context_management, request.tier)

        # 4. Compose call kwargs.
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": request.max_output_tokens,
            "system": system,
            "messages": list(request.messages),
            "tools": tools,
            "tool_choice": {"type": "tool", "name": STRUCTURED_OUTPUT_TOOL_NAME},
        }
        if ctx_mgmt is not None:
            kwargs["context_management"] = ctx_mgmt
        if betas:
            kwargs["betas"] = betas

        self._logger.info(
            "llm_call_dispatch",
            agent_name=request.agent_name,
            run_id=request.run_id,
            tier=request.tier.value,
            model=model,
            n_stable_blocks=len(request.stable_system_blocks),
            n_messages=len(request.messages),
            n_tools=len(tools),
            betas=list(betas),
            has_context_management=ctx_mgmt is not None,
        )

        # 5. Make the call. The `client.beta.messages.create` path accepts
        #    both legacy and beta kwargs; calls with empty `betas` are
        #    equivalent to a normal call.
        try:
            response = self._client.beta.messages.create(**kwargs)
        except anthropic.AnthropicError as e:
            # The SDK's error repr does not include the API key, but defend
            # against future changes by stringifying explicitly here.
            raise LlmClientError(f"Anthropic API error: {type(e).__name__}: {e}") from e
        except Exception as e:
            raise LlmClientError(f"unexpected error from Anthropic SDK: {e!r}") from e

        # 6. Parse the response: locate the tool_use block, validate input.
        tool_input = _extract_tool_input(response)
        try:
            parsed = output_model(**tool_input)
        except ValidationError as e:
            raise LlmClientError(
                f"model output failed schema validation for "
                f"{output_model.__name__}: {e}"
            ) from e

        # 7. Token accounting.
        usage = _extract_usage(response)
        meta = AgentResponse(
            agent_name=request.agent_name,
            run_id=request.run_id,
            model=model,
            tier=request.tier,
            stop_reason=getattr(response, "stop_reason", "") or "",
            usage=usage,
            betas_sent=tuple(betas),
            applied_context_edits=_extract_applied_edits(response),
        )
        if self._usage_recorder is not None:
            self._usage_recorder.record(
                agent_name=request.agent_name,
                run_id=request.run_id,
                model=model,
                usage=usage,
                extra={"tier": request.tier.value, **request.extra_metadata},
            )
        return parsed, meta


# ---------------------------------------------------------------------------
# Helpers (response parsing)
# ---------------------------------------------------------------------------


def _build_structured_output_tool(output_model: type[BaseModel]) -> dict[str, Any]:
    """Build the forced-tool definition for structured output."""
    schema = output_model.model_json_schema()
    return {
        "name": STRUCTURED_OUTPUT_TOOL_NAME,
        "description": (
            f"Submit the structured response as a {output_model.__name__}. "
            "Call this tool exactly once with all required fields filled."
        ),
        "input_schema": schema,
    }


def _extract_tool_input(response: Any) -> dict[str, Any]:
    """Find the structured-output tool's input in the response content.

    The Anthropic SDK response.content is a list of blocks. We skip thinking
    / text blocks and pick the tool_use one whose ``name`` matches our
    synthetic tool. If we don't find one, the model didn't honour tool_choice
    — that's an :class:`LlmClientError`, not a parse-as-best-effort case.
    """
    content = getattr(response, "content", None) or []
    for block in content:
        # Anthropic blocks are objects; tests may pass plain dicts.
        block_type = getattr(block, "type", None) or (
            block.get("type") if isinstance(block, dict) else None
        )
        if block_type != "tool_use":
            continue
        name = getattr(block, "name", None) or (
            block.get("name") if isinstance(block, dict) else None
        )
        if name != STRUCTURED_OUTPUT_TOOL_NAME:
            continue
        input_obj = getattr(block, "input", None) or (
            block.get("input") if isinstance(block, dict) else None
        )
        if not isinstance(input_obj, dict):
            raise LlmClientError(
                f"tool_use block for {STRUCTURED_OUTPUT_TOOL_NAME} had non-dict input"
            )
        return input_obj
    raise LlmClientError(
        f"response did not include a tool_use for {STRUCTURED_OUTPUT_TOOL_NAME}"
    )


def _extract_usage(response: Any) -> TokenUsage:
    """Read `response.usage.*`, defaulting missing fields to 0.

    Compaction iterations (when enabled and triggered) are not aggregated
    here yet — that's a future extension; this stage's only consumer
    (market_context_agent) is a single-turn structured output.
    """
    usage_obj = getattr(response, "usage", None)
    if usage_obj is None:
        return TokenUsage(0, 0, 0, 0)
    return TokenUsage(
        input_tokens=int(getattr(usage_obj, "input_tokens", 0) or 0),
        cache_read_input_tokens=int(getattr(usage_obj, "cache_read_input_tokens", 0) or 0),
        cache_creation_input_tokens=int(
            getattr(usage_obj, "cache_creation_input_tokens", 0) or 0
        ),
        output_tokens=int(getattr(usage_obj, "output_tokens", 0) or 0),
    )


def _extract_applied_edits(response: Any) -> list[dict[str, Any]]:
    cm = getattr(response, "context_management", None)
    if cm is None:
        return []
    applied = getattr(cm, "applied_edits", None) or []
    out: list[dict[str, Any]] = []
    for edit in applied:
        if isinstance(edit, dict):
            out.append(edit)
        else:
            # Best-effort: dump attributes that look like fields.
            out.append({k: getattr(edit, k) for k in dir(edit) if not k.startswith("_")})
    return out


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _default_logger() -> structlog.stdlib.BoundLogger:
    if not structlog.is_configured():
        structlog.configure(
            processors=[
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso", utc=True),
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )
    return structlog.get_logger("core.llm_client")
