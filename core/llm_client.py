"""Shared LLM client for the slow loop.

**All runtime LLM calls in 4xPrima must go through this module.** Direct calls to
`anthropic.Anthropic().messages.create(...)` from anywhere else are forbidden by
project convention (`CLAUDE.md`) — they bypass the cache layout, model routing,
and usage accounting this wrapper enforces.

This file is a typed STUB. No network calls are made yet. The shapes here are the
contract every agent will build against once Stage 3 starts (see `PLAN.md`).

Conventions implemented (see `docs/llm-conventions.md` for the full rationale):

1. Message structure: tools → system [stable spec + skills, with one
   cache_control breakpoint] → messages [volatile tail].
2. Model routing by tier (HAIKU / SONNET / OPUS), pinned model strings.
3. Optional server-side context management (`clear_thinking_20251015` listed
   before `clear_tool_uses_20250919`) for long-running agents.
4. Optional Tool Search Tool (`tool_search_tool_bm25_20251119`) with
   `defer_loading: true` on non-essential tools.
5. Token + cache-hit accounting recorded for every call via
   `core.usage_accounting`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Protocol

# ---------------------------------------------------------------------------
# Pinned model strings (single source of truth — see docs/llm-conventions.md §4).
# Switching models invalidates the prompt cache, so pin and bump explicitly.
# ---------------------------------------------------------------------------

MODEL_HAIKU: str = "claude-haiku-4-5-20251001"
MODEL_SONNET: str = "claude-sonnet-4-6"
MODEL_OPUS: str = "claude-opus-4-7"

# Beta header required for the context_management parameter.
CONTEXT_MANAGEMENT_BETA: str = "context-management-2025-06-27"

# Tool Search Tool identifiers.
TOOL_SEARCH_REGEX: str = "tool_search_tool_regex_20251119"
TOOL_SEARCH_BM25: str = "tool_search_tool_bm25_20251119"

# Context-editing identifiers (note: the project brief used a placeholder
# `compact_20260112`; the verified identifiers per the Anthropic docs are these).
CONTEXT_EDIT_CLEAR_THINKING: str = "clear_thinking_20251015"
CONTEXT_EDIT_CLEAR_TOOL_USES: str = "clear_tool_uses_20250919"

# Memory tool identifier.
MEMORY_TOOL_TYPE: str = "memory_20250818"
MEMORY_TOOL_NAME: str = "memory"


class ModelTier(str, Enum):
    """Which model class to route to. Picked per agent in its spec."""

    HAIKU = "haiku"
    SONNET = "sonnet"
    OPUS = "opus"


def model_for_tier(tier: ModelTier) -> str:
    """Resolve a tier to its pinned model string. Bump constants above to change."""
    return {
        ModelTier.HAIKU: MODEL_HAIKU,
        ModelTier.SONNET: MODEL_SONNET,
        ModelTier.OPUS: MODEL_OPUS,
    }[tier]


# ---------------------------------------------------------------------------
# Message-assembly types.
#
# The cacheable system prefix is built out of *blocks*: an agent spec block plus
# one block per referenced skill. The wrapper places exactly one `cache_control`
# breakpoint on the last block of the stable prefix.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StableBlock:
    """A piece of the stable prefix (agent spec, a skill, taxonomy, etc.).

    `kind` is a short tag used by the cache regression test and the usage log
    so we can tell which blocks compose the prefix. The text must be
    byte-identical across calls for the cache to hit.
    """

    kind: Literal["spec", "skill", "taxonomy", "schema", "policy_stable"]
    name: str
    text: str


@dataclass(frozen=True, slots=True)
class VolatilePolicy:
    """Per-call policy overrides that go AFTER the cache breakpoint.

    Use sparingly. Anything that changes per call belongs here, not in
    `StableBlock`s — putting it in the stable prefix breaks caching.
    """

    text: str


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """One tool definition. `defer_loading=True` keeps it out of the eager prefix.

    Default to deferred for everything that isn't called every turn. Keep the
    Tool Search Tool itself eager (`defer_loading=False`) — it cannot be
    deferred or the API rejects the request.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    defer_loading: bool = True


@dataclass(frozen=True, slots=True)
class ContextManagementConfig:
    """Server-side context-editing config for long-running agents.

    The wrapper emits the edits in the order `clear_thinking_20251015` first,
    then `clear_tool_uses_20250919`, as required by the API.
    """

    clear_thinking_keep_turns: int = 2
    clear_tool_uses_trigger_input_tokens: int = 60_000
    clear_tool_uses_keep: int = 5
    clear_tool_uses_clear_at_least_input_tokens: int = 10_000
    exclude_tools_from_clear: tuple[str, ...] = ("memory",)


@dataclass(frozen=True, slots=True)
class AgentRequest:
    """Everything `LLMClient.call` needs.

    Fields ordered to mirror the on-the-wire layout: identification, then the
    stable prefix, then volatile tail, then config knobs.
    """

    # Identification (for accounting; never sent to the model).
    agent_name: str
    run_id: str

    # Model routing.
    tier: ModelTier

    # Tools. The wrapper splits these into eager and deferred groups and emits a
    # Tool Search Tool when any deferred tools are present.
    tools: tuple[ToolSpec, ...] = ()

    # Stable prefix — order is preserved exactly.
    stable_system_blocks: tuple[StableBlock, ...] = ()

    # Per-call policy that lives AFTER the cache breakpoint inside `system`.
    volatile_policy: VolatilePolicy | None = None

    # Conversation. The final user turn carries the volatile payload (timestamps,
    # candidate JSON, etc.). Anthropic's hierarchy means everything in `messages`
    # is naturally after the prefix.
    messages: tuple[dict[str, Any], ...] = ()

    # Output sizing.
    max_output_tokens: int = 2048

    # Optional knobs.
    context_management: ContextManagementConfig | None = None
    enable_memory_tool: bool = False
    cache_ttl: Literal["5m", "1h"] = "5m"
    tool_search_variant: Literal["bm25", "regex"] = "bm25"

    # Free-form metadata stored on the usage row (e.g. cycle phase).
    extra_metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Token accounting for a single call. Mirrors the Anthropic usage object.

    `cache_hit_ratio` is derived; never trust a hand-computed total — sum the
    three input fields and compare.
    """

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
    """What `LLMClient.call` returns.

    `raw_blocks` is the model's content array (mixed text / tool_use / etc.); the
    caller is responsible for shape-checking against its agent's output schema.
    """

    agent_name: str
    run_id: str
    model: str
    stop_reason: str
    usage: TokenUsage
    raw_blocks: list[dict[str, Any]]
    # If context editing fired, what it cleared. Empty otherwise.
    applied_context_edits: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Wire-format assembly.
#
# These helpers translate `AgentRequest` into the Anthropic Messages API shape.
# They are pure: identical input → identical bytes → cache hit. Test that
# property.
# ---------------------------------------------------------------------------


def build_system_blocks(
    stable_blocks: tuple[StableBlock, ...],
    volatile_policy: VolatilePolicy | None,
    cache_ttl: Literal["5m", "1h"],
) -> list[dict[str, Any]]:
    """Build the `system` array.

    The single explicit `cache_control` marker is placed on the LAST stable
    block. Volatile policy (if any) is appended after with no cache_control —
    so it never enters the cache and never invalidates it.

    Raises:
        ValueError: if `stable_blocks` is empty (we always cache the spec).
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
        # Explicitly no cache_control here.
        system.append({"type": "text", "text": volatile_policy.text})

    return system


def build_tools(
    tools: tuple[ToolSpec, ...],
    tool_search_variant: Literal["bm25", "regex"],
    enable_memory_tool: bool,
) -> list[dict[str, Any]]:
    """Assemble the `tools` array.

    Adds a Tool Search Tool entry (eager) whenever any user-supplied tool is
    deferred, so the model can discover the deferred ones at runtime. Adds the
    memory tool (eager) if `enable_memory_tool=True`.
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

    # The API rejects requests where every tool is deferred. With Tool Search /
    # memory above we usually satisfy this; the guard is a belt-and-braces check.
    if out and all(item.get("defer_loading") for item in out):
        raise ValueError(
            "all tools deferred — keep the Tool Search Tool eager or mark one tool "
            "non-deferred"
        )

    return out


def build_context_management(
    config: ContextManagementConfig | None,
) -> dict[str, Any] | None:
    """Build the `context_management` parameter body, or None to omit it.

    Order matters: `clear_thinking_20251015` MUST be listed before
    `clear_tool_uses_20250919` per the Anthropic API contract.
    """
    if config is None:
        return None
    return {
        "edits": [
            {
                "type": CONTEXT_EDIT_CLEAR_THINKING,
                "keep": {"type": "thinking_turns", "value": config.clear_thinking_keep_turns},
            },
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
            },
        ]
    }


# ---------------------------------------------------------------------------
# Client.
# ---------------------------------------------------------------------------


class UsageRecorder(Protocol):
    """Anything that can persist a usage row. See `core.usage_accounting`."""

    def record(
        self,
        *,
        agent_name: str,
        run_id: str,
        model: str,
        usage: TokenUsage,
        extra: dict[str, str],
    ) -> None: ...


class LLMClient:
    """Thin wrapper around the Anthropic Messages API.

    Not implemented yet — this stub fixes the surface every agent will build
    against in Stage 3. When implemented, `call()` will:

      1. Resolve the tier to a pinned model string.
      2. Build the `tools`, `system`, and (passthrough) `messages` arrays.
      3. Call `messages.create(...)` with the `context-management-2025-06-27`
         beta header iff `context_management` is set.
      4. Read `usage` off the response, log it via `UsageRecorder`.
      5. Return a typed `AgentResponse`.

    The wrapper never holds broker credentials, never places trades, and is
    never invoked from the fast loop.
    """

    def __init__(
        self,
        *,
        anthropic_api_key: str,
        usage_recorder: UsageRecorder,
    ) -> None:
        # Hold creds and recorder. The actual SDK client is constructed lazily
        # in `call()` so unit tests don't need network access.
        self._anthropic_api_key = anthropic_api_key
        self._usage_recorder = usage_recorder

    def call(self, request: AgentRequest) -> AgentResponse:
        """Issue one structured agent call.

        STUB: raises NotImplementedError. The wire-format assembly helpers above
        are real and unit-testable; the network round-trip will land in Stage 3.
        """
        # Sanity-check the request shape now so future tests have something
        # to assert on. `build_*` helpers raise on misuse.
        _ = build_tools(request.tools, request.tool_search_variant, request.enable_memory_tool)
        _ = build_system_blocks(
            request.stable_system_blocks,
            request.volatile_policy,
            request.cache_ttl,
        )
        _ = build_context_management(request.context_management)
        _ = model_for_tier(request.tier)

        raise NotImplementedError(
            "core.llm_client.LLMClient.call is a stub. Implement in Stage 3 — see PLAN.md."
        )
