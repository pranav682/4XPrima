"""Provider-agnostic LLM client for the slow loop.

**All runtime LLM calls in 4xPrima must go through this module.** Direct calls
to the OpenAI SDK from anywhere else are forbidden by project convention
(``CLAUDE.md``) — they bypass the message-structure discipline, tier routing,
and usage accounting this wrapper enforces.

Two abstractions:

- :class:`LLMProvider` — the Protocol agents depend on. Its workhorse method is
  :meth:`LLMProvider.generate_structured`, which takes a stable system prefix,
  a volatile user payload, and a pydantic output model.
- :class:`OpenAIProvider` — the runtime implementation against the OpenAI SDK.
  Uses ``client.chat.completions.parse(response_format=PydanticModel)`` for
  structured output (the SDK's pydantic-native Structured Outputs path).

Conventions encoded here (see ``docs/llm-conventions.md`` for the rationale):

- Caching is AUTOMATIC. We do not set ``cache_control``; we structure each call
  as ``[system: STABLE PREFIX] + [user: VOLATILE PAYLOAD]`` and rely on
  OpenAI to cache the system prefix when it crosses the ~1024 token threshold.
- Tier routing pins to current GPT-5 family names (CHEAP / DEFAULT / HEAVY).
- Token accounting records ``prompt_tokens``, ``cached_tokens`` (from
  ``usage.prompt_tokens_details.cached_tokens``), and ``completion_tokens``.
- All SDK errors are wrapped in :class:`LlmClientError`; the API key never
  appears in the error message.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final, Protocol, TypeVar

import openai
import structlog
from pydantic import BaseModel, ValidationError

from core.config import OpenAISettings

# ---------------------------------------------------------------------------
# Pinned model strings — see docs/llm-conventions.md §3.
# Bumping a model is a deliberate change reviewed against the docs.
# ---------------------------------------------------------------------------

MODEL_CHEAP: Final[str] = "gpt-5.4-nano"
MODEL_DEFAULT: Final[str] = "gpt-5.4"
MODEL_HEAVY: Final[str] = "gpt-5.5"


class ModelTier(StrEnum):
    """Which model class to route to. Picked per agent in its spec."""

    CHEAP = "cheap"
    DEFAULT = "default"
    HEAVY = "heavy"


def model_for_tier(tier: ModelTier) -> str:
    return {
        ModelTier.CHEAP: MODEL_CHEAP,
        ModelTier.DEFAULT: MODEL_DEFAULT,
        ModelTier.HEAVY: MODEL_HEAVY,
    }[tier]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Token accounting for one call.

    ``cached_tokens`` is a SUBSET of ``prompt_tokens`` (OpenAI reports the
    cached share within the prompt). Cache hit ratio is
    ``cached_tokens / prompt_tokens``.
    """

    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int

    @property
    def cache_hit_ratio(self) -> float:
        return (self.cached_tokens / self.prompt_tokens) if self.prompt_tokens else 0.0


@dataclass(frozen=True, slots=True)
class AgentResponse:
    """Metadata about one LLM call.

    Decoupled from the parsed output so structured-output callers can return
    ``(parsed_model, AgentResponse)``.
    """

    agent_name: str
    run_id: str
    tier: ModelTier
    model: str
    finish_reason: str
    usage: TokenUsage
    extra_metadata: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class LlmClientError(Exception):
    """Unrecoverable LLM call failure.

    Wraps SDK errors and schema-validation failures. Error messages never
    include the API key (the SDK's errors redact it; we also never
    interpolate ``SecretStr.get_secret_value()`` into a message).
    """


# ---------------------------------------------------------------------------
# Accounting interop
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
# Provider Protocol — agents depend on THIS, not on OpenAI
# ---------------------------------------------------------------------------


T = TypeVar("T", bound=BaseModel)


class LLMProvider(Protocol):
    """The Protocol agents depend on.

    Switching providers later is a one-file change (a new implementation of
    this Protocol), not a refactor of every agent.
    """

    def generate_structured(
        self,
        *,
        agent_name: str,
        run_id: str,
        tier: ModelTier,
        stable_system: str,
        volatile_user: str,
        output_model: type[T],
        max_output_tokens: int = 2048,
        extra_metadata: dict[str, str] | None = None,
    ) -> tuple[T, AgentResponse]:
        """Issue one structured call. Returns the parsed pydantic instance
        and a typed response metadata object.

        Raises:
            LlmClientError: on SDK failure, schema-validation failure, or
                any other unrecoverable failure.
        """
        ...


# ---------------------------------------------------------------------------
# OpenAI implementation
# ---------------------------------------------------------------------------


class OpenAIProvider:
    """OpenAI-backed implementation of :class:`LLMProvider`.

    Constructed with an :class:`OpenAISettings` (SecretStr-wrapped key);
    the underlying ``openai.OpenAI`` client is injectable for tests so we
    never need to monkey-patch the SDK module.
    """

    def __init__(
        self,
        settings: OpenAISettings,
        *,
        openai_client: Any | None = None,
        usage_recorder: UsageRecorder | None = None,
        logger: structlog.stdlib.BoundLogger | None = None,
    ) -> None:
        self._settings = settings
        self._usage_recorder = usage_recorder
        if openai_client is None:
            kwargs: dict[str, Any] = {"api_key": settings.api_key.get_secret_value()}
            if settings.project:
                kwargs["project"] = settings.project
            if settings.org:
                kwargs["organization"] = settings.org
            self._client = openai.OpenAI(**kwargs)
        else:
            self._client = openai_client
        self._logger = (
            logger if logger is not None else _default_logger()
        ).bind(component="openai_provider")

    # ---------------------------------------------------------- workhorse

    def generate_structured(
        self,
        *,
        agent_name: str,
        run_id: str,
        tier: ModelTier,
        stable_system: str,
        volatile_user: str,
        output_model: type[T],
        max_output_tokens: int = 2048,
        extra_metadata: dict[str, str] | None = None,
    ) -> tuple[T, AgentResponse]:
        if not stable_system:
            raise ValueError("stable_system must not be empty — cache the agent spec")
        if not volatile_user:
            raise ValueError("volatile_user must not be empty")

        model = model_for_tier(tier)
        messages = [
            {"role": "system", "content": stable_system},
            {"role": "user", "content": volatile_user},
        ]
        meta = dict(extra_metadata or {})

        self._logger.info(
            "llm_call_dispatch",
            agent_name=agent_name,
            run_id=run_id,
            tier=tier.value,
            model=model,
            stable_system_chars=len(stable_system),
            volatile_user_chars=len(volatile_user),
            max_output_tokens=max_output_tokens,
        )

        try:
            completion = self._client.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=output_model,
                max_completion_tokens=max_output_tokens,
            )
        except openai.OpenAIError as e:
            # The SDK's error string already redacts the key; we also never
            # interpolate the key here. Type name + message is enough for
            # debugging without ever exposing a secret.
            raise LlmClientError(f"OpenAI API error: {type(e).__name__}: {e}") from e
        except Exception as e:
            raise LlmClientError(f"unexpected error from OpenAI SDK: {e!r}") from e

        # Parse the response: the SDK gives us a pydantic instance directly.
        parsed_raw = _extract_parsed(completion, output_model)

        # Belt-and-braces re-validation at our boundary. ``parsed_raw`` is
        # already a pydantic instance, but if the SDK ever changes how
        # strict-mode is enforced we catch any drift here.
        try:
            parsed = output_model.model_validate(parsed_raw.model_dump())
        except ValidationError as e:
            raise LlmClientError(
                f"model output failed schema validation for "
                f"{output_model.__name__}: {e}"
            ) from e

        # Usage accounting.
        usage = _extract_usage(completion)
        finish_reason = _extract_finish_reason(completion)
        response_meta = AgentResponse(
            agent_name=agent_name,
            run_id=run_id,
            tier=tier,
            model=model,
            finish_reason=finish_reason,
            usage=usage,
            extra_metadata=meta,
        )
        if self._usage_recorder is not None:
            self._usage_recorder.record(
                agent_name=agent_name,
                run_id=run_id,
                model=model,
                usage=usage,
                extra={"tier": tier.value, **meta},
            )
        return parsed, response_meta


# ---------------------------------------------------------------------------
# Helpers (response parsing)
# ---------------------------------------------------------------------------


def _extract_parsed(completion: Any, output_model: type[BaseModel]) -> BaseModel:
    """Pull the parsed pydantic instance out of a chat-completions.parse() result.

    The SDK puts the parsed instance on ``choices[0].message.parsed``. If a
    safety refusal happened, ``choices[0].message.refusal`` carries text.
    """
    choices = getattr(completion, "choices", None) or []
    if not choices:
        raise LlmClientError("OpenAI response had no choices")
    msg = getattr(choices[0], "message", None)
    if msg is None:
        raise LlmClientError("OpenAI response choice had no message")
    refusal = getattr(msg, "refusal", None)
    if refusal:
        raise LlmClientError(f"model refused the request: {refusal}")
    parsed = getattr(msg, "parsed", None)
    if parsed is None:
        raise LlmClientError(
            f"OpenAI response did not include a parsed {output_model.__name__}"
        )
    if not isinstance(parsed, BaseModel):
        # The SDK should always return a pydantic instance when we passed a
        # BaseModel class as response_format; this guards against shape drift.
        raise LlmClientError(
            f"parsed payload was not a pydantic model "
            f"(got {type(parsed).__name__})"
        )
    return parsed


def _extract_usage(completion: Any) -> TokenUsage:
    """Read ``completion.usage.*``, defaulting missing fields to 0."""
    usage_obj = getattr(completion, "usage", None)
    if usage_obj is None:
        return TokenUsage(0, 0, 0)
    prompt = int(getattr(usage_obj, "prompt_tokens", 0) or 0)
    completion_t = int(getattr(usage_obj, "completion_tokens", 0) or 0)
    details = getattr(usage_obj, "prompt_tokens_details", None)
    cached = 0
    if details is not None:
        cached = int(getattr(details, "cached_tokens", 0) or 0)
    return TokenUsage(
        prompt_tokens=prompt, cached_tokens=cached, completion_tokens=completion_t
    )


def _extract_finish_reason(completion: Any) -> str:
    choices = getattr(completion, "choices", None) or []
    if not choices:
        return ""
    return str(getattr(choices[0], "finish_reason", "") or "")


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


__all__ = [
    "MODEL_CHEAP",
    "MODEL_DEFAULT",
    "MODEL_HEAVY",
    "AgentResponse",
    "LLMProvider",
    "LlmClientError",
    "ModelTier",
    "OpenAIProvider",
    "TokenUsage",
    "UsageRecorder",
    "model_for_tier",
]
