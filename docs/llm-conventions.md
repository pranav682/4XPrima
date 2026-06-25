# 4xPrima — LLM Conventions

**This document is the single source of truth for prompt structure, model routing, structured output, and token accounting.** Every agent and `core/llm_client.py` must follow these conventions. If something here is wrong, fix this doc first, then update the code.

> **Source verification.** All endpoint shapes, parameter names, and numbers below are taken from the current OpenAI docs (`developers.openai.com/api/docs`) as of **2026-06-25**. When you change the LLM client or upgrade the SDK, re-verify against the docs and bump the "Verified" stamp at the bottom of each section.

---

## 1. Provider abstraction

Agents depend on `core.llm_client.LLMProvider` — a thin Protocol with a single workhorse method, `generate_structured(...)`. The runtime implementation is `OpenAIProvider`, but no agent imports it directly. Switching providers later is a one-file change (a new implementation of the Protocol), not a refactor of every agent.

Why not LiteLLM / langchain / agent-framework: those add another layer of caching/retry that we want to own ourselves at the boundary so the accounting and behavior stay deterministic. We can revisit if/when we wire in a second provider.

---

## 2. Message structure (the only one)

OpenAI's prompt cache reads the **prefix** of every chat-completion request. To keep cache hits high, every agent call MUST be assembled as:

```
messages = [
  { "role": "system", "content": <STABLE PREFIX> },   # ≥ 1024 tokens, byte-identical across calls
  { "role": "user",   "content": <VOLATILE PAYLOAD> } # this call's distilled brief
]
```

- The system message is **the only stable block** and holds the agent's instructions plus its output-schema notes / skill text. It must be byte-identical across calls for the same agent — any whitespace churn breaks the cache.
- The user message carries everything that changes per call: timestamps, prices, run IDs, the distilled brief.
- **No `cache_control`, no betas, no `context_management`.** These don't exist in OpenAI's API. Caching is automatic — the only thing we control is the prefix-stability discipline above.

If the stable prefix is shorter than **1024 tokens**, OpenAI will not cache it at all. Either pad the prefix with genuinely useful, stable content (the schema notes are a good source) or admit the agent isn't worth caching.

Verified: 2026-06-25.

### Why not a multi-turn approach with assistant turns mid-prompt?

We do single-shot structured calls. The whole point of the brief is to flatten everything an agent needs into one volatile user turn so the previous (system) turn is the cached prefix. Multi-turn would push variable content into earlier positions and kill caching.

---

## 3. Model routing (cost discipline)

Default routing, enforced by `core/llm_client.py` via a model-tier enum:

| Tier | Default model | Use for | Rationale |
| --- | --- | --- | --- |
| `CHEAP` | `gpt-5.4-nano` | Cheap classifiers, status summaries, low-stakes routing decisions, live-smoke wiring tests. | Cheapest input/output in the GPT-5 family; great for high-volume short calls. |
| `DEFAULT` | `gpt-5.4` | Most slow-loop agents: `market_context_agent`, strategy lab, backtest interpretation, optimization, orchestrator decisions, reporting deep-dives. | Best perf/cost balance — $2.50/$15 per MTok. Automatic caching makes the long stable prefix cheap to re-send. |
| `HEAVY` | `gpt-5.5` | The critic, novel strategy design, anything where a wrong "accept" is genuinely costly. | We pay HEAVY rates ($5/$30 per MTok) only where reasoning quality is worth it. |

Pin the **exact model string** in code (`core/llm_client.py`), do not let it drift. Bumping a model is a deliberate change reviewed against the docs.

When you upgrade a default, bump the constant, re-run the live smoke (the new model may have a different cache-prefix hash), and note the change in this file's changelog.

> **Pinning note.** The OpenAI docs currently expose bare names (e.g. `gpt-5.4`) without dated snapshot suffixes for the GPT-5 family. The bare names are themselves the pinned IDs we use. If/when dated forms become available (e.g. `gpt-5.4-2026-XX-XX`), prefer those for stricter reproducibility and bump in one PR.

Verified: 2026-06-25.

---

## 4. Structured output

All agent calls go through `OpenAIProvider.generate_structured(output_model=SomeModel, ...)`. Internally this calls `client.chat.completions.parse(model=..., messages=..., response_format=SomeModel)`, which is OpenAI's pydantic-native Structured Outputs path. The SDK derives the strict JSON schema, the model is required to return a valid instance, and the parsed pydantic object comes back as `response.choices[0].message.parsed`.

We accept the parsed model only if it survives the post-API pydantic validation a second time at our boundary — defence in depth against any drift between the SDK's parse and the strict schema constraints in the model. Validation failure → `LlmClientError`.

Verified: 2026-06-25.

---

## 5. Token accounting

Every call records three numbers via `core.usage_accounting`:

- `prompt_tokens` — total input tokens (from `response.usage.prompt_tokens`).
- `cached_tokens` — the subset of `prompt_tokens` served from cache (from `response.usage.prompt_tokens_details.cached_tokens`). 0 when no cache hit.
- `completion_tokens` — output tokens (from `response.usage.completion_tokens`).

Cache-hit ratio is `cached_tokens / prompt_tokens`. A healthy DEFAULT-tier agent settles at 70%+ once warm. If the ratio is 0 on the second call of a same-prefix sequence, the cache is broken — see §2.

Verified: 2026-06-25.

---

## 6. Mandatory checklist for every new agent

- [ ] Spec exists in `specs/agents/<name>.md` and lists inputs, tool deps, model tier, and the structured-output model.
- [ ] All LLM calls go through `core/llm_client.py` — no direct `import openai` anywhere else.
- [ ] System message is a single byte-identical string per agent version. No timestamps, no run_id, no per-call data inside it.
- [ ] System message is ≥ 1024 tokens (or you've explicitly accepted that this agent won't cache).
- [ ] User message carries every per-call value.
- [ ] Tier picked per §3 with one-line rationale in the spec.
- [ ] Output is a frozen pydantic model with extra="forbid" and field-level validation.
- [ ] First call shows `cached_tokens == 0`; the second call of an identical-prefix sequence shows `cached_tokens > 0` (asserted by live smoke).
- [ ] Per-call token totals stay within the spec's budget (regression test asserts).

---

## Changelog

- **2026-06-25** — Switched the LLM layer to OpenAI. Retired the Anthropic-specific scaffolding (manual `cache_control`, compaction edits, context-management beta, tier-aware compaction guard) — none apply to OpenAI. New tier mapping: `CHEAP=gpt-5.4-nano`, `DEFAULT=gpt-5.4`, `HEAVY=gpt-5.5`. Caching is automatic at ≥1024 tokens with `usage.prompt_tokens_details.cached_tokens` reporting hits. Structured output via `client.chat.completions.parse(response_format=PydanticModel)`. Agents now depend on the provider-agnostic `LLMProvider` Protocol.
- **2026-06-24** — *(superseded)* Corrections to the Anthropic conventions: bumped Opus tier to `claude-opus-4-8`; restored `compact_20260112` as a real edit type with the `compact-2026-01-12` beta. Retired by the 2026-06-25 OpenAI switch.
- **2026-06-23** — *(superseded)* Initial Anthropic version. Retired by the 2026-06-25 OpenAI switch.
