# 4xPrima — LLM Conventions

**This document is the single source of truth for prompt caching, context management, model routing, and token accounting.** Every agent and `core/llm_client.py` must implement these conventions. If something here is wrong, fix this doc first, then update the code.

> **Source verification.** All identifiers, defaults, and limits below are taken from the Anthropic docs as of **2026-06-23** (`platform.claude.com/docs/en/...`). When you change the LLM client or upgrade the SDK, re-verify against the docs and bump the "Verified" stamp at the bottom of each section.

---

## 1. Message structure & cache layout

### Prefix ordering

Anthropic caches in this hierarchy: **`tools` → `system` → `messages`**. Changes at one level invalidate the cache for that level and everything below it.

Build every agent request in this order, from most-stable to most-volatile:

```
1. tools                        ← stable per agent version
2. system: [
     spec block (cacheable),    ← stable per agent version
     skill blocks (cacheable),  ← stable per agent version
     dynamic policy (NOT cached) ← may include runtime overrides
   ]
3. messages: [
     prior-turn history,
     final user/assistant turn with VOLATILE data
     (timestamps, prices, run_id, candidate JSON, etc.)
   ]
```

Anything that changes per request — timestamps, current prices, run IDs, this-cycle's candidate spec — goes **after** the cache breakpoint, not before.

Verified: 2026-06-23.

### Explicit cache breakpoints

- Place an explicit `cache_control: {"type": "ephemeral"}` on the **last block of the stable prefix** (typically the last skill block in `system`).
- Use **automatic caching** (top-level `cache_control` field) only for very simple agents whose entire `system` is stable.
- **Max 4 explicit `cache_control` markers per request.** Reserve them for the boundaries that actually need them; do not sprinkle.
- The system checks **at most 20 positions** per breakpoint for matching cache entries — keep the prefix byte-identical across runs.

Verified: 2026-06-23.

### Cache minimums (must hit these or no cache write)

| Model | Min cacheable tokens |
| --- | --- |
| Sonnet 4.5 / 4.6 | **1,024** |
| Opus 4.5 / 4.6 | **4,096** |
| Opus 4.7 (preview) | 2,048 |
| Opus 4.8 | 1,024 |
| Haiku 4.5 | **4,096** |

Implication: if a Haiku/Opus 4.5/4.6 agent's stable prefix doesn't reach 4k tokens, the cache write is silently skipped. Either pad the prefix (with genuinely useful, stable content) or stop pretending it's cacheable.

Verified: 2026-06-23.

### TTL

- Default: `5m` (no `ttl` field, or `"ttl": "5m"`).
- Long-running scheduled cycles can use `"ttl": "1h"` — costs 2× base input on the cache write, but pays off if the prefix is reused across multiple agent calls in the same cycle.

**Pick TTL per scheduled cadence:** slow-loop cycles ≤ 5 minutes apart → 5m. Longer cycles → 1h, **only if** the same prefix is reused within the hour.

Verified: 2026-06-23.

### Things that invalidate the cache

Treat each of these as **a new cache lane**:

- Model switch (e.g. Sonnet → Opus for the critic).
- Tool-list change (adding/removing a tool, even to MCP definitions).
- Tool-choice change, image presence, thinking parameters → invalidates `system` and `messages` caches.
- Web search / citations toggles → invalidate `tools` cache (and everything below).

Implication: do **not** dynamically swap tools in/out mid-cycle for the same agent. Either declare them all up-front (and use the Tool Search Tool with `defer_loading` — see §3) or accept the cache miss.

Verified: 2026-06-23.

### Token accounting per call

Every LLM call must log, at minimum:

- `model`
- `agent_name`
- `run_id`
- `input_tokens` (tokens NOT served from cache, after the last breakpoint)
- `cache_read_input_tokens`
- `cache_creation_input_tokens`
- `output_tokens`
- a derived `cache_hit_ratio = cache_read / (cache_read + cache_creation + input)`

Total billable input = `cache_read_input_tokens + cache_creation_input_tokens + input_tokens`. Do not approximate.

Verified: 2026-06-23.

---

## 2. Context management (long-horizon agents)

Anthropic exposes server-side context editing under the `context_management` request parameter, behind the beta header `anthropic-beta: context-management-2025-06-27`.

> **Note (project brief used a placeholder ID).** The brief mentioned `compact_20260112`. The actual current edit types per Anthropic docs are `clear_tool_uses_20250919` (tool-result clearing) and `clear_thinking_20251015` (thinking-block clearing). We use those.

### When to enable

Enable for any agent that:

- accumulates many tool results in one session (`backtest_agent` running many backtests; `orchestrator_agent` mid-cycle),
- runs extended thinking blocks.

For one-shot agents that return a structured JSON in a single turn (e.g. `market_context_agent`), context editing is unnecessary — the call ends before the threshold matters.

### Configuration we use

```python
context_management = {
    "edits": [
        # IMPORTANT: clear_thinking must be listed before clear_tool_uses when both are used.
        {
            "type": "clear_thinking_20251015",
            "keep": {"type": "thinking_turns", "value": 2},
        },
        {
            "type": "clear_tool_uses_20250919",
            "trigger": {"type": "input_tokens", "value": 60000},
            "keep": {"type": "tool_uses", "value": 5},
            "clear_at_least": {"type": "input_tokens", "value": 10000},
            "exclude_tools": ["memory"],
        },
    ]
}
```

`clear_at_least` is set so that a clearing actually invalidates enough prefix to be worth the cache cost. `exclude_tools=["memory"]` keeps memory-tool I/O intact (see §2.2).

Verified: 2026-06-23.

### 2.2 Memory tool (optional, opt-in per agent)

For agents that must remember things *across sessions* (e.g. `orchestrator_agent` tracking which candidate is on which validation step), enable the memory tool:

- Tool type: `memory_20250818`, name: `memory`.
- Anthropic-hosted file-based storage; Claude reads/writes its own memory files.
- Combine with `clear_tool_uses_20250919` and `exclude_tools=["memory"]`.

Do **not** put trading-critical state in memory — that lives in our own audit log and run-state files. Memory is for the agent's reasoning continuity.

Verified: 2026-06-23.

---

## 3. Tool Search Tool & deferred loading

For agents that have access to many tools (MCP-heavy or multi-domain), use the Tool Search Tool so we don't burn the prefix on tool definitions the agent won't call this turn.

- Tool types: `tool_search_tool_regex_20251119` (regex queries) or `tool_search_tool_bm25_20251119` (natural-language queries).
- Mark non-critical tools with `defer_loading: true`.
- Keep 3–5 most-used tools **non-deferred** for speed.
- The tool search tool itself must **never** have `defer_loading: true`.
- At least one tool must be non-deferred (API returns 400 otherwise).

Default in `core/llm_client.py`: every agent uses `tool_search_tool_bm25_20251119` plus its 3–5 essential tools eager; everything else (broker adapters, vendor APIs, browser tools) is deferred.

`defer_loading` does **not** break prompt caching — deferred tools are appended inline when discovered, leaving the prefix untouched.

Verified: 2026-06-23.

---

## 4. Model routing (cost discipline)

Default routing, enforced by `core/llm_client.py` via a model-tier enum:

| Tier | Default model | Use for | Rationale |
| --- | --- | --- | --- |
| `HAIKU` | `claude-haiku-4-5-20251001` | Cheap classifiers, routing-only orchestrator ticks, status summarisers, reporting agent for short status messages. | Cheapest input/output; cache min is 4k so reserve for short stable prefixes. |
| `SONNET` (default) | `claude-sonnet-4-6` | Most slow-loop agents: market context, strategy lab, backtest, optimization, orchestrator decisions, reporting deep dives. | Best perf/cost balance; 1k cache minimum means almost any prefix caches. |
| `OPUS` | `claude-opus-4-7` | The critic, novel strategy-design sessions. | We pay Opus rates only where adversarial reasoning is worth it. |

Pin the **exact model string** in code (`core/llm_client.py`), do not let it drift. Model strings are part of the cache lane — switching models invalidates everything.

If you upgrade a default, bump the constant, re-run the prefix size check (cache minimums!), and note it in this file's changelog.

Verified: 2026-06-23.

---

## 5. Mid-conversation system updates

Sometimes an agent needs to apply a new instruction mid-cycle (e.g. orchestrator tells reporting: "now switch from summary mode to approval-request mode"). Two options:

- **Bad:** mutate the system prompt → invalidates the entire `system` cache.
- **Good:** push a new **user** or **assistant** turn that updates instructions ("From this turn forward, behave as follows…"). The stable prefix stays intact; only the messages tail extends.

Default to the good option. Treat the `system` block as immutable for the life of a session.

---

## 6. Build vs. buy — Claude Agent SDK

**Recommendation: start with a thin custom loop wrapping the `anthropic` Python SDK, packaged behind `core/llm_client.py`. Re-evaluate Claude Agent SDK adoption at the start of Stage 3.**

### The options

| Option | What it gives you | What it costs |
| --- | --- | --- |
| **Direct `anthropic` SDK + our `llm_client.py`** | Total control over message assembly, exact cache layout, model routing, accounting. | We implement the agent loop, tool dispatch, retries. |
| **Claude Agent SDK** (`pip install claude-agent-sdk`, Python ≥3.10) | Built-in agent loop; built-in tools (Read, Edit, Bash, Glob, Grep, WebSearch, WebFetch, AskUserQuestion); hooks (`PreToolUse`, `PostToolUse`, …); sessions; subagents; MCP integration; loads `.claude/skills/*/SKILL.md` automatically. | Less direct control over the exact cache-control placement and per-call accounting; designed for code-assistant workloads, not for our prompt-cache-heavy structured-output agents. |
| **Managed Agents** (hosted) | Anthropic runs the loop + sandbox; great for long async sessions. | Wrong shape for us — our agents read fast-loop data on our infrastructure and must return strict structured outputs. |

### Why thin custom loop first

1. Our agents are **mostly structured-output** (single-turn, JSON-schema'd). The Agent SDK shines for autonomous multi-tool loops; we don't need a loop for `market_context_agent`.
2. The cache economics matter a lot — see §1. We want to **own** the message assembly and the cache_control placement.
3. The two agents that *are* multi-tool (`backtest_agent`, `orchestrator_agent`) are simple enough that a few hundred lines of loop code is cheaper than another framework dependency.
4. We can still **use Agent SDK's skill loader convention** (`.claude/skills/*/SKILL.md`) for our `skills/` directory regardless of which loop we run — the format is the same.

### When to revisit

Adopt the Agent SDK when **any** of these becomes true:

- We need its hooks (`PreToolUse`, `PostToolUse`) for audit/permission gating and don't want to reimplement them.
- We add ≥3 MCP servers to the slow loop.
- We want subagents inside `orchestrator_agent` rather than top-level routing.
- We need its session/resume mechanics for long async runs.

If we adopt it, `core/llm_client.py` stays — it becomes a thin layer that constructs the SDK's `ClaudeAgentOptions` while still enforcing our cache layout and accounting.

---

## 7. Mandatory checklist for every new agent

Before merging a new agent or modifying an existing one:

- [ ] Spec exists in `specs/agents/<name>.md` and lists every tool, skill, and output field.
- [ ] All LLM calls go through `core/llm_client.py` (no direct `anthropic.Anthropic()` calls).
- [ ] Stable prefix (tools + system spec + skills) is byte-identical across runs and hits the model's cache minimum.
- [ ] Exactly one explicit `cache_control` breakpoint at end of stable prefix; nothing volatile before it.
- [ ] Volatile data (timestamps, prices, run_id, candidate JSON) is in the last user message, after the breakpoint.
- [ ] Model tier picked per §4 with one-line rationale in the agent spec.
- [ ] Tools beyond the eager 3–5 marked `defer_loading: true`.
- [ ] If long-running: `context_management` configured per §2 (thinking edit listed first).
- [ ] First call logs `cache_creation_input_tokens > 0`; subsequent calls log `cache_read_input_tokens > 0`. Regression test asserts this.
- [ ] Token-budget regression test asserts input + output stay within the spec's stated budget.

---

## Changelog

- **2026-06-23** — Initial version. Verified against `platform.claude.com/docs` and `code.claude.com/docs`. Corrected user brief's `compact_20260112` placeholder to the actual identifiers `clear_thinking_20251015` and `clear_tool_uses_20250919`. Pinned model strings: `claude-haiku-4-5-20251001`, `claude-sonnet-4-6`, `claude-opus-4-7`.
