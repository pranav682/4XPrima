---
name: prompt-caching-and-token-budget
description: Practical, per-call checklist derived from docs/llm-conventions.md. Read this before assembling any agent's prompt; the canonical doc is the source of truth for numbers.
---

# prompt-caching-and-token-budget

## When to use

Every agent call. Every. Single. One.

The full rationale and verification stamps live in `docs/llm-conventions.md`. This skill is the at-a-glance checklist.

## The five rules

1. **Stable first, volatile last.** Order:
   ```
   tools  →  system [spec, skills, taxonomy]  →  CACHE BREAKPOINT  →  per-call policy  →  messages [volatile user turn]
   ```
   Anything that changes per call (timestamp, prices, run_id, candidate JSON) goes in the last user message.

2. **One explicit cache_control per agent.** Put `cache_control: {"type": "ephemeral"}` on the last block of the stable system prefix. Max 4 per request — do not exceed; do not sprinkle.

3. **Hit the model's cache minimum or skip the breakpoint.** Sonnet 4.5/4.6 = 1,024 tokens. Opus 4.5/4.6 and Haiku 4.5 = **4,096** tokens. Below the minimum, the cache write is silently dropped — wasted intent.

4. **Model switch = new cache lane.** Don't switch tiers mid-conversation. If the orchestrator changes tier between status_check (Haiku) and full cycle (Sonnet), they're separate caches — fine, expected.

5. **Defer tools you don't use every call.** Use `tool_search_tool_bm25_20251119` (or regex variant) with `defer_loading: true` on non-essential tools. Keep 3–5 essential tools eager. The search tool itself must NOT be `defer_loading`.

## Per-call checklist (paste into PR descriptions for any new agent code)

- [ ] Stable prefix (tools + system spec + skills) is byte-identical across calls.
- [ ] Exactly **one** `cache_control` breakpoint, at end of stable system.
- [ ] Volatile data (timestamps, prices, run_id, candidate JSON) is **after** the breakpoint — in the last user message.
- [ ] Stable prefix size ≥ model's cache minimum (Haiku/Opus 4.5/4.6 = 4k; Sonnet = 1k).
- [ ] Model string is pinned (no model drift).
- [ ] Eager tools ≤ 5; rest deferred via Tool Search Tool.
- [ ] If long-running: `context_management` configured (thinking edit listed first, see `docs/llm-conventions.md` §2).
- [ ] Token-budget regression test asserts input + output within the agent spec's budget.
- [ ] Caching regression test asserts: first call has `cache_creation_input_tokens > 0`, second has `cache_read_input_tokens > 0`.

## Things that silently break caching (watch list)

- Embedding the current time in the system prompt.
- Reordering JSON keys in the stable prefix (use a deterministic serializer).
- Adding a "today is …" line to the system prompt — keep it in the messages tail.
- Adding/removing a tool between calls (any tool, including MCP).
- Switching `tool_choice` or toggling images/thinking.
- Switching the model (Sonnet ↔ Opus) — accepted, but you do pay the cache write the first time on the new tier.

## Things that look like they break caching but don't

- `defer_loading: true` on tools — deferred tools are appended *after* discovery and do not invalidate the prefix.
- Adding new user/assistant turns at the end of `messages` — that is normal cache extension.
- A mid-conversation user message that pushes new instructions ("from now on, behave as…") — fine; mutating `system` is not.
