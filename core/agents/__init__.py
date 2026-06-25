"""Slow-loop agents.

Every agent in this package goes through :mod:`core.llm_client` for its LLM
calls — see ``CLAUDE.md`` invariant: ALL runtime LLM calls MUST go through
``core/llm_client.py``. Direct anthropic SDK use anywhere here is forbidden.
"""
