#!/usr/bin/env bash
#
# THE quality gate. Run this before every commit.
#
# Runs ruff, black --check, and mypy --strict over ONE shared enforced-clean
# set, and fails if any tool reports any error in any of those files. All three
# tools cover the EXACT same set on purpose — a file is "enforced" only when it
# is simultaneously ruff-clean, black-clean, and mypy --strict-clean. That rule
# is what keeps the three gates from drifting onto different file sets.
#
# This is a RATCHET, not a full-repo gate. The repo is not yet clean under any
# of the three tools (see docs/quality-debt.md). What this guarantees is that
# COMPLETED code can't silently rot. As a debt-laden file is cleaned under all
# three tools, add it to ENFORCED below and delete its row from
# docs/quality-debt.md. Do NOT add a file while it still fails any tool.
#
# Note: mypy reports errors only for files passed explicitly, so debt modules
# pulled in via import (e.g. core/risk_manager.py, imported by the backtester)
# are followed for type info but don't fail the gate.
#
# Run:   scripts/check.sh      (or: bash scripts/check.sh)
# Exits non-zero if any check fails.
#
set -uo pipefail

cd "$(dirname "$0")/.."

# --- The enforced-clean set. Keep in sync with CLAUDE.md + docs/quality-debt.md.
ENFORCED=(
  core/models.py            # frozen domain models
  core/strategy.py          # strategy contract + reference strategy
  core/backtest/            # deterministic backtester
  core/analysis/            # structural pair screener
  core/agents/strategy_lab_agent.py  # strategy proposal agent (rest of core/agents/ is debt)
  core/agents/backtest_harness.py    # deterministic backtest harness
  core/agents/backtest_agent.py      # backtest interpretation agent
  core/broker.py            # broker Protocol
  core/config.py            # pydantic-settings config
  core/usage_accounting.py  # per-agent token/cache log
)

fail=0
run() {
  local label=$1
  shift
  echo "=== ${label} ==="
  if "$@"; then
    echo "  OK"
  else
    echo "  FAILED"
    fail=1
  fi
  echo
}

run "ruff check"    ruff check "${ENFORCED[@]}"
run "black --check"  black --check "${ENFORCED[@]}"
run "mypy --strict"  mypy --strict "${ENFORCED[@]}"

if [ "${fail}" -ne 0 ]; then
  echo "QUALITY GATE FAILED — see output above."
  exit 1
fi
echo "QUALITY GATE PASSED — ruff + black + mypy --strict clean over the enforced set."
