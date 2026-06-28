#!/usr/bin/env bash
#
# Enforce `mypy --strict` over the COMPLETED, type-clean modules only.
#
# This is a RATCHET, not a full-repo gate. The repo is not `--strict` clean yet
# (see docs/type-debt.md). What this script guarantees is narrower and honest:
# the modules listed below ARE clean today, and this fails CI/pre-commit the
# moment a new --strict error appears in any of them — so completed code can't
# silently rot back into debt.
#
# Policy: every NEW module must land --strict clean and be added here. As a
# debt-laden package stabilises and is cleaned, add it to ENFORCED and delete
# its row from docs/type-debt.md. Do NOT add a package with known debt.
#
# Run:   scripts/typecheck.sh      (or: bash scripts/typecheck.sh)
# Exits non-zero if any --strict error is found in the enforced set.
#
set -euo pipefail

cd "$(dirname "$0")/.."

# --- The enforced-clean set (keep in sync with CLAUDE.md + docs/type-debt.md).
# mypy only reports errors for files passed explicitly; imported-but-unlisted
# debt modules (e.g. core/risk_manager.py, imported by core/backtest/engine.py)
# are followed for type info but their errors are NOT reported here. That's why
# this set passes even though the wider repo does not.
ENFORCED=(
  core/models.py            # frozen domain models
  core/strategy.py          # strategy contract + reference strategy
  core/backtest/            # deterministic backtester
  core/broker.py            # broker Protocol
  core/config.py            # pydantic-settings config
  core/usage_accounting.py  # per-agent token/cache log
)

echo "mypy --strict (enforced-clean set):"
printf '  %s\n' "${ENFORCED[@]}"
echo

exec mypy --strict "${ENFORCED[@]}"
