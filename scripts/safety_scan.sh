#!/usr/bin/env bash
# Hermetic secret-leak scan for 4xPrima.
#
# RULES (see CLAUDE.md → "Security hygiene"):
#   1. This script MUST NEVER contain literal secret values — only PATTERNS.
#   2. We never echo a matched secret to stdout. Only file:line is reported.
#   3. Run this before every push, and before sharing this repo or its
#      commit messages.
#
# Use:
#   ./scripts/safety_scan.sh
# Exit 0 = clean; exit 1 = at least one finding.

set -uo pipefail

cd "$(git rev-parse --show-toplevel)" || exit 2

errors=0

# 1. .env must be gitignored. An accidentally-tracked .env is a hard fail
#    regardless of its contents.
if [ -e .env ]; then
    if ! git check-ignore -q .env; then
        echo "FAIL: .env exists in the working tree but is NOT gitignored." >&2
        errors=$((errors + 1))
    fi
fi
if git ls-files --error-unmatch .env >/dev/null 2>&1; then
    echo "FAIL: .env appears in 'git ls-files' — it is tracked." >&2
    errors=$((errors + 1))
fi

# 2. Pattern scan over tracked files. The patterns below DO NOT contain
#    any real key prefix uniquely identifying any account. Add a pattern
#    when a new provider's token format is adopted; never paste a real
#    value.
PATTERNS=(
    # OpenAI / Anthropic: covers `sk-` plus optional service prefix.
    'sk-(ant-|svcacct-|proj-|admin-)?[A-Za-z0-9_-]{20,}'
    # GitHub PAT / fine-grained / OAuth / runner / refresh tokens.
    'gh[opsru]_[A-Za-z0-9]{36,}'
    # AWS access-key IDs.
    'AKIA[0-9A-Z]{16}'
    # Slack bot / app / user tokens.
    'xox[abprs]-[A-Za-z0-9-]{10,}'
    # Google Cloud / generic RSA private keys baked into JSON or PEM files.
    '-----BEGIN (RSA |EC )?PRIVATE KEY-----'
    # JWT-style bearer tokens (three base64url segments separated by dots).
    'eyJ[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}'
)

joined=$(IFS='|'; echo "${PATTERNS[*]}")

# --cached restricts to tracked files; -I skips binaries. We deliberately
# discard the matched-line content via awk so we never echo a secret.
if hits=$(git grep -InE --cached "$joined" -- . 2>/dev/null); then
    echo "FAIL: token-shaped strings found in tracked files:" >&2
    # Print file:line only; never the matched text.
    echo "$hits" | awk -F: '{ print "  " $1 ":" $2 }' >&2
    errors=$((errors + 1))
fi

# 3. Recommended escalation: a real secret scanner (gitleaks / trufflehog)
#    catches entropy-based secrets that pattern matching misses. Try
#    `gitleaks detect --no-banner` if installed; non-zero exit is a fail.
if command -v gitleaks >/dev/null 2>&1; then
    if ! gitleaks detect --no-banner --redact -q; then
        echo "FAIL: gitleaks reported findings (output redacted)." >&2
        errors=$((errors + 1))
    fi
fi

if [ "$errors" -gt 0 ]; then
    echo "" >&2
    echo "safety_scan: $errors issue(s) detected. NOT safe to push." >&2
    exit 1
fi

echo "safety_scan: clean."
echo "  .env: gitignored and untracked"
echo "  tracked files: no token-pattern matches"
if command -v gitleaks >/dev/null 2>&1; then
    echo "  gitleaks: no findings"
else
    echo "  gitleaks: not installed (recommended — see CLAUDE.md)"
fi
exit 0
