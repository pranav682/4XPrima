"""Deterministic risk manager — the safety keystone of the fast loop.

This module is **pure Python, no I/O, no LLM**. The fast loop must route every
order through :meth:`RiskManager.evaluate`; there is no bypass. Higher-level
components (the order router, execution) treat ``RiskDecision.accepted`` as the
*only* signal that an order may go to the broker.

See ``specs/components/risk_manager.md`` for the spec and ``CLAUDE.md`` for the
hard invariants this module enforces (invariant #2 in particular).
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Final

import structlog

from core.models import (
    AccountState,
    DecisionKind,
    OrderRequest,
    RejectionReason,
    RiskConfig,
    RiskDecision,
)

# ---------------------------------------------------------------------------
# Kill switch
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class KillSwitchState:
    """Latching switch state. Owned by ``RiskManager``.

    Once ``engaged=True``, only :meth:`RiskManager.reset` flips it back. This
    deliberately survives across many ``evaluate`` calls — that is the *latch*.
    """

    engaged: bool = False
    reason: str | None = None
    tripped_at: datetime | None = None
    tripped_by: str | None = None  # short tag: "drawdown", "daily_loss", "manual", ...


# ---------------------------------------------------------------------------
# Config hashing (for audit)
# ---------------------------------------------------------------------------


def _hash_config(config: RiskConfig) -> str:
    """Stable short hash of a ``RiskConfig`` for audit-trail attribution.

    Uses sort_keys to keep the hash deterministic regardless of dict ordering.
    Truncated to 16 hex chars — collision risk is irrelevant at the rate we
    change configs, and short hashes read better in logs.
    """
    payload = json.dumps(config.model_dump(mode="json"), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# RiskManager
# ---------------------------------------------------------------------------


_DEFAULT_REASONS_LABEL: Final[dict[RejectionReason, str]] = {
    RejectionReason.KILL_SWITCH: "kill switch is engaged",
    RejectionReason.PER_TRADE_CAP: "per-trade risk cap",
    RejectionReason.MAX_CONCURRENT_POSITIONS: "max concurrent positions reached",
    RejectionReason.PORTFOLIO_RISK_CAP: "aggregate portfolio risk cap",
    RejectionReason.PER_PAIR_EXPOSURE_CAP: "per-pair notional exposure cap",
    RejectionReason.CORRELATED_EXPOSURE_CAP: "correlated notional exposure cap",
    RejectionReason.DAILY_LOSS_LIMIT: "daily loss limit breached",
    RejectionReason.DRAWDOWN_CAP: "max-drawdown cap breached",
    RejectionReason.INVALID_INPUT: "invalid order input",
    RejectionReason.STOP_DISTANCE_NONPOSITIVE: "stop distance must be positive",
    RejectionReason.NONPOSITIVE_EQUITY: "equity is non-positive",
}


class RiskManager:
    """Single deterministic gate between strategy signals and broker orders.

    Construction:

        >>> rm = RiskManager(config)

    Each evaluation:

        >>> decision = rm.evaluate(order, account)
        >>> if decision.accepted:
        ...     route_to_broker(decision.sized_order)  # uses the (possibly resized) order

    Concurrency: instances are NOT thread-safe. The fast loop is single-threaded
    by design; if that ever changes, wrap mutations in a lock.
    """

    def __init__(
        self,
        config: RiskConfig,
        *,
        logger: structlog.stdlib.BoundLogger | None = None,
        initial_kill_switch: KillSwitchState | None = None,
    ) -> None:
        self._config = config
        self._config_hash = _hash_config(config)
        self._kill_switch = initial_kill_switch or KillSwitchState()
        self._logger: structlog.stdlib.BoundLogger = (
            logger if logger is not None else _default_logger()
        ).bind(component="risk_manager", config_hash=self._config_hash)

    # ------------------------------------------------------------------ props

    @property
    def config(self) -> RiskConfig:
        """The active config. Cannot be replaced on an existing instance —
        config changes mean constructing a fresh ``RiskManager``."""
        return self._config

    @property
    def config_hash(self) -> str:
        return self._config_hash

    @property
    def kill_switch_engaged(self) -> bool:
        return self._kill_switch.engaged

    @property
    def kill_switch_state(self) -> KillSwitchState:
        """Snapshot copy so callers can't mutate ours."""
        ks = self._kill_switch
        return KillSwitchState(
            engaged=ks.engaged,
            reason=ks.reason,
            tripped_at=ks.tripped_at,
            tripped_by=ks.tripped_by,
        )

    # ---------------------------------------------------------- kill switch

    def trip(
        self,
        reason: str,
        *,
        tripped_by: str = "manual",
        now: datetime | None = None,
    ) -> None:
        """Engage the kill switch. Idempotent: re-tripping preserves the first
        reason (we want the *root cause* in the audit, not the latest)."""
        if self._kill_switch.engaged:
            self._logger.warning(
                "kill_switch_already_engaged",
                attempted_reason=reason,
                attempted_by=tripped_by,
                original_reason=self._kill_switch.reason,
                original_tripped_by=self._kill_switch.tripped_by,
            )
            return
        self._kill_switch = KillSwitchState(
            engaged=True,
            reason=reason,
            tripped_at=now or _utcnow(),
            tripped_by=tripped_by,
        )
        tripped_at_iso = (
            self._kill_switch.tripped_at.isoformat()
            if self._kill_switch.tripped_at
            else None
        )
        self._logger.critical(
            "kill_switch_tripped",
            reason=reason,
            tripped_by=tripped_by,
            tripped_at=tripped_at_iso,
        )

    def reset(self, *, operator: str, confirmation: str) -> None:
        """Clear the kill switch. Requires an operator name and the exact
        confirmation token ``"I_UNDERSTAND_RESET"`` to make accidental resets
        from automated callers near-impossible.

        Raises:
            PermissionError: if the confirmation token doesn't match.
        """
        if confirmation != "I_UNDERSTAND_RESET":
            self._logger.error(
                "kill_switch_reset_rejected_bad_confirmation",
                operator=operator,
            )
            raise PermissionError("invalid kill-switch reset confirmation")
        prior = self.kill_switch_state
        self._kill_switch = KillSwitchState()
        self._logger.critical(
            "kill_switch_reset",
            operator=operator,
            prior_reason=prior.reason,
            prior_tripped_by=prior.tripped_by,
            prior_tripped_at=prior.tripped_at.isoformat() if prior.tripped_at else None,
        )

    # ---------------------------------------------------------------- main

    def evaluate(self, order: OrderRequest, account: AccountState) -> RiskDecision:
        """Decide whether ``order`` may go to the broker, possibly resized.

        Order of checks — each gate either rejects outright or hands off to the
        next. The kill switch is checked first because no matter what other
        invariants hold, a tripped switch denies everything.

        Steps:

        1. Kill switch.
        2. Input validity (stop distance > 0, equity > 0).
        3. Drawdown breach → trip + reject.
        4. Daily-loss-limit breach → trip + reject.
        5. Max concurrent positions.
        6. Per-trade risk cap → RESIZE down if exceeded.
        7. Per-pair notional exposure cap.
        8. Correlated-group notional exposure cap.
        9. Aggregate portfolio risk cap.
        10. Approve (as APPROVE or RESIZE).
        """
        now = account.as_of
        decision_id = uuid.uuid4().hex

        # 1. Kill switch
        if self._kill_switch.engaged:
            return self._reject(
                decision_id,
                now,
                (RejectionReason.KILL_SWITCH,),
                reason=(
                    f"kill switch engaged "
                    f"(tripped_by={self._kill_switch.tripped_by!r}, "
                    f"reason={self._kill_switch.reason!r})"
                ),
                order=order,
                account=account,
            )

        # 2. Input validity
        if order.stop_distance <= 0:
            return self._reject(
                decision_id,
                now,
                (RejectionReason.STOP_DISTANCE_NONPOSITIVE,),
                reason="stop distance must be positive",
                order=order,
                account=account,
            )
        if account.equity <= 0:
            return self._reject(
                decision_id,
                now,
                (RejectionReason.NONPOSITIVE_EQUITY,),
                reason=f"equity is non-positive ({account.equity})",
                order=order,
                account=account,
            )

        # 3. Drawdown — trip + reject if breached.
        if account.drawdown_pct >= self._config.max_drawdown_pct:
            self.trip(
                f"drawdown {account.drawdown_pct} >= cap {self._config.max_drawdown_pct}",
                tripped_by="drawdown",
                now=now,
            )
            return self._reject(
                decision_id,
                now,
                (RejectionReason.DRAWDOWN_CAP, RejectionReason.KILL_SWITCH),
                reason=(
                    f"drawdown {account.drawdown_pct:.4%} breached cap "
                    f"{self._config.max_drawdown_pct:.4%}; kill switch engaged"
                ),
                order=order,
                account=account,
            )

        # 4. Daily loss limit — trip + reject if breached.
        if account.daily_loss_pct >= self._config.daily_loss_limit_pct:
            self.trip(
                f"daily_loss {account.daily_loss_pct} >= cap {self._config.daily_loss_limit_pct}",
                tripped_by="daily_loss",
                now=now,
            )
            return self._reject(
                decision_id,
                now,
                (RejectionReason.DAILY_LOSS_LIMIT, RejectionReason.KILL_SWITCH),
                reason=(
                    f"daily loss {account.daily_loss_pct:.4%} breached cap "
                    f"{self._config.daily_loss_limit_pct:.4%}; kill switch engaged"
                ),
                order=order,
                account=account,
            )

        # 5. Max concurrent positions
        if len(account.open_positions) >= self._config.max_concurrent_positions:
            return self._reject(
                decision_id,
                now,
                (RejectionReason.MAX_CONCURRENT_POSITIONS,),
                reason=(
                    f"{len(account.open_positions)} open positions ≥ cap "
                    f"{self._config.max_concurrent_positions}"
                ),
                order=order,
                account=account,
            )

        # 6. Per-trade risk cap → resize down if needed.
        max_trade_risk = self._config.max_risk_per_trade_pct * account.equity
        sized_order = order
        resized_for_per_trade = False
        if order.risk_at_stop > max_trade_risk:
            # New size = max_trade_risk / stop_distance. Always positive here
            # since stop_distance > 0 and max_trade_risk > 0.
            new_size = max_trade_risk / order.stop_distance
            if new_size <= 0:
                # Defence in depth — the arithmetic above can't yield this with
                # validated inputs, but if it ever did we reject rather than
                # silently approve a zero-sized order.
                return self._reject(
                    decision_id,
                    now,
                    (RejectionReason.PER_TRADE_CAP,),
                    reason="per-trade risk cap leaves no positive size",
                    order=order,
                    account=account,
                )
            sized_order = order.with_size(new_size)
            resized_for_per_trade = True

        # 7. Per-pair notional exposure cap
        existing_pair_notional = sum(
            (p.notional for p in account.open_positions if p.pair == sized_order.pair),
            start=Decimal("0"),
        )
        per_pair_cap = self._config.max_exposure_per_pair_pct * account.equity
        if existing_pair_notional + sized_order.notional > per_pair_cap:
            return self._reject(
                decision_id,
                now,
                (RejectionReason.PER_PAIR_EXPOSURE_CAP,),
                reason=(
                    f"per-pair {sized_order.pair} notional "
                    f"{existing_pair_notional + sized_order.notional} > cap {per_pair_cap}"
                ),
                order=order,
                account=account,
            )

        # 8. Correlated exposure cap — checked per group that contains the pair.
        groups = self._config.groups_containing(sized_order.pair)
        correlated_cap = self._config.max_correlated_exposure_pct * account.equity
        for group_name in groups:
            members = set(self._config.correlation_groups[group_name])
            group_notional = sum(
                (p.notional for p in account.open_positions if p.pair in members),
                start=Decimal("0"),
            )
            projected = group_notional + sized_order.notional
            if projected > correlated_cap:
                return self._reject(
                    decision_id,
                    now,
                    (RejectionReason.CORRELATED_EXPOSURE_CAP,),
                    reason=(
                        f"correlated group {group_name!r} notional {projected} "
                        f"> cap {correlated_cap}"
                    ),
                    order=order,
                    account=account,
                )

        # 9. Aggregate portfolio risk-at-stop cap
        existing_portfolio_risk = sum(
            (p.risk_at_stop for p in account.open_positions),
            start=Decimal("0"),
        )
        portfolio_cap = self._config.max_portfolio_risk_pct * account.equity
        if existing_portfolio_risk + sized_order.risk_at_stop > portfolio_cap:
            return self._reject(
                decision_id,
                now,
                (RejectionReason.PORTFOLIO_RISK_CAP,),
                reason=(
                    f"portfolio risk {existing_portfolio_risk + sized_order.risk_at_stop} "
                    f"> cap {portfolio_cap}"
                ),
                order=order,
                account=account,
            )

        # 10. Approve (RESIZE if step 6 trimmed the order).
        kind = DecisionKind.RESIZE if resized_for_per_trade else DecisionKind.APPROVE
        decision = RiskDecision(
            decision_id=decision_id,
            kind=kind,
            sized_order=sized_order,
            rejected_by=(),
            reason=(
                "resized down to fit per-trade risk cap"
                if resized_for_per_trade
                else "approved"
            ),
            limiting_rule=RejectionReason.PER_TRADE_CAP if resized_for_per_trade else None,
            config_hash=self._config_hash,
            as_of=now,
        )
        self._log_decision(decision, order, account)
        return decision

    # ----------------------------------------------------------- internals

    def _reject(
        self,
        decision_id: str,
        now: datetime,
        rejected_by: tuple[RejectionReason, ...],
        *,
        reason: str,
        order: OrderRequest,
        account: AccountState,
    ) -> RiskDecision:
        decision = RiskDecision(
            decision_id=decision_id,
            kind=DecisionKind.REJECT,
            sized_order=None,
            rejected_by=rejected_by,
            reason=reason,
            limiting_rule=rejected_by[0] if rejected_by else None,
            config_hash=self._config_hash,
            as_of=now,
        )
        self._log_decision(decision, order, account)
        return decision

    def _log_decision(
        self,
        decision: RiskDecision,
        order: OrderRequest,
        account: AccountState,
    ) -> None:
        # One structured record per decision. Keep field names stable —
        # downstream analytics depend on them.
        self._logger.info(
            "risk_decision",
            decision_id=decision.decision_id,
            kind=decision.kind.value,
            limiting_rule=(
                decision.limiting_rule.value if decision.limiting_rule else None
            ),
            rejected_by=[r.value for r in decision.rejected_by],
            reason=decision.reason,
            order_pair=order.pair,
            order_direction=order.direction.value,
            order_size_in=str(order.size),
            order_size_out=(
                str(decision.sized_order.size) if decision.sized_order else None
            ),
            order_entry=str(order.entry_price),
            order_stop=str(order.stop_price),
            order_risk_at_stop=str(order.risk_at_stop),
            account_equity=str(account.equity),
            account_drawdown_pct=str(account.drawdown_pct),
            account_daily_loss_pct=str(account.daily_loss_pct),
            account_open_positions=len(account.open_positions),
            kill_switch_engaged=self._kill_switch.engaged,
        )


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(tz=UTC)


def _default_logger() -> structlog.stdlib.BoundLogger:
    """Best-effort default. Real deployments configure structlog at startup."""
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
    return structlog.get_logger("core.risk_manager")
