"""ExecutionEngine — the single path from a strategy signal to a broker fill.

**Invariant.** No code anywhere in the fast loop should call
:meth:`core.broker.Broker.place_order` or
:meth:`core.broker.Broker.close_position` directly. Every order goes through
:meth:`ExecutionEngine.submit`, which:

1. Refuses if the kill switch is already engaged.
2. Pulls a fresh :class:`AccountState` from the broker.
3. Calls :meth:`RiskManager.evaluate` to gate the order.
4. Places the (possibly RESIZED) order if accepted.
5. Trips the kill switch on any broker-side exception.
6. Emits one structured log record per submission.

The engine is intentionally tiny — its job is not to add logic, it's to make
the contract impossible to break.
"""

from __future__ import annotations

import logging
import uuid

import structlog
from pydantic import BaseModel, ConfigDict

from core.broker import Broker
from core.models import (
    AccountState,
    DecisionKind,
    Fill,
    OrderRequest,
    RejectionReason,
    RiskDecision,
)
from core.risk_manager import RiskManager


class ExecutionResult(BaseModel):
    """Outcome of one :meth:`ExecutionEngine.submit` call.

    ``fill`` is set iff the risk decision was APPROVE or RESIZE *and* the
    broker accepted the order. A RESIZE decision with a broker-side
    exception will surface as ``decision.accepted = True, fill = None``,
    plus a tripped kill switch on the engine's risk manager.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    submission_id: str
    decision: RiskDecision
    fill: Fill | None
    placed_order: OrderRequest | None
    """The order actually sent to the broker — equals ``decision.sized_order``
    when one was placed, ``None`` otherwise. Tests assert against this when
    verifying that a RESIZE decision sends the *resized* order downstream."""

    @property
    def placed(self) -> bool:
        return self.fill is not None


class ExecutionEngine:
    """The only path that places orders. See module docstring."""

    def __init__(
        self,
        *,
        broker: Broker,
        risk_manager: RiskManager,
        logger: structlog.stdlib.BoundLogger | None = None,
    ) -> None:
        self._broker = broker
        self._risk_manager = risk_manager
        self._logger = (
            logger if logger is not None else _default_logger()
        ).bind(
            component="execution",
            config_hash=risk_manager.config_hash,
        )

    @property
    def risk_manager(self) -> RiskManager:
        """Exposed so callers (and tests) can inspect kill-switch state without
        the engine becoming a passthrough for risk-manager mutations."""
        return self._risk_manager

    def submit(self, order: OrderRequest) -> ExecutionResult:
        submission_id = uuid.uuid4().hex

        # 1. Pre-flight: kill switch refusal happens *before* any broker I/O.
        #    This is redundant with the risk manager's first check, but the
        #    point of the invariant is precisely to never make a broker call
        #    when the switch is engaged.
        if self._risk_manager.kill_switch_engaged:
            decision = self._synthetic_kill_switch_decision(order)
            self._log_result(submission_id, order, decision, fill=None, placed=None)
            return ExecutionResult(
                submission_id=submission_id,
                decision=decision,
                fill=None,
                placed_order=None,
            )

        # 2. Pull account state; broker failure trips the switch.
        try:
            account = self._broker.get_account_state()
        except Exception as e:
            self._risk_manager.trip(
                f"broker.get_account_state failed: {e!r}",
                tripped_by="exception",
            )
            decision = self._synthetic_kill_switch_decision(order)
            self._log_result(
                submission_id, order, decision, fill=None, placed=None, error=str(e)
            )
            return ExecutionResult(
                submission_id=submission_id,
                decision=decision,
                fill=None,
                placed_order=None,
            )

        # 3. Risk gate.
        decision = self._risk_manager.evaluate(order, account)
        if not decision.accepted or decision.sized_order is None:
            self._log_result(submission_id, order, decision, fill=None, placed=None)
            return ExecutionResult(
                submission_id=submission_id,
                decision=decision,
                fill=None,
                placed_order=None,
            )

        # 4. Place the (possibly resized) order. The order that hits the
        #    broker is decision.sized_order — NOT the original `order` —
        #    which is what makes RESIZE meaningful end-to-end.
        order_to_place = decision.sized_order
        try:
            fill = self._broker.place_order(order_to_place)
        except Exception as e:
            self._risk_manager.trip(
                f"broker.place_order failed: {e!r}",
                tripped_by="exception",
            )
            self._log_result(
                submission_id,
                order,
                decision,
                fill=None,
                placed=order_to_place,
                error=str(e),
            )
            return ExecutionResult(
                submission_id=submission_id,
                decision=decision,
                fill=None,
                placed_order=order_to_place,
            )

        self._log_result(
            submission_id, order, decision, fill=fill, placed=order_to_place
        )
        return ExecutionResult(
            submission_id=submission_id,
            decision=decision,
            fill=fill,
            placed_order=order_to_place,
        )

    # ----------------------------------------------------------- internals

    def _synthetic_kill_switch_decision(self, order: OrderRequest) -> RiskDecision:
        """Build a RiskDecision for the pre-flight refusal path.

        We don't have an AccountState here (we refused before fetching one),
        so we cannot use ``RiskManager.evaluate`` directly — synthesise a
        decision shaped exactly like one ``evaluate`` would have returned
        had it run the kill-switch branch.
        """
        ks = self._risk_manager.kill_switch_state
        from datetime import UTC, datetime  # local import keeps module deps small

        return RiskDecision(
            decision_id=uuid.uuid4().hex,
            kind=DecisionKind.REJECT,
            sized_order=None,
            rejected_by=(RejectionReason.KILL_SWITCH,),
            reason=(
                f"execution refused: kill switch engaged "
                f"(tripped_by={ks.tripped_by!r}, reason={ks.reason!r})"
            ),
            limiting_rule=RejectionReason.KILL_SWITCH,
            config_hash=self._risk_manager.config_hash,
            as_of=datetime.now(UTC),
        )

    def _log_result(
        self,
        submission_id: str,
        original_order: OrderRequest,
        decision: RiskDecision,
        *,
        fill: Fill | None,
        placed: OrderRequest | None,
        error: str | None = None,
    ) -> None:
        self._logger.info(
            "execution_submit",
            submission_id=submission_id,
            decision_id=decision.decision_id,
            decision_kind=decision.kind.value,
            limiting_rule=(
                decision.limiting_rule.value if decision.limiting_rule else None
            ),
            reason=decision.reason,
            order_pair=original_order.pair,
            order_direction=original_order.direction.value,
            order_size_in=str(original_order.size),
            order_size_placed=(str(placed.size) if placed is not None else None),
            fill_price=(str(fill.fill_price) if fill is not None else None),
            commission=(str(fill.commission) if fill is not None else None),
            error=error,
            kill_switch_engaged=self._risk_manager.kill_switch_engaged,
        )


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
    return structlog.get_logger("core.execution")


# Re-export AccountState so type-checking callers can `from core.execution
# import ExecutionResult, ExecutionEngine` without dragging in core.models for
# the type hint on ``submit``.
__all__ = ["AccountState", "ExecutionEngine", "ExecutionResult"]
