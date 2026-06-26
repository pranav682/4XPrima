"""Walk-forward windowing + held-out OOS isolation.

Mirrors the contract in ``specs/agents/backtest_agent.md`` and
``skills/backtesting-methodology``: the held-out OOS slice is the most
recent contiguous tail of the data, and it MUST NOT be touched during
fitting or in-sample evaluation.

This module enforces that structurally. The :class:`DataSplit` exposes
``in_sample`` directly but gates ``out_of_sample`` behind a confirmation
token (cf. :meth:`RiskManager.reset`'s token). Touches are *counted*, not
just rejected — downstream tooling (the eventual ``backtest_agent``) can
refuse to promote a strategy whose OOS access count exceeds 1.

The token is the literal string ``"I_AM_DONE_TUNING"``. Reaching for it
in code without good cause is a smell; the test suite for this module
exercises the structural isolation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Final

from core.models import Candle

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class OutOfSampleAccessError(Exception):
    """Raised on an attempt to access the held-out OOS slice without the
    confirmation token, or when the slice is otherwise mishandled."""


# ---------------------------------------------------------------------------
# Held-out OOS split
# ---------------------------------------------------------------------------


_OOS_TOKEN: Final[str] = "I_AM_DONE_TUNING"  # noqa: S105 — confirmation token, not a secret


class DataSplit:
    """In-sample + OOS holdout, structurally isolated.

    The split is taken **once** at construction. The OOS slice is the most
    recent ``oos_fraction`` of bars; in-sample is everything before. There
    is no walk-forward inside the OOS slice — it's a single one-shot
    evaluation per strategy version.

    Use:

        split = DataSplit(bars, oos_fraction=0.2)
        # ... fit / walk-forward on split.in_sample ...
        oos = split.access_out_of_sample(token="I_AM_DONE_TUNING")
        # ... single OOS evaluation ...

    If a caller asks for OOS twice, ``oos_access_count`` ticks. Downstream
    tooling should refuse to promote on ``oos_access_count > 1``.
    """

    __slots__ = ("_in_sample", "_oos", "_oos_accesses")

    def __init__(self, bars: Sequence[Candle], *, oos_fraction: float = 0.2) -> None:
        if not 0.0 < oos_fraction < 1.0:
            raise ValueError("oos_fraction must be in (0, 1)")
        if len(bars) < 2:
            raise ValueError("DataSplit needs at least 2 bars")
        sorted_bars = tuple(sorted(bars, key=lambda b: b.time))
        n = len(sorted_bars)
        oos_size = max(1, round(n * oos_fraction))
        oos_size = min(oos_size, n - 1)  # always leave ≥ 1 in-sample bar
        oos_start = n - oos_size
        self._in_sample: tuple[Candle, ...] = sorted_bars[:oos_start]
        self._oos: tuple[Candle, ...] = sorted_bars[oos_start:]
        self._oos_accesses = 0

    @property
    def in_sample(self) -> tuple[Candle, ...]:
        return self._in_sample

    @property
    def oos_access_count(self) -> int:
        return self._oos_accesses

    @property
    def oos_burned(self) -> bool:
        """``True`` once the OOS slice has been touched at all. Distinct
        from ``oos_access_count > 1`` (which signals multiple touches)."""
        return self._oos_accesses > 0

    def access_out_of_sample(self, *, token: str) -> tuple[Candle, ...]:
        """Return the OOS slice. The token MUST be ``"I_AM_DONE_TUNING"``.

        Tracks accesses; touching twice is a smell the orchestrator will
        notice. Anything except the exact token raises
        :class:`OutOfSampleAccessError` — typo-tolerance is the wrong
        direction here.
        """
        if token != _OOS_TOKEN:
            raise OutOfSampleAccessError(
                "OOS access requires the literal confirmation token "
                f"{_OOS_TOKEN!r}; got a different string."
            )
        self._oos_accesses += 1
        return self._oos


# ---------------------------------------------------------------------------
# Walk-forward windowing (used inside the in-sample slice only)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WalkForwardConfig:
    """Rolling train/test windowing over IN-SAMPLE data.

    All durations are calendar-time; the engine slices bars by ``time``.
    A default of ``train=6 months / test=1 month / step=1 month`` mirrors
    the backtesting-methodology skill.
    """

    train_window: timedelta
    test_window: timedelta
    step: timedelta

    def __post_init__(self) -> None:
        if self.train_window.total_seconds() <= 0:
            raise ValueError("train_window must be positive")
        if self.test_window.total_seconds() <= 0:
            raise ValueError("test_window must be positive")
        if self.step.total_seconds() <= 0:
            raise ValueError("step must be positive")


def walk_forward_windows(
    bars: Sequence[Candle],
    cfg: WalkForwardConfig,
) -> list[tuple[tuple[Candle, ...], tuple[Candle, ...]]]:
    """Generate (train, test) windows over ``bars`` per ``cfg``.

    Train slices are the lookback fitting window; test slices are the
    forward-stepping evaluation window. Both are tuples (immutable).
    Returns an empty list if ``bars`` is too short to contain even one
    train+test pair.

    NOTE: this function does NOT touch :class:`DataSplit.out_of_sample`.
    Pass ``split.in_sample`` if you want walk-forward inside the
    in-sample portion.
    """
    if not bars:
        return []
    ordered = sorted(bars, key=lambda b: b.time)
    span_start = ordered[0].time
    span_end = ordered[-1].time

    windows: list[tuple[tuple[Candle, ...], tuple[Candle, ...]]] = []
    cursor_start = span_start
    while True:
        train_end = cursor_start + cfg.train_window
        test_end = train_end + cfg.test_window
        if test_end > span_end + cfg.step:
            # No more full windows fit; loop is done. Allow the final
            # test_end to land slightly past span_end to capture the
            # tail when bars align awkwardly with calendar offsets.
            break
        train = tuple(b for b in ordered if cursor_start <= b.time < train_end)
        test = tuple(b for b in ordered if train_end <= b.time < test_end)
        if train and test:
            windows.append((train, test))
        cursor_start = cursor_start + cfg.step
        if cursor_start >= span_end:
            break
    return windows


__all__ = [
    "DataSplit",
    "OutOfSampleAccessError",
    "WalkForwardConfig",
    "walk_forward_windows",
]
