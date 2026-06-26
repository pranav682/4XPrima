"""Tests for walk-forward windowing and the token-gated out-of-sample holdout.

The held-out OOS slice is the structural defence against the cardinal sin of
optimisation: peeking at the data you'll judge yourself on. These tests pin the
split boundaries, prove the OOS slice cannot be reached without the literal
confirmation token, and prove every access is counted.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from decimal import Decimal

import pytest

from core.backtest.walkforward import (
    DataSplit,
    OutOfSampleAccessError,
    WalkForwardConfig,
    walk_forward_windows,
)
from core.models import Candle

# ---------------------------------------------------------------------------
# DataSplit — held-out OOS isolation
# ---------------------------------------------------------------------------


def test_split_boundaries_put_recent_tail_in_oos(
    make_bars: Callable[..., list[Candle]],
) -> None:
    bars = make_bars([str(Decimal("1.10") + Decimal("0.001") * i) for i in range(10)])
    split = DataSplit(bars, oos_fraction=0.2)

    # 20% of 10 = 2 bars held out; the rest in-sample.
    assert len(split.in_sample) == 8
    oos = split.access_out_of_sample(token="I_AM_DONE_TUNING")
    assert len(oos) == 2
    # OOS is the most recent contiguous tail.
    assert oos[0].time > split.in_sample[-1].time
    assert oos == tuple(bars[8:])


def test_oos_requires_the_literal_token(
    make_bars: Callable[..., list[Candle]],
) -> None:
    bars = make_bars([str(Decimal("1.10") + Decimal("0.001") * i) for i in range(10)])
    split = DataSplit(bars, oos_fraction=0.2)

    assert split.oos_access_count == 0
    assert split.oos_burned is False

    with pytest.raises(OutOfSampleAccessError):
        split.access_out_of_sample(token="i_am_done_tuning")  # wrong case
    with pytest.raises(OutOfSampleAccessError):
        split.access_out_of_sample(token="")

    # A failed access must NOT count as a touch.
    assert split.oos_access_count == 0
    assert split.oos_burned is False


def test_oos_access_is_counted(
    make_bars: Callable[..., list[Candle]],
) -> None:
    bars = make_bars([str(Decimal("1.10") + Decimal("0.001") * i) for i in range(10)])
    split = DataSplit(bars, oos_fraction=0.3)

    split.access_out_of_sample(token="I_AM_DONE_TUNING")
    assert split.oos_access_count == 1
    assert split.oos_burned is True

    # A second touch is the smell downstream tooling refuses to promote on.
    split.access_out_of_sample(token="I_AM_DONE_TUNING")
    assert split.oos_access_count == 2


def test_split_rejects_bad_fraction_and_tiny_inputs(
    make_bars: Callable[..., list[Candle]],
) -> None:
    bars = make_bars(["1.10", "1.11", "1.12", "1.13"])
    with pytest.raises(ValueError):
        DataSplit(bars, oos_fraction=0.0)
    with pytest.raises(ValueError):
        DataSplit(bars, oos_fraction=1.0)
    with pytest.raises(ValueError):
        DataSplit(make_bars(["1.10"]), oos_fraction=0.2)


def test_split_always_leaves_at_least_one_in_sample_bar(
    make_bars: Callable[..., list[Candle]],
) -> None:
    bars = make_bars(["1.10", "1.11"])
    split = DataSplit(bars, oos_fraction=0.99)
    assert len(split.in_sample) >= 1
    oos = split.access_out_of_sample(token="I_AM_DONE_TUNING")
    assert len(oos) >= 1
    assert len(split.in_sample) + len(oos) == 2


# ---------------------------------------------------------------------------
# walk_forward_windows
# ---------------------------------------------------------------------------


def test_walk_forward_windows_are_ordered_and_non_overlapping(
    make_bars: Callable[..., list[Candle]],
) -> None:
    # 30 days of hourly bars.
    closes = [str(Decimal("1.10") + Decimal("0.0001") * i) for i in range(24 * 30)]
    bars = make_bars(closes)
    cfg = WalkForwardConfig(
        train_window=timedelta(days=10),
        test_window=timedelta(days=5),
        step=timedelta(days=5),
    )
    windows = walk_forward_windows(bars, cfg)

    assert len(windows) >= 2
    prev_test_start = None
    for train, test in windows:
        assert train and test
        # Train strictly precedes test (no leakage of test bars into train).
        assert train[-1].time < test[0].time
        # Train window respects its configured length.
        assert test[0].time - train[0].time >= cfg.train_window - timedelta(hours=1)
        # Test windows step forward in time.
        if prev_test_start is not None:
            assert test[0].time > prev_test_start
        prev_test_start = test[0].time


def test_walk_forward_returns_empty_when_too_short(
    make_bars: Callable[..., list[Candle]],
) -> None:
    bars = make_bars(["1.10", "1.11", "1.12"])
    cfg = WalkForwardConfig(
        train_window=timedelta(days=10),
        test_window=timedelta(days=5),
        step=timedelta(days=5),
    )
    assert walk_forward_windows(bars, cfg) == []
    assert walk_forward_windows([], cfg) == []


def test_walk_forward_config_validates_durations() -> None:
    with pytest.raises(ValueError):
        WalkForwardConfig(
            train_window=timedelta(0),
            test_window=timedelta(days=5),
            step=timedelta(days=5),
        )
