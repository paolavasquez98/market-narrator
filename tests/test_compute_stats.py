from datetime import date

import pandas as pd
import pytest

from finrag.ingestion.compute_stats import compute_period_stats, iter_period_stats


def _df(dates: list[str], closes: list[float], highs=None, lows=None, opens=None, volumes=None):
    n = len(dates)
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(dates),
            "Open": opens or closes,
            "High": highs or [c * 1.01 for c in closes],
            "Low": lows or [c * 0.99 for c in closes],
            "Close": closes,
            "Volume": volumes or [1_000_000] * n,
        }
    )


def test_return_pct_is_start_to_end_close():
    df = _df(["2024-01-02", "2024-01-03", "2024-01-04"], [100.0, 105.0, 110.0])
    stats = compute_period_stats(df, "TEST", "weekly")

    assert stats.start_price == 100.0
    assert stats.end_price == 110.0
    assert stats.return_pct == pytest.approx(10.0)
    assert stats.period_start == date(2024, 1, 2)
    assert stats.period_end == date(2024, 1, 4)


def test_max_drawdown_from_running_peak():
    # peaks at 120 on day 2, then falls to 90 on day 4: drawdown = (90-120)/120 = -25%
    df = _df(
        ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
        [100.0, 120.0, 105.0, 90.0],
    )
    stats = compute_period_stats(df, "TEST", "weekly")

    assert stats.max_drawdown_pct == pytest.approx(-25.0)


def test_notable_moves_flags_big_single_day_changes():
    # day 3 is a +10% jump from day 2 (100 -> 110), well above the 3% threshold
    df = _df(
        ["2024-01-02", "2024-01-03", "2024-01-04"],
        [100.0, 100.0, 110.0],
    )
    stats = compute_period_stats(df, "TEST", "weekly")

    assert len(stats.notable_moves) == 1
    assert stats.notable_moves[0].move_date == date(2024, 1, 4)
    assert stats.notable_moves[0].pct_change == pytest.approx(10.0)


def test_compute_period_stats_raises_on_empty_period():
    with pytest.raises(ValueError, match="empty period"):
        compute_period_stats(_df([], []), "TEST", "weekly")


def test_iter_period_stats_splits_by_month_and_skips_single_day_periods():
    df = _df(
        ["2024-01-02", "2024-01-03", "2024-02-01"],  # Jan has 2 days, Feb has 1 (skipped)
        [100.0, 101.0, 102.0],
    )
    results = list(iter_period_stats(df, "TEST", "monthly"))

    assert len(results) == 1
    assert results[0].period_start == date(2024, 1, 2)
    assert results[0].period_end == date(2024, 1, 3)
