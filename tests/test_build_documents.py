from datetime import date

import pytest

from finrag.ingestion.build_documents import (
    build_documents_for_ticker,
    render_narrative,
)
from finrag.ingestion.compute_stats import NotableMove, PeriodStats


def test_render_narrative_contains_every_number_exactly():
    stats = PeriodStats(
        ticker="AAPL",
        granularity="weekly",
        period_start=date(2022, 1, 3),
        period_end=date(2022, 1, 7),
        start_price=180.0,
        end_price=170.0,
        return_pct=-5.56,
        high=182.0,
        low=168.0,
        max_drawdown_pct=-7.5,
        avg_volume=95_000_000,
        volatility_annualized_pct=42.3,
        notable_moves=[NotableMove(move_date=date(2022, 1, 5), pct_change=-4.2)],
    )

    text = render_narrative(stats, sector="Technology")

    assert "AAPL" in text
    assert "Technology" in text
    assert "2022-01-03" in text and "2022-01-07" in text
    assert "$180.00" in text and "$170.00" in text
    assert "-5.56%" in text
    assert "-7.50%" in text
    assert "42.30%" in text
    assert "2022-01-05 (-4.2%)" in text


def test_render_narrative_handles_no_notable_moves():
    stats = PeriodStats(
        ticker="SPY",
        granularity="monthly",
        period_start=date(2023, 6, 1),
        period_end=date(2023, 6, 30),
        start_price=400.0,
        end_price=410.0,
        return_pct=2.5,
        high=412.0,
        low=398.0,
        max_drawdown_pct=-2.0,
        avg_volume=70_000_000,
        volatility_annualized_pct=12.0,
        notable_moves=[],
    )

    text = render_narrative(stats, sector="ETF")

    assert "No single-day moves of 3% or more." in text


def test_build_documents_for_ticker_rejects_unknown_ticker():
    with pytest.raises(ValueError, match="not in the configured ticker universe"):
        build_documents_for_ticker("NOT_A_TICKER", settings=None)
