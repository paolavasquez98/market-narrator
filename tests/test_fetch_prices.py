import pandas as pd
import pytest

from finrag.ingestion.fetch_prices import _clean_single_ticker_frame


def _raw_frame() -> pd.DataFrame:
    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0],
            "High": [105.0, 106.0, 107.0],
            "Low": [99.0, 100.0, 101.0],
            "Close": [104.0, 105.0, 106.0],
            "Volume": [1_000_000, 1_100_000, 1_200_000],
        },
        index=index,
    )


def test_clean_single_ticker_frame_shape_and_columns():
    cleaned = _clean_single_ticker_frame(_raw_frame(), "AAPL")

    assert list(cleaned.columns) == ["Ticker", "Date", "Open", "High", "Low", "Close", "Volume"]
    assert (cleaned["Ticker"] == "AAPL").all()
    assert len(cleaned) == 3


def test_clean_single_ticker_frame_drops_all_nan_rows():
    raw = _raw_frame()
    raw.loc[pd.Timestamp("2024-01-05")] = [None, None, None, None, None]

    cleaned = _clean_single_ticker_frame(raw, "AAPL")

    assert len(cleaned) == 3


def test_clean_single_ticker_frame_raises_on_empty_data():
    empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    with pytest.raises(ValueError, match="no usable price data"):
        _clean_single_ticker_frame(empty, "BADTICKER")
