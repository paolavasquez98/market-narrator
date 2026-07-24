from finrag.config.tickers import TICKER_UNIVERSE, all_tickers, sector_for


def test_all_tickers_are_unique_and_sorted():
    tickers = all_tickers()
    assert tickers == sorted(tickers)
    assert len(tickers) == len(set(tickers))


def test_universe_has_26_tickers_across_8_sectors():
    assert len(TICKER_UNIVERSE) == 8
    assert len(all_tickers()) == 26


def test_sector_for_known_and_unknown_ticker():
    assert sector_for("AAPL") == "Technology"
    assert sector_for("SPY") == "ETF"
    assert sector_for("NOT_A_TICKER") is None
