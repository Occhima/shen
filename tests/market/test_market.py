import datetime as dt

import polars as pl
import pytest
import shen


def test_duplicate_market_quotes_fail():
    df = pl.DataFrame(
        {
            "curve_id": ["c", "c"],
            "pillar_date": [dt.date(2026, 1, 1)] * 2,
            "log_df": [0.0, 0.0],
        }
    )
    with pytest.raises(shen.DuplicateQuoteError):
        shen.Market.load({shen.Curve: df}, valuation_date=dt.date(2026, 1, 1))


def test_raw_di_canonicalization():
    raw = pl.DataFrame(
        {
            "curve_id": ["DI"],
            "pillar_date": [dt.date(2027, 1, 1)],
            "rate": [0.1],
            "anchor_date": [dt.date(2026, 1, 1)],
        }
    )
    out = shen.canonicalize_di(raw).collect()
    assert out["log_df"][0] < 0
