import datetime as dt

import polars as pl
import shen
from shen.products.fx.ndf import ndf


def test_dv01_returns_lazy():
    m = shen.Market.load(
        {
            shen.Curve: pl.DataFrame(
                {
                    "curve_id": ["BRL", "BRL"],
                    "pillar_date": [dt.date(2026, 7, 3), dt.date(2026, 8, 3)],
                    "log_df": [0.0, 0.0],
                }
            ),
            shen.FxReference: pl.DataFrame(
                {
                    "reference_id": ["PTAX"],
                    "observation_date": [dt.date(2026, 7, 31)],
                    "rate": [5.1],
                    "status": ["projected"],
                    "source": ["x"],
                    "observed_at": [dt.datetime(2026, 7, 3)],
                }
            ),
        },
        valuation_date=dt.date(2026, 7, 3),
    )
    t = pl.DataFrame(
        {
            "instrument_id": ["N"],
            "product_type": ["ndf"],
            "base_currency": ["USD"],
            "quote_currency": ["BRL"],
            "strike": [5.0],
            "fixing_date": [dt.date(2026, 7, 31)],
            "settlement_date": [dt.date(2026, 8, 3)],
            "reference_id": ["PTAX"],
            "settlement_curve": ["BRL"],
            "settlement_currency": ["BRL"],
        }
    )
    assert isinstance(
        shen.dv01(t, m, universe=shen.PricingUniverse.from_pricers(ndf)), pl.LazyFrame
    )
