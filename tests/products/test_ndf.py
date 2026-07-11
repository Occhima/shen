import datetime as dt

import polars as pl
import shen
from shen.products.fx.ndf import NdfTicket, ndf

REF = dt.date(2026, 7, 3)


def market(rate=5.1485):
    curves = pl.DataFrame(
        {
            "curve_id": ["BRL", "BRL"],
            "pillar_date": [REF, dt.date(2026, 8, 3)],
            "log_df": [0.0, 0.0],
        }
    )
    fx = pl.DataFrame(
        {
            "reference_id": ["PTAX"],
            "observation_date": [dt.date(2026, 7, 31)],
            "rate": [rate],
            "status": ["projected"],
            "source": ["internal"],
            "observed_at": [dt.datetime(2026, 7, 3, 12)],
        }
    )
    return shen.Market.load(
        {shen.Curve: curves, shen.FxReference: fx}, valuation_date=REF
    )


def terms():
    return pl.DataFrame(
        {
            "instrument_id": ["NDF1"],
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


def test_ndf_price_and_mtm_long_short():
    v = ndf.price(terms(), market()).collect()
    assert abs(v["value"][0] - 0.1485) < 1e-12
    pos = shen.Position.validate(
        pl.DataFrame(
            {
                "book": ["FX"],
                "instrument_id": ["NDF1"],
                "quantity": [750000.0],
                "quantity_unit": ["USD"],
            }
        )
    )
    book = shen.mtm(pos, v).collect()
    assert abs(book["mtm"][0] - 111375.0) < 1e-6


def test_ndf_missing_reference_is_unresolved():
    out = ndf.price(
        terms().with_columns(reference_id=pl.lit("BAD")),
        market(),
        strict=False,
        trace=True,
    ).collect()
    assert out["_lookup_error"][0]


def test_ndf_ticket_parse():
    raw = pl.DataFrame(
        {
            "trade_id": ["T1"],
            "contract_id": ["NDF1"],
            "trade_date": [REF],
            "book": ["FX"],
            "counterparty": [None],
            "trader": [None],
            "external_id": [None],
            "base_currency": ["USD"],
            "quote_currency": ["BRL"],
            "strike": [5.0],
            "fixing_date": [dt.date(2026, 7, 31)],
            "settlement_date": [dt.date(2026, 8, 3)],
            "reference_id": ["PTAX"],
            "settlement_curve": ["BRL"],
            "settlement_currency": ["BRL"],
            "qty": [1.0],
        }
    )
    parsed = NdfTicket.parse(raw)
    assert (
        parsed.tickets.collect().height == 1 and parsed.components.collect().height == 1
    )
