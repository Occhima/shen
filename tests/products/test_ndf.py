import datetime as dt

import polars as pl
import shen
from shen.products.fx.ndf import NdfTicket, ndf

REF = dt.date(2026, 7, 3)


def market(rate=5.1485, curve_scen=False):
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
            "fixing_date": [dt.date(2026, 7, 31)],
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
            "contract_id": ["NDF1"],
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
    ).lazy()


def test_ndf_price_and_mtm_long_short():
    v = ndf.price(terms(), market()).collect()
    assert v["scenario_id"][0] == "base"
    assert v["currency"][0] == "BRL"
    assert v["unit"][0] == "BRL/USD"
    assert abs(v["value"][0] - 0.1485) < 1e-12
    pos = shen.Position.validate(
        pl.DataFrame({"contract_id": ["NDF1", "NDF1"], "qty": [1_000_000.0, -250_000.0]})
    )
    book = shen.mtm(pos, v.lazy()).collect()
    assert abs(book["mtm"][0] - 111375.0) < 1e-6


def test_ndf_missing_reference_is_unresolved():
    bad = terms().with_columns(reference_id=pl.lit("BAD"))
    out = ndf.price(bad, market(), strict=False).collect()
    assert out["value"][0] is None


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
    assert parsed.trade.collect().height == 1 and parsed.positions.collect().height == 1
