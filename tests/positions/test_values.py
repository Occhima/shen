import datetime as dt

import polars as pl
import shen


def test_pnl_joins_full_value_grain():
    v = pl.DataFrame(
        {
            "contract_id": ["c", "c"],
            "product_type": ["p", "p"],
            "scenario_id": ["base", "up"],
            "valuation_date": [dt.date(2026, 1, 1)] * 2,
            "measure": ["pv", "pv"],
            "currency": ["BRL", "BRL"],
            "unit": ["u", "u"],
            "value": [10.0, 11.0],
            "qty": [2.0, 2.0],
            "mtm": [20.0, 22.0],
        }
    ).lazy()
    p = shen.pnl(v, v.with_columns(mtm=pl.col("mtm") - 1)).collect()
    assert p.height == 2 and p["pnl"].to_list() == [1.0, 1.0]
