import datetime as dt

import polars as pl
import shen


def test_pnl_can_compare_different_valuation_dates():
    today = pl.DataFrame(
        {
            "instrument_id": ["i"],
            "product_type": ["p"],
            "scenario_id": ["base"],
            "valuation_date": [dt.date(2026, 1, 2)],
            "measure": ["pv"],
            "currency": ["BRL"],
            "unit": ["u"],
            "mtm": [12.0],
        }
    )
    prev = pl.DataFrame(
        {
            "instrument_id": ["i"],
            "product_type": ["p"],
            "scenario_id": ["base"],
            "valuation_date": [dt.date(2026, 1, 1)],
            "measure": ["pv"],
            "currency": ["BRL"],
            "unit": ["u"],
            "mtm": [10.0],
        }
    )
    assert shen.pnl(today, prev).collect()["pnl"][0] == 2.0
