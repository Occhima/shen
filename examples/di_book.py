"""Value a position across multiple DI futures maturities."""

import datetime as dt

import pandas as pd

import plib.pricers  # noqa: F401
from plib import Curve, Market, Position, dv01, mtm, price, unresolved

REF = dt.date(2026, 7, 3)

# DI curve from the lake (bootstrapped bus/252 upstream; we take DFs)
di_curve = pd.DataFrame({
    "curve_id":        ["DI"] * 4,
    "pillar_date":     [dt.date(2027, 1, 4), dt.date(2027, 7, 1),
                        dt.date(2028, 1, 3), dt.date(2029, 1, 2)],
    "discount_factor": [0.9450, 0.8950, 0.8470, 0.7620],
})
mkt = Market.load({Curve: di_curve}, ref_date=REF)

# the book: three maturities, entry PUs from the blotter
trades = pd.DataFrame({
    "instrument_id":   ["DI1-F27", "DI1-N27", "DI1-F29"],
    "instrument_type": ["di_future"] * 3,
    "strike_pu":       [94_380.0, 89_610.0, 76_050.0],
    "maturity":        [dt.date(2027, 1, 4), dt.date(2027, 7, 1),
                        dt.date(2029, 1, 2)],
    "di_curve":        ["DI"] * 3,
})

# positions: signed contracts (long/short PU)
positions = Position.validate(pd.DataFrame({
    "instrument_id": ["DI1-F27", "DI1-N27", "DI1-F29"],
    "qty":           [500.0, -200.0, 150.0],
}))

assert unresolved(trades, mkt).collect().is_empty()   # book fully binds

values = price(trades, mkt)
print(mtm(positions, values).collect())               # per-contract + BRL mtm
print(dv01(trades, mkt).collect())                    # curve risk per maturity
