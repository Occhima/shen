"""How Market context is injected — shown, not told.

Context reaches a valuation in exactly two ways, both as DATA:

  1. mkt.frames  — Lookups join against them (the environment is frames)
  2. mkt.ref_date — the engine injects it as a literal `ref_date` COLUMN
     before lookups run, so Lookup(Spot, at="ref_date") binds "latest
     fixing as of the valuation date" with no date on the trade.

The calculator sees neither Market nor dates: by the time its exprs
run, context is columns. Repricing under another context is therefore
just calling price() with another Market VALUE — two contexts can
coexist in one expression, which is how pnl-between-dates works.
"""

import datetime as dt

import pandas as pd
import polars as pl

import plib.pricers  # noqa: F401
from plib import Book, Curve, Fx, Market, Spot, price

swap = pd.DataFrame({
    "instrument_id": ["SWP-1"], "instrument_type": ["bullet_swap"],
    "end_date": [dt.date(2027, 7, 1)],
    "active_curve": ["DI"], "active_index": ["IPCA"],
    "active_index_base": [100.0],
    "active_ccy_pair": ["BRLBRL"], "active_fx_base": [1.0],
    "passive_curve": ["DI"], "passive_index": ["CDI"],
    "passive_index_base": [100.0],
    "passive_ccy_pair": ["BRLBRL"], "passive_fx_base": [1.0],
})
curves = pd.DataFrame({
    "curve_id": ["DI"] * 2,
    "pillar_date": [dt.date(2026, 7, 6), dt.date(2027, 7, 1)],
    "discount_factor": [1.0, 0.90],
})
fx = pd.DataFrame({"pair": ["BRLBRL"], "date": [dt.date(2026, 1, 1)], "rate": [1.0]})
fixings = pd.DataFrame({           # note: TWO IPCA prints, June and July
    "index":       ["IPCA", "IPCA", "CDI", "CDI"],
    "fixing_date": [dt.date(2026, 6, 15), dt.date(2026, 7, 15),
                    dt.date(2026, 6, 15), dt.date(2026, 7, 15)],
    "value":       [104.0, 105.2, 101.5, 102.4],
    "publication_lag_days": [0] * 4,
})

# same frames, two CONTEXTS: only ref_date differs
jun = Market.load({Curve: curves, Spot: fixings, Fx: fx},
                  ref_date=dt.date(2026, 6, 30))
jul = Market.load({Curve: curves, Spot: fixings, Fx: fx},
                  ref_date=dt.date(2026, 7, 31))

# trace makes the injection visible: ref_date arrives as a column and
# the SAME lookup binds a different fixing under each context
for label, mkt in [("JUN", jun), ("JUL", jul)]:
    t = price(swap, mkt, trace=True).collect()
    print(label, t.select("sign", "ref_date", "index", "index_fwd").rows())

# contexts are values -> both live in one expression:
book = Book.load(swap, pd.DataFrame({"instrument_id": ["SWP-1"], "qty": [1e6]}))
print(book.pnl(jul, jun).collect())   # month P&L = mtm(jul) - mtm(jun)
