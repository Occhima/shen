"""Full EOD valuation: a book of swaps + DI1 futures.

The whole ritual: load market through the gates -> consolidate fungible
lots -> audit that everything binds -> value -> MTM -> risk. One lazy
plan per step; collect only at the prints.
"""

import datetime as dt

import pandas as pd
import polars as pl

import plib.pricers  # noqa: F401
from plib import (Book, Curve, Fx, Market, Spot, dv01, sensitivities,
                  unpriced, unresolved)
from plib.pricers.di import DiFuture

REF = dt.date(2026, 7, 3)

# ── 1. market data through the contract gates ─────────────────────────
mkt = Market.load({
    Curve: pd.DataFrame({                       # DI curve, DFs from the lake
        "curve_id":        ["DI"] * 4,
        "pillar_date":     [dt.date(2026, 7, 6), dt.date(2027, 1, 4),
                            dt.date(2027, 7, 1), dt.date(2028, 1, 3)],
        "discount_factor": [1.0, 0.9450, 0.8950, 0.8470],
    }),
    Spot: pd.DataFrame({                        # indexer fixings, vendor feed
        "index":                ["IPCA", "CDI"],
        "fixing_date":          [dt.date(2026, 7, 2)] * 2,
        "value":                [106.30, 103.10],
        "publication_lag_days": [0, 0],
    }),
    Fx: pd.DataFrame({
        "pair": ["USDBRL", "BRLBRL"], "date": [REF] * 2, "rate": [5.4310, 1.0],
    })},
    ref_date=REF,
)

# ── 2. the book: swaps + DI lots, one union frame ─────────────────────
trades = pd.DataFrame({
    "instrument_id":   ["SWP-1", "SWP-2", "L1", "L2", "L3"],
    "instrument_type": ["bullet_swap"] * 2 + ["di_future"] * 3,
    # swap economics
    "end_date":            [dt.date(2027, 7, 1), dt.date(2028, 1, 3), None, None, None],
    "active_curve":        ["DI", "DI", None, None, None],
    "active_index":        ["IPCA", "CDI", None, None, None],
    "active_index_base":   [100.0, 100.0, None, None, None],
    "active_ccy_pair":     ["BRLBRL", "USDBRL", None, None, None],
    "active_fx_base":      [1.0, 5.20, None, None, None],
    "passive_curve":       ["DI", "DI", None, None, None],
    "passive_index":       ["CDI", "IPCA", None, None, None],
    "passive_index_base":  [100.0, 100.0, None, None, None],
    "passive_ccy_pair":    ["BRLBRL", "BRLBRL", None, None, None],
    "passive_fx_base":     [1.0, 1.0, None, None, None],
    # DI economics (L1/L2 same contract -> fungível; L3 another maturity)
    "strike_pu":  [None, None, 89_400.0, 89_650.0, 94_300.0],
    "maturity":   [None, None, dt.date(2027, 7, 1), dt.date(2027, 7, 1),
                   dt.date(2027, 1, 4)],
    "di_curve":   [None, None, "DI", "DI", "DI"],
})
positions = pd.DataFrame({
    "instrument_id": ["SWP-1", "SWP-2", "L1", "L2", "L3"],
    "qty":           [2_000_000.0, -1_500_000.0, 300.0, 200.0, -150.0],
})

book = Book.load(trades, positions).consolidate(DiFuture)   # fungibilidade

# ── 3. audits BEFORE trusting numbers ──────────────────────────────────
assert unpriced(book.trades).collect().is_empty()            # every type covered
assert unresolved(book.trades, mkt).collect().is_empty()     # every lookup binds

# ── 4. valuation ────────────────────────────────────────────────────────
print(book.mtm(mkt).collect().sort("instrument_id"))
print("book MTM:", round(book.mtm(mkt).collect()["mtm"].sum(), 2))

# ── 5. risk ─────────────────────────────────────────────────────────────
print(dv01(book.trades, mkt).collect().sort("instrument_id"))
print(sensitivities(book.trades, mkt).collect()
      .sort("instrument_id", "factor")
      .filter(pl.col("dv").abs() > 1e-9))
