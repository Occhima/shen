"""plib quickstart — data from various places, one lazy pipeline out."""

import datetime as dt

import pandas as pd
import polars as pl

import plib.pricers  # noqa: F401 — registers commodity_forward + bullet_swap
from plib import Curve, Fx, Market, Position, Spot, dv01, mtm, price, sensitivities

REF = dt.date(2026, 7, 3)

# -- 1. raw frames, heterogeneous sources (pandas API, parquet lake...) --
trades = pd.DataFrame({
    "instrument_id":   ["FWD-001", "FWD-002", "SWP-001"],
    "instrument_type": ["commodity_forward", "commodity_forward", "bullet_swap"],
    "strike":          [95.0, 101.5, None],
    "disc_curve":      ["USD-OIS", "USD-OIS", None],
    "index":           ["BRENT", "WTI", None],
    "ccy_pair":        ["USDBRL", "USDBRL", None],
    "fixing_date":     [dt.date(2026, 12, 10)] * 2 + [None],
    "pay_date":        [dt.date(2026, 12, 15)] * 2 + [None],
    "end_date":        [None, None, dt.date(2027, 7, 5)],
    "active_curve":    [None, None, "USD-OIS"],
    "active_index":    [None, None, "IPCA"],
    "active_index_base":  [None, None, 100.0],
    "active_ccy_pair": [None, None, "USDBRL"],   # dollar-linked leg
    "active_fx_base":  [None, None, 5.20],
    "passive_curve":   [None, None, "USD-OIS"],
    "passive_index":   [None, None, "CDI"],
    "passive_index_base": [None, None, 100.0],
    "passive_ccy_pair": [None, None, "BRLBRL"],  # domestic: identity pair
    "passive_fx_base": [None, None, 1.0],
    "trader":          ["ana", "beto", "ana"],   # junk: strict="filter" drops it
})

curves = pl.LazyFrame({   # in prod: pl.scan_parquet("s3://md/curves/...")
    "curve_id":        ["USD-OIS"] * 3,
    "pillar_date":     [dt.date(2026, 7, 6), dt.date(2027, 1, 4), dt.date(2027, 7, 5)],
    "discount_factor": [1.0, 0.981, 0.962],
})
spots = pd.DataFrame({
    "index":                ["BRENT", "WTI", "IPCA", "CDI"],
    "fixing_date":          [dt.date(2026, 12, 9), dt.date(2026, 12, 9),
                             dt.date(2026, 7, 2), dt.date(2026, 7, 2)],
    "value":                [104.20, 99.85, 106.3, 103.1],
    "publication_lag_days": [1, 1, 0, 0],
})
fx = pd.DataFrame({"pair": ["USDBRL", "BRLBRL"],
                   "date": [REF, REF],
                   "rate": [5.4310, 1.0]})

# -- 2. the gate ---------------------------------------------------------
mkt = Market.load({Curve: curves, Spot: spots, Fx: fx}, ref_date=REF)

# -- 3. value vector (lazy), then positions own the qty ------------------
values = price(trades, mkt)
positions = Position.validate(   # the gate; mtm() is typed-pure past it
    pd.DataFrame({"instrument_id": ["FWD-001", "FWD-002", "SWP-001"],
                  "qty": [1_000.0, -500.0, 1_000_000.0]}))  # swap notional = qty
book = mtm(positions, values)
print(book.collect())

# -- 4. risk -------------------------------------------------------------
print(sensitivities(trades, mkt).collect())   # per-factor dv, per instrument
print(dv01(trades, mkt).collect())            # curve bump via scenario dim
