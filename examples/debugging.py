"""Debugging a suspicious price with trace=True.

Scenario: the desk says WDO-N27 looks off and SWP-001 doubled overnight.
Instead of print-statements inside a pricer (there is no code path to
put them in — calculators are pure), you ask for the unprojected frame
and READ the valuation like data.
"""

import datetime as dt

import pandas as pd
import polars as pl

import plib.pricers  # noqa: F401
from plib import Curve, Fx, Market, Spot, price

REF = dt.date(2026, 7, 3)

trades = pd.DataFrame({
    "instrument_id":   ["WDO-N27", "SWP-001"],
    "instrument_type": ["wdo_future", "bullet_swap"],
    # wdo fields
    "strike":       [5.35, None],
    "maturity":     [dt.date(2027, 7, 1), None],
    "di_curve":     ["DI", None],
    "cupom_curve":  ["CUPOM", None],
    "ccy_pair":     ["USDBRL", None],
    # swap fields
    "end_date":            [None, dt.date(2027, 7, 1)],
    "active_curve":        [None, "DI"],
    "active_index":        [None, "IPCA"],
    "active_index_base":   [None, 100.0],
    "active_ccy_pair":     [None, "BRLBRL"],
    "active_fx_base":      [None, 1.0],
    "passive_curve":       [None, "DI"],
    "passive_index":       [None, "CDI"],
    "passive_index_base":  [None, 100.0],
    "passive_ccy_pair":    [None, "BRLBRL"],
    "passive_fx_base":     [None, 1.0],
})

mkt = Market.load({
    Curve: pd.DataFrame({
        "curve_id":        ["DI", "DI", "CUPOM", "CUPOM"],
        "pillar_date":     [dt.date(2026, 7, 6), dt.date(2027, 7, 1)] * 2,
        "discount_factor": [1.0, 0.90, 1.0, 0.98],
    }),
    Spot: pd.DataFrame({
        "index":                ["IPCA", "CDI"],
        "fixing_date":          [dt.date(2026, 7, 2)] * 2,
        "value":                [105.0, 102.0],
        "publication_lag_days": [0, 0],
    }),
    Fx: pd.DataFrame({
        "pair": ["USDBRL", "BRLBRL"], "date": [REF] * 2, "rate": [5.4310, 1.0],
    })},
    ref_date=REF,
)

# 1. the headline numbers the desk is questioning
print(price(trades, mkt).collect())

# 2. open the hood: same computation, no projection, leg grain
t = price(trades, mkt, trace=True).collect()

# 3. interrogate it like any dataframe — e.g. every input that touched WDO:
wdo = t.filter(pl.col("instrument_id") == "WDO-N27")
print(wdo.select("spot", "strike", "log_df_di", "log_df_cupom", "value"))
#    -> log_df_di = ln(0.90): the DI pillar was fat-fingered (0.90 vs ~0.96),
#       inflating fwd = spot * exp(log_df_cupom - log_df_di). Found in one select.

# 4. and the swap, leg by leg — is one leg's factor stale?
#    (_w is the structure's combinator weight: active +1, passive -1)
swp = t.filter(pl.col("instrument_id") == "SWP-001")
print(swp.select("_w", "index", "index_fwd", "index_base",
                 "fx_fwd", "log_df_end", "value"))

# 5. reconciliation invariant: weighted trace legs == the reduced price
official = price(trades, mkt).collect().filter(
    pl.col("instrument_id") == "SWP-001")["value"][0]
assert abs((swp["value"] * swp["_w"]).sum() - official) < 1e-12
print("legs reconcile to price():", official)
