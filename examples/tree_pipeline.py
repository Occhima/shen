"""The complete pipeline, as a pricing tree.

Given: a swap dataframe, raw DI1 lots + positions, and market data.
The DAG does: fungibilidade on the DI book -> price each family ->
merge value vectors -> merge positions -> book MTM. One tree, because
everything on the edges is a LazyFrame; positions are just nodes.

Then the ambient MarketContext earns its keep: the SAME tree runs
under the base market, a bumped market, and a second reference date —
zero changes to any node.
"""

import datetime as dt

import pandas as pd
import polars as pl

import plib.pricers  # noqa: F401 — fills the registry the tree recovers from
from plib import Book, Curve, Fx, Market, Position, Spot, mtm
from plib.contracts.market import as_lazy
from plib.pricers.di import DiFuture
from plib.plugins import MarketContext, Tree, market

REF = dt.date(2026, 7, 3)

# ---- raw inputs (from anywhere) --------------------------------------
swaps_df = pd.DataFrame({
    "instrument_id": ["SWP-1"], "instrument_type": ["bullet_swap"],
    "end_date": [dt.date(2027, 7, 1)],
    "active_curve": ["DI"], "active_index": ["IPCA"], "active_index_base": [100.0],
    "active_ccy_pair": ["BRLBRL"], "active_fx_base": [1.0],
    "passive_curve": ["DI"], "passive_index": ["CDI"], "passive_index_base": [100.0],
    "passive_ccy_pair": ["BRLBRL"], "passive_fx_base": [1.0],
})
swap_pos_df = pd.DataFrame({"instrument_id": ["SWP-1"], "qty": [2_000_000.0]})

di_lots_df = pd.DataFrame({           # three lots, two economic contracts
    "instrument_id": ["L1", "L2", "L3"], "instrument_type": ["di_future"] * 3,
    "strike_pu": [89_400.0, 89_650.0, 94_300.0],
    "maturity": [dt.date(2027, 7, 1), dt.date(2027, 7, 1), dt.date(2027, 1, 4)],
    "di_curve": ["DI"] * 3,
})
di_pos_df = pd.DataFrame({"instrument_id": ["L1", "L2", "L3"],
                          "qty": [300.0, 200.0, -150.0]})

# ---- the tree ---------------------------------------------------------
g = Tree("eod")

@g.node
def di_book():                        # fungibilidade inside the DAG
    return Book.load(di_lots_df, di_pos_df).consolidate(DiFuture)

@g.instruments("di_future")           # registry recovery by NAME
def di_values(di_book):
    return di_book.trades             # priced under the ambient market

@g.instruments("bullet_swap")
def swap_values():
    return swaps_df

@g.node
def values(swap_values, di_values):
    return pl.concat([swap_values, di_values], how="diagonal")

@g.node
def positions(di_book):
    return pl.concat([di_book.positions, Position.validate(swap_pos_df)])

@g.node
def book_mtm(values, positions):
    return mtm(positions, values)

print("evaluation order:", g.order("book_mtm"))
print("edges:", sorted(g.graph.edges))

# ---- one tree, many contexts ------------------------------------------
data = {
    Curve: pd.DataFrame({
        "curve_id": ["DI"] * 3,
        "pillar_date": [dt.date(2026, 7, 6), dt.date(2027, 1, 4), dt.date(2027, 7, 1)],
        "discount_factor": [1.0, 0.945, 0.895],
    }),
    Spot: pd.DataFrame({
        "index": ["IPCA", "IPCA", "CDI", "CDI"],
        "fixing_date": [dt.date(2026, 6, 15), dt.date(2026, 7, 15)] * 2,
        "value": [104.0, 105.2, 101.5, 102.4],
        "publication_lag_days": [0] * 4,
    }),
    Fx: pd.DataFrame({"pair": ["BRLBRL"], "date": [dt.date(2026, 1, 1)], "rate": [1.0]}),
}

with MarketContext(REF, data=data):
    base = g.run("book_mtm").collect()
    print(base)

    # bump: nested context, same tree, shocked market — then gone on exit
    scen = pl.LazyFrame({"curve_id": ["DI"], "shift": [0.0025]})
    with MarketContext.bump(Curve, scen,
                            apply=(pl.col("log_df") - pl.col("shift")).alias("log_df")):
        stressed = g.run("book_mtm").collect()
        print("stress dMTM:", stressed["mtm"].sum() - base["mtm"].sum())

    # multiple reference dates: inherit frames, rebind ref_date
    with MarketContext(dt.date(2026, 7, 31)):
        eom = g.run("book_mtm").collect()
        print("intramonth dMTM (new fixings bind):",
              eom["mtm"].sum() - base["mtm"].sum())

# or fully explicit, no ambient context at all:
print(g.run("book_mtm", mkt=Market.load(data, ref_date=REF)).collect()["mtm"].sum())
