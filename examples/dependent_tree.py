"""A pricing tree where instruments DEPEND on each other.

The canonical Brazilian case: DI1 futures quotes IMPLY the DI curve,
and swaps price off that implied curve. Same market data (fixings, fx),
two instrument families, a real dependency between them.

The mechanism needs no new tree machinery: dependency between
instruments = one instrument's OUTPUT becoming another's MARKET OBJECT.
In graph terms: di_quotes -> implied_curve -> (Market.with_data) ->
swap_values. The edge carries a frame; with_data validates it through
the Curve gate like any vendor frame.

Genuine simultaneity (A needs B needs A) is deliberately impossible as
graph edges — networkx rejects cycles — because a fixed point is a
SOLVER, not a dependency: it lives INSIDE one node (a bootstrap that
iterates to convergence), keeping the DAG acyclic and honest.
"""

import datetime as dt

import pandas as pd
import polars as pl

import plib.pricers  # noqa: F401
from plib import Fx, MarketContext, Spot, unresolved
from plib.pricers.di import di_future
from plib.pricers.swap import bullet_swap
from plib.plugins import Tree, market

REF = dt.date(2026, 7, 3)
g = Tree("dependent")


@g.node
def di_quotes() -> pd.DataFrame:
    """Instrument A's market quotes: DI1 settlement rates per maturity."""
    return pd.DataFrame({
        "maturity": [dt.date(2027, 1, 4), dt.date(2027, 7, 1), dt.date(2028, 1, 3)],
        "rate":     [0.1125, 0.1140, 0.1132],
    })


@g.node
def implied_curve(di_quotes) -> pl.LazyFrame:
    """A DI1 quote IS a curve point: the futures strip defines the curve.
    Emit rate+anchor rows; the Curve gate derives DFs (bus/252)."""
    return (pl.from_pandas(di_quotes).lazy()
            .select(curve_id=pl.lit("DI-IMPLIED"),
                    pillar_date=pl.col("maturity"),
                    rate=pl.col("rate"),
                    anchor_date=pl.lit(market().ref_date)))


@g.node
def derived_market(implied_curve):
    """Ambient market (fixings, fx) + the instrument-implied curve."""
    return market().with_data(curve=implied_curve)


@g.node
def di_values(derived_market) -> pl.LazyFrame:
    """Instrument A priced off the curve it implied: entry at the quote
    PU means every value must be ~0 — the consistency check."""
    trades = pd.DataFrame({
        "instrument_id": ["DI1-F27"], "instrument_type": ["di_future"],
        "strike_pu": [None], "maturity": [dt.date(2027, 1, 4)],
        "di_curve": ["DI-IMPLIED"],
    })
    du = pl.business_day_count(pl.lit(market().ref_date), pl.lit(dt.date(2027, 1, 4)))
    quote_pu = (pl.LazyFrame({"r": [0.1125]})
                .select(pu=100_000 * (1 + pl.col("r")) ** -(du / 252))
                .collect()["pu"][0])
    trades["strike_pu"] = [quote_pu]
    return di_future.price(trades, derived_market)


@g.node
def swap_values(derived_market) -> pl.LazyFrame:
    """Instrument B, priced off instrument A's implied curve + the SAME
    ambient fixings/fx frames."""
    swaps = pd.DataFrame({
        "instrument_id": ["SWP-1"], "instrument_type": ["bullet_swap"],
        "end_date": [dt.date(2027, 7, 1)],
        "active_curve": ["DI-IMPLIED"], "active_index": ["IPCA"],
        "active_index_base": [100.0],
        "active_ccy_pair": ["BRLBRL"], "active_fx_base": [1.0],
        "passive_curve": ["DI-IMPLIED"], "passive_index": ["CDI"],
        "passive_index_base": [100.0],
        "passive_ccy_pair": ["BRLBRL"], "passive_fx_base": [1.0],
    })
    assert unresolved(swaps, derived_market).collect().is_empty()
    return bullet_swap.price(swaps, derived_market)


# NOTE: the ambient context has NO curve — the curve is instrument-implied.
with MarketContext(REF,
                   spot=pd.DataFrame({
                       "index": ["IPCA", "CDI"],
                       "fixing_date": [dt.date(2026, 7, 2)] * 2,
                       "value": [106.30, 103.10],
                       "publication_lag_days": [0, 0]}),
                   fx=pd.DataFrame({"pair": ["BRLBRL"], "date": [REF], "rate": [1.0]})):
    out = g.run("di_values", "swap_values")

print("order:", g.order("swap_values"))
di, swp = out["di_values"].collect(), out["swap_values"].collect()
print(di)
print(swp)
assert abs(di["value"][0]) < 1e-6      # future at its own quote: value ~ 0
print("consistency OK: DI1 priced off its own implied curve is at par")
