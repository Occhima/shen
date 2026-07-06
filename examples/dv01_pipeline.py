"""DV01 pipeline for a DI1 futures book — pure user-land shen.

Everything below is written as if `pip install shen[tree]` just ran:
only public API. The pipeline is a Tree; the market is an ambient
context; the outputs are (1) per-contract and book DV01 in BRL and
(2) a key-rate DV01 ladder per pillar, built from nothing but the
scenario dimension — no risk machinery added.
"""

import datetime as dt

import pandas as pd
import polars as pl
import shen.pricers  # noqa: F401 — registers di_future
from shen import Book, Curve, dv01, unresolved
from shen.core import MarketContext, Tree, market
from shen.pricers.di import DiFuture

REF = dt.date(2026, 7, 3)
BPS = 1.0

# ── raw inputs (vendor/blotter shapes) ─────────────────────────────────
di_rates = pd.DataFrame({                    # rates in, DFs derived by the
    "curve_id":    ["DI"] * 4,               # Curve gate (bus/252, log1p)
    "pillar_date": [dt.date(2027, 1, 4), dt.date(2027, 7, 1),
                    dt.date(2028, 1, 3), dt.date(2029, 1, 2)],
    "rate":        [0.1125, 0.1140, 0.1132, 0.1120],
    "anchor_date": [REF] * 4,
})
lots = pd.DataFrame({
    "instrument_id":   ["L1", "L2", "L3", "L4"],
    "instrument_type": ["di_future"] * 4,
    "strike_pu":       [94_310.0, 94_365.0, 89_120.0, 76_400.0],
    "maturity":        [dt.date(2027, 1, 4), dt.date(2027, 1, 4),
                        dt.date(2027, 7, 1), dt.date(2029, 1, 2)],
    "di_curve":        ["DI"] * 4,
})
blotter_qty = pd.DataFrame({
    "instrument_id": ["L1", "L2", "L3", "L4"],
    "qty":           [400.0, -150.0, 300.0, -80.0],
})

# ── the pipeline ────────────────────────────────────────────────────────
g = Tree("dv01_di")


@g.node
def book() -> Book:
    b = Book.load(lots, blotter_qty).consolidate(DiFuture)   # fungibilidade
    if not unresolved(b.trades, market()).collect().is_empty():
        raise RuntimeError("book does not fully bind — run unresolved()")
    return b


@g.node
def parallel_dv01(book) -> pl.LazyFrame:
    """Per-contract dv01 (BRL/contract/bp) scaled by position -> BRL/bp."""
    per_contract = dv01(book.trades, market(), bps=BPS)
    return (per_contract
            .join(book.positions, on="instrument_id")
            .with_columns(dv01_brl=pl.col("dv01") * pl.col("qty"))
            .select("instrument_id", "qty", "dv01", "dv01_brl"))


@g.node
def krd_ladder(book) -> pl.LazyFrame:
    """Key-rate DV01: one scenario per pillar, bumping only that pillar
    by BPS in rate space (bus/252 year fraction). The scenario dimension
    does the rest — same price(), taller frame, pivot at the end."""
    mkt = market()
    pillars = mkt[Curve].select("pillar_date").unique().collect()["pillar_date"]
    scen = pl.LazyFrame({
        "curve_id":    ["DI"] * (len(pillars) + 1),
        "scenario_id": ["base", *(f"krd:{p}" for p in pillars)],
        "bump_pillar": [None, *pillars],
        "shift_bps":   [0.0] + [BPS] * len(pillars),
    })
    du252 = pl.business_day_count(pl.lit(mkt.ref_date), pl.col("pillar_date")) / 252
    shocked = mkt.with_shocks(
        Curve, scen,
        apply=pl.when(pl.col("pillar_date") == pl.col("bump_pillar"))
                .then(pl.col("log_df") - pl.col("shift_bps") * 1e-4 * du252)
                .otherwise(pl.col("log_df"))
                .alias("log_df"),
    )
    from shen import price
    wide = (price(book.trades, shocked).collect()
            .pivot("scenario_id", index="instrument_id", values="value"))
    krd_cols = [c for c in wide.columns if c.startswith("krd:")]
    return (wide.lazy()
            .join(book.positions, on="instrument_id")
            .with_columns(
                *((pl.col(c) - pl.col("base")).mul(pl.col("qty")).alias(c)
                  for c in krd_cols))
            .select("instrument_id", "qty", *krd_cols))


@g.node
def report(parallel_dv01, krd_ladder) -> dict:
    par, krd = parallel_dv01.collect(), krd_ladder.collect()
    krd_cols = [c for c in krd.columns if c.startswith("krd:")]
    return {
        "per_contract": par,
        "ladder": krd,
        "book_dv01_brl": par["dv01_brl"].sum(),
        "ladder_total_brl": sum(krd[c].sum() for c in krd_cols),
    }


# ── run ─────────────────────────────────────────────────────────────────
with MarketContext(REF, curve=di_rates):   # kwarg = registered contract name
    out = g.run("report")

print("order:", g.order("report"))
print(out["per_contract"])
print(out["ladder"])
print(f"book DV01: {out['book_dv01_brl']:,.2f} BRL/bp")
print(f"KRD sum:   {out['ladder_total_brl']:,.2f} BRL/bp")
assert abs(out["book_dv01_brl"] - out["ladder_total_brl"]) < 1e-4 * abs(out["book_dv01_brl"])
print("invariant OK: sum(key-rate) == parallel")
