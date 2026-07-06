"""Risk — two mechanisms, zero new pricing machinery.

sensitivities() — per-instrument, per-risk-factor dValue/dFactor by
finite difference IN EXPRESSION SPACE: because the calculator is a pure
function of Exprs, we re-instantiate it with one injected column
shifted; base and all shifted valuations live in ONE lazy plan (Polars
CSEs the shared subexpressions). This is the answer to "qual a
contribuição de cada fator de risco": dv is the local sensitivity, and
dv * factor_level is the Euler-style price attribution for the
homogeneous factors (spot, fx, index_fwd).

dv01() — market-level parallel bump ridden in as a scenario dimension:
build a shocked Market, call the SAME price().
"""

from __future__ import annotations

import polars as pl

from plib.contracts.market import Curve, MarketObject
from plib.engine import ID_COLS, Market, _resolve, _resolved, price
from plib.registry import MEASURE


def sensitivities(instruments, mkt: Market | None = None, h: float = 1e-4,
                  *, pricers=None) -> pl.LazyFrame:
    """Long frame: (instrument_id, instrument_type, [scenario_id],
    value, factor, dv) — one row per bound risk factor."""
    mkt = _resolve(mkt)
    def one(p, bound: pl.LazyFrame, dims: list[str]) -> pl.LazyFrame:
        base = p.base_expr()
        factors = [a for lk in p.lookups for a in lk.aliases if a in p.params]
        shifted = {
            f: p.calc(*(pl.col(x) + h if x == f else pl.col(x)
                        for x in p.params))
            for f in factors
        }
        out = bound.with_columns(
            base.alias(MEASURE),
            *(((s - base) / h).alias(f) for f, s in shifted.items()),
        )
        if p.reduce == "sum":
            w = (pl.col("_w") if "_w" in out.collect_schema()
                 else pl.lit(1.0))
            out = out.group_by(*ID_COLS, *dims).agg(
                *((pl.col(c) * w).sum().alias(c) for c in (MEASURE, *factors)))
        return (out.select(*ID_COLS, *dims, MEASURE, *factors)
                   .unpivot(index=[*ID_COLS, *dims, MEASURE],
                            variable_name="factor", value_name="dv"))

    return pl.concat(one(*r) for r in _resolved(instruments, mkt, pricers))


def dv01(instruments, mkt: Market | None = None,
         curve: type[MarketObject] = Curve, bps: float = 1.0,
         *, pricers=None) -> pl.LazyFrame:
    """Parallel curve bump via the scenario dimension."""
    mkt = _resolve(mkt)
    if mkt.ref_date is None:
        raise ValueError("dv01 needs Market.ref_date for year fractions")
    scen = (mkt[curve].select(pl.col(curve._key)).unique()
            .join(pl.LazyFrame({"scenario_id": ["base", "up"],
                                "shift_bps": [0.0, bps]}), how="cross"))
    yf = pl.business_day_count(pl.lit(mkt.ref_date), pl.col(curve._at)) / 252
    shocked = mkt.with_shocks(
        curve, scen,
        apply=(pl.col("log_df") - pl.col("shift_bps") * 1e-4 * yf).alias("log_df"),
    )
    wide = (price(instruments, shocked, pricers=pricers).collect()
            .pivot("scenario_id", index=list(ID_COLS), values=MEASURE))
    return wide.with_columns(dv01=pl.col("up") - pl.col("base")).lazy()
