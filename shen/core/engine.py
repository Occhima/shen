from __future__ import annotations

import polars as pl

from shen.contracts.base import as_lazy
from shen.contracts.market import Market
from shen.core.registry import Pricer, PricingUniverse


def _ensure_terms(lf: pl.LazyFrame, p: Pricer) -> pl.LazyFrame:
    names = set(lf.collect_schema().names())
    if "product_type" not in names:
        lf = lf.with_columns(product_type=pl.lit(p.product_type))
    return p.schema.validate(lf.filter(pl.col("product_type") == p.product_type))


def _scenarios(market: Market) -> pl.LazyFrame:
    parts = [
        f.select("scenario_id")
        for f in market.frames.values()
        if "scenario_id" in f.collect_schema().names()
    ]
    return pl.concat(parts).unique() if parts else pl.LazyFrame({"scenario_id": ["base"]})


def _bind(rows: pl.LazyFrame, p: Pricer, market: Market) -> pl.LazyFrame:
    if market.valuation_date is not None:
        rows = rows.with_columns(valuation_date=pl.lit(market.valuation_date))
    elif "valuation_date" not in rows.collect_schema().names():
        raise ValueError("Market.valuation_date is required")
    if "scenario_id" not in rows.collect_schema().names():
        rows = rows.join(_scenarios(market), how="cross")
    if p.derive:
        rows = rows.with_columns(**p.derive)
    for lk in p.lookups:
        rows = lk(rows, market)
    return rows


def price(
    terms,
    market: Market,
    *,
    universe: PricingUniverse,
    strict: bool = True,
    trace: bool = False,
) -> pl.LazyFrame:
    lf = as_lazy(terms)
    parts = []
    for p in universe.pricers.values():
        rows = _ensure_terms(lf, p)
        if p.explode:
            rows = p.explode(rows)
            rows = p.legs.validate(rows.drop("_w")) if p.legs else rows
            if "_w" not in rows.collect_schema().names():
                rows = p.explode(_ensure_terms(lf, p))
        bound = _bind(rows, p, market)
        out = bound.with_columns(value=p.base_expr())
        if p.explode or p.weights:
            w = pl.col("_w") if "_w" in out.collect_schema().names() else pl.lit(1.0)
            out = out.group_by(
                "contract_id", "product_type", "scenario_id", "valuation_date"
            ).agg((pl.col("value") * w).sum().alias("value"))
        if isinstance(p.output.currency, pl.Expr):
            cur = p.output.currency
        else:
            cur = pl.lit(p.output.currency)
        unit = (
            p.output.unit if isinstance(p.output.unit, pl.Expr) else pl.lit(p.output.unit)
        )
        out = out.with_columns(measure=pl.lit(p.output.measure), currency=cur, unit=unit)
        if trace:
            parts.append(out)
        else:
            parts.append(
                out.select(
                    "contract_id",
                    "product_type",
                    "scenario_id",
                    "valuation_date",
                    "measure",
                    "currency",
                    "unit",
                    "value",
                )
            )
    res = pl.concat(parts, how="diagonal") if parts else pl.LazyFrame({})
    if strict and not trace and "_lookup_error" in res.collect_schema().names():
        res = res.with_columns(
            value=pl.when(pl.col("_lookup_error"))
            .then(pl.lit(float("nan")))
            .otherwise(pl.col("value"))
        )
    return res


def unresolved(terms, market: Market, *, universe: PricingUniverse) -> pl.LazyFrame:
    return price(terms, market, universe=universe, strict=False).filter(
        pl.col("value").is_null() | pl.col("value").is_nan()
    )
