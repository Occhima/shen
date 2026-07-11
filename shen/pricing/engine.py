from __future__ import annotations

import polars as pl

from shen.domain.contracts.base import as_lazy
from shen.domain.market import Market
from shen.pricing.pricer import Pricer, Registry


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


def bind(rows: pl.LazyFrame, p: Pricer, market: Market) -> pl.LazyFrame:
    if market.valuation_date is not None:
        rows = rows.with_columns(valuation_date=pl.lit(market.valuation_date))
    elif "valuation_date" not in rows.collect_schema().names():
        raise ValueError("Market.valuation_date is required")
    if "scenario_id" not in rows.collect_schema().names():
        rows = rows.join(_scenarios(market), how="cross")
    if p.derive:
        rows = rows.with_columns(**p.derive)
    for f in p.fetches:
        rows = f(rows, market)
    return rows


def price(
    terms, market: Market, *, universe: Registry, strict: bool = True, trace: bool = False
) -> pl.LazyFrame:
    lf = as_lazy(terms)
    parts = []
    for p in universe.pricers.values():
        rows = _ensure_terms(lf, p)
        bound = bind(rows, p, market)
        out = bound.with_columns(value=p.base_expr())
        cur = (
            p.output.currency
            if isinstance(p.output.currency, pl.Expr)
            else pl.lit(p.output.currency)
        )
        unit = (
            p.output.unit if isinstance(p.output.unit, pl.Expr) else pl.lit(p.output.unit)
        )
        out = out.with_columns(measure=pl.lit(p.output.measure), currency=cur, unit=unit)
        if strict and "_lookup_error" in out.collect_schema().names():
            out = out.with_columns(
                value=pl.when(pl.col("_lookup_error"))
                .then(None)
                .otherwise(pl.col("value"))
            )
        parts.append(
            out
            if trace
            else out.select(
                "instrument_id",
                "product_type",
                "scenario_id",
                "valuation_date",
                "measure",
                "currency",
                "unit",
                "value",
            )
        )
    return pl.concat(parts, how="diagonal") if parts else pl.LazyFrame({})


def unresolved(terms, market: Market, *, universe: Registry) -> pl.LazyFrame:
    return price(terms, market, universe=universe, strict=False).filter(
        pl.col("_lookup_error") | pl.col("value").is_null() | pl.col("value").is_nan()
    )
