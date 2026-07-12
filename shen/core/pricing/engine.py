from __future__ import annotations

from typing import Any
from uuid import uuid4

import pandas as pd
import polars as pl

from shen.core.pricing.pricer import Pricer
from shen.core.pricing.registry import Registry
from shen.core.pricing.tree import PricingTree
from shen.domain.contracts.base import as_lazy
from shen.domain.contracts.market import Curve, MarketContract
from shen.domain.contracts.values import UNIT_VALUE_GRAIN
from shen.domain.market import Market
from shen.inject import InjectedMethod, validate_injections


def _contract_discriminator(pricer: Pricer) -> str:
    return pricer.dispatch_key


def _terms_for(universe: pl.LazyFrame, pricer: Pricer) -> pl.LazyFrame:
    names = universe.collect_schema().names()
    discriminator = "instrument_type" if "instrument_type" in names else "product_type"
    rows = universe
    if discriminator not in names:
        rows = rows.with_columns(**{discriminator: pl.lit(pricer.dispatch_key)})
    subset = rows.filter(pl.col(discriminator) == pricer.dispatch_key)
    return pricer.terms.resolve(subset)


def _scenario_ids(market: Market) -> pl.LazyFrame:
    parts = [
        frame.select("scenario_id")
        for frame in market.frames.values()
        if "scenario_id" in frame.collect_schema().names()
    ]
    return pl.concat(parts).unique() if parts else pl.LazyFrame({"scenario_id": ["base"]})


def _coordinates(rows: pl.LazyFrame, market: Market) -> pl.LazyFrame:
    names = rows.collect_schema().names()
    if market.valuation_date is not None:
        rows = rows.with_columns(valuation_date=pl.lit(market.valuation_date))
    elif "valuation_date" not in names:
        raise ValueError("Market.valuation_date is required")
    if "scenario_id" not in rows.collect_schema().names():
        rows = rows.join(_scenario_ids(market), how="cross")
    return rows


def _apply_one_injection(
    rows: pl.LazyFrame,
    terms: type,
    method: InjectedMethod,
    dependency: pl.LazyFrame,
) -> pl.LazyFrame:
    metadata = method.metadata
    row_id = f"__inject_row_{uuid4().hex}"
    source_name = f"__inject_source_{uuid4().hex}"
    left = rows.with_row_index(row_id).collect().to_pandas()
    right_names = dependency.collect_schema().names()
    dimensions = [
        name
        for name in ("scenario_id", "valuation_date")
        if name in left.columns and name in right_names
    ]
    required = list(
        dict.fromkeys([*metadata.on.values(), *dimensions, metadata.take])
    )
    missing = set(required) - set(right_names)
    if missing:
        raise KeyError(
            f"{terms.__name__}.{method.name}: dependency output misses {sorted(missing)}"
        )
    right_key_names = [f"__inject_key_{uuid4().hex}" for _ in metadata.on]
    right_dimension_names = [f"__inject_dim_{uuid4().hex}" for _ in dimensions]
    right_renames = {
        **dict(zip(metadata.on.values(), right_key_names, strict=True)),
        **dict(zip(dimensions, right_dimension_names, strict=True)),
        metadata.take: source_name,
    }
    right = (
        dependency.select(*required)
        .rename(right_renames)
        .collect()
        .to_pandas()
    )
    merged = left.merge(
        right,
        how="left",
        left_on=[*metadata.on, *dimensions],
        right_on=[*right_key_names, *right_dimension_names],
        sort=False,
        validate="many_to_one",
    ).sort_values(row_id, kind="stable")
    merged.index = left.index
    source = merged[source_name].copy()
    frame = merged[left.columns].copy()
    if metadata.output not in frame:
        frame[metadata.output] = pd.Series(pd.NA, index=frame.index)
    calculated = method.calculate(terms, frame, source)
    if metadata.overwrite:
        frame[metadata.output] = calculated
    else:
        frame[metadata.output] = frame[metadata.output].combine_first(calculated)
    previous_error = (
        frame["_injection_error"].fillna(False).astype(bool)
        if "_injection_error" in frame
        else pd.Series(False, index=frame.index)
    )
    frame["_injection_error"] = previous_error | frame[metadata.output].isna()
    frame = frame.sort_values(row_id, kind="stable").drop(columns=[row_id])
    eager = pl.from_pandas(frame, include_index=False)
    return eager.lazy()


def _apply_injections(
    rows: pl.LazyFrame,
    pricer: Pricer,
    dependency_values: dict[str, pl.LazyFrame],
) -> pl.LazyFrame:
    for method in validate_injections(pricer.terms):
        rows = _apply_one_injection(
            rows,
            pricer.terms,
            method,
            dependency_values[method.metadata.source_key],
        )
    return rows


def _literal_or_expression(value: str | pl.Expr) -> pl.Expr:
    return value if isinstance(value, pl.Expr) else pl.lit(value)


def price(
    terms: Any,
    market: Market,
    *,
    using: Registry,
    strategy: str = "default",
    tree: PricingTree | None = None,
    strict: bool = True,
    trace: bool = False,
) -> pl.LazyFrame:
    universe = as_lazy(terms)
    run = (tree or PricingTree()).run(
        {"market": market, "valuation_date": market.valuation_date}
    )
    cache: dict[str, pl.LazyFrame] = {}

    def evaluate(key: str) -> pl.LazyFrame:
        if key in cache:
            return cache[key]
        pricer = using[key]
        rows = _coordinates(_terms_for(universe, pricer), market)
        dependency_values = {
            source: evaluate(source) for source in using.dependencies(key)
        }
        if dependency_values:
            rows = _apply_injections(rows, pricer, dependency_values)
        prepared = pricer.prepare(rows, market, strategy=strategy, tree=run)
        if "product_type" not in prepared.collect_schema().names():
            prepared = prepared.with_columns(product_type=pl.lit(pricer.dispatch_key))
        output = prepared.with_columns(value=pricer.expression()).with_columns(
            measure=pl.lit(pricer.output.measure),
            currency=_literal_or_expression(pricer.output.currency),
            unit=_literal_or_expression(pricer.output.unit),
        )
        error_columns = [
            name
            for name in ("_lookup_error", "_injection_error")
            if name in output.collect_schema().names()
        ]
        if strict and error_columns:
            failed = pl.any_horizontal(*(pl.col(name).fill_null(False) for name in error_columns))
            output = output.with_columns(
                value=pl.when(failed).then(None).otherwise(pl.col("value"))
            )
        if not trace:
            output = output.select(*UNIT_VALUE_GRAIN, "value")
        cache[key] = output
        return output

    parts = [evaluate(key) for key in using.order()]
    return pl.concat(parts, how="diagonal_relaxed") if parts else pl.LazyFrame({})


def unresolved(terms: Any, market: Market, *, using: Registry) -> pl.LazyFrame:
    values = price(terms, market, using=using, strict=False, trace=True)
    names = values.collect_schema().names()
    errors = [
        pl.col(name).fill_null(False)
        for name in ("_lookup_error", "_injection_error")
        if name in names
    ]
    predicate = pl.col("value").is_null() | pl.col("value").is_nan()
    if errors:
        predicate = predicate | pl.any_horizontal(*errors)
    return values.filter(predicate)


def compare_values(
    up: pl.LazyFrame,
    down: pl.LazyFrame,
    *,
    step: float,
    name: str,
) -> pl.LazyFrame:
    keys = [key for key in UNIT_VALUE_GRAIN if key != "scenario_id"]
    lhs = up.select(*keys, pl.col("value").alias("__up"))
    rhs = down.select(*keys, pl.col("value").alias("__down"))
    return lhs.join(rhs, on=keys, how="inner").with_columns(
        ((pl.col("__up") - pl.col("__down")) / (2 * step)).alias(name)
    ).drop("__up", "__down")


def market_scenario(market: Market, scenario_id: str) -> Market:
    return Market(
        {
            contract: frame.with_columns(scenario_id=pl.lit(scenario_id))
            for contract, frame in market.frames.items()
        },
        market.valuation_date,
    )


def parallel_curve_shift(
    market: Market,
    *,
    curve: type[MarketContract] = Curve,
    shift: float,
    scenario_id: str,
    basis: float = 1e-4,
) -> Market:
    if market.valuation_date is None:
        raise ValueError("Market.valuation_date is required for a curve shift")
    shocked = market_scenario(market, scenario_id)
    axis = getattr(curve, "_at", None) or getattr(curve, "_axis", None)
    if axis is None:
        raise TypeError(f"{curve.__name__} must define _at or _axis")
    year_fraction = (
        pl.col(axis) - pl.lit(market.valuation_date)
    ).dt.total_days() / 365.0
    frame = shocked[curve].with_columns(
        log_df=pl.col("log_df") - shift * basis * year_fraction
    )
    return shocked.with_frame(curve, frame)


__all__ = [
    "compare_values",
    "market_scenario",
    "parallel_curve_shift",
    "price",
    "unresolved",
]
