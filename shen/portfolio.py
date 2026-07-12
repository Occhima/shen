from __future__ import annotations

from collections.abc import Iterable

import polars as pl

from shen.domain.contracts.base import as_lazy
from shen.domain.contracts.values import UNIT_VALUE_GRAIN


def mtm_value(quantity, value):
    return quantity * value


def pnl_value(today_mtm, previous_mtm):
    return today_mtm - previous_mtm


def basis_value(model_value, mark_value):
    return model_value - mark_value


def _common_identity(left: pl.LazyFrame, right: pl.LazyFrame) -> list[str]:
    preferred = [
        "instrument_id",
        "product_type",
        "scenario_id",
        "valuation_date",
        "currency",
        "unit",
        "measure",
    ]
    left_names = set(left.collect_schema().names())
    right_names = set(right.collect_schema().names())
    return [name for name in preferred if name in left_names and name in right_names]


def component_values(components, unit_values) -> pl.LazyFrame:
    component = as_lazy(components)
    values = as_lazy(unit_values)
    keys = _common_identity(component, values)
    if "instrument_id" not in keys:
        keys.insert(0, "instrument_id")
    return component.join(values, on=keys, how="left").with_columns(
        value=mtm_value(pl.col("quantity"), pl.col("value"))
    )


def ticket_values(components, unit_values) -> pl.LazyFrame:
    values = component_values(components, unit_values)
    grain = [
        name
        for name in (
            "ticket_id",
            "scenario_id",
            "valuation_date",
            "measure",
            "currency",
            "unit",
        )
        if name in values.collect_schema().names()
    ]
    return values.group_by(grain).agg(pl.col("value").sum())


def aggregate_positions(tickets, components) -> pl.LazyFrame:
    return (
        as_lazy(components)
        .join(as_lazy(tickets).select("ticket_id", "book"), on="ticket_id")
        .group_by("book", "instrument_id", "quantity_unit")
        .agg(pl.col("quantity").sum().alias("quantity"))
    )


def mark_positions(positions, values, *, value_name: str = "value") -> pl.LazyFrame:
    position = as_lazy(positions)
    priced = as_lazy(values)
    keys = _common_identity(position, priced)
    if "instrument_id" not in keys:
        keys.insert(0, "instrument_id")
    return position.join(priced, on=keys, how="left").with_columns(
        mtm=mtm_value(pl.col("quantity"), pl.col(value_name))
    )


def mark_book(positions, model_values, marks=None) -> pl.LazyFrame:
    model = mark_positions(positions, model_values).rename(
        {"value": "model_value", "mtm": "model_mtm"}
    )
    if marks is None:
        return model
    mark = as_lazy(marks)
    keys = _common_identity(model, mark)
    marked = model.join(
        mark.select(*keys, pl.col("value").alias("mark_value")),
        on=keys,
        how="left",
    )
    return marked.with_columns(
        mark_mtm=mtm_value(pl.col("quantity"), pl.col("mark_value")),
        basis=basis_value(pl.col("model_value"), pl.col("mark_value")),
        basis_mtm=mtm_value(
            pl.col("quantity"),
            basis_value(pl.col("model_value"), pl.col("mark_value")),
        ),
    )


def book_mtm(positions, values, *, by: Iterable[str] = ("book",)) -> pl.LazyFrame:
    marked = mark_positions(positions, values)
    grain = [name for name in by if name in marked.collect_schema().names()]
    for name in ("scenario_id", "valuation_date", "currency", "unit", "measure"):
        if name in marked.collect_schema().names() and name not in grain:
            grain.append(name)
    return marked.group_by(grain).agg(pl.col("mtm").sum())


def pnl(today, previous) -> pl.LazyFrame:
    current = as_lazy(today)
    prior = as_lazy(previous)
    keys = [
        key
        for key in UNIT_VALUE_GRAIN
        if key != "valuation_date"
        and key in current.collect_schema().names()
        and key in prior.collect_schema().names()
    ]
    prior = prior.select(
        *keys,
        pl.col("mtm").alias("previous_mtm"),
        pl.col("valuation_date").alias("previous_valuation_date"),
    )
    return current.join(prior, on=keys, how="inner").with_columns(
        pnl=pnl_value(pl.col("mtm"), pl.col("previous_mtm"))
    )


def basis(model, marks) -> pl.LazyFrame:
    model_frame = as_lazy(model)
    mark_frame = as_lazy(marks)
    keys = _common_identity(model_frame, mark_frame)
    return model_frame.join(
        mark_frame.select(*keys, pl.col("value").alias("mark_value")),
        on=keys,
        how="left",
    ).with_columns(basis=basis_value(pl.col("value"), pl.col("mark_value")))


mtm = mark_positions

__all__ = [
    "aggregate_positions",
    "basis",
    "basis_value",
    "book_mtm",
    "component_values",
    "mark_book",
    "mark_positions",
    "mtm",
    "mtm_value",
    "pnl",
    "pnl_value",
    "ticket_values",
]
