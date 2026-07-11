from __future__ import annotations

import polars as pl

from shen.domain.contracts.base import as_lazy
from shen.domain.contracts.values import UNIT_VALUE_GRAIN


def component_values(components, unit_values):
    return (
        as_lazy(components)
        .join(as_lazy(unit_values), on="instrument_id", how="inner")
        .with_columns(value=pl.col("quantity") * pl.col("value"))
    )


def ticket_values(components, unit_values):
    cv = component_values(components, unit_values)
    return cv.group_by(
        "ticket_id", "scenario_id", "valuation_date", "measure", "currency", "unit"
    ).agg(pl.col("value").sum())


def aggregate_positions(tickets, components):
    return (
        as_lazy(components)
        .join(as_lazy(tickets).select("ticket_id", "book"), on="ticket_id")
        .group_by("book", "instrument_id", "quantity_unit")
        .agg(pl.col("quantity").sum().alias("quantity"))
    )


def mtm(positions, values):
    return (
        as_lazy(positions)
        .join(as_lazy(values), on="instrument_id", how="inner")
        .with_columns(mtm=pl.col("quantity") * pl.col("value"))
    )


def pnl(today, previous):
    keys = [k for k in UNIT_VALUE_GRAIN if k != "valuation_date"]
    prev = as_lazy(previous).select(
        *keys,
        pl.col("mtm").alias("_prev_mtm"),
        pl.col("valuation_date").alias("previous_valuation_date"),
    )
    return (
        as_lazy(today)
        .join(prev, on=keys, how="inner")
        .with_columns(pnl=pl.col("mtm") - pl.col("_prev_mtm"))
        .drop("_prev_mtm")
    )


def basis(model, marks):
    m = as_lazy(marks).select(*UNIT_VALUE_GRAIN, pl.col("value").alias("mark"))
    return (
        as_lazy(model)
        .join(m, on=list(UNIT_VALUE_GRAIN), how="inner")
        .with_columns(basis=pl.col("value") - pl.col("mark"))
    )
