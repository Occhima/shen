from __future__ import annotations

import polars as pl

from shen.contracts.values import VALUE_KEYS


def mtm(positions, values):
    return positions.join(values, on="contract_id", how="inner").with_columns(
        mtm=pl.col("qty") * pl.col("value")
    )


def pnl(today, previous):
    prev = previous.select(*VALUE_KEYS, pl.col("mtm").alias("_prev"))
    return (
        today.join(prev, on=list(VALUE_KEYS), how="inner")
        .with_columns(pnl=pl.col("mtm") - pl.col("_prev"))
        .drop("_prev")
    )


def basis(model, marks):
    m = marks.select(*VALUE_KEYS, pl.col("value").alias("mark"))
    return model.join(m, on=list(VALUE_KEYS), how="inner").with_columns(
        basis=pl.col("value") - pl.col("mark")
    )
