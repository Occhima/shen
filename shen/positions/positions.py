"""Positions — holdings, marks, and P&L over any value vector.

Everything here is a TYPED PURE FUNCTION: contracts appear as function
annotations and pa.check_types enforces them at the boundary. No
validate() calls inside bodies — conversion from raw frames happens at
the gates (Position.validate), exactly like market data.

The relevant point about markings (marcação): mtm/pnl take
LazyFrame[PriceVector] and do not care where the vector came from —
shen.price() (model) or official desk/exchange marks. Both are just
PriceVector sources, so model-vs-mark comparison is one join: basis().
"""

from __future__ import annotations

import pandera.polars as pa
import polars as pl
from pandera import check_types
from pandera.typing.polars import LazyFrame

from shen.contracts.market import as_lazy


class Position(pa.DataFrameModel):
    instrument_id: str
    qty: float

    class Config:
        strict = "filter"
        coerce = True

    @classmethod
    def validate(cls, frame, *args, **kwargs) -> pl.LazyFrame:  # type: ignore[override]
        """The gate: raw lots in, canonical NET positions out — one row
        per instrument_id (qty summed), so downstream joins can never
        double-count."""
        netted = (as_lazy(frame)
                  .group_by("instrument_id")
                  .agg(pl.col("qty").sum()))
        return super().validate(netted, *args, **kwargs)


class PriceVector(pa.DataFrameModel):
    """Any value vector: model output (price()) or official marks.
    Not strict — dimensions like scenario_id may ride along."""

    instrument_id: str
    value: float

    class Config:
        coerce = True


class Valuation(pa.DataFrameModel):
    instrument_id: str
    qty: float
    value: float
    mtm: float

    class Config:
        coerce = True


class Pnl(Valuation):
    pnl: float


@check_types
def mtm(positions: LazyFrame[Position],
        values: LazyFrame[PriceVector]) -> LazyFrame[Valuation]:
    return (positions
            .join(values.select("instrument_id", "value"),
                  on="instrument_id", how="left")
            .with_columns(mtm=pl.col("qty") * pl.col("value")))


@check_types
def pnl(today: LazyFrame[Valuation],
        previous: LazyFrame[Valuation]) -> LazyFrame[Pnl]:
    prev = previous.select("instrument_id", pl.col("mtm").alias("_prev"))
    return (today.join(prev, on="instrument_id", how="left")
                 .with_columns(pnl=pl.col("mtm") - pl.col("_prev"))
                 .drop("_prev"))


@check_types
def basis(model: LazyFrame[PriceVector],
          marks: LazyFrame[PriceVector]) -> pl.LazyFrame:
    """Model value vs official mark, per instrument."""
    m = marks.select("instrument_id", pl.col("value").alias("mark"))
    return (model.select("instrument_id", "value")
                 .join(m, on="instrument_id", how="left")
                 .with_columns(basis=pl.col("value") - pl.col("mark")))
