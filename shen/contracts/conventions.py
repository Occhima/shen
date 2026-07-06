"""Convention contracts: ingestion shapes that route to a canonical
contract via _to_canonical. A convention exists only as an entry
point — its rows are never stored under its own type, always merged
into the canonical contract's frame."""

from __future__ import annotations

from typing import ClassVar

import polars as pl

from shen.contracts.market import Curve, MarketObject


class DICurve(MarketObject):
    """Brazilian DI (bus/252, annual compounding) convention.

    Input: curve_id, pillar_date, rate, anchor_date.
    _to_canonical derives log_df = -(du/252) * ln(1+rate) where du is
    the business-day count between anchor_date and pillar_date using
    Curve._holidays. Routed to Curve's frame by Market.load.
    """

    _key: ClassVar[str] = "curve_id"
    _at: ClassVar[str] = "pillar_date"
    _canonical: ClassVar[type[MarketObject] | None] = Curve

    @classmethod
    def _to_canonical(cls, lf: pl.LazyFrame) -> pl.LazyFrame:
        du = pl.business_day_count(
            pl.col("anchor_date"), pl.col("pillar_date"),
            holidays=list(Curve._holidays))
        return (lf.with_columns(log_df=-(du / 252) * pl.col("rate").log1p())
                .select("curve_id", "pillar_date", "log_df"))
