from __future__ import annotations

import polars as pl

from shen.contracts.base import ShenFrame, as_lazy


class Position(ShenFrame):
    contract_id: str
    qty: float

    @classmethod
    def validate(cls, frame, *args, **kwargs) -> pl.LazyFrame:  # type: ignore[override]
        return super().validate(
            as_lazy(frame).group_by("contract_id").agg(pl.col("qty").sum()),
            *args,
            **kwargs,
        )
