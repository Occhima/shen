from __future__ import annotations

from typing import Any

import pandera.pandas  # noqa: F401
import pandera.polars as pa
import polars as pl


def as_lazy(frame: Any) -> pl.LazyFrame:
    if isinstance(frame, pl.LazyFrame):
        return frame
    if isinstance(frame, pl.DataFrame):
        return frame.lazy()
    return pl.from_pandas(frame).lazy()


class ShenFrame(pa.DataFrameModel):
    class Config:
        strict = "filter"
        coerce = True

    @classmethod
    def validate(cls, frame, *args, **kwargs) -> pl.LazyFrame:  # type: ignore[override]
        return super().validate(as_lazy(frame), *args, **kwargs)
