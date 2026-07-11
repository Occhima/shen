from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import polars as pl

from shen.domain.contracts.market import (
    Curve,
    Fx,
    FxReference,
    IndexObservation,
    MarketContract,
    VolSurface,
)
from shen.exceptions import DuplicateQuoteError, MissingMarketObjectError


@dataclass(frozen=True, slots=True)
class Market:
    frames: Mapping[type[MarketContract], pl.LazyFrame] = field(default_factory=dict)
    valuation_date: dt.date | None = None

    @classmethod
    def load(
        cls,
        data: Mapping[type[MarketContract], Any] | None = None,
        *,
        valuation_date: dt.date | None = None,
        ref_date: dt.date | None = None,
        collisions: Literal["error", "last"] = "error",
        **named,
    ):
        entries = dict(data or {})
        name_map = {
            c.__name__.lower(): c
            for c in (Curve, IndexObservation, Fx, FxReference, VolSurface)
        }
        for name, raw in named.items():
            entries[name_map[name]] = raw
        frames = {}
        for c, raw in entries.items():
            lf = c.validate(raw)
            keys = [k for k in c.economic_keys() if k in lf.collect_schema().names()]
            dup = lf.group_by(keys).len().filter(pl.col("len") > 1).limit(1).collect()
            if dup.height and collisions == "error":
                raise DuplicateQuoteError(f"duplicate {c.__name__} quotes for {keys}")
            if dup.height:
                lf = lf.unique(subset=keys, keep="last")
            frames[c] = lf
        return cls(frames, valuation_date or ref_date)

    def __getitem__(self, c: type[MarketContract]) -> pl.LazyFrame:
        try:
            return self.frames[c]
        except KeyError as e:
            raise MissingMarketObjectError(c.__name__) from e

    def with_frame(self, c: type[MarketContract], lf: pl.LazyFrame):
        return Market({**self.frames, c: lf}, self.valuation_date)

    def with_data(self, data: Mapping[type[MarketContract], Any] | None = None, **named):
        add = Market.load(data, valuation_date=self.valuation_date, **named)
        return Market({**self.frames, **add.frames}, self.valuation_date)
