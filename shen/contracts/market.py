from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

import pandera.polars as pa
import polars as pl

from shen.contracts.base import ShenFrame, as_lazy
from shen.exceptions import DuplicateQuoteError, MissingMarketObjectError


class MarketObject(ShenFrame):
    key: ClassVar[str]
    at: ClassVar[str]
    value_cols: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def coordinate(cls, at: pl.Expr, *, ref_date: pl.Expr | None = None) -> pl.Expr:
        return at.dt.epoch("d")

    @classmethod
    def economic_keys(cls) -> tuple[str, ...]:
        return (cls.key, cls.at, "scenario_id")


class Curve(MarketObject):
    key: ClassVar[str] = "curve_id"
    at: ClassVar[str] = "pillar_date"
    value_cols: ClassVar[tuple[str, ...]] = ("log_df",)
    curve_id: str
    pillar_date: pl.Date
    log_df: float
    source: str | None = None
    as_of: pl.Datetime | None = None
    priority: int | None = None
    quality: str | None = None
    scenario_id: str = "base"

    @classmethod
    def validate(cls, frame, *args, **kwargs) -> pl.LazyFrame:  # type: ignore[override]
        lf = as_lazy(frame)
        names = set(lf.collect_schema().names())
        if "log_df" not in names and "discount_factor" in names:
            lf = lf.with_columns(log_df=pl.col("discount_factor").log())
        if "scenario_id" not in names:
            lf = lf.with_columns(scenario_id=pl.lit("base"))
        return super().validate(lf, *args, **kwargs)


class RawDICurve(ShenFrame):
    curve_id: str
    pillar_date: pl.Date
    rate: float
    anchor_date: pl.Date


def canonicalize_di(raw, convention: RateConvention | None = None) -> pl.LazyFrame:
    lf = RawDICurve.validate(raw)
    denom = 252.0 if convention is None or convention.day_count == "BUS/252" else 365.0
    days = (
        pl.business_day_count(pl.col("anchor_date"), pl.col("pillar_date"))
        if denom == 252.0
        else (pl.col("pillar_date") - pl.col("anchor_date")).dt.total_days()
    )
    return lf.with_columns(log_df=-(days / denom) * pl.col("rate").log1p()).select(
        "curve_id", "pillar_date", "log_df"
    )


class Spot(MarketObject):
    key: ClassVar[str] = "index"
    at: ClassVar[str] = "fixing_date"
    value_cols: ClassVar[tuple[str, ...]] = ("value",)
    index: str
    fixing_date: pl.Date
    value: float
    scenario_id: str = "base"


class Fx(MarketObject):
    key: ClassVar[str] = "pair"
    at: ClassVar[str] = "date"
    value_cols: ClassVar[tuple[str, ...]] = ("rate",)
    pair: str
    date: pl.Date
    rate: float = pa.Field(gt=0)
    scenario_id: str = "base"


class FxReference(MarketObject):
    key: ClassVar[str] = "reference_id"
    at: ClassVar[str] = "fixing_date"
    value_cols: ClassVar[tuple[str, ...]] = ("rate",)
    reference_id: str
    fixing_date: pl.Date
    rate: float = pa.Field(gt=0)
    status: str
    source: str
    observed_at: pl.Datetime
    scenario_id: str = "base"


class VolSurface(MarketObject):
    key: ClassVar[str] = "surface_id"
    at: ClassVar[str] = "expiry"
    value_cols: ClassVar[tuple[str, ...]] = ("atm", "skew", "curv")
    surface_id: str
    expiry: pl.Date
    atm: float = pa.Field(gt=0)
    skew: float
    curv: float
    scenario_id: str = "base"


@dataclass(frozen=True, slots=True)
class Calendar:
    name: str
    holidays: tuple[dt.date, ...] = ()


DayCount = Literal["BUS/252", "ACT/365"]
Compounding = Literal["continuous", "annual"]


@dataclass(frozen=True, slots=True)
class RateConvention:
    day_count: DayCount = "BUS/252"
    compounding: Compounding = "annual"
    calendar: Calendar = Calendar("weekends")


@dataclass(frozen=True, slots=True)
class Scenario:
    scenario_id: str
    scenario_group: str = "default"
    description: str | None = None


@dataclass(frozen=True, slots=True)
class Market:
    frames: Mapping[type[MarketObject], pl.LazyFrame] = field(default_factory=dict)
    valuation_date: dt.date | None = None

    @classmethod
    def load(
        cls,
        data: Mapping[type[MarketObject], Any] | None = None,
        *,
        valuation_date: dt.date | None = None,
        ref_date: dt.date | None = None,
        collisions: Literal["error", "last"] = "error",
        **named,
    ) -> Market:
        entries = dict(data or {})
        name_map = {
            c.__name__.lower(): c for c in (Curve, Spot, Fx, FxReference, VolSurface)
        }
        for name, raw in named.items():
            entries[name_map[name]] = raw
        frames = {}
        for c, raw in entries.items():
            lf = c.validate(raw)
            keys = [k for k in c.economic_keys() if k in lf.collect_schema().names()]
            if collisions == "error":
                dup = lf.group_by(keys).len().filter(pl.col("len") > 1).limit(1).collect()
                if dup.height:
                    raise DuplicateQuoteError(f"duplicate {c.__name__} quotes for {keys}")
            else:
                lf = lf.unique(subset=keys, keep="last")
            frames[c] = lf
        return cls(frames, valuation_date or ref_date)

    def __getitem__(self, c: type[MarketObject]) -> pl.LazyFrame:
        try:
            return self.frames[c]
        except KeyError as e:
            raise MissingMarketObjectError(c.__name__) from e

    def with_data(
        self, data: Mapping[type[MarketObject], Any] | None = None, **named
    ) -> Market:
        add = Market.load(data, valuation_date=self.valuation_date, **named)
        return Market({**self.frames, **add.frames}, self.valuation_date)

    def with_frame(self, c: type[MarketObject], lf: pl.LazyFrame) -> Market:
        return Market({**self.frames, c: lf}, self.valuation_date)
