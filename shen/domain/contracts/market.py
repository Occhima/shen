from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import ClassVar, Literal

import pandera.polars as pa
import polars as pl

from shen.domain.contracts.base import ShenFrame, derived


class MarketContract(ShenFrame):
    _key: ClassVar[str]
    _axis: ClassVar[str]
    _value_columns: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def economic_keys(cls):
        return (cls._key, cls._axis, "scenario_id")


class Curve(MarketContract):
    _key: ClassVar[str] = "curve_id"
    _axis: ClassVar[str] = "pillar_date"
    _value_columns: ClassVar[tuple[str, ...]] = ("log_df",)
    curve_id: str
    pillar_date: pl.Date
    log_df: float = pa.Field(nullable=True, default=None)
    discount_factor: float = pa.Field(nullable=True, default=None)
    source: str = pa.Field(nullable=True, default=None)
    as_of: pl.Datetime = pa.Field(nullable=True, default=None)
    priority: int = pa.Field(nullable=True, default=None)
    quality: str = pa.Field(nullable=True, default=None)
    scenario_id: str = pa.Field(default="base")

    @derived("log_df")
    def _derive_log_df(frame: pl.LazyFrame) -> pl.Expr:
        return pl.coalesce(pl.col("log_df"), pl.col("discount_factor").log()).alias(
            "log_df"
        )


class RawDICurve(ShenFrame):
    curve_id: str
    pillar_date: pl.Date
    rate: float
    anchor_date: pl.Date


class IndexObservation(MarketContract):
    _key: ClassVar[str] = "index_id"
    _axis: ClassVar[str] = "observation_date"
    _value_columns: ClassVar[tuple[str, ...]] = ("value",)
    index_id: str
    observation_date: pl.Date
    value: float
    scenario_id: str = pa.Field(default="base")


class Fx(MarketContract):
    _key: ClassVar[str] = "pair"
    _axis: ClassVar[str] = "observation_date"
    _value_columns: ClassVar[tuple[str, ...]] = ("rate",)
    pair: str
    observation_date: pl.Date
    rate: float = pa.Field(gt=0)
    scenario_id: str = pa.Field(default="base")


class FxReference(MarketContract):
    _key: ClassVar[str] = "reference_id"
    _axis: ClassVar[str] = "observation_date"
    _value_columns: ClassVar[tuple[str, ...]] = ("rate",)
    reference_id: str
    observation_date: pl.Date
    rate: float = pa.Field(gt=0)
    status: str
    source: str
    observed_at: pl.Datetime
    scenario_id: str = pa.Field(default="base")


class VolSurface(MarketContract):
    _key: ClassVar[str] = "surface_id"
    _axis: ClassVar[str] = "expiry"
    _value_columns: ClassVar[tuple[str, ...]] = ("atm", "skew", "curvature")
    surface_id: str
    expiry: pl.Date
    atm: float = pa.Field(gt=0)
    skew: float
    curvature: float
    scenario_id: str = pa.Field(default="base")


@dataclass(frozen=True, slots=True)
class Calendar:
    name: str
    holidays: tuple[dt.date, ...] = ()


DayCount = Literal["BUS/252", "ACT/365"]
Compounding = Literal["continuous", "annual"]
BUSINESS_DAYS = 252.0


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


def canonicalize_di(raw, convention: RateConvention | None = None) -> pl.LazyFrame:
    lf = RawDICurve.validate(raw)
    denom = 252.0 if convention is None or convention.day_count == "BUS/252" else 365.0
    days = (
        pl.business_day_count(pl.col("anchor_date"), pl.col("pillar_date"))
        if denom == BUSINESS_DAYS
        else (pl.col("pillar_date") - pl.col("anchor_date")).dt.total_days()
    )
    return lf.with_columns(log_df=-(days / denom) * pl.col("rate").log1p()).select(
        "curve_id", "pillar_date", "log_df"
    )
