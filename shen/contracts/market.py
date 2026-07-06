"""Market-object contracts.

Interpolation prep, fixing logic, and factor derivation live HERE — not
in the calculator, not in the engine. `validate()` is overridden to be
the single gate: parse -> pandera-validate -> LazyFrame. It accepts
anything frame-shaped (pandas from a vendor, polars eager/lazy from a
cache) and emits a canonical LazyFrame. There is no path around it, and
the schema checks the DERIVED fields — validation as postcondition of
parsing.

Each contract carries its own join geometry (_key, _at): a Curve knows
it is keyed by curve_id and time-indexed by pillar_date. Binds only say
which *instrument* columns map onto that geometry.
"""

from __future__ import annotations

from typing import ClassVar

import pandera.polars as pa
import polars as pl


def as_lazy(frame) -> pl.LazyFrame:
    """pandas / polars eager / polars lazy -> LazyFrame (frame-agnostic edge)."""
    match frame:
        case pl.LazyFrame():
            return frame
        case pl.DataFrame():
            return frame.lazy()
        case _:
            return pl.from_pandas(frame).lazy()


def _snake(name: str) -> str:
    return "".join("_" + c.lower() if c.isupper() else c for c in name).lstrip("_")


MARKET_TYPES: dict[str, type["MarketObject"]] = {}
"""Name -> contract, filled by MarketObject.__init_subclass__. This is
what lets Market.load(curve=df) and MarketContext(ref, curve=df) exist:
defining `class Vol(MarketObject)` REGISTERS `vol=` as a valid kwarg —
the open-world version of a dataclass field."""


class MarketObject(pa.DataFrameModel):
    _key: ClassVar[str]  # entity column, e.g. curve_id
    _at: ClassVar[str]   # time column joined against instrument dates

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = _snake(cls.__name__)
        other = MARKET_TYPES.get(name)
        if other is not None and other is not cls:
            raise TypeError(
                f"market contract name {name!r} already registered by {other.__name__}")
        MARKET_TYPES[name] = cls

    class Config:
        strict = "filter"
        coerce = True

    @classmethod
    def _parse(cls, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Derive fields, dedup, sort. Identity by default."""
        return lf.unique(subset=[cls._key, cls._at], keep="last").sort(cls._key, cls._at)

    @classmethod
    def validate(cls, frame, *args, **kwargs) -> pl.LazyFrame:  # type: ignore[override]
        """THE gate: anything frame-shaped in, canonical LazyFrame out."""
        return super().validate(cls._parse(as_lazy(frame)), *args, **kwargs)


class Curve(MarketObject):
    """Discount curve. Prep: log-space factors, sorted deduped pillars.

    The compounding/day-count convention is a CONTRACT characteristic:
    the gate accepts whichever representation the source has and
    derives the rest —
        log_df                          (bootstrap output: passthrough)
        discount_factor                 -> log_df = ln(df)
        rate + anchor_date              -> bus/252 exponential (Brazil):
                                           log_df = -(du/252) * ln(1+rate)
    Holidays for the business-day count are injected per deployment via
    Curve.holidays(anbima_dates) — weekends-only by default.
    """

    _key: ClassVar[str] = "curve_id"
    _at: ClassVar[str] = "pillar_date"
    _holidays: ClassVar[tuple] = ()

    curve_id: str
    pillar_date: pl.Date
    discount_factor: float = pa.Field(gt=0)
    log_df: float  # derived; schema proves the parse happened

    @classmethod
    def holidays(cls, dates) -> None:
        cls._holidays = tuple(dates)

    @classmethod
    def _parse(cls, lf: pl.LazyFrame) -> pl.LazyFrame:
        names = set(lf.collect_schema().names())
        if "log_df" not in names:
            if "discount_factor" in names:
                lf = lf.with_columns(log_df=pl.col("discount_factor").log())
            elif {"rate", "anchor_date"} <= names:
                du = pl.business_day_count(
                    pl.col("anchor_date"), pl.col("pillar_date"),
                    holidays=list(cls._holidays))
                lf = lf.with_columns(
                    log_df=-(du / 252) * pl.col("rate").log1p())
            else:
                raise ValueError(
                    "Curve needs log_df, discount_factor, or (rate, anchor_date)")
        if "discount_factor" not in names:
            lf = lf.with_columns(discount_factor=pl.col("log_df").exp())
        return super()._parse(lf)


class Spot(MarketObject):
    """Fixings / index observations. Fixing logic: publication lag shift,
    last-good-observation dedup. Calculators only ever see `value`."""

    _key: ClassVar[str] = "index"
    _at: ClassVar[str] = "fixing_date"

    index: str
    fixing_date: pl.Date
    value: float
    publication_lag_days: int = pa.Field(ge=0, default=0)

    @classmethod
    def _parse(cls, lf: pl.LazyFrame) -> pl.LazyFrame:
        return super()._parse(
            lf.with_columns(
                fixing_date=pl.col("fixing_date")
                + pl.duration(days=pl.col("publication_lag_days"))
            )
        )


class Fx(MarketObject):
    _key: ClassVar[str] = "pair"
    _at: ClassVar[str] = "date"

    pair: str
    date: pl.Date
    rate: float = pa.Field(gt=0)


class VolSurface(MarketObject):
    """Volatility matrix as a PARAMETRIZED smile per (surface, expiry):
    sigma(k) = atm + skew*k + curv*k^2 with k = ln(K/F).

    Design choice: storing smile parameters (instead of a raw strike
    grid) keeps interpolation inside the ONE existing Lookup primitive
    — three lerp lookups over expiry — and makes the strike dimension
    pure calculator arithmetic. A raw K-grid would need a 2D lookup;
    that is the extension point if you ever quote by strike directly.
    """

    _key: ClassVar[str] = "surface_id"
    _at: ClassVar[str] = "expiry"

    surface_id: str
    expiry: pl.Date
    atm: float = pa.Field(gt=0)
    skew: float
    curv: float
