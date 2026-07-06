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

import re
from typing import ClassVar

import pandera.pandas  # noqa: F401 — registers builtin checks (gt/unique/...) into the polars backend's CHECK_FUNCTION_REGISTRY; without this, pandera.polars.Field(gt=0) raises KeyError at class definition.
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
    """Acronym-aware: DICurve -> di_curve, VolSurface -> vol_surface.
    Boundaries: lower/digit->Upper, and Upper->Upper+lower."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])",
                  "_", name).lower()


MARKET_TYPES: dict[str, type["MarketObject"]] = {}
"""Name -> contract, filled by MarketObject.__init_subclass__. This is
what lets Market.load(curve=df) and MarketContext(ref, curve=df) exist:
defining `class Vol(MarketObject)` REGISTERS `vol=` as a valid kwarg —
the open-world version of a dataclass field."""


class MarketObject(pa.DataFrameModel):
    _key: ClassVar[str]  # entity column, e.g. curve_id
    _at: ClassVar[str]   # time column joined against instrument dates
    _canonical: ClassVar[type["MarketObject"] | None] = None
    """Convention -> canonical contract. None = self is canonical (the
    normal case). A convention (e.g. DICurve) sets this to its target
    (e.g. Curve) so Market.load routes its rows through _to_canonical
    and merges into the canonical contract's frame — the convention
    exists only as an ingestion shape, never as a stored frame."""

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
    def _to_canonical(cls, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Convention form -> canonical form. Identity by default;
        conventions override to do their specific derivation
        (rate -> log_df, etc.). Only called when _canonical is set."""
        return lf

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
        has_log = "log_df" in names
        has_df = "discount_factor" in names
        if has_log and has_df:
            lf = lf.with_columns(
                log_df=pl.when(pl.col("log_df").is_null())
                         .then(pl.col("discount_factor").log())
                         .otherwise(pl.col("log_df")),
                discount_factor=pl.when(pl.col("discount_factor").is_null())
                                  .then(pl.col("log_df").exp())
                                  .otherwise(pl.col("discount_factor")))
        elif not has_log:
            if has_df:
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
            lf = lf.with_columns(discount_factor=pl.col("log_df").exp())
        elif not has_df:
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
