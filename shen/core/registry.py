from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

import pandera.polars as pa
import polars as pl

from shen.contracts.market import Market, MarketObject
from shen.contracts.pricing import PricingTerms
from shen.contracts.values import ValueSpec
from shen.exceptions import (
    LookupResolutionError,
)

Calculator = Callable[..., object]


@dataclass(frozen=True, slots=True)
class LookupPolicy:
    interpolation: Literal["exact", "previous", "linear"] = "previous"
    extrapolation: Literal["error", "flat", "linear"] = "error"
    missing: Literal["error", "null"] = "error"
    tolerance: object | None = None


@dataclass(frozen=True, slots=True)
class Lookup:
    src: type[MarketObject]
    by: str
    at: str
    take: str | tuple[str, ...]
    As: str | tuple[str, ...] | None = None
    policy: LookupPolicy = LookupPolicy()

    @property
    def takes(self):
        return (self.take,) if isinstance(self.take, str) else self.take

    @property
    def aliases(self):
        return (
            self.takes
            if self.As is None
            else ((self.As,) if isinstance(self.As, str) else self.As)
        )

    def __call__(self, lf: pl.LazyFrame, market: Market) -> pl.LazyFrame:
        rhs = market[self.src]
        takes = self.takes
        aliases = self.aliases
        if len(takes) != len(aliases):
            raise LookupResolutionError("take/alias mismatch")
        dims = (
            ["scenario_id"]
            if "scenario_id" in lf.collect_schema().names()
            and "scenario_id" in rhs.collect_schema().names()
            else []
        )
        rhs = rhs.select(
            pl.col(self.src.key).alias("_rkey"),
            pl.col(self.src.at).alias("_rat"),
            *(pl.col(t).alias(f"_rv{i}") for i, t in enumerate(takes)),
            *dims,
        ).sort("_rat")
        by_left = [self.by, *dims]
        by_right = ["_rkey", *dims]
        if self.policy.interpolation == "exact":
            out = lf.join(
                rhs,
                left_on=[self.by, self.at, *dims],
                right_on=["_rkey", "_rat", *dims],
                how="left",
            )
            return out.rename({f"_rv{i}": a for i, a in enumerate(aliases)})
        lo = (
            lf.sort(self.at)
            .join_asof(
                rhs,
                left_on=self.at,
                right_on="_rat",
                by_left=by_left,
                by_right=by_right,
                strategy="backward",
            )
            .rename({"_rat": "_t0", **{f"_rv{i}": f"_lo{i}" for i in range(len(takes))}})
        )
        if self.policy.interpolation == "previous":
            if self.policy.extrapolation == "error":
                lo = lo.with_columns(_lookup_error=pl.col("_t0").is_null())
            return lo.rename({f"_lo{i}": a for i, a in enumerate(aliases)}).drop("_t0")
        hi = lo.join_asof(
            rhs,
            left_on=self.at,
            right_on="_rat",
            by_left=by_left,
            by_right=by_right,
            strategy="forward",
        ).rename({"_rat": "_t1", **{f"_rv{i}": f"_hi{i}" for i in range(len(takes))}})
        if self.policy.extrapolation == "error":
            hi = hi.with_columns(
                _lookup_error=pl.col("_t0").is_null() | pl.col("_t1").is_null()
            )
        w = (
            (
                (pl.col(self.at) - pl.col("_t0")).dt.total_days()
                / (pl.col("_t1") - pl.col("_t0")).dt.total_days()
            )
            .fill_nan(0)
            .fill_null(0)
        )
        exprs = [
            ((1 - w) * pl.col(f"_lo{i}") + w * pl.col(f"_hi{i}")).alias(a)
            for i, a in enumerate(aliases)
        ]
        return hi.with_columns(*exprs).drop(
            [
                "_t0",
                "_t1",
                *(f"_lo{i}" for i in range(len(takes))),
                *(f"_hi{i}" for i in range(len(takes))),
            ]
        )


@dataclass(frozen=True, slots=True)
class PricerInfo:
    product_type: str
    terms_schema: type[PricingTerms]
    required_market_objects: tuple[type[MarketObject], ...]
    required_columns: tuple[str, ...]
    lookup_aliases: tuple[str, ...]
    output: ValueSpec


@dataclass(frozen=True, slots=True)
class Pricer:
    product_type: str
    schema: type[PricingTerms]
    lookups: tuple[Lookup, ...]
    calc: Calculator
    params: tuple[str, ...]
    output: ValueSpec
    derive: Mapping[str, pl.Expr] | None = None
    legs: type[pa.DataFrameModel] | None = None
    explode: Callable[[pl.LazyFrame], pl.LazyFrame] | None = None
    weights: Mapping[str, float] | None = None

    def __call__(self, *args, **kwargs):
        return self.calc(*args, **kwargs)

    def base_expr(self):
        return self.calc(*(pl.col(p) for p in self.params))

    def price(self, terms, market, **kw):
        from shen.core.engine import price

        return price(terms, market, universe=PricingUniverse.from_pricers(self), **kw)

    def unresolved(self, terms, market):
        return self.price(terms, market, strict=False).filter(pl.col("value").is_null())

    def explain(self):
        return self.requirements()

    def requirements(self):
        return PricerInfo(
            self.product_type,
            self.schema,
            tuple(lookup.src for lookup in self.lookups),
            self.params,
            tuple(a for lookup in self.lookups for a in lookup.aliases),
            self.output,
        )

    def output_schema(self):
        from shen.contracts.values import InstrumentValue

        return InstrumentValue

    def risk_factors(self):
        return tuple(a for lookup in self.lookups for a in lookup.aliases)


@dataclass(frozen=True, slots=True)
class PricingUniverse:
    pricers: Mapping[str, Pricer]

    @classmethod
    def from_pricers(cls, *pricers: Pricer):
        return cls({p.product_type: p for p in pricers})

    def get(self, product_type: str) -> Pricer:
        try:
            return self.pricers[product_type]
        except KeyError as e:
            from shen.exceptions import UnknownProductTypeError

            raise UnknownProductTypeError(product_type) from e


def pricer(
    schema: type[PricingTerms],
    lookups: tuple[Lookup, ...] = (),
    *,
    output: ValueSpec,
    derive: Mapping[str, pl.Expr] | None = None,
):
    def deco(fn: Calculator) -> Pricer:
        return Pricer(
            schema.product_type,
            schema,
            lookups,
            fn,
            tuple(inspect.signature(fn).parameters),
            output,
            derive,
        )

    return deco


def price_legs(schema: type[PricingTerms], leg: Pricer, *, weights: Mapping[str, float]):
    def deco(fn: Callable[..., object]) -> Pricer:
        sides = tuple(inspect.signature(fn).parameters)
        if set(sides) != set(weights):
            raise TypeError("weights must match leg parameters")

        def explode(lf: pl.LazyFrame) -> pl.LazyFrame:
            parts = []
            leg_cols = list(leg.schema.to_schema().columns)
            for side, w in weights.items():
                exprs = []
                for c in leg_cols:
                    exprs.append(
                        pl.col(f"{side}_{c}").alias(c)
                        if f"{side}_{c}" in lf.collect_schema().names()
                        else pl.col(c)
                        if c in lf.collect_schema().names()
                        else pl.lit(
                            schema.product_type if c == "product_type" else None
                        ).alias(c)
                    )
                exprs.append(pl.lit(w).alias("_w"))
                parts.append(lf.select(exprs))
            return pl.concat(parts)

        return Pricer(
            schema.product_type,
            schema,
            leg.lookups,
            leg.calc,
            leg.params,
            leg.output,
            leg.derive,
            leg.schema,
            explode,
            weights,
        )

    return deco
