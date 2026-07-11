from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import polars as pl

from shen.domain.contracts.instruments import InstrumentTerms
from shen.domain.contracts.values import UnitValue
from shen.pricing.fetch import Fetch

Calculator = Callable[..., object]


@dataclass(frozen=True, slots=True)
class ValueSpec:
    measure: str
    currency: str | pl.Expr
    unit: str | pl.Expr
    quantity_unit: str | pl.Expr | None = None


@dataclass(frozen=True, slots=True)
class PricerInfo:
    product_type: str
    terms_schema: type[InstrumentTerms]
    required_market_objects: tuple[type, ...]
    required_columns: tuple[str, ...]
    lookup_aliases: tuple[str, ...]
    output: ValueSpec


@dataclass(frozen=True, slots=True)
class Pricer:
    product_type: str
    schema: type[InstrumentTerms]
    fetches: tuple[Fetch, ...]
    calc: Calculator
    params: tuple[str, ...]
    output: ValueSpec
    derive: Mapping[str, pl.Expr] | None = None

    @property
    def lookups(self):
        return self.fetches

    def __call__(self, *a, **kw):
        return self.calc(*a, **kw)

    def base_expr(self):
        return self.calc(*(pl.col(p) for p in self.params))

    def price(self, terms, market, **kw):
        engine = __import__("shen.pricing.engine", fromlist=["price"])
        return engine.price(terms, market, universe=Registry.from_pricers(self), **kw)

    def requirements(self):
        return PricerInfo(
            self.product_type,
            self.schema,
            tuple(f.src for f in self.fetches),
            self.params,
            tuple(a for f in self.fetches for a in f.aliases),
            self.output,
        )

    explain = requirements

    def output_schema(self):
        return UnitValue

    def risk_factors(self):
        return tuple(a for f in self.fetches for a in f.aliases)


@dataclass(frozen=True, slots=True)
class Registry:
    pricers: Mapping[str, Pricer]

    @classmethod
    def from_pricers(cls, *pricers: Pricer):
        return cls({p.product_type: p for p in pricers})

    def get(self, product_type: str):
        return self.pricers[product_type]


def pricer(
    schema: type[InstrumentTerms],
    fetches: tuple[Fetch, ...] = (),
    *,
    output: ValueSpec,
    derive: Mapping[str, pl.Expr] | None = None,
    lookups: tuple[Fetch, ...] | None = None,
):
    if lookups is not None:
        fetches = lookups

    def deco(fn: Calculator) -> Pricer:
        return Pricer(
            schema._product_type,
            schema,
            fetches,
            fn,
            tuple(inspect.signature(fn).parameters),
            output,
            derive,
        )

    return deco


PricingUniverse = Registry
