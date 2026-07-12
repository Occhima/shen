from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import polars as pl

from shen.domain.contracts.base import ShenFrame
from shen.domain.contracts.instruments import InstrumentTerms
from shen.domain.contracts.values import UnitValue
from shen.inject import contract_key, validate_injections

if TYPE_CHECKING:
    from shen.core.pricing.tree import PricingRun
    from shen.domain.market import Market

type Calculator = Callable[..., Any]
type ParameterFunction = Callable[..., pl.LazyFrame]


class PricingTerms(ShenFrame):
    """Prepared fields understood by a calculator."""

    class Config:
        strict = False
        coerce = True
        unique_column_names = True
        add_missing_columns = True


@dataclass(frozen=True, slots=True)
class ValueSpec:
    measure: str
    currency: str | pl.Expr
    unit: str | pl.Expr


@dataclass(frozen=True, slots=True)
class PricerInfo:
    dispatch_key: str
    terms: type[InstrumentTerms]
    pricing_terms: type[PricingTerms]
    parameter_strategies: tuple[str, ...]
    calculator_fields: tuple[str, ...]
    injection_sources: tuple[str, ...]
    output: ValueSpec


@dataclass(slots=True)
class Pricer:
    terms: type[InstrumentTerms]
    pricing_terms: type[PricingTerms]
    output: ValueSpec
    _parameters: dict[str, ParameterFunction] = field(default_factory=dict, init=False)
    _calculator: Calculator | None = field(default=None, init=False)
    _calculator_fields: tuple[str, ...] = field(default=(), init=False)

    @property
    def dispatch_key(self) -> str:
        return contract_key(self.terms)

    @property
    def product_type(self) -> str:
        return self.dispatch_key

    @property
    def schema(self) -> type[InstrumentTerms]:
        return self.terms

    def parameters(
        self, function: ParameterFunction | None = None, *, strategy: str = "default"
    ):
        def register(fn: ParameterFunction) -> ParameterFunction:
            if strategy in self._parameters:
                raise ValueError(
                    f"duplicate {self.dispatch_key!r} parameter strategy {strategy!r}"
                )
            count = len(inspect.signature(fn).parameters)
            if count not in (2, 3):
                raise TypeError(
                    f"{fn.__qualname__} must accept (terms, market) or "
                    "(terms, market, tree)"
                )
            self._parameters[strategy] = fn
            return fn

        return register(function) if function is not None else register

    def calculator(self, function: Calculator) -> Calculator:
        if self._calculator is not None:
            raise ValueError(f"calculator already registered for {self.dispatch_key!r}")
        fields = tuple(inspect.signature(function).parameters)
        declared = set(self.pricing_terms.to_schema().columns)
        missing = set(fields) - declared
        if missing:
            raise TypeError(
                f"calculator fields absent from {self.pricing_terms.__name__}: "
                f"{sorted(missing)}"
            )
        self._calculator = function
        self._calculator_fields = fields
        return function

    @property
    def parameter_strategies(self) -> Mapping[str, ParameterFunction]:
        return MappingProxyType(self._parameters)

    @property
    def calculator_fields(self) -> tuple[str, ...]:
        return self._calculator_fields

    def finalize(self) -> Pricer:
        if "default" not in self._parameters:
            raise ValueError(f"{self.dispatch_key!r} has no default parameters strategy")
        if self._calculator is None:
            raise ValueError(f"{self.dispatch_key!r} has no calculator")
        validate_injections(self.terms)
        return self

    def prepare(
        self,
        terms: pl.LazyFrame,
        market: Market,
        *,
        strategy: str = "default",
        tree: PricingRun | None = None,
    ) -> pl.LazyFrame:
        try:
            function = self._parameters[strategy]
        except KeyError as error:
            raise KeyError(
                f"unknown {self.dispatch_key!r} parameter strategy {strategy!r}"
            ) from error
        if len(inspect.signature(function).parameters) == 3:
            prepared = function(terms, market, tree)
        else:
            prepared = function(terms, market)
        return self.pricing_terms.resolve(prepared)

    def expression(self) -> pl.Expr:
        if self._calculator is None:
            raise ValueError(f"{self.dispatch_key!r} has no calculator")
        return self._calculator(*(pl.col(name) for name in self._calculator_fields))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._calculator is None:
            raise ValueError(f"{self.dispatch_key!r} has no calculator")
        return self._calculator(*args, **kwargs)

    def requirements(self) -> PricerInfo:
        declared = validate_injections(self.terms)
        return PricerInfo(
            dispatch_key=self.dispatch_key,
            terms=self.terms,
            pricing_terms=self.pricing_terms,
            parameter_strategies=tuple(self._parameters),
            calculator_fields=self._calculator_fields,
            injection_sources=tuple(item.metadata.source_key for item in declared),
            output=self.output,
        )

    explain = requirements

    @staticmethod
    def output_schema() -> type[UnitValue]:
        return UnitValue

    def price(self, terms: Any, market: Market, **kwargs: Any) -> pl.LazyFrame:
        from shen.core.pricing.engine import price
        from shen.core.pricing.registry import Registry

        return price(terms, market, using=Registry.from_pricers(self), **kwargs)


__all__ = ["Pricer", "PricerInfo", "PricingTerms", "ValueSpec"]
