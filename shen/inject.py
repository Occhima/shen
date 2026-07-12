from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import pandas as pd

type InjectionFunction = Callable[[type, pd.DataFrame, pd.Series], pd.Series]


def contract_key(contract: type) -> str:
    instrument_type = getattr(contract, "instrument_type", None)
    if callable(instrument_type):
        return str(instrument_type())
    try:
        return str(contract._product_type)
    except AttributeError as error:
        raise TypeError(
            f"{contract.__name__} must define instrument_type() or _product_type"
        ) from error


@dataclass(frozen=True, slots=True)
class Injection:
    output: str
    source_terms: type
    on: Mapping[str, str]
    take: str
    overwrite: bool = False

    def __post_init__(self) -> None:
        if not self.output:
            raise ValueError("injection output cannot be empty")
        if not self.take:
            raise ValueError("injection take cannot be empty")
        if not self.on:
            raise ValueError("injection join mapping cannot be empty")
        if self.take in self.on.values():
            raise ValueError("injection take cannot also be a right join key")
        object.__setattr__(self, "on", MappingProxyType(dict(self.on)))

    @property
    def source_key(self) -> str:
        return contract_key(self.source_terms)

    @property
    def kind(self) -> str:
        return "inject"

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.output,)


class InjectedMethod:
    """Descriptor installed by @inject; registration occurs during class creation."""

    metadata: Injection
    function: InjectionFunction
    name: str

    def __init__(self, metadata: Injection, function: InjectionFunction) -> None:
        self.metadata = metadata
        self.function = function
        self.name = function.__name__

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name
        inherited = tuple(getattr(owner, "__pricing_injections__", ()))
        own = tuple(item for item in inherited if item.name != name)
        setattr(owner, "__pricing_injections__", (*own, self))

    def __get__(self, instance: object, owner: type | None = None) -> Callable:
        contract = owner or type(instance)

        def bound(frame: pd.DataFrame, source: pd.Series) -> pd.Series:
            return self.function(contract, frame, source)

        return bound

    def calculate(
        self, contract: type, frame: pd.DataFrame, source: pd.Series
    ) -> pd.Series:
        result = self.function(contract, frame, source)
        if not isinstance(result, pd.Series):
            raise TypeError(
                f"{contract.__name__}.{self.name} must return pandas.Series"
            )
        if not result.index.equals(frame.index):
            raise ValueError(
                f"{contract.__name__}.{self.name} returned a misaligned Series"
            )
        return result


def inject(
    *,
    name: str,
    source: type,
    on: Mapping[str, str],
    take: str = "value",
    overwrite: bool = False,
) -> Callable[[InjectionFunction], InjectedMethod]:
    metadata = Injection(name, source, on, take, overwrite)

    def decorate(function: InjectionFunction) -> InjectedMethod:
        parameters = tuple(inspect.signature(function).parameters)
        if len(parameters) != 3:
            raise TypeError(
                f"@inject method {function.__qualname__} must have "
                "(cls, frame, source)"
            )
        return InjectedMethod(metadata, function)

    return decorate


def injections(contract: type) -> tuple[InjectedMethod, ...]:
    return tuple(getattr(contract, "__pricing_injections__", ()))


def validate_injections(contract: type) -> tuple[InjectedMethod, ...]:
    declared = injections(contract)
    if not declared:
        return ()
    schema = contract.to_schema()
    fields = set(schema.columns)
    outputs: set[str] = set()
    for method in declared:
        metadata = method.metadata
        if metadata.output not in fields:
            raise TypeError(
                f"{contract.__name__}.{method.name}: output "
                f"{metadata.output!r} is not a declared contract field"
            )
        if metadata.output in outputs:
            raise TypeError(
                f"{contract.__name__}: duplicate injection output "
                f"{metadata.output!r}"
            )
        missing = set(metadata.on) - fields
        if missing:
            raise TypeError(
                f"{contract.__name__}.{method.name}: missing left join fields "
                f"{sorted(missing)}"
            )
        contract_key(metadata.source_terms)
        outputs.add(metadata.output)
    return declared


__all__ = [
    "InjectedMethod",
    "Injection",
    "contract_key",
    "inject",
    "injections",
    "validate_injections",
]
