from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from types import MappingProxyType

from shen.core.pricing.pricer import Pricer
from shen.inject import validate_injections


@dataclass(frozen=True, slots=True)
class Registry:
    _pricers: Mapping[str, Pricer]

    @classmethod
    def from_pricers(cls, *pricers: Pricer) -> Registry:
        indexed: dict[str, Pricer] = {}
        for pricer in pricers:
            pricer.finalize()
            key = pricer.dispatch_key
            if key in indexed:
                raise ValueError(f"duplicate pricer key: {key!r}")
            indexed[key] = pricer
        registry = cls(MappingProxyType(indexed))
        registry.order()
        return registry

    @property
    def pricers(self) -> Mapping[str, Pricer]:
        return self._pricers

    def get(self, key: str) -> Pricer:
        try:
            return self._pricers[key]
        except KeyError as error:
            raise KeyError(f"unknown pricer: {key!r}") from error

    def __getitem__(self, key: str) -> Pricer:
        return self.get(key)

    def __iter__(self) -> Iterator[Pricer]:
        return iter(self._pricers.values())

    def dependencies(self, key: str) -> tuple[str, ...]:
        pricer = self.get(key)
        sources = {
            method.metadata.source_key for method in validate_injections(pricer.terms)
        }
        missing = sources - self._pricers.keys()
        if missing:
            raise KeyError(
                f"{key!r} references unregistered injection sources: {sorted(missing)}"
            )
        return tuple(source for source in self._pricers if source in sources)

    def graph(self) -> Mapping[str, tuple[str, ...]]:
        return MappingProxyType(
            {key: self.dependencies(key) for key in self._pricers}
        )

    def order(self) -> tuple[str, ...]:
        graph = dict(self.graph())
        try:
            return tuple(TopologicalSorter(graph).static_order())
        except CycleError as error:
            raise ValueError("cycle in @inject pricer dependencies") from error


__all__ = ["Registry"]
