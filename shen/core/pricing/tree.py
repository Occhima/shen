from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, field
from graphlib import CycleError, TopologicalSorter
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, slots=True)
class NodeKey:
    namespace: str
    name: str
    dimensions: tuple[tuple[str, Hashable], ...] = ()

    @classmethod
    def of(cls, namespace: str, name: str, **dimensions: Hashable) -> NodeKey:
        return cls(namespace, name, tuple(sorted(dimensions.items())))


type NodeFunction = Callable[[PricingRun, Mapping[str, Any]], Any]


@dataclass(frozen=True, slots=True)
class PricingNode:
    key: NodeKey
    function: NodeFunction
    dependencies: tuple[NodeKey, ...] = ()


@dataclass(slots=True)
class PricingTree:
    """Explicit calculation graph. It is independent from the @inject pricer DAG."""

    _nodes: dict[NodeKey, PricingNode] = field(default_factory=dict)

    def add(
        self,
        key: NodeKey,
        function: NodeFunction,
        *,
        dependencies: tuple[NodeKey, ...] = (),
    ) -> PricingTree:
        if key in self._nodes:
            raise ValueError(f"duplicate pricing-tree node: {key}")
        self._nodes[key] = PricingNode(key, function, tuple(dependencies))
        return self

    def node(
        self,
        key: NodeKey,
        *,
        dependencies: tuple[NodeKey, ...] = (),
    ) -> Callable[[NodeFunction], NodeFunction]:
        def decorate(function: NodeFunction) -> NodeFunction:
            self.add(key, function, dependencies=dependencies)
            return function

        return decorate

    @property
    def nodes(self) -> Mapping[NodeKey, PricingNode]:
        return MappingProxyType(self._nodes)

    def order(self) -> tuple[NodeKey, ...]:
        graph = {key: node.dependencies for key, node in self._nodes.items()}
        missing = {
            dependency
            for node in self._nodes.values()
            for dependency in node.dependencies
            if dependency not in self._nodes
        }
        if missing:
            raise KeyError(f"missing pricing-tree nodes: {sorted(map(str, missing))}")
        try:
            return tuple(TopologicalSorter(graph).static_order())
        except CycleError as error:
            raise ValueError("cycle in pricing tree") from error

    def run(self, context: Mapping[str, Any] | None = None) -> PricingRun:
        self.order()
        return PricingRun(self, dict(context or {}))


@dataclass(slots=True)
class PricingRun:
    tree: PricingTree
    context: dict[str, Any] = field(default_factory=dict)
    _cache: dict[NodeKey, Any] = field(default_factory=dict)
    _active: set[NodeKey] = field(default_factory=set)

    def get(self, key: NodeKey) -> Any:
        if key in self._cache:
            return self._cache[key]
        if key in self._active:
            raise ValueError(f"recursive pricing-tree evaluation: {key}")
        try:
            node = self.tree.nodes[key]
        except KeyError as error:
            raise KeyError(f"unknown pricing-tree node: {key}") from error
        self._active.add(key)
        try:
            for dependency in node.dependencies:
                self.get(dependency)
            value = node.function(self, MappingProxyType(self.context))
            self._cache[key] = value
            return value
        finally:
            self._active.remove(key)

    def put(self, key: NodeKey, value: Any) -> Any:
        self._cache[key] = value
        return value

    def cached(self, key: NodeKey) -> bool:
        return key in self._cache

    @property
    def values(self) -> Mapping[NodeKey, Any]:
        return MappingProxyType(self._cache)


__all__ = ["NodeKey", "PricingNode", "PricingRun", "PricingTree"]
