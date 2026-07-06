"""Tree — pricing DAGs over networkx (plugin).

The same primitive the whole library runs on, lifted one level: a node
is a decorated function whose PARAMETER NAMES are its dependencies —
exactly how calculators receive columns, now receiving other nodes'
results. networkx owns the graph theory (topological order, ancestors,
cycle detection); the plugin owns nothing clever.

Two decorators:

    @tree.node                      generic computation (frames in/out)
    @tree.instruments("di_future")  the function returns an INSTRUMENT
                                    frame; the node's value is it PRICED
                                    with that pricer under the ambient
                                    MarketContext. Accepts a Pricer, a
                                    decorated calculator, or a registry
                                    NAME — recovery is what the registry
                                    is for.

Positions need no special tree: a positions node is just a node that
returns a canonical Position frame; Book/fungibilidade are nodes too.
Valuation and portfolio live in ONE DAG because everything flowing on
the edges is a LazyFrame — and everything stays lazy until the caller
collects, so a run() materializes nothing by itself.

Market context is deliberately NOT an edge: it is the ambient
environment (see plugins.context), which is why nodes don't thread mkt
through their signatures. tree.run(target, mkt=...) binds one for the
duration of the run when you'd rather be explicit.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass

import networkx as nx

from shen.core.engine import MarketContext, market
from shen.core.registry import REGISTRY, Pricer, as_pricer


def _recover(pricer) -> Pricer:
    if isinstance(pricer, str):
        try:
            return REGISTRY[pricer]
        except KeyError:
            raise KeyError(
                f"{pricer!r} not in registry; known: {sorted(REGISTRY)}") from None
    return as_pricer(pricer)


@dataclass(frozen=True, slots=True)
class _Node:
    fn: Callable
    deps: tuple[str, ...]


class Tree:
    def __init__(self, name: str = "pricing"):
        self.name = name
        self.graph = nx.DiGraph()
        self._nodes: dict[str, _Node] = {}

    # -- the two decorators --------------------------------------------
    def node(self, fn=None, *, name: str | None = None):
        def deco(f: Callable) -> Callable:
            self._register(name or f.__name__, f, f)
            return f
        return deco(fn) if fn is not None else deco

    def instruments(self, pricer, *, name: str | None = None):
        p = _recover(pricer)

        def deco(f: Callable) -> Callable:
            def valued(*deps):
                return p.price(f(*deps))          # ambient market resolves
            self._register(name or f.__name__, valued, f)
            return f
        return deco

    def _register(self, n: str, run_fn: Callable, sig_fn: Callable) -> None:
        if n in self._nodes:
            raise ValueError(f"node {n!r} already defined in tree {self.name!r}")
        deps = tuple(inspect.signature(sig_fn).parameters)
        self._nodes[n] = _Node(run_fn, deps)
        self.graph.add_node(n)
        for d in deps:
            self.graph.add_edge(d, n)

    # -- execution -------------------------------------------------------
    def run(self, *targets: str, mkt=None):
        """Evaluate targets (and only their ancestors), in one ambient
        market. Returns the single result, or {name: result} for many."""
        if not targets:
            raise TypeError("run() needs at least one target node")
        needed = set(targets)
        for t in targets:
            if t not in self.graph:
                raise KeyError(f"unknown node {t!r}; have {sorted(self._nodes)}")
            needed |= nx.ancestors(self.graph, t)
        if undeclared := [n for n in needed if n not in self._nodes]:
            raise KeyError(f"nodes referenced but never defined: {undeclared}")

        sub = self.graph.subgraph(needed)
        ctx = MarketContext(data=mkt) if mkt is not None else nullcontext()
        results: dict[str, object] = {}
        with ctx:
            for n in nx.topological_sort(sub):   # raises on cycles
                node = self._nodes[n]
                results[n] = node.fn(*(results[d] for d in node.deps))
        if len(targets) == 1:
            return results[targets[0]]
        return {t: results[t] for t in targets}

    def order(self, target: str) -> list[str]:
        """The evaluation order for a target — the tree, legible."""
        needed = nx.ancestors(self.graph, target) | {target}
        return list(nx.topological_sort(self.graph.subgraph(needed)))
