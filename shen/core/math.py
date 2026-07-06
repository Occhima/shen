"""shen.core.math — a deliberately tiny numeric DSL.

Calculators import functions from here and NEVER call Expr methods, so
the identical calculator body runs on pl.Expr (engine), floats (unit
tests), numpy arrays, or any backend pushed with use():

    import jax.numpy as jnp
    shen.core.math.use(jnp)      # exp/log/sqrt now fall back to jax

Dispatch order: the operand's own method (pl.Expr has .exp) -> registered
backend modules, most recent first -> stdlib math. Because calculators
are plain functions of their arguments, "exporting" a pricing formula to
jax for autodiff is just calling it with jax arrays. No adapters, no
codegen — the DSL is the function signature.
"""

from __future__ import annotations

import math as _stdlib
from typing import Any

import polars as pl

_BACKENDS: list[Any] = [_stdlib]


def use(module: Any) -> None:
    """Push a numeric backend (jax.numpy, autograd.numpy, numpy, ...)."""
    _BACKENDS.insert(0, module)


def _dispatch(name: str, x):
    method = getattr(x, name, None)
    if callable(method):
        return method()
    for mod in _BACKENDS:
        if fn := getattr(mod, name, None):
            return fn(x)
    raise TypeError(f"no backend provides {name!r} for {type(x).__name__}")


def exp(x):
    return _dispatch("exp", x)


def log(x):
    return _dispatch("log", x)


def sqrt(x):
    return _dispatch("sqrt", x)


def where(cond, a, b):
    """DSL where: bool -> ternary; pl.Expr -> when/then/otherwise;
    backend array (jax/numpy) -> backend .where; else ternary. Compiles
    per-backend so the IDENTICAL calculator body routes call/put on
    polars, floats, and jax without branching in the formula."""
    return match((cond, a), default=b)


def match(*cases, default):
    """Multiway first-match-wins. cases = (cond, val) pairs; default
    when no cond holds. where(c, a, b) == match((c, a), default=b).

    R6b: ALL branches are evaluated eagerly on every backend (polars
    when/then/otherwise, numpy.select, scalar ternary) — every val MUST
    be TOTAL over the whole input domain. Lazy short-circuit is NOT
    available; design branches to not blow up on the unmatched side."""
    if not cases:
        return default
    conds = [c for c, _ in cases]
    vals = [v for _, v in cases]
    if any(isinstance(c, pl.Expr) for c in conds):
        def _lit(v):
            return v if isinstance(v, pl.Expr) else pl.lit(v)
        expr = pl.when(conds[0]).then(_lit(vals[0]))
        for c, v in zip(conds[1:], vals[1:], strict=True):
            expr = expr.when(c).then(_lit(v))
        return expr.otherwise(_lit(default))
    for mod in _BACKENDS:
        if hasattr(mod, "select"):
            return mod.select(conds, vals, default=default)
    for c, v in cases:
        if bool(c):
            return v
    return default


SQRT_2PI = 2.5066282746310002


def norm_cdf(x):
    """Standard normal CDF via Bowling's logistic approximation
    (|err| < 1.5e-4), built from exp() and where() only.

    Reflected to |x| so it is TOTAL: for ax >= 0 the exponent
    -0.07056*ax^3 - 1.5976*ax is <= 0, so exp() can never overflow —
    which matters because where() evaluates BOTH branches (routed
    calculators compute the put branch even for calls)."""
    ax = where(x >= 0, x, -x)
    pos = 1.0 / (1.0 + exp(-0.07056 * ax * ax * ax - 1.5976 * ax))
    return where(x >= 0, pos, 1.0 - pos)


def norm_pdf(x):
    return exp(-0.5 * x * x) / SQRT_2PI
