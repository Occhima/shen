from __future__ import annotations

import contextlib
import contextvars
import math as _math
from typing import Any

import polars as pl

_backend: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "shen_math_backend", default=_math
)


@contextlib.contextmanager
def math_backend(module: Any):
    token = _backend.set(module)
    try:
        yield
    finally:
        _backend.reset(token)


def _dispatch(name, x):
    m = getattr(x, name, None)
    if callable(m):
        return m()
    return (getattr(_backend.get(), name, None) or getattr(_math, name))(x)


def exp(x):
    return _dispatch("exp", x)


def log(x):
    return _dispatch("log", x)


def sqrt(x):
    return _dispatch("sqrt", x)


def where(cond, a, b):
    if isinstance(cond, pl.Expr):
        return (
            pl.when(cond)
            .then(a if isinstance(a, pl.Expr) else pl.lit(a))
            .otherwise(b if isinstance(b, pl.Expr) else pl.lit(b))
        )
    mod = _backend.get()
    return mod.where(cond, a, b) if hasattr(mod, "where") else (a if cond else b)


def norm_cdf(x):
    ax = where(x >= 0, x, -x)
    pos = 1 / (1 + exp(-0.07056 * ax * ax * ax - 1.5976 * ax))
    return where(x >= 0, pos, 1 - pos)


def norm_pdf(x):
    return exp(-0.5 * x * x) / 2.5066282746310002
