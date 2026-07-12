from __future__ import annotations

import contextlib
import contextvars
import math as _math
from collections.abc import Iterator
from typing import Any

_backend: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "shen_math_backend", default=_math
)


@contextlib.contextmanager
def math_backend(module: Any) -> Iterator[None]:
    """Temporarily select the scalar/array math backend."""
    token = _backend.set(module)
    try:
        yield
    finally:
        _backend.reset(token)


def _unary(name: str, value: Any) -> Any:
    method = getattr(value, name, None)
    if callable(method):
        return method()
    backend = _backend.get()
    function = getattr(backend, name, None) or getattr(_math, name)
    return function(value)


def exp(value: Any) -> Any:
    return _unary("exp", value)


def log(value: Any) -> Any:
    return _unary("log", value)


def sqrt(value: Any) -> Any:
    return _unary("sqrt", value)


def abs_(value: Any) -> Any:
    method = getattr(value, "abs", None)
    return method() if callable(method) else abs(value)


def where(condition: Any, when_true: Any, when_false: Any) -> Any:
    """Backend-neutral conditional supporting scalars, arrays and Polars Expr."""
    if condition.__class__.__module__.startswith("polars"):
        import polars as pl

        true_expr = when_true if isinstance(when_true, pl.Expr) else pl.lit(when_true)
        false_expr = when_false if isinstance(when_false, pl.Expr) else pl.lit(when_false)
        return pl.when(condition).then(true_expr).otherwise(false_expr)
    backend = _backend.get()
    function = getattr(backend, "where", None)
    return function(condition, when_true, when_false) if function else (
        when_true if condition else when_false
    )


def norm_cdf(value: Any) -> Any:
    absolute = where(value >= 0, value, -value)
    positive = 1 / (1 + exp(-0.07056 * absolute**3 - 1.5976 * absolute))
    return where(value >= 0, positive, 1 - positive)


def norm_pdf(value: Any) -> Any:
    return exp(-0.5 * value * value) / 2.5066282746310002


__all__ = [
    "abs_",
    "exp",
    "log",
    "math_backend",
    "norm_cdf",
    "norm_pdf",
    "sqrt",
    "where",
]
