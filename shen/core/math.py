"""plib.math — a deliberately tiny numeric DSL.

Calculators import functions from here and NEVER call Expr methods, so
the identical calculator body runs on pl.Expr (engine), floats (unit
tests), numpy arrays, or any backend pushed with use():

    import jax.numpy as jnp
    plib.math.use(jnp)      # exp/log/sqrt now fall back to jax

Dispatch order: the operand's own method (pl.Expr has .exp) -> registered
backend modules, most recent first -> stdlib math. Because calculators
are plain functions of their arguments, "exporting" a pricing formula to
jax for autodiff is just calling it with jax arrays. No adapters, no
codegen — the DSL is the function signature.
"""

from __future__ import annotations

import math as _stdlib
from typing import Any

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


SQRT_2PI = 2.5066282746310002


def norm_cdf(x):
    """Standard normal CDF via Bowling's logistic approximation
    (|err| < 1.5e-4), built from exp() only so it dispatches through
    the same structural backends as everything else."""
    return 1.0 / (1.0 + exp(-0.07056 * x * x * x - 1.5976 * x))


def norm_pdf(x):
    return exp(-0.5 * x * x) / SQRT_2PI
