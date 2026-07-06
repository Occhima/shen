"""Greeks by autodiff — one calculator body, three backends.

The DSL contract pays out here: vanilla_option is a pure function whose
math routes through plib.math (structural dispatch). So:

    polars   — price the book lazily (Exprs have .exp/.log methods)
    floats   — unit-test the same body
    jax      — plib.math.use(jnp) makes exp/log/sqrt/norm_cdf fall back
               to jax.numpy, and jax.grad/vmap give EXACT greeks of the
               exact pricing function. No second implementation, no
               bump-and-reprice error, nothing to keep in sync.

The reconciliation at the end is the point: jax's autodiff delta agrees
with plib.sensitivities' expression-space finite difference — two
independent mechanisms differentiating the SAME function.
"""

import datetime as dt

import jax
import jax.numpy as jnp
import pandas as pd

import plib.math as pmath
import plib.pricers  # noqa: F401
from plib import Market, price, sensitivities
from plib.pricers.vanilla import vanilla_option

jax.config.update("jax_enable_x64", True)   # match polars f64
pmath.use(jnp)                              # DSL: jax joins the backends

REF = dt.date(2026, 7, 3)

mkt = Market.load(
    curve=pd.DataFrame({
        "curve_id": ["USD"] * 2,
        "pillar_date": [dt.date(2026, 7, 6), dt.date(2027, 7, 1)],
        "discount_factor": [1.0, 0.955]}),
    spot=pd.DataFrame({
        "index": ["BRENT"], "fixing_date": [dt.date(2026, 7, 2)],
        "value": [100.0], "publication_lag_days": [0]}),
    vol_surface=pd.DataFrame({
        "surface_id": ["BRENT-VOL"] * 2,
        "expiry": [dt.date(2026, 10, 1), dt.date(2027, 7, 1)],
        "atm": [0.22, 0.26], "skew": [-0.10, -0.08], "curv": [0.40, 0.30]}),
    ref_date=REF)

book = pd.DataFrame({
    "instrument_id":   ["C-90", "C-100", "P-100", "C-110"],
    "instrument_type": ["vanilla_option"] * 4,
    "cp":     [1.0, 1.0, -1.0, 1.0],
    "strike": [90.0, 100.0, 100.0, 110.0],
    "expiry": [dt.date(2027, 1, 4)] * 4,
    "index":  ["BRENT"] * 4,
    "disc_curve": ["USD"] * 4,
    "surface": ["BRENT-VOL"] * 4,
})

# 1. polars: lazy valuation + the trace carries every resolved input
values = price(book, mkt).collect()
tr = (price(book, mkt, trace=True).collect()
      .sort("instrument_id"))

# 2. jax: feed the trace columns into the SAME function
PARAMS = ("cp", "strike", "tau", "spot", "log_df_exp", "atm", "skew", "curv")
args = tuple(jnp.asarray(tr[c].to_numpy()) for c in PARAMS)

px = jax.vmap(vanilla_option)(*args)
assert jnp.allclose(px, jnp.asarray(tr["value"].to_numpy()), atol=1e-9)
print("backend equivalence: jax prices == polars prices  OK")

d_spot, d_tau, d_atm = 3, 2, 5              # positions in PARAMS
delta = jax.vmap(jax.grad(vanilla_option, argnums=d_spot))(*args)
gamma = jax.vmap(jax.grad(jax.grad(vanilla_option, argnums=d_spot),
                          argnums=d_spot))(*args)
vega  = jax.vmap(jax.grad(vanilla_option, argnums=d_atm))(*args)
theta = -jax.vmap(jax.grad(vanilla_option, argnums=d_tau))(*args)

import numpy as np
import polars as pl

greeks = tr.select("instrument_id", "value").with_columns(
    pl.Series("delta", np.asarray(delta)),
    pl.Series("gamma", np.asarray(gamma)),
    pl.Series("vega", np.asarray(vega)),
    pl.Series("theta", np.asarray(theta)))
print(greeks)

# 3. reconciliation: autodiff vs plib's expression-space FD (h=1e-4)
fd = sensitivities(book, mkt, h=1e-4).collect()
fd_spot = (fd.filter(fd["factor"] == "spot")
             .sort("instrument_id")["dv"].to_numpy())
assert jnp.allclose(delta, jnp.asarray(fd_spot), atol=1e-4)
print("reconciliation: jax autodiff delta == polars FD delta (1e-4)  OK")
