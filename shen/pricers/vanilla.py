"""Plain vanilla options (Black-76 on the forward) + butterfly structure.

The pieces, each in its native slot:
  * sigma interpolation — three Lookups against VolSurface (atm, skew,
    curv lerp'd over expiry) + the smile polynomial in the calculator:
    sigma = atm + skew*k + curv*k^2, k = ln(K/F).
  * tau — dates are frame algebra, so `derive` computes bus/252 year
    fraction declaratively; the calculator receives it as a number.
  * the calculator — pure plib.math DSL (exp, log, sqrt, norm_cdf), so
    the IDENTICAL body prices lazily under polars and differentiates
    under jax for exact greeks (see examples/option_greeks_jax.py).
  * butterfly — a combinator: the body `low - 2*mid + high` IS the
    structure; coefficients (+1, -2, +1) are probed from it at import.

cp is +1.0 for calls, -1.0 for puts — a float so the math stays total.
"""

from __future__ import annotations

import polars as pl

from plib.contracts.instruments import Instrument
from plib.contracts.market import Curve, Spot, VolSurface
from plib.math import exp, log, norm_cdf, sqrt
from plib.registry import Lookup, price_legs, pricer


class VanillaOption(Instrument):
    cp: float            # +1 call / -1 put
    strike: float
    expiry: pl.Date
    index: str           # -> Spot._key
    disc_curve: str      # -> Curve._key
    surface: str         # -> VolSurface._key


class Butterfly(Instrument):
    """Three strikes, one expiry: long wings, short 2x the body."""

    cp: float
    expiry: pl.Date
    index: str
    disc_curve: str
    surface: str
    low_strike: float
    mid_strike: float
    high_strike: float


@pricer(
    VanillaOption,
    lookups=(
        Lookup(Spot, by="index", at="ref_date", take="value", As="spot"),
        Lookup(Curve, by="disc_curve", at="expiry", take="log_df",
               As="log_df_exp", interp="lerp"),
        Lookup(VolSurface, by="surface", at="expiry",
               take=("atm", "skew", "curv"), interp="lerp"),
    ),
    derive={"tau": pl.business_day_count(pl.col("ref_date"),
                                         pl.col("expiry")) / 252},
)
def vanilla_option(cp, strike, tau, spot, log_df_exp, atm, skew, curv):
    f = spot / exp(log_df_exp)                 # forward
    k = log(strike / f)                        # log-moneyness
    sigma = atm + skew * k + curv * k * k      # the smile
    v = sigma * sqrt(tau)
    d1 = -k / v + 0.5 * v
    d2 = d1 - v
    return exp(log_df_exp) * cp * (
        f * norm_cdf(cp * d1) - strike * norm_cdf(cp * d2))


@price_legs(Butterfly, vanilla_option)
def butterfly(low, mid, high):
    """strike <- low_strike / mid_strike / high_strike by prefix;
    everything else shared; coefficients probed: (+1, -2, +1)."""
    return low - 2 * mid + high
