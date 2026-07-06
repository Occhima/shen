"""Bullet swap = structure over bullets.

The swap's unit value is "just a product of factors" per leg:

    sign * (index_fwd / index_base) * (fx_fwd / fx_base) * exp(log_df)

@pricer(Bullet) owns that valuation — and registers "bullet" as a
priceable type in its own right. @price_legs(BulletSwap) owns only the
structure: explode each swap row into two signed Bullet rows, reduce
leg values back by sum (active - passive via the sign). Notional is
absent by design: it is quantity, i.e. plib.positions' business.
"""

from __future__ import annotations

import polars as pl

from plib.contracts.instruments import BulletSwap, Instrument
from plib.contracts.market import Curve, Fx, Spot
from plib.math import exp
from plib.registry import Lookup, price_legs, pricer


class Bullet(Instrument):
    """Leg-grain contract: one indexed, currency-linked zero bullet."""

    sign: float
    end_date: pl.Date
    curve: str
    index: str
    index_base: float
    ccy_pair: str
    fx_base: float


@pricer(
    Bullet,
    lookups=(
        Lookup(Spot, by="index", at="ref_date", take="value", As="index_fwd"),
        Lookup(Fx, by="ccy_pair", at="end_date", take="rate",
               As="fx_fwd", interp="lerp"),
        Lookup(Curve, by="curve", at="end_date", take="log_df",
               As="log_df_end", interp="lerp"),
    ),
)
def bullet(sign, index_fwd, index_base, fx_fwd, fx_base, log_df_end):
    return (sign
            * (index_fwd / index_base)
            * (fx_fwd / fx_base)
            * exp(log_df_end))


@price_legs(BulletSwap, bullet)
def bullet_swap(active, passive):
    """The structure IS this function: parameter names declare the leg
    prefixes, the body declares the fold. Everything else — signs,
    explode, drift checks — is derived from it at import."""
    return active - passive
