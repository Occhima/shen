import polars as pl

from shen.contracts.market import Curve, Spot, VolSurface
from shen.contracts.pricing import ButterflyTerms, VanillaOptionTerms
from shen.contracts.values import ValueSpec
from shen.core.math import exp, log, norm_cdf, sqrt, where
from shen.core.registry import Lookup, LookupPolicy, price_legs, pricer


@pricer(
    VanillaOptionTerms,
    lookups=(
        Lookup(
            Spot, "index", "valuation_date", "value", "spot", LookupPolicy("previous")
        ),
        Lookup(
            Curve, "disc_curve", "expiry", "log_df", "log_df_exp", LookupPolicy("linear")
        ),
        Lookup(
            VolSurface,
            "surface",
            "expiry",
            ("atm", "skew", "curv"),
            None,
            LookupPolicy("linear"),
        ),
    ),
    derive={
        "tau": pl.business_day_count(pl.col("valuation_date"), pl.col("expiry")) / 252
    },
    output=ValueSpec("present_value", "BRL", "BRL/option"),
)
def vanilla_option(cp, strike, tau, spot, log_df_exp, atm, skew, curv):
    f = spot / exp(log_df_exp)
    k = log(strike / f)
    sigma = atm + skew * k + curv * k * k
    v = sigma * sqrt(tau)
    d1 = -k / v + 0.5 * v
    d2 = d1 - v
    df = exp(log_df_exp)
    call = df * (f * norm_cdf(d1) - strike * norm_cdf(d2))
    put = df * (strike * norm_cdf(-d2) - f * norm_cdf(-d1))
    return where(cp > 0, call, put)


@price_legs(
    ButterflyTerms, vanilla_option, weights={"low": 1.0, "mid": -2.0, "high": 1.0}
)
def butterfly(low, mid, high):
    return low - 2 * mid + high
