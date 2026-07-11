import polars as pl

from shen.domain.contracts.instruments import VanillaOptionTerms
from shen.domain.contracts.market import Curve, IndexObservation, VolSurface
from shen.pricing.fetch import Fetch, FetchPolicy
from shen.pricing.math import exp, log, norm_cdf, sqrt, where
from shen.pricing.pricer import ValueSpec, pricer


@pricer(
    VanillaOptionTerms,
    lookups=(
        Fetch(
            IndexObservation,
            "index_id",
            "valuation_date",
            "value",
            "spot",
            FetchPolicy("previous"),
        ),
        Fetch(
            Curve, "disc_curve", "expiry", "log_df", "log_df_exp", FetchPolicy("linear")
        ),
        Fetch(
            VolSurface,
            "surface",
            "expiry",
            ("atm", "skew", "curvature"),
            None,
            FetchPolicy("linear"),
        ),
    ),
    derive={
        "tau": pl.business_day_count(pl.col("valuation_date"), pl.col("expiry")) / 252
    },
    output=ValueSpec("present_value", "BRL", "BRL/option"),
)
def vanilla_option(cp, strike, tau, spot, log_df_exp, atm, skew, curvature):
    f = spot / exp(log_df_exp)
    k = log(strike / f)
    sigma = atm + skew * k + curvature * k * k
    v = sigma * sqrt(tau)
    d1 = -k / v + 0.5 * v
    d2 = d1 - v
    df = exp(log_df_exp)
    call = df * (f * norm_cdf(d1) - strike * norm_cdf(d2))
    put = df * (strike * norm_cdf(-d2) - f * norm_cdf(-d1))
    return where(cp > 0, call, put)
