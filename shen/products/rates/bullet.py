from shen.contracts.market import Curve, Fx, Spot
from shen.contracts.pricing import BulletSwapTerms, BulletTerms
from shen.contracts.values import ValueSpec
from shen.core.math import exp
from shen.core.registry import Lookup, LookupPolicy, price_legs, pricer


@pricer(
    BulletTerms,
    lookups=(
        Lookup(
            Spot,
            "index",
            "valuation_date",
            "value",
            "index_fwd",
            LookupPolicy("previous"),
        ),
        Lookup(Fx, "ccy_pair", "end_date", "rate", "fx_fwd", LookupPolicy("previous")),
        Lookup(
            Curve, "curve", "end_date", "log_df", "log_df_end", LookupPolicy("linear")
        ),
    ),
    output=ValueSpec("present_value", "BRL", "unit"),
)
def bullet(sign, index_fwd, index_base, fx_fwd, fx_base, log_df_end):
    return sign * (index_fwd / index_base) * (fx_fwd / fx_base) * exp(log_df_end)


@price_legs(BulletSwapTerms, bullet, weights={"active": 1.0, "passive": -1.0})
def bullet_swap(active, passive):
    return active - passive
