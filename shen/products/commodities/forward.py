from shen.contracts.market import Curve, Fx, Spot
from shen.contracts.pricing import CommodityForwardTerms
from shen.contracts.values import ValueSpec
from shen.core.math import exp
from shen.core.registry import Lookup, LookupPolicy, pricer


@pricer(
    CommodityForwardTerms,
    lookups=(
        Lookup(Spot, "index", "fixing_date", "value", "spot", LookupPolicy("previous")),
        Lookup(
            Curve,
            "disc_curve",
            "pay_date",
            "log_df",
            "log_df_pay",
            LookupPolicy("linear"),
        ),
        Lookup(Fx, "ccy_pair", "pay_date", "rate", "fx", LookupPolicy("previous")),
    ),
    output=ValueSpec("present_value", "USD", "currency/index"),
)
def commodity_forward(strike, spot, log_df_pay, fx):
    return (spot - strike) * exp(log_df_pay) * fx
