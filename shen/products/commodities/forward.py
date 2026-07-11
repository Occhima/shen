from shen.domain.contracts.instruments import CommodityForwardTerms
from shen.domain.contracts.market import Curve, Fx, IndexObservation
from shen.pricing.fetch import Fetch, FetchPolicy
from shen.pricing.math import exp
from shen.pricing.pricer import ValueSpec, pricer


@pricer(
    CommodityForwardTerms,
    lookups=(
        Fetch(
            IndexObservation,
            "index_id",
            "fixing_date",
            "value",
            "spot",
            FetchPolicy("previous"),
        ),
        Fetch(
            Curve,
            "disc_curve",
            "payment_date",
            "log_df",
            "log_df_pay",
            FetchPolicy("linear"),
        ),
        Fetch(Fx, "ccy_pair", "payment_date", "rate", "fx", FetchPolicy("previous")),
    ),
    output=ValueSpec("present_value", "USD", "currency/index"),
)
def commodity_forward(strike, spot, log_df_pay, fx):
    return (spot - strike) * exp(log_df_pay) * fx
