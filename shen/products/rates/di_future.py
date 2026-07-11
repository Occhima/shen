from shen.domain.contracts.instruments import DiFutureTerms
from shen.domain.contracts.market import Curve
from shen.pricing.fetch import Fetch, FetchPolicy
from shen.pricing.math import exp
from shen.pricing.pricer import ValueSpec, pricer

NOTIONAL_PU = 100000.0


@pricer(
    DiFutureTerms,
    lookups=(
        Fetch(
            Curve, "di_curve", "maturity", "log_df", "log_df_mat", FetchPolicy("linear")
        ),
    ),
    output=ValueSpec("present_value", "BRL", "BRL/contract"),
)
def di_future(strike_pu, log_df_mat):
    return NOTIONAL_PU * exp(log_df_mat) - strike_pu
