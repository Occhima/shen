from shen.contracts.market import Curve
from shen.contracts.pricing import DiFutureTerms
from shen.contracts.values import ValueSpec
from shen.core.math import exp
from shen.core.registry import Lookup, LookupPolicy, pricer

NOTIONAL_PU = 100000.0


@pricer(
    DiFutureTerms,
    lookups=(
        Lookup(
            Curve, "di_curve", "maturity", "log_df", "log_df_mat", LookupPolicy("linear")
        ),
    ),
    output=ValueSpec("present_value", "BRL", "BRL/contract"),
)
def di_future(strike_pu, log_df_mat):
    return NOTIONAL_PU * exp(log_df_mat) - strike_pu
