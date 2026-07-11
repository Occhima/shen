from shen.contracts.market import Curve, Fx
from shen.contracts.pricing import WdoFutureTerms
from shen.contracts.values import ValueSpec
from shen.core.math import exp
from shen.core.registry import Lookup, LookupPolicy, pricer


@pricer(
    WdoFutureTerms,
    lookups=(
        Lookup(
            Fx, "ccy_pair", "valuation_date", "rate", "spot", LookupPolicy("previous")
        ),
        Lookup(
            Curve, "di_curve", "maturity", "log_df", "log_df_di", LookupPolicy("linear")
        ),
        Lookup(
            Curve,
            "cupom_curve",
            "maturity",
            "log_df",
            "log_df_cupom",
            LookupPolicy("linear"),
        ),
    ),
    output=ValueSpec("present_value", "BRL", "BRL/USD"),
)
def wdo_future(strike, spot, log_df_di, log_df_cupom):
    return spot * exp(log_df_cupom - log_df_di) - strike
