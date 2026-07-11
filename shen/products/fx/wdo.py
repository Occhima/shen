from shen.domain.contracts.instruments import WdoFutureTerms
from shen.domain.contracts.market import Curve, Fx
from shen.pricing.fetch import Fetch, FetchPolicy
from shen.pricing.math import exp
from shen.pricing.pricer import ValueSpec, pricer


@pricer(
    WdoFutureTerms,
    lookups=(
        Fetch(Fx, "ccy_pair", "valuation_date", "rate", "spot", FetchPolicy("previous")),
        Fetch(
            Curve, "di_curve", "maturity", "log_df", "log_df_di", FetchPolicy("linear")
        ),
        Fetch(
            Curve,
            "cupom_curve",
            "maturity",
            "log_df",
            "log_df_cupom",
            FetchPolicy("linear"),
        ),
    ),
    output=ValueSpec("present_value", "BRL", "BRL/USD"),
)
def wdo_future(strike, spot, log_df_di, log_df_cupom):
    return spot * exp(log_df_cupom - log_df_di) - strike
