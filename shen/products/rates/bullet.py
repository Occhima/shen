from shen.domain.contracts.instruments import BulletTerms
from shen.domain.contracts.market import Curve, Fx, IndexObservation
from shen.pricing.fetch import Fetch, FetchPolicy
from shen.pricing.math import exp
from shen.pricing.pricer import ValueSpec, pricer


@pricer(
    BulletTerms,
    lookups=(
        Fetch(
            IndexObservation,
            "index_id",
            "end_date",
            "value",
            "index_fwd",
            FetchPolicy("previous"),
        ),
        Fetch(Fx, "ccy_pair", "end_date", "rate", "fx_fwd", FetchPolicy("previous")),
        Fetch(Curve, "curve", "end_date", "log_df", "log_df_end", FetchPolicy("linear")),
    ),
    output=ValueSpec("present_value", "BRL", "unit"),
)
def bullet(index_fwd, index_base, fx_fwd, fx_base, log_df_end):
    return (index_fwd / index_base) * (fx_fwd / fx_base) * exp(log_df_end)
