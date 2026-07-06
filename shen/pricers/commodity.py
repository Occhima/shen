"""Commodity forward — pure calculator, returns the unit value vector."""

from __future__ import annotations

from shen.contracts.instruments import CommodityForward
from shen.contracts.market import Curve, Fx, Spot
from shen.core.math import exp
from shen.core.registry import Lookup, pricer


@pricer(
    CommodityForward,
    lookups=(
        Lookup(Spot, by="index", at="fixing_date", take="value", As="spot"),
        Lookup(Curve, by="disc_curve", at="pay_date", take="log_df",
               As="log_df_pay", interp="lerp"),
        Lookup(Fx, by="ccy_pair", at="pay_date", take="rate", As="fx"),
    ),
)
def commodity_forward(strike, spot, log_df_pay, fx):
    return (spot - strike) * exp(log_df_pay) * fx
