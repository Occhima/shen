"""WDO (B3 mini US dollar futures) — the "I have all the data" case.

Nothing new is needed. Covered interest parity over the objects that
already exist: fwd = spot * DF_cupom / DF_DI, i.e. in log space
spot * exp(log_df_cupom - log_df_di). Daily settlement means the
mark-to-market difference is undiscounted: unit value = fwd - strike,
in quote points (qty carries contracts x point value, in positions).

Note the two Lookups against the SAME Curve contract with different
`by` columns — multi-curve is just two joins, not a framework feature.
"""

from __future__ import annotations

from typing import ClassVar

import polars as pl

from plib.contracts.instruments import Instrument
from plib.contracts.market import Curve, Fx
from plib.math import exp
from plib.registry import Lookup, pricer


class WdoFuture(Instrument):
    _fungible_on: ClassVar[tuple[str, ...]] = (
        "ccy_pair", "maturity", "di_curve", "cupom_curve")
    _averaged: ClassVar[tuple[str, ...]] = ("strike",)

    strike: float        # negotiated price, BRL per USD (quote points)
    maturity: pl.Date
    di_curve: str        # domestic discounting (DI/PRE)
    cupom_curve: str     # cupom cambial (onshore USD)
    ccy_pair: str        # USDBRL


@pricer(
    WdoFuture,
    lookups=(
        Lookup(Fx, by="ccy_pair", at="ref_date", take="rate", As="spot"),
        Lookup(Curve, by="di_curve", at="maturity", take="log_df",
               As="log_df_di", interp="lerp"),
        Lookup(Curve, by="cupom_curve", at="maturity", take="log_df",
               As="log_df_cupom", interp="lerp"),
    ),
)
def wdo_future(strike, spot, log_df_di, log_df_cupom):
    fwd = spot * exp(log_df_cupom - log_df_di)
    return fwd - strike
