"""DI1 futures (B3) — one lookup, one line of math.

PU convention: the theoretical unit price is the discounted notional,
PU = 100_000 * DF_DI(maturity) = 100_000 * exp(log_df). Entry is stored
as the traded PU (strike_pu), so unit value = PU_theo - strike_pu, in
BRL per contract — daily settlement, undiscounted. Position qty is
signed contracts (long PU = receiving fixed = "dado"; short = "tomado").

NOTE: exact B3 PU uses bus/252 day count on the traded rate; storing
entry as PU sidesteps that until the calendar module exists. The curve
itself should be bootstrapped bus/252 upstream — the contract only
requires discount factors.
"""

from __future__ import annotations

from typing import ClassVar

import polars as pl

from plib.contracts.instruments import Instrument
from plib.contracts.market import Curve
from plib.math import exp
from plib.registry import Lookup, pricer

NOTIONAL_PU = 100_000.0


class DiFuture(Instrument):
    # fungibilidade: same curve + same maturity = same exchange contract;
    # entry PU merges as the qty-weighted preço médio
    _fungible_on: ClassVar[tuple[str, ...]] = ("maturity", "di_curve")
    _averaged: ClassVar[tuple[str, ...]] = ("strike_pu",)

    strike_pu: float     # entry PU (traded price)
    maturity: pl.Date
    di_curve: str


@pricer(
    DiFuture,
    lookups=(
        Lookup(Curve, by="di_curve", at="maturity", take="log_df",
               As="log_df_mat", interp="lerp"),
    ),
)
def di_future(strike_pu, log_df_mat):
    return NOTIONAL_PU * exp(log_df_mat) - strike_pu
