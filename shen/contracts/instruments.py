"""Instrument contracts: economics + join keys, nothing else.

Note where base factors live: index_base / fx_base are TRADE ECONOMICS,
fixed at inception — so they are columns on the instrument contract and
travel through explode(). Forward values are MARKET and arrive via
Lookups. factor = fwd / base is therefore one line of calculator math.

qty is deliberately ABSENT: pricers return unit value; quantity belongs
to shen.positions.
"""

from __future__ import annotations

from typing import ClassVar

import pandera.polars as pa
import polars as pl

from shen.contracts.market import as_lazy


class Instrument(pa.DataFrameModel):
    _fungible_on: ClassVar[tuple[str, ...]] = ()   # columns defining "same contract"
    _averaged: ClassVar[tuple[str, ...]] = ()      # price fields, qty-weighted mean

    instrument_id: str
    instrument_type: str

    class Config:
        strict = "filter"
        coerce = True

    @classmethod
    def validate(cls, frame, *args, **kwargs) -> pl.LazyFrame:  # type: ignore[override]
        return super().validate(as_lazy(frame), *args, **kwargs)


class CommodityForward(Instrument):
    _fungible_on: ClassVar[tuple[str, ...]] = (
        "index", "ccy_pair", "disc_curve", "fixing_date", "pay_date")
    _averaged: ClassVar[tuple[str, ...]] = ("strike",)

    strike: float
    disc_curve: str      # -> Curve._key
    index: str           # -> Spot._key
    ccy_pair: str        # -> Fx._key
    fixing_date: pl.Date
    pay_date: pl.Date


class BulletSwap(Instrument):
    """One row = one swap: an active and a passive bullet leg.

    No notional here — pricers return UNIT value ("some product of
    factors"); notional is a quantity and belongs to shen.positions.
    Base factors (index_base, fx_base) are inception economics, per leg.
    Domestic legs use the identity pair (e.g. BRLBRL, rate 1.0) so the
    calculator stays total — no nullable branches.
    """

    end_date: pl.Date
    active_curve: str
    active_index: str
    active_index_base: float
    active_ccy_pair: str
    active_fx_base: float
    passive_curve: str
    passive_index: str
    passive_index_base: float
    passive_ccy_pair: str
    passive_fx_base: float
