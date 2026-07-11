from __future__ import annotations

from typing import ClassVar

import pandera.polars as pa
import polars as pl

from shen.contracts.base import ShenFrame


class PricingTerms(ShenFrame):
    product_type: ClassVar[str] = "pricing_terms"
    contract_id: str


class CommodityForwardTerms(PricingTerms):
    product_type: ClassVar[str] = "commodity_forward"
    strike: float
    disc_curve: str
    index: str
    ccy_pair: str
    fixing_date: pl.Date
    pay_date: pl.Date


class DiFutureTerms(PricingTerms):
    product_type: ClassVar[str] = "di_future"
    strike_pu: float
    maturity: pl.Date
    di_curve: str


class WdoFutureTerms(PricingTerms):
    product_type: ClassVar[str] = "wdo_future"
    strike: float
    maturity: pl.Date
    di_curve: str
    cupom_curve: str
    ccy_pair: str


class BulletTerms(PricingTerms):
    product_type: ClassVar[str] = "bullet"
    sign: float = pa.Field(isin=[-1.0, 1.0])
    end_date: pl.Date
    curve: str
    index: str
    index_base: float = pa.Field(gt=0)
    ccy_pair: str
    fx_base: float = pa.Field(gt=0)


class BulletSwapTerms(PricingTerms):
    product_type: ClassVar[str] = "bullet_swap"
    end_date: pl.Date
    active_curve: str
    active_index: str
    active_index_base: float = pa.Field(gt=0)
    active_ccy_pair: str
    active_fx_base: float = pa.Field(gt=0)
    passive_curve: str
    passive_index: str
    passive_index_base: float = pa.Field(gt=0)
    passive_ccy_pair: str
    passive_fx_base: float = pa.Field(gt=0)


class VanillaOptionTerms(PricingTerms):
    product_type: ClassVar[str] = "vanilla_option"
    cp: float = pa.Field(isin=[-1.0, 1.0])
    strike: float = pa.Field(gt=0)
    expiry: pl.Date
    index: str
    disc_curve: str
    surface: str


class ButterflyTerms(PricingTerms):
    product_type: ClassVar[str] = "butterfly"
    cp: float = pa.Field(isin=[-1.0, 1.0])
    expiry: pl.Date
    index: str
    disc_curve: str
    surface: str
    low_strike: float = pa.Field(gt=0)
    mid_strike: float = pa.Field(gt=0)
    high_strike: float = pa.Field(gt=0)


class NdfTerms(PricingTerms):
    product_type: ClassVar[str] = "ndf"
    base_currency: str
    quote_currency: str
    strike: float = pa.Field(gt=0)
    fixing_date: pl.Date
    settlement_date: pl.Date
    reference_id: str
    settlement_curve: str
    settlement_currency: str
