from __future__ import annotations

from typing import ClassVar

import pandera.polars as pa
import polars as pl

from shen.domain.contracts.base import ShenFrame


class InstrumentTerms(ShenFrame):
    _product_type: ClassVar[str]
    instrument_id: str = pa.Field(str_length={"min_value": 1}, unique=True)
    product_type: str


class CommodityForwardTerms(InstrumentTerms):
    _product_type: ClassVar[str] = "commodity_forward"
    strike: float
    disc_curve: str
    index_id: str
    ccy_pair: str
    fixing_date: pl.Date
    payment_date: pl.Date


class DiFutureTerms(InstrumentTerms):
    _product_type: ClassVar[str] = "di_future"
    strike_pu: float
    maturity: pl.Date
    di_curve: str


class WdoFutureTerms(InstrumentTerms):
    _product_type: ClassVar[str] = "wdo_future"
    strike: float
    maturity: pl.Date
    di_curve: str
    cupom_curve: str
    ccy_pair: str


class BulletTerms(InstrumentTerms):
    _product_type: ClassVar[str] = "bullet"
    end_date: pl.Date
    curve: str
    index_id: str
    index_base: float = pa.Field(gt=0)
    ccy_pair: str
    fx_base: float = pa.Field(gt=0)


class VanillaOptionTerms(InstrumentTerms):
    _product_type: ClassVar[str] = "vanilla_option"
    cp: float = pa.Field(isin=[-1.0, 1.0])
    strike: float = pa.Field(gt=0)
    expiry: pl.Date
    index_id: str
    disc_curve: str
    surface: str


class NdfTerms(InstrumentTerms):
    _product_type: ClassVar[str] = "ndf"
    base_currency: str
    quote_currency: str
    strike: float = pa.Field(gt=0)
    fixing_date: pl.Date
    settlement_date: pl.Date
    reference_id: str
    settlement_curve: str
    settlement_currency: str
