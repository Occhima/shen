from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from shen.contracts.base import ShenFrame

VALUE_KEYS = (
    "contract_id",
    "product_type",
    "scenario_id",
    "valuation_date",
    "currency",
    "unit",
    "measure",
)


@dataclass(frozen=True, slots=True)
class ValueSpec:
    measure: str
    currency: str | pl.Expr
    unit: str | pl.Expr
    quantity_unit: str | pl.Expr | None = None


class InstrumentValue(ShenFrame):
    contract_id: str
    product_type: str
    scenario_id: str
    valuation_date: pl.Date
    measure: str
    currency: str
    unit: str
    value: float


class Valuation(InstrumentValue):
    qty: float
    mtm: float
