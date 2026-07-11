from __future__ import annotations

import polars as pl

from shen.domain.contracts.base import ShenFrame

UNIT_VALUE_GRAIN = (
    "instrument_id",
    "product_type",
    "scenario_id",
    "valuation_date",
    "currency",
    "unit",
    "measure",
)
COMPONENT_VALUE_GRAIN = (
    "ticket_id",
    "component_id",
    "instrument_id",
    "scenario_id",
    "valuation_date",
    "currency",
    "unit",
    "measure",
)
TICKET_VALUE_GRAIN = (
    "ticket_id",
    "scenario_id",
    "valuation_date",
    "currency",
    "unit",
    "measure",
)
POSITION_VALUE_GRAIN = (
    "book",
    "instrument_id",
    "quantity_unit",
    "scenario_id",
    "valuation_date",
    "currency",
    "unit",
    "measure",
)
VALUE_KEYS = UNIT_VALUE_GRAIN


class UnitValue(ShenFrame):
    instrument_id: str
    product_type: str
    scenario_id: str
    valuation_date: pl.Date
    measure: str
    currency: str
    unit: str
    value: float


class ComponentValue(ShenFrame):
    ticket_id: str
    component_id: str
    instrument_id: str
    scenario_id: str
    valuation_date: pl.Date
    measure: str
    currency: str
    unit: str
    quantity: float
    quantity_unit: str
    value: float


class TicketValue(ShenFrame):
    ticket_id: str
    scenario_id: str
    valuation_date: pl.Date
    measure: str
    currency: str
    unit: str
    value: float


class PositionValue(ShenFrame):
    book: str
    instrument_id: str
    quantity_unit: str
    scenario_id: str
    valuation_date: pl.Date
    measure: str
    currency: str
    unit: str
    quantity: float
    value: float
