from __future__ import annotations

import pandera.polars as pa
import polars as pl

from shen.domain.contracts.base import ShenFrame, as_lazy


class Ticket(ShenFrame):
    ticket_id: str = pa.Field(str_length={"min_value": 1}, unique=True)
    ticket_type: str = pa.Field(str_length={"min_value": 1})
    trade_date: pl.Date
    book: str = pa.Field(str_length={"min_value": 1})
    counterparty: str = pa.Field(nullable=True, default=None)
    trader: str = pa.Field(nullable=True, default=None)
    external_id: str = pa.Field(nullable=True, default=None)


class TicketComponent(ShenFrame):
    ticket_id: str = pa.Field(str_length={"min_value": 1})
    component_id: str = pa.Field(str_length={"min_value": 1})
    instrument_id: str = pa.Field(str_length={"min_value": 1})
    quantity: float = pa.Field(ne=0)
    quantity_unit: str = pa.Field(str_length={"min_value": 1})
    role: str = pa.Field(nullable=True, default=None)


class Position(ShenFrame):
    book: str = pa.Field(str_length={"min_value": 1})
    instrument_id: str = pa.Field(str_length={"min_value": 1})
    quantity: float
    quantity_unit: str = pa.Field(str_length={"min_value": 1})


def _fail_if_any(lf: pl.LazyFrame, msg: str) -> None:
    if lf.limit(1).collect().height:
        raise ValueError(msg)


def validate_ticket_book(tickets, components, instruments):
    t = Ticket.validate(tickets)
    c = TicketComponent.validate(components)
    i = as_lazy(instruments)
    _fail_if_any(
        c.group_by("ticket_id", "component_id").len().filter(pl.col("len") > 1),
        "duplicate ticket components",
    )
    _fail_if_any(
        i.group_by("instrument_id").len().filter(pl.col("len") > 1),
        "duplicate instruments",
    )
    _fail_if_any(
        t.join(c.select("ticket_id").unique(), on="ticket_id", how="anti"),
        "ticket without components",
    )
    _fail_if_any(
        c.join(t.select("ticket_id"), on="ticket_id", how="anti"),
        "component references missing ticket",
    )
    _fail_if_any(
        c.join(i.select("instrument_id"), on="instrument_id", how="anti"),
        "component references missing instrument",
    )
    return t, c, i
