from dataclasses import dataclass

import polars as pl

from shen.domain.contracts.base import as_lazy
from shen.domain.contracts.instruments import NdfTerms
from shen.domain.contracts.market import Curve, FxReference
from shen.domain.contracts.tickets import Ticket, TicketComponent
from shen.pricing.fetch import Fetch, FetchPolicy
from shen.pricing.math import exp
from shen.pricing.pricer import ValueSpec, pricer


@dataclass(frozen=True, slots=True)
class ParsedTrade:
    tickets: pl.LazyFrame
    terms: pl.LazyFrame
    components: pl.LazyFrame

    @property
    def trade(self):
        return self.tickets

    @property
    def positions(self):
        return self.components


class NdfTicket:
    @staticmethod
    def parse(raw) -> ParsedTrade:
        lf = as_lazy(raw)
        ticket_id = (
            "ticket_id" if "ticket_id" in lf.collect_schema().names() else "trade_id"
        )
        instrument_id = (
            "instrument_id"
            if "instrument_id" in lf.collect_schema().names()
            else "contract_id"
        )
        qty = "quantity" if "quantity" in lf.collect_schema().names() else "qty"
        tickets = Ticket.validate(
            lf.select(
                pl.col(ticket_id).alias("ticket_id"),
                pl.lit("ndf").alias("ticket_type"),
                "trade_date",
                "book",
                "counterparty",
                "trader",
                "external_id",
            )
        )
        terms = NdfTerms.validate(
            lf.select(
                pl.col(instrument_id).alias("instrument_id"),
                "base_currency",
                "quote_currency",
                "strike",
                "fixing_date",
                "settlement_date",
                "reference_id",
                "settlement_curve",
                "settlement_currency",
            ).with_columns(product_type=pl.lit("ndf"))
        )
        comps = TicketComponent.validate(
            lf.select(
                pl.col(ticket_id).alias("ticket_id"),
                pl.lit("main").alias("component_id"),
                pl.col(instrument_id).alias("instrument_id"),
                pl.col(qty).alias("quantity"),
                pl.col("base_currency").alias("quantity_unit"),
            )
        )
        return ParsedTrade(tickets, terms, comps)


@pricer(
    NdfTerms,
    lookups=(
        Fetch(
            FxReference,
            "reference_id",
            "fixing_date",
            "rate",
            "reference_rate",
            FetchPolicy("exact"),
        ),
        Fetch(
            Curve,
            "settlement_curve",
            "settlement_date",
            "log_df",
            "log_df_settlement",
            FetchPolicy("linear"),
        ),
    ),
    output=ValueSpec(
        "present_value",
        pl.col("settlement_currency"),
        pl.concat_str([pl.col("quote_currency"), pl.lit("/"), pl.col("base_currency")]),
        pl.col("base_currency"),
    ),
)
def ndf(strike, reference_rate, log_df_settlement):
    return (reference_rate - strike) * exp(log_df_settlement)
