from dataclasses import dataclass

import polars as pl

from shen.contracts.base import as_lazy
from shen.contracts.market import Curve, FxReference
from shen.contracts.position import Position
from shen.contracts.pricing import NdfTerms
from shen.contracts.trade import Trade
from shen.contracts.values import ValueSpec
from shen.core.math import exp
from shen.core.registry import Lookup, LookupPolicy, pricer


@dataclass(frozen=True, slots=True)
class ParsedTrade:
    trade: pl.LazyFrame
    terms: pl.LazyFrame
    positions: pl.LazyFrame


class NdfTicket:
    @staticmethod
    def parse(raw) -> ParsedTrade:
        lf = as_lazy(raw)
        trade = Trade.validate(
            lf.select(
                "trade_id",
                "contract_id",
                "trade_date",
                "book",
                "counterparty",
                "trader",
                "external_id",
            )
        )
        terms = NdfTerms.validate(
            lf.select(
                "contract_id",
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
        pos = Position.validate(lf.select("contract_id", "qty"))
        return ParsedTrade(trade, terms, pos)


@pricer(
    NdfTerms,
    lookups=(
        Lookup(
            FxReference,
            "reference_id",
            "fixing_date",
            "rate",
            "reference_rate",
            LookupPolicy("exact"),
        ),
        Lookup(
            Curve,
            "settlement_curve",
            "settlement_date",
            "log_df",
            "log_df_settlement",
            LookupPolicy("linear"),
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
