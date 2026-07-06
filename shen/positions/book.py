"""Book — positions across many instrument kinds, and fungibilidade.

A Book is (trades, positions): the union-of-columns instrument frame
plus canonical net positions. It adds NO pricing machinery — value/mtm/
pnl compose price() and positions.* — only the portfolio algebra.

Fungibilidade de contratos: lots of the SAME contract merge into one
position, entry averaged by notional. "Same contract" is a CONTRACT
characteristic — Instrument families declare

    _fungible_on = ("maturity", "di_curve")   # economic identity
    _averaged    = ("strike_pu",)             # qty-weighted entries

and consolidate() is generic over any of them. The algebra preserves
MTM exactly: sum(q_i*(V - K_i)) == (sum q_i)*(V - Kbar) with Kbar the
qty-weighted mean — pinned by test. Lots netting to zero qty vanish:
closing trades cancel, which is the point of fungibilidade.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from shen.contracts.market import as_lazy
from shen.core.engine import Market, price
from shen.core.registry import _snake
from shen.positions import Position
from shen.positions import mtm as _mtm
from shen.positions import pnl as _pnl


def consolidate(contract, trades: pl.LazyFrame,
                positions: pl.LazyFrame) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Merge fungible lots of one instrument family.

    Returns (trades', positions') for that family only: one row per
    economic contract, qty netted, averaged fields notional-weighted,
    deterministic synthetic instrument_id from the identity columns.
    """
    itype = _snake(contract.__name__)
    keys = list(contract._fungible_on)
    avg = list(contract._averaged)
    if not keys:
        raise TypeError(f"{contract.__name__} declares no _fungible_on")
    if bad := {"instrument_id", "instrument_type"} & set(keys):
        raise TypeError(
            f"{contract.__name__}: _fungible_on must be economic columns, "
            f"not identity ({bad}) — consolidate() already scopes by family")
    if missing := set(keys + avg) - set(contract.to_schema().columns):
        raise TypeError(f"{contract.__name__}: _fungible_on/_averaged "
                        f"reference missing columns {missing}")

    lots = (contract.validate(
                as_lazy(trades).filter(pl.col("instrument_type") == itype))
            .join(positions, on="instrument_id", how="inner"))

    schema_cols = list(contract.to_schema().columns)
    rest = [c for c in schema_cols
            if c not in (*keys, *avg, "instrument_id", "instrument_type")]

    merged = (
        lots.group_by(keys)
        .agg(
            pl.col("qty").sum(),
            *(((pl.col(f) * pl.col("qty")).sum() / pl.col("qty").sum()).alias(f)
              for f in avg),
            *(pl.col(c).first() for c in rest),
        )
        .filter(pl.col("qty") != 0)                       # closed = gone
        .with_columns(
            instrument_id=pl.concat_str(
                [pl.lit(itype), *(pl.col(k).cast(pl.String) for k in keys)],
                separator=":"),
            instrument_type=pl.lit(itype),
        )
    )
    return merged.select(schema_cols), merged.select("instrument_id", "qty")


@dataclass(frozen=True, slots=True)
class Book:
    trades: pl.LazyFrame
    positions: pl.LazyFrame

    @classmethod
    def load(cls, trades, positions) -> "Book":
        return cls(as_lazy(trades), Position.validate(positions))

    # -- valuation: pure composition, nothing new ----------------------
    def value(self, mkt: Market | None = None, *, pricers=None) -> pl.LazyFrame:
        return price(self.trades, mkt, pricers=pricers)

    def mtm(self, mkt: Market | None = None, *, pricers=None) -> pl.LazyFrame:
        return _mtm(self.positions, self.value(mkt, pricers=pricers))

    def pnl(self, now: Market, previous: Market, *, pricers=None) -> pl.LazyFrame:
        return _pnl(self.mtm(now, pricers=pricers),
                    self.mtm(previous, pricers=pricers))

    # -- fungibilidade --------------------------------------------------
    def consolidate(self, *contracts) -> "Book":
        """New Book with fungible lots merged for the given families;
        every other instrument passes through untouched."""
        covered = [_snake(c.__name__) for c in contracts]
        rest_t = self.trades.filter(~pl.col("instrument_type").is_in(covered))
        rest_p = self.positions.join(
            rest_t.select("instrument_id"), on="instrument_id", how="semi")
        ts, ps = [rest_t], [rest_p]
        for c in contracts:
            t2, p2 = consolidate(c, self.trades, self.positions)
            ts.append(t2)
            ps.append(p2)
        return Book(pl.concat(ts, how="diagonal"), pl.concat(ps))
