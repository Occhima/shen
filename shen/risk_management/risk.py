from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import polars as pl

from shen.contracts.market import Curve, Market
from shen.contracts.values import VALUE_KEYS
from shen.core.engine import price
from shen.core.registry import PricingUniverse


@dataclass(frozen=True, slots=True)
class RiskFactorSpec:
    name: str
    bump_type: Literal["absolute", "relative"]
    bump_size: float
    unit: str


def sensitivities(
    terms,
    market: Market,
    *,
    universe: PricingUniverse,
    spec: RiskFactorSpec | None = None,
):
    spec = spec or RiskFactorSpec("factor", "absolute", 1e-4, "unit")
    base = price(terms, market, universe=universe).with_columns(
        factor=pl.lit(spec.name), dv=pl.lit(0.0)
    )
    return base.select(*VALUE_KEYS, "value", "factor", "dv")


def dv01(
    terms, market: Market, *, universe: PricingUniverse, curve=Curve, bps: float = 1.0
):
    yf = (pl.col(curve.at) - pl.lit(market.valuation_date)).dt.total_days() / 365
    up = market[curve].with_columns(
        scenario_id=pl.lit("up"), log_df=pl.col("log_df") - bps * 1e-4 * yf
    )
    dn = market[curve].with_columns(
        scenario_id=pl.lit("down"), log_df=pl.col("log_df") + bps * 1e-4 * yf
    )
    shocked = market.with_frame(curve, pl.concat([market[curve], up, dn]))
    vals = price(terms, shocked, universe=universe)
    idx = [k for k in VALUE_KEYS if k != "scenario_id"]
    return vals.group_by(idx).agg(
        (
            (
                pl.col("value").filter(pl.col("scenario_id") == "up").first()
                - pl.col("value").filter(pl.col("scenario_id") == "down").first()
            )
            / (2 * bps)
        ).alias("dv01")
    )
