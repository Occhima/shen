from __future__ import annotations

import polars as pl

from shen.domain.contracts.market import Curve
from shen.domain.contracts.values import UNIT_VALUE_GRAIN
from shen.domain.market import Market
from shen.pricing.engine import price
from shen.pricing.pricer import Registry


def dv01(terms, market: Market, *, universe: Registry, curve=Curve, bps: float = 1.0):
    yf = (pl.col(curve._axis) - pl.lit(market.valuation_date)).dt.total_days() / 365
    base = market[curve]
    up = base.with_columns(
        scenario_id=pl.lit("up"), log_df=pl.col("log_df") - bps * 1e-4 * yf
    )
    dn = base.with_columns(
        scenario_id=pl.lit("down"), log_df=pl.col("log_df") + bps * 1e-4 * yf
    )
    shocked = market.with_frame(curve, pl.concat([base, up, dn]))
    vals = price(terms, shocked, universe=universe)
    idx = [k for k in UNIT_VALUE_GRAIN if k != "scenario_id"]
    return vals.group_by(idx).agg(
        (
            (
                pl.col("value").filter(pl.col("scenario_id") == "up").first()
                - pl.col("value").filter(pl.col("scenario_id") == "down").first()
            )
            / (2 * bps)
        ).alias("dv01")
    )
