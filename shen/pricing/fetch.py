from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import polars as pl

from shen.domain.contracts.market import MarketContract
from shen.domain.market import Market
from shen.exceptions import LookupResolutionError


@dataclass(frozen=True, slots=True)
class FetchPolicy:
    interpolation: Literal["exact", "previous", "linear"] = "previous"
    extrapolation: Literal["error", "flat"] = "error"
    missing: Literal["error", "null"] = "error"
    tolerance: object | None = None


@dataclass(frozen=True, slots=True)
class Fetch:
    src: type[MarketContract]
    by: str
    at: str
    take: str | tuple[str, ...]
    As: str | tuple[str, ...] | None = None
    policy: FetchPolicy = FetchPolicy()

    @property
    def takes(self):
        return (self.take,) if isinstance(self.take, str) else self.take

    @property
    def aliases(self):
        return (
            self.takes
            if self.As is None
            else ((self.As,) if isinstance(self.As, str) else self.As)
        )

    def __call__(self, lf: pl.LazyFrame, market: Market) -> pl.LazyFrame:
        rhs = market[self.src]
        dims = (
            ["scenario_id"]
            if "scenario_id" in lf.collect_schema().names()
            and "scenario_id" in rhs.collect_schema().names()
            else []
        )
        if len(self.takes) != len(self.aliases):
            raise LookupResolutionError("take/alias mismatch")
        rhs = rhs.select(
            pl.col(self.src._key).alias("_rkey"),
            pl.col(self.src._axis).alias("_rat"),
            *(pl.col(t).alias(f"_rv{i}") for i, t in enumerate(self.takes)),
            *dims,
        ).sort("_rat")
        if self.policy.interpolation == "exact":
            out = lf.join(
                rhs,
                left_on=[self.by, self.at, *dims],
                right_on=["_rkey", "_rat", *dims],
                how="left",
            )
            out = out.rename({f"_rv{i}": a for i, a in enumerate(self.aliases)})
            miss = pl.any_horizontal([pl.col(a).is_null() for a in self.aliases])
            return out.with_columns(
                _lookup_error=pl.coalesce(pl.col("_lookup_error"), pl.lit(False)) | miss
            )
        by_left = [self.by, *dims]
        by_right = ["_rkey", *dims]
        lo = (
            lf.sort(self.at)
            .join_asof(
                rhs,
                left_on=self.at,
                right_on="_rat",
                by_left=by_left,
                by_right=by_right,
                strategy="backward",
            )
            .rename(
                {"_rat": "_t0", **{f"_rv{i}": f"_lo{i}" for i in range(len(self.takes))}}
            )
        )
        if self.policy.interpolation == "previous":
            out = lo.rename({f"_lo{i}": a for i, a in enumerate(self.aliases)}).drop(
                "_t0"
            )
            miss = pl.any_horizontal([pl.col(a).is_null() for a in self.aliases])
            return out.with_columns(
                _lookup_error=pl.coalesce(pl.col("_lookup_error"), pl.lit(False)) | miss
            )
        hi = lo.join_asof(
            rhs,
            left_on=self.at,
            right_on="_rat",
            by_left=by_left,
            by_right=by_right,
            strategy="forward",
        ).rename(
            {"_rat": "_t1", **{f"_rv{i}": f"_hi{i}" for i in range(len(self.takes))}}
        )
        w = (
            (
                (pl.col(self.at) - pl.col("_t0")).dt.total_days()
                / (pl.col("_t1") - pl.col("_t0")).dt.total_days()
            )
            .fill_nan(0)
            .fill_null(0)
        )
        exprs = [
            ((1 - w) * pl.col(f"_lo{i}") + w * pl.col(f"_hi{i}")).alias(a)
            for i, a in enumerate(self.aliases)
        ]
        return hi.with_columns(
            *exprs,
            _lookup_error=pl.coalesce(pl.col("_lookup_error"), pl.lit(False))
            | pl.col("_t0").is_null()
            | pl.col("_t1").is_null(),
        ).drop(
            [
                "_t0",
                "_t1",
                *(f"_lo{i}" for i in range(len(self.takes))),
                *(f"_hi{i}" for i in range(len(self.takes))),
            ]
        )
