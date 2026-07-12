from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal
from uuid import uuid4

import polars as pl

from shen.domain.contracts.market import MarketContract
from shen.domain.market import Market

type FetchMode = Literal["exact", "previous", "next", "linear"]
type Extrapolation = Literal["error", "flat"]


@dataclass(frozen=True, slots=True)
class FetchPolicy:
    mode: FetchMode = "previous"
    extrapolation: Extrapolation = "error"

    @classmethod
    def exact(cls) -> FetchPolicy:
        return cls("exact")

    @classmethod
    def previous(cls) -> FetchPolicy:
        return cls("previous")

    @classmethod
    def next(cls) -> FetchPolicy:
        return cls("next")

    @classmethod
    def linear(cls, *, extrapolation: Extrapolation = "error") -> FetchPolicy:
        return cls("linear", extrapolation)


@dataclass(frozen=True, slots=True)
class Fetch:
    source: type[MarketContract]
    at: str
    take: str | Mapping[str, str]
    on: str | Mapping[str, str] | None = None
    policy: FetchPolicy = FetchPolicy()

    def __post_init__(self) -> None:
        if isinstance(self.take, str):
            take = {self.take: self.take}
        else:
            take = dict(self.take)
        if not take:
            raise ValueError("Fetch.take cannot be empty")
        object.__setattr__(self, "take", MappingProxyType(take))
        if self.on is None:
            key = self.source._key
            on = {key: key}
        elif isinstance(self.on, str):
            on = {self.on: self.on}
        else:
            on = dict(self.on)
        object.__setattr__(self, "on", MappingProxyType(on))

    @property
    def outputs(self) -> tuple[str, ...]:
        return tuple(self.take.values())

    @property
    def right_axis(self) -> str:
        axis = getattr(self.source, "_at", None) or getattr(self.source, "_axis", None)
        if axis is None:
            raise TypeError(f"{self.source.__name__} must define _at or _axis")
        return axis

    def __call__(self, rows: pl.LazyFrame, market: Market) -> pl.LazyFrame:
        left_names = rows.collect_schema().names()
        right = market[self.source]
        right_names = right.collect_schema().names()
        missing_left = (set(self.on) | {self.at}) - set(left_names)
        missing_right = (set(self.on.values()) | set(self.take) | {self.right_axis}) - set(
            right_names
        )
        if missing_left:
            raise KeyError(f"Fetch missing left fields: {sorted(missing_left)}")
        if missing_right:
            raise KeyError(f"Fetch missing market fields: {sorted(missing_right)}")
        dimensions = [
            name
            for name in ("scenario_id", "valuation_date")
            if name in left_names and name in right_names
        ]
        prefix = f"__fetch_{uuid4().hex}"
        row_id = f"{prefix}_row"
        right_key_names = [f"{prefix}_key_{index}" for index in range(len(self.on))]
        right_value_names = [f"{prefix}_value_{index}" for index in range(len(self.take))]
        right_axis = f"{prefix}_axis"
        left_with_id = rows.with_row_index(row_id)
        market_rows = right.select(
            *(
                pl.col(name).alias(alias)
                for name, alias in zip(self.on.values(), right_key_names, strict=True)
            ),
            pl.col(self.right_axis).alias(right_axis),
            *(
                pl.col(name).alias(alias)
                for name, alias in zip(self.take, right_value_names, strict=True)
            ),
            *dimensions,
        ).sort([*right_key_names, *dimensions, right_axis])
        if self.policy.mode == "exact":
            result = self._exact(
                left_with_id,
                market_rows,
                row_id,
                right_key_names,
                right_axis,
                right_value_names,
                dimensions,
            )
        elif self.policy.mode in ("previous", "next"):
            result = self._asof(
                left_with_id,
                market_rows,
                row_id,
                right_key_names,
                right_axis,
                right_value_names,
                dimensions,
                self.policy.mode,
            )
        else:
            result = self._linear(
                left_with_id,
                market_rows,
                row_id,
                right_key_names,
                right_axis,
                right_value_names,
                dimensions,
                prefix,
            )
        outputs = list(self.take.values())
        missing = pl.any_horizontal(*(pl.col(name).is_null() for name in outputs))
        previous_error = (
            pl.col("_lookup_error")
            if "_lookup_error" in result.collect_schema().names()
            else pl.lit(False)
        )
        base_names = [
            name
            for name in left_names
            if name not in outputs and name != "_lookup_error"
        ]
        return (
            result.with_columns(_lookup_error=previous_error.fill_null(False) | missing)
            .sort(row_id)
            .drop(row_id)
            .select(*base_names, *outputs, "_lookup_error")
        )

    def _exact(
        self,
        left: pl.LazyFrame,
        right: pl.LazyFrame,
        row_id: str,
        right_keys: list[str],
        right_axis: str,
        right_values: list[str],
        dimensions: list[str],
    ) -> pl.LazyFrame:
        result = left.join(
            right,
            left_on=[*self.on, self.at, *dimensions],
            right_on=[*right_keys, right_axis, *dimensions],
            how="left",
        )
        return result.rename(dict(zip(right_values, self.take.values(), strict=True)))

    def _asof(
        self,
        left: pl.LazyFrame,
        right: pl.LazyFrame,
        row_id: str,
        right_keys: list[str],
        right_axis: str,
        right_values: list[str],
        dimensions: list[str],
        mode: Literal["previous", "next"],
    ) -> pl.LazyFrame:
        result = left.sort(self.at).join_asof(
            right,
            left_on=self.at,
            right_on=right_axis,
            by_left=[*self.on, *dimensions],
            by_right=[*right_keys, *dimensions],
            strategy="backward" if mode == "previous" else "forward",
        )
        return result.rename(dict(zip(right_values, self.take.values(), strict=True))).drop(
            right_axis
        )

    def _linear(
        self,
        left: pl.LazyFrame,
        right: pl.LazyFrame,
        row_id: str,
        right_keys: list[str],
        right_axis: str,
        right_values: list[str],
        dimensions: list[str],
        prefix: str,
    ) -> pl.LazyFrame:
        lower_axis = f"{prefix}_lower_axis"
        upper_axis = f"{prefix}_upper_axis"
        lower_values = [f"{name}_lower" for name in right_values]
        upper_values = [f"{name}_upper" for name in right_values]
        lower = left.sort(self.at).join_asof(
            right.rename(
                {right_axis: lower_axis, **dict(zip(right_values, lower_values, strict=True))}
            ),
            left_on=self.at,
            right_on=lower_axis,
            by_left=[*self.on, *dimensions],
            by_right=[*right_keys, *dimensions],
            strategy="backward",
        )
        upper_right = right.rename(
            {right_axis: upper_axis, **dict(zip(right_values, upper_values, strict=True))}
        )
        joined = lower.join_asof(
            upper_right,
            left_on=self.at,
            right_on=upper_axis,
            by_left=[*self.on, *dimensions],
            by_right=[*right_keys, *dimensions],
            strategy="forward",
        )
        x = pl.col(self.at).cast(pl.Int64).cast(pl.Float64)
        x0 = pl.col(lower_axis).cast(pl.Int64).cast(pl.Float64)
        x1 = pl.col(upper_axis).cast(pl.Int64).cast(pl.Float64)
        weight = (x - x0) / (x1 - x0)
        expressions: list[pl.Expr] = []
        for lower_name, upper_name, output in zip(
            lower_values, upper_values, self.take.values(), strict=True
        ):
            observed = pl.col(lower_name)
            interpolated = observed + weight * (pl.col(upper_name) - observed)
            if self.policy.extrapolation == "flat":
                interpolated = pl.coalesce(interpolated, pl.col(lower_name), pl.col(upper_name))
            expressions.append(
                pl.when(x0 == x1).then(observed).otherwise(interpolated).alias(output)
            )
        return joined.with_columns(*expressions).drop(
            lower_axis, upper_axis, *lower_values, *upper_values
        )


__all__ = ["Fetch", "FetchPolicy"]
