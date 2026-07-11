from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, ClassVar, overload

import pandera.polars as pa
import polars as pl

type FrameLike = pl.DataFrame | pl.LazyFrame


def as_lazy(frame: Any) -> pl.LazyFrame:
    if isinstance(frame, pl.LazyFrame):
        return frame
    if isinstance(frame, pl.DataFrame):
        return frame.lazy()
    if frame.__class__.__module__.startswith("pandas"):
        return pl.from_pandas(frame).lazy()
    return pl.DataFrame(frame).lazy()


def _as_eager(frame: Any) -> pl.DataFrame:
    return (
        frame.collect() if isinstance(frame, pl.LazyFrame) else as_lazy(frame).collect()
    )


class _Derived:
    def __init__(
        self,
        outputs: tuple[str, ...],
        fn: Callable[[pl.LazyFrame], pl.Expr | Sequence[pl.Expr]],
    ):
        self.outputs = outputs
        self.fn = fn
        fn._shen_derived_outputs = outputs

    def __get__(self, obj: object, objtype: type | None = None):
        return self.fn


@overload
def derived(
    output: str, /, *outputs: str
) -> Callable[[Callable[[pl.LazyFrame], pl.Expr | Sequence[pl.Expr]]], _Derived]: ...


def derived(output: str, /, *outputs: str):
    """Declare vectorized derived columns.

    Derivations run in class declaration order after inherited derivations. Each
    derivation is applied in its own ``with_columns`` call, so declaration order
    is the dependency order; no topological sorting is performed.
    """
    names = (output, *outputs)

    def deco(fn: Callable[[pl.LazyFrame], pl.Expr | Sequence[pl.Expr]]) -> _Derived:
        return _Derived(names, fn)

    return deco


class ShenFrame(pa.DataFrameModel):
    _product_type: ClassVar[str]
    _key: ClassVar[str]
    _axis: ClassVar[str]
    _value_columns: ClassVar[tuple[str, ...]] = ()

    class Config:
        strict = "filter"
        coerce = True
        unique_column_names = True
        add_missing_columns = True

    @classmethod
    def _derivations(cls):
        items = []
        for base in reversed(cls.mro()):
            for obj in getattr(base, "__dict__", {}).values():
                if isinstance(obj, _Derived):
                    items.append(obj)
        return items

    @classmethod
    def _apply_derivations(cls, lf: pl.LazyFrame) -> pl.LazyFrame:
        for d in cls._derivations():
            exprs = d.fn(lf)
            if isinstance(exprs, pl.Expr):
                exprs = [exprs]
            lf = lf.with_columns(*exprs)
        return lf

    @classmethod
    def resolve(cls, frame: Any) -> pl.LazyFrame:
        """Return a lazy canonical frame with lazy schema validation only.

        This path never collects, so backend value-level checks, uniqueness and
        dataframe checks are not fully executed. Use it only for trusted internal
        canonical frames.
        """
        return super().validate(cls._apply_derivations(as_lazy(frame)))

    @classmethod
    def validate(cls, check_obj: Any, *args: Any, **kwargs: Any) -> pl.LazyFrame:
        """Validate at a public boundary, collecting exactly once."""
        eager = cls._apply_derivations(as_lazy(check_obj)).collect()
        return super().validate(eager, *args, **kwargs).lazy()
