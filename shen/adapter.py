from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

import pandas as pd
import polars as pl

from shen.domain.contracts.base import ShenFrame

ContractT = TypeVar("ContractT", bound=ShenFrame)


def _as_pandas(data: Any) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, pl.LazyFrame):
        return data.collect().to_pandas()
    if isinstance(data, pl.DataFrame):
        return data.to_pandas()
    to_pandas = getattr(data, "to_pandas", None)
    if callable(to_pandas):
        converted = to_pandas()
        return converted if isinstance(converted, pd.DataFrame) else pd.DataFrame(converted)
    return pd.DataFrame(data)


def _as_lazy_result(data: Any) -> pl.LazyFrame:
    if isinstance(data, pl.LazyFrame):
        return data
    if isinstance(data, pl.DataFrame):
        return data.lazy()
    if isinstance(data, pd.DataFrame):
        return pl.from_pandas(data, include_index=False).lazy()
    return pl.DataFrame(data).lazy()


@dataclass(frozen=True, slots=True)
class BaseAdapter(Generic[ContractT]):
    contract: type[ContractT]

    def prepare(self, data: Any) -> Any:
        return data

    def from_data(self, data: Any) -> pl.LazyFrame:
        frame = _as_pandas(self.prepare(data))
        return _as_lazy_result(self.contract.validate(frame))

    def from_records(self, records: Any, **options: Any) -> pl.LazyFrame:
        return self.from_data(pd.DataFrame.from_records(records, **options))

    def from_dict(self, data: dict[str, Any], **options: Any) -> pl.LazyFrame:
        return self.from_data(pd.DataFrame.from_dict(data, **options))

    def from_pandas(self, frame: pd.DataFrame) -> pl.LazyFrame:
        return self.from_data(frame)

    def from_polars(self, frame: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
        return self.from_data(frame)

    def from_arrow(self, table: Any) -> pl.LazyFrame:
        return self.from_data(pl.from_arrow(table))

    def from_csv(self, source: Any, **options: Any) -> pl.LazyFrame:
        return self.from_data(pl.read_csv(source, **options))

    def from_parquet(self, source: Any, **options: Any) -> pl.LazyFrame:
        return self.from_data(pl.read_parquet(source, **options))

    def from_json(self, source: Any, **options: Any) -> pl.LazyFrame:
        return self.from_data(pl.read_json(source, **options))

    def from_ndjson(self, source: Any, **options: Any) -> pl.LazyFrame:
        return self.from_data(pl.read_ndjson(source, **options))

    def from_excel(
        self,
        source: Any,
        *,
        sheet_id: int | None = None,
        sheet_name: str | None = None,
        table_name: str | None = None,
        engine: str = "calamine",
        **options: Any,
    ) -> pl.LazyFrame:
        selected = sum(value is not None for value in (sheet_id, sheet_name, table_name))
        if selected > 1:
            raise ValueError("select only one of sheet_id, sheet_name or table_name")
        result = pl.read_excel(
            source,
            sheet_id=sheet_id,
            sheet_name=sheet_name,
            table_name=table_name,
            engine=engine,
            **options,
        )
        if isinstance(result, dict):
            raise ValueError("from_excel requires exactly one sheet/table")
        return self.from_data(result)

    def from_file(self, source: str | Path, **options: Any) -> pl.LazyFrame:
        suffix = Path(source).suffix.lower()
        readers = {
            ".csv": self.from_csv,
            ".parquet": self.from_parquet,
            ".json": self.from_json,
            ".jsonl": self.from_ndjson,
            ".ndjson": self.from_ndjson,
            ".xlsx": self.from_excel,
            ".xls": self.from_excel,
        }
        try:
            reader = readers[suffix]
        except KeyError as error:
            raise ValueError(f"unsupported input suffix: {suffix!r}") from error
        return reader(source, **options)


__all__ = ["BaseAdapter"]
