from __future__ import annotations

import polars as pl

from shen.contracts.base import ShenFrame


class Trade(ShenFrame):
    trade_id: str
    contract_id: str
    trade_date: pl.Date
    book: str
    counterparty: str | None = None
    trader: str | None = None
    external_id: str | None = None
