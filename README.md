# Shen

Shen is a small, lazy, vectorized pricing library for tabular financial contracts.
The public namespace is:

```python
import shen
```

The architecture is intentionally explicit:

```text
PricingTerms + Market -> UnitValue
Trade -> Position
Position x UnitValue -> MTM
```

## Concepts

- **Product**: a family such as NDF, DI Future, WDO Future, Commodity Forward,
  Vanilla Option, Bullet Swap.
- **PricingTerms**: economic terms that change payoff. They are Pandera/Polars
  table contracts keyed by `contract_id` and stable `product_type` values.
- **Trade**: booking metadata (`trade_id`, `book`, `counterparty`, `trader`).
- **Position**: signed exposure (`contract_id`, `qty`). Pricers return unit values.
- **Market**: immutable valuation-date market frames: curves, FX references, fixings,
  volatility surfaces, scenarios and provenance.
- **Pricer**: a callable object that combines terms schema, declarative lookups,
  derived columns, a pure mathematical calculator and typed output metadata.

## NDF example

```python
import datetime as dt
import polars as pl
import shen
from shen.products.fx.ndf import ndf

valuation_date = dt.date(2026, 7, 3)

terms = pl.DataFrame({
    "contract_id": ["NDF1"],
    "product_type": ["ndf"],
    "base_currency": ["USD"],
    "quote_currency": ["BRL"],
    "strike": [5.00],
    "fixing_date": [dt.date(2026, 7, 31)],
    "settlement_date": [dt.date(2026, 8, 3)],
    "reference_id": ["PTAX"],
    "settlement_curve": ["BRL"],
    "settlement_currency": ["BRL"],
}).lazy()

market = shen.Market.load({
    shen.FxReference: pl.DataFrame({
        "reference_id": ["PTAX"],
        "fixing_date": [dt.date(2026, 7, 31)],
        "rate": [5.1485],
        "status": ["projected"],
        "source": ["internal"],
        "observed_at": [dt.datetime(2026, 7, 3, 12)],
    }),
    shen.Curve: pl.DataFrame({
        "curve_id": ["BRL", "BRL"],
        "pillar_date": [valuation_date, dt.date(2026, 8, 3)],
        "log_df": [0.0, 0.0],
    }),
}, valuation_date=valuation_date)

unit_value = ndf.price(terms, market)
position = shen.Position.validate(pl.DataFrame({
    "contract_id": ["NDF1"],
    "qty": [1_000_000.0],
}))
valuation = shen.mtm(position, unit_value)
```

For the data above, the unit value is `0.1485 BRL/USD` and the MTM is
`148_500 BRL` for a `1_000_000 USD` long position.

## Scenarios

Market objects can carry `scenario_id`. The base scenario is always `"base"`.
Scenario identifiers are preserved in pricing and MTM outputs; they are never
implemented as Python loops.

## Explicit universe

Book pricing uses an explicit immutable universe:

```python
values = shen.price(terms, market, universe=shen.STANDARD)
```

Individual pricers remain ergonomic:

```python
values = ndf.price(terms, market)
```

## Development checks

```bash
ruff check .
ruff format --check .
ty check shen
pytest
python -m build
```
