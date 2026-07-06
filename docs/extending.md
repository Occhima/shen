# Extending Schenberg

Keep extensions small, local, inspectable, and lazy.

## Add a new pricer

1. Declare `MarketRole`s for the market columns you need.
2. Build an input schema with `With[role]` mixins.
3. Build a `FormulaGraph` over those input columns.
4. Register formulas with `@g.formula(...)`.
5. Resolve market data with `bind` and run `graph.plan(...)`.

```python
import polars as pl
from schenberg import FormulaGraph, MarketSnapshot, With, bind, exp, market_role
from schenberg.domain.base import SchenbergDataFrameModel

MyRate = (
    market_role("my_rate")
    .read("my_curves", "rate")
    .by(indexer="id_indexador", payment_days="tenor_days")
)

class MyInput(With[MyRate], SchenbergDataFrameModel):
    instrument_id: str
    indexer: str
    payment_days: int
    strike: float

g = FormulaGraph("my_pricer", input=MyInput)

@g.formula(symbol="T")
def year_fraction(payment_days):
    return payment_days / 252.0

@g.formula(symbol="DF")
def discount_factor(my_rate, year_fraction):
    return exp(-my_rate * year_fraction)

@g.formula(symbol="FV")
def future_value(my_rate, strike):
    return my_rate - strike

@g.formula(symbol="PV")
def present_value(future_value, discount_factor):
    return future_value * discount_factor

g.returns("output", instrument_id="instrument_id", value="present_value")

def price_my_instrument(trades: pl.LazyFrame, market: MarketSnapshot) -> pl.LazyFrame:
    enriched = bind(trades, market, MyInput)
    return g.plan(enriched, view="output")
```

The decorator preserves the current symbolic IR: LaTeX, Mermaid, `explain`,
`stage`, required-input analysis, lazy Polars execution, and future derivative
support continue to work. Use `g.let(...)` only when you need the lower-level
primitive directly.

## Reuse a graph with another market source

Reuse the same formula by defining a different input schema whose market roles
publish the same semantic columns. Keep this wiring in your application or in an
example notebook; do not create a central module of example pricers.

## Add a market fixing

A `Fixing` derives a date join key from contract columns. The derived key is used
by `bind` before the graph runs:

```python
from schenberg import Fixing, market_role
from schenberg.market_data.date_rules import previous_business_days

PtaxFixing = (
    market_role("ptax")
    .read("ptax_fixings", "fixing_value")
    .by(currency_pair="currency_pair")
    .fixing("ptax_fixing_date", Fixing.rule(previous_business_days("tenor", n=5)))
)
```

`previous_business_days` uses a simplified Monday-Friday calendar. For production
calendars, provide a precomputed fixing date or a more specific date-rule helper.

## Add a position measure

Custom measures are small callables that register one term on a `PositionView`:

```python
from schenberg.position.view import Measure, PositionView
from schenberg.core.expr import Expr


def my_measure(*, name: str = "my_col") -> Measure:
    def register(view: PositionView) -> Expr:
        return view.let(name, view.col("exposure") * view.col("value") * 0.5)
    return Measure(register)
```

Position measures may use `side` and `quantity`; pure pricing graphs should not.

## HTML examples

Examples live under `docs/examples/*.qmd` and render with Quarto:

```bash
uv run poe examples-html
```

## Extending with market roles and pricer boundaries

When adding an instrument, keep the instrument example local (for example under
`docs/examples/`) and keep the public API small. Declare semantic market reads
outside the graph, bind them to the trade frame, then plan the graph:

```python
Spot = FIXINGS.value("USD/BRL", as_="spot").source("fixings")
Vol = (
    VOLS.implied("USD/BRL", as_="vol")
    .source("vol_surface")
    .for_expiry("expiry")
    .for_strike("strike")
)
RiskFree = CURVES.zero_rate("BRL_DI", as_="risk_free_rate").source("curves").for_tenor(
    "payment_days"
)

class VanillaOptionInput(With[Spot], With[Vol], With[RiskFree], SchenbergDataFrameModel):
    ...

@price_function
def price_option(trades, market):
    enriched = bind(trades, market, VanillaOptionInput)
    return option_graph.plan(enriched, view="output")
```

The graph should use only resolved columns. Position direction (`side`, book,
quantity) belongs in position or structure composition layers, while structures
are simply tables of already-priceable legs joined to `InstrumentValue`.
