# Concepts

Schenberg is a small lazy pricing DSL over symbolic expression trees that compile
to Polars.

## Terms and formula graphs

A `FormulaGraph` is a directed acyclic graph of named terms:

- **Input terms** are columns supplied by the trade frame after market binding.
- **Formula terms** are symbolic `Expr` nodes over inputs and earlier terms.
- **Views** are named output mappings from result columns to terms or inputs.

The preferred declaration style is the formula decorator:

```python
from schenberg import FormulaGraph
from schenberg.core.expr import exp

g = FormulaGraph("my_pricer", input=MyInput)

@g.formula(symbol="T")
def year_fraction(payment_days):
    return payment_days / 252.0

@g.formula(symbol="DF")
def discount_factor(risk_free_rate, year_fraction):
    return exp(-risk_free_rate * year_fraction)
```

The function signature is the dependency list, declared as **headless
parameters**: each argument resolves to a symbolic `var` — an already-defined
term first, otherwise an input-schema column (contract or pre-resolved market
data). Unknown parameters fail early with a clear error. The legacy namespace
names `c`, `contract`, `input`, and `inputs` still receive the whole input
namespace for backward compatibility.

`@g.formula` registers the returned `Expr` through the same infrastructure as
`g.let(...)`; it does not create Polars closures, Python UDFs, row-wise loops, or
collect data. `g.let(...)` remains available as the low-level primitive.

## Lazy interpretation

`graph.plan(frame, view="output")` returns a `pl.LazyFrame`. It validates that
required columns are present, materializes reachable terms lazily in dependency
order, then selects only the requested output view. It never calls `.collect()`.
Pass `materialize_terms=False` for the legacy recursive inlining path. Market
data is resolved before this boundary by `bind`, so a formula graph never reads a
`MarketSnapshot` directly.

`graph.stage(frame, view="output")` returns a lazy debug frame with intermediate
terms materialised in dependency order.

## Introspection

Because formulas are symbolic, Schenberg can inspect the same declaration in
multiple ways:

- `graph.formulas()` and `graph.formula_of("term")` render LaTeX from the IR.
- `graph.explain(view="output")` reports inputs, formulas, and returns.
- `graph.to_mermaid(math_labels=True)` draws dependencies.
- `graph.info(view="output")`, `graph.required_inputs(...)`, and
  `graph.topological_order()` expose structured graph metadata.

## Market data boundary

Use `market_role(...).read(...).by(...)` and `With[role]` mixins on an input
schema. Semantic helpers such as `CURVES.zero_rate("BRL_DI")` bind that argument
as a literal quote-side key; dynamic per-row keys remain available with `.by(...)`.
`bind(raw_trades, market_snapshot, InputSchema)` performs the joins and returns
an enriched lazy frame with market columns available as ordinary inputs. It fails
loudly if a market output would overwrite an existing non-key column. Fixing
rules, including custom date keys, also live at this boundary.

`MarketSnapshot.at(...).source(...).build()` does not validate `unique_by` by
default because uniqueness checks may collect market source data. Use
`build(validate=True)` or `market.validate()` at an explicit market-data boundary
when duplicate-key checking is required.

## Position boundary

Pricing graphs are pure instrument functions: no `side`, no `long_short`, no
pay/receive sign. Direction, quantity, reporting FX, and book aggregation belong
to `PositionView`, reusable position measures, and `Fold` rollups.

## Examples and HTML

Instrument-specific example pricers live in `docs/examples/*.qmd` as self-contained
Quarto notebooks that use the public Schenberg API. Render them with:

```bash
uv run poe examples-html
```

## Market semantics stop at `bind`

A formula graph is a pure symbolic program. It must not look up market data or
know whether a resolved input came from a curve, fixing table, or volatility
surface. Market semantics live in `MarketRole` declarations, including the light
semantic DSL:

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

enriched = bind(trades, market, VanillaOptionInput)
priced = option_graph.plan(enriched, view="output")
```

The graph receives ordinary columns such as `spot`, `strike`, `risk_free_rate`,
`vol`, and `time_to_maturity`. Public pricer functions should keep Pandera-style
boundary validation (`@price_function`, `check_input`, `check_output`, or
`check_io`) around schemas; graph planning remains lazy formula compilation.
